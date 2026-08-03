import numpy as np
from PIL import Image, ImageOps

from .evaluate import compute_iou
from .postprocessing import (
    classifications_to_rails,
    rails_to_mask,
    regression_to_rails,
    scale_mask,
    scale_rails,
)

# Default acceptance thresholds. They are deliberately permissive on the geometric
# side and strict on the consistency side, and should be recalibrated on annotated
# images (see the calibration mode of pseudo_label.py) before a large generation run.
DEFAULT_THRESHOLDS = {
    "min_prob_confidence": 0.90,  # mean max(p, 1-p) of the predicted probabilities
    "max_uncertain_ratio": 0.10,  # ratio of pixels with an ambiguous probability
    "min_flip_iou": 0.85,  # agreement between the prediction and its mirrored one
    "min_ensemble_iou": 0.85,  # agreement between the teacher and the extra models
    "max_bottom_gap": 0.02,  # distance between the path start and the image bottom
    "min_height_ratio": 0.20,  # vertical extent of the path
    "min_width_monotonicity": 0.80,  # ratio of rows where the path does not widen
    "max_jitter": 0.01,  # RMS deviation of the path center from a quadratic fit
}


def softmax(x, axis=-1):
    """Computes the softmax of an array along the given axis.

    Args:
        x (numpy.ndarray): Input array.
        axis (int, optional): Axis along which the softmax is computed. Defaults to -1.

    Returns:
        numpy.ndarray: Softmax of the input array.
    """
    shifted = x - np.max(x, axis=axis, keepdims=True)  # for numerical stability
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def mask_to_rows(mask, min_width=3):
    """Extracts the ego-path boundaries of a binary mask, row by row, from the bottom.

    The scan stops as soon as the path becomes empty, too narrow or discontinuous, so
    that a fragmented mask does not produce a path spanning unrelated regions.

    Args:
        mask (PIL.Image.Image or numpy.ndarray): Binary mask of the ego-path region.
        min_width (int, optional): Minimum width (in pixels) of a valid row. Defaults to 3.

    Returns:
        numpy.ndarray or None: Array of (y, xleft, xright) rows sorted from bottom to
            top, or None if fewer than two valid rows were found.
    """
    mask = np.array(mask) if isinstance(mask, Image.Image) else mask
    mask = mask.astype(bool)
    rows = []
    previous = None
    for y in range(mask.shape[0] - 1, -1, -1):
        cols = np.nonzero(mask[y, :])[0]
        if cols.size == 0:
            if not rows:
                continue  # the path does not necessarily start on the last row
            break
        runs = np.split(cols, np.where(np.diff(cols) > 1)[0] + 1)  # contiguous segments
        run = max(runs, key=len)  # keep the widest one to discard isolated noise
        xleft, xright = int(run[0]), int(run[-1])
        if xright - xleft + 1 < min_width:
            if not rows:
                continue
            break
        if previous is not None and (xright < previous[0] or xleft > previous[1]):
            break  # no overlap with the previous row, the path is discontinuous
        rows.append((y, xleft, xright))
        previous = (xleft, xright)
    return np.array(rows) if len(rows) >= 2 else None


def rails_to_rows(rails):
    """Converts left and right rails lists of points to an array of rows.

    Args:
        rails (list): List containing the left and right rails lists of point coordinates (x, y).

    Returns:
        numpy.ndarray or None: Array of (y, xleft, xright) rows, or None if fewer than two points.
    """
    left_rail, right_rail = rails
    if len(left_rail) < 2 or len(left_rail) != len(right_rail):
        return None
    rows = [[ly, lx, rx] for (lx, ly), (rx, _) in zip(left_rail, right_rail)]
    return np.array(rows)


def rows_to_rails(rows, num_points=64):
    """Converts an array of rows to left and right rails lists of points.

    Args:
        rows (numpy.ndarray): Array of (y, xleft, xright) rows sorted from bottom to top.
        num_points (int, optional): Maximum number of points per rail. Defaults to 64.

    Returns:
        list: List containing the left and right rails lists of point coordinates (x, y).
    """
    indices = np.linspace(0, len(rows) - 1, min(num_points, len(rows)))
    indices = np.unique(np.round(indices).astype(int))  # always keeps both extremities
    selected = rows[indices]
    left_rail = [[int(xleft), int(y)] for y, xleft, _ in selected]
    right_rail = [[int(xright), int(y)] for y, _, xright in selected]
    return [left_rail, right_rail]


def extend_rows_to_bottom(rows, height):
    """Extends the path vertically down to the last row of the image.

    Training requires the annotated path to reach the bottom row of the image, which a
    segmentation mask does not always do. The bottom-most boundaries are therefore
    repeated on the last row, which is only sound for a gap of a few pixels (the caller
    is responsible for rejecting larger ones).

    Args:
        rows (numpy.ndarray): Array of (y, xleft, xright) rows sorted from bottom to top.
        height (int): Height of the image.

    Returns:
        numpy.ndarray: Array of rows starting on the last row of the image.
    """
    if rows[0, 0] >= height - 1:
        return rows
    bottom = np.array([[height - 1, rows[0, 1], rows[0, 2]]])
    return np.concatenate((bottom, rows), axis=0)


def geometry_metrics(rows, img_shape):
    """Computes the geometric plausibility metrics of a detected ego-path.

    Args:
        rows (numpy.ndarray): Array of (y, xleft, xright) rows sorted from bottom to top.
        img_shape (tuple): Shape (W, H) of the image.

    Returns:
        dict: Geometric metrics of the path.
    """
    width, height = img_shape
    ys, xlefts, xrights = rows[:, 0], rows[:, 1], rows[:, 2]
    widths = xrights - xlefts
    centers = (xrights + xlefts) / 2
    if len(rows) >= 2:  # rows go up, so the path should not widen with distance
        tolerance = max(1, 0.005 * width)
        monotonicity = float(np.mean(np.diff(widths) <= tolerance))
    else:
        monotonicity = 1.0
    if len(rows) >= 4:  # deviation from a quadratic fit, independent of the sampling
        ys_normalized = (ys - ys.min()) / max(ys.max() - ys.min(), 1)
        residuals = centers - np.polyval(
            np.polyfit(ys_normalized, centers, 2), ys_normalized
        )
        jitter = float(np.sqrt(np.mean(residuals**2)) / width)
    else:
        jitter = 0.0
    return {
        "bottom_gap": float((height - 1 - ys.max()) / height),
        "height_ratio": float((ys.max() - ys.min()) / max(height - 1, 1)),
        "min_width": float(widths.min() / width),
        "width_monotonicity": monotonicity,
        "jitter": jitter,
    }


def check_confidence(metrics, thresholds):
    """Lists the confidence criteria that a pseudo-label fails to meet.

    Metrics that are not available for the teacher method (e.g. the probabilistic
    confidence of a regression model) are skipped rather than counted as failures.

    Args:
        metrics (dict): Metrics of the pseudo-label.
        thresholds (dict): Acceptance thresholds (see DEFAULT_THRESHOLDS).

    Returns:
        list: Names of the failed criteria (empty if the pseudo-label is accepted).
    """
    comparisons = [
        ("min_prob_confidence", "prob_confidence", np.less),
        ("max_uncertain_ratio", "uncertain_ratio", np.greater),
        ("min_flip_iou", "flip_iou", np.less),
        ("min_ensemble_iou", "ensemble_iou", np.less),
        ("max_bottom_gap", "bottom_gap", np.greater),
        ("min_height_ratio", "height_ratio", np.less),
        ("min_width_monotonicity", "width_monotonicity", np.less),
        ("max_jitter", "jitter", np.greater),
    ]
    failed = []
    for threshold_name, metric_name, failing in comparisons:
        threshold = thresholds.get(threshold_name)
        metric = metrics.get(metric_name)
        if threshold is None or metric is None:
            continue
        if failing(metric, threshold):
            failed.append(metric_name)
    return failed


class PseudoLabeler:
    def __init__(
        self,
        detector,
        ensemble=None,
        thresholds=None,
        num_points=64,
        min_mask_width=3,
        flip_tta=True,
        bottom_tolerance=0.02,
    ):
        """Generates ego-path pseudo-labels for unannotated images with a teacher model.

        Confidence is estimated without ground truth, by combining the probabilistic
        output of the teacher, its consistency under horizontal flipping (the training
        pipeline makes the model equivariant to it), the geometric plausibility of the
        path and, optionally, the agreement with additional models.

        Args:
            detector (Detector): Teacher model used to produce the pseudo-labels.
            ensemble (list, optional): Additional detectors used to measure agreement. Defaults to None.
            thresholds (dict, optional): Acceptance thresholds. Defaults to DEFAULT_THRESHOLDS.
            num_points (int, optional): Maximum number of points per rail. Defaults to 64.
            min_mask_width (int, optional): Minimum width (in pixels) of a valid mask row. Defaults to 3.
            flip_tta (bool, optional): Whether to measure the flipping consistency. Defaults to True.
            bottom_tolerance (float, optional): Maximum gap to the image bottom that is extended instead of rejected. Defaults to 0.02.
        """
        self.detector = detector
        self.ensemble = ensemble if ensemble is not None else []
        self.thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
        self.num_points = num_points
        self.min_mask_width = min_mask_width
        self.flip_tta = flip_tta
        self.bottom_tolerance = bottom_tolerance

    def predict(self, detector, img, crop_coords):
        """Infers a detector on an image, keeping the raw model output for confidence estimation.

        Args:
            detector (Detector): Detector to infer.
            img (PIL.Image.Image): Input image.
            crop_coords (tuple or None): Inclusive absolute coordinates (xleft, ytop, xright, ybottom) of the cropped region.

        Returns:
            dict: Prediction with its mask, its rows and its probabilistic metrics.
        """
        original_shape = img.size
        if crop_coords is not None:
            xleft, ytop, xright, ybottom = crop_coords
            img = img.crop((xleft, ytop, xright + 1, ybottom + 1))
        if detector.runtime == "pytorch":
            pred = detector.infer_model_pytorch(img)
        else:
            pred = detector.infer_model_tensorrt(img)
        config = detector.config

        if config["method"] == "segmentation":
            logits = pred.squeeze(0).squeeze(0)
            probs = 1 / (1 + np.exp(-logits))  # sigmoid
            prob_confidence = float(np.mean(np.maximum(probs, 1 - probs)))
            uncertain_ratio = float(np.mean((probs > 0.1) & (probs < 0.9)))
            mask = Image.fromarray((logits > 0).astype(np.uint8) * 255)
            mask = scale_mask(mask, crop_coords, original_shape)
            rows = mask_to_rows(mask, self.min_mask_width)
        else:
            if config["method"] == "classification":
                logits = pred.reshape(2, config["anchors"], config["classes"] + 1)
                probs = softmax(logits, axis=2)
                prob_confidence = float(np.mean(np.max(probs, axis=2)))
                uncertain_ratio = None
                rails = classifications_to_rails(
                    np.argmax(logits, axis=2), config["classes"]
                )
            else:  # regression, whose output is not probabilistic
                prob_confidence = None
                uncertain_ratio = None
                traj = pred[:, :-1].reshape(2, config["anchors"])
                ylim = 1 / (1 + np.exp(-pred[:, -1].item()))  # sigmoid
                rails = regression_to_rails(traj, ylim)
            rails = scale_rails(rails, crop_coords, original_shape)
            rails = np.round(rails).astype(int).tolist()
            rows = rails_to_rows(rails)
            mask = (
                rails_to_mask(rails, original_shape)
                if rows is not None
                else Image.new("L", original_shape, 0)
            )

        return {
            "mask": mask,
            "rows": rows,
            "prob_confidence": prob_confidence,
            "uncertain_ratio": uncertain_ratio,
        }

    def label(self, img, crop_coords=None):
        """Generates the ego-path pseudo-label of an image and estimates its confidence.

        Args:
            img (PIL.Image.Image): Input image.
            crop_coords (tuple or None, optional): Inclusive absolute coordinates (xleft, ytop, xright, ybottom) of the cropped region. Defaults to None.

        Returns:
            dict: Pseudo-label with its rails, its mask, its metrics and its acceptance status.
        """
        prediction = self.predict(self.detector, img, crop_coords)
        metrics = {
            "prob_confidence": prediction["prob_confidence"],
            "uncertain_ratio": prediction["uncertain_ratio"],
        }
        rows = prediction["rows"]
        if rows is None:
            return {
                "rails": None,
                "mask": prediction["mask"],
                "metrics": metrics,
                "accepted": False,
                "reasons": ["no_path"],
            }

        metrics.update(geometry_metrics(rows, img.size))

        if self.flip_tta:
            flipped_crop = crop_coords
            if crop_coords is not None:
                xleft, ytop, xright, ybottom = crop_coords
                width = img.size[0]
                flipped_crop = (width - 1 - xright, ytop, width - 1 - xleft, ybottom)
            flipped = self.predict(self.detector, ImageOps.mirror(img), flipped_crop)
            metrics["flip_iou"] = compute_iou(
                prediction["mask"], ImageOps.mirror(flipped["mask"])
            )

        if self.ensemble:
            ious = [
                compute_iou(
                    prediction["mask"], self.predict(model, img, crop_coords)["mask"]
                )
                for model in self.ensemble
            ]
            metrics["ensemble_iou"] = float(np.mean(ious))

        # the path has to reach the bottom row of the image to be usable for training
        if 0 < metrics["bottom_gap"] <= self.bottom_tolerance:
            rows = extend_rows_to_bottom(rows, img.size[1])

        reasons = check_confidence(metrics, self.thresholds)
        return {
            "rails": rows_to_rails(rows, self.num_points),
            "mask": prediction["mask"],
            "metrics": metrics,
            "accepted": len(reasons) == 0,
            "reasons": reasons,
        }
