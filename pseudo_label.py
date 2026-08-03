import argparse
import json
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
from PIL import Image

from src.utils.autocrop import Autocropper
from src.utils.common import simple_logger
from src.utils.evaluate import compute_iou
from src.utils.interface import Detector
from src.utils.postprocessing import rails_to_mask
from src.utils.pseudo_labeling import DEFAULT_THRESHOLDS, PseudoLabeler
from src.utils.visualization import draw_egopath

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Ego-Path Detection Pseudo-Labeling Script"
    )
    parser.add_argument(
        "model",
        type=str,
        help="Name of the trained model to use as teacher (e.g., 'twinkling-rocket-21').",
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the directory containing the images to pseudo-label.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the destination annotations file. If not specified, the annotations are saved as 'pseudo_egopath.json' in the input directory.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to the destination file for the per-image confidence report. If not specified, no report is saved.",
    )
    parser.add_argument(
        "--visualize",
        type=str,
        default=None,
        help="Path to the destination directory for the visual outputs (prefixed with their acceptance status). If not specified, no visual output is saved.",
    )
    parser.add_argument(
        "--ensemble",
        type=str,
        nargs="+",
        default=None,
        help="Names of additional trained models whose agreement with the teacher is used as an extra confidence signal (e.g., 'chromatic-laughter-5 fortuitous-goat-12').",
    )
    parser.add_argument(
        "--calibrate",
        type=str,
        default=None,
        help="Path to an annotations file to run in calibration mode: instead of generating pseudo-labels, reports how the confidence metrics relate to the IoU actually achieved on these annotated images.",
    )
    parser.add_argument(
        "--crop",
        type=str,
        default="none",
        help="Coordinates to use for cropping the input images ('auto' for automatic cropping, only relevant for consecutive frames of a same video, 'x_left,y_top,x_right,y_bottom' inclusive absolute coordinates for manual cropping, or 'none' to disable cropping).",
    )
    parser.add_argument(
        "--no-flip-tta",
        action="store_true",
        help="If enabled, disables the flipping consistency check (halves the inference cost, at the price of the most informative confidence signal).",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=64,
        help="Maximum number of points per rail in the generated annotations.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If enabled, searches the input directory recursively.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images to process. If not specified, processes all of them.",
    )
    parser.add_argument(
        "--keep-rejected",
        action="store_true",
        help="If enabled, writes the rejected pseudo-labels to the annotations file as well (they remain flagged in the report).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda", "mps"]
        + [f"cuda:{x}" for x in range(torch.cuda.device_count())],
        help="Device to use ('cpu', 'cuda', 'cuda:x' or 'mps').",
    )
    for name, value in DEFAULT_THRESHOLDS.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=float,
            default=value,
            help=f"Acceptance threshold '{name}' (default: {value}). Set to 'nan' to disable this criterion.",
        )
    return parser.parse_args()


def list_images(input_path, recursive, limit):
    """Lists the images of a directory, as paths relative to it.

    Args:
        input_path (str): Path to the directory containing the images.
        recursive (bool): Whether to search the directory recursively.
        limit (int or None): Maximum number of images to return.

    Returns:
        list: Sorted list of image paths, relative to the input directory.
    """
    if recursive:
        images = [
            os.path.relpath(os.path.join(root, file), input_path)
            for root, _, files in os.walk(input_path)
            for file in files
            if file.lower().endswith(IMAGE_EXTENSIONS)
        ]
    else:
        images = [
            file
            for file in os.listdir(input_path)
            if file.lower().endswith(IMAGE_EXTENSIONS)
        ]
    images = sorted(images)
    return images[:limit] if limit is not None else images


def correlation(x, y):
    """Computes the Pearson and Spearman correlation coefficients between two samples.

    Args:
        x (list): First sample.
        y (list): Second sample.

    Returns:
        tuple: Pearson and Spearman correlation coefficients (nan if undefined).
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    ranks = [np.argsort(np.argsort(sample)).astype(float) for sample in (x, y)]
    spearman = (
        float("nan")
        if np.std(ranks[0]) == 0 or np.std(ranks[1]) == 0
        else float(np.corrcoef(*ranks)[0, 1])
    )
    return float(np.corrcoef(x, y)[0, 1]), spearman


def report_calibration(records, logger):
    """Reports how the confidence metrics relate to the IoU achieved on annotated images.

    For each metric, the correlation with the true IoU is reported, along with a sweep
    over candidate thresholds showing the resulting kept ratio and the mean IoU of the
    kept and dropped images. A useful criterion keeps most of the images while dropping
    those with a clearly lower IoU.

    Args:
        records (list): List of per-image records containing their metrics and true IoU.
        logger (logging.Logger): Logger to print the report with.
    """
    ious = [record["iou"] for record in records]
    logger.info(
        f"\nCalibration on {len(records)} annotated images"
        + f" | mean IoU: {np.mean(ious):.4f}"
        + f" | worst decile IoU: {np.percentile(ious, 10):.4f}"
    )
    metric_names = sorted({name for record in records for name in record["metrics"]})
    for name in metric_names:
        pairs = [
            (record["metrics"][name], record["iou"])
            for record in records
            if record["metrics"].get(name) is not None
        ]
        if len(pairs) < 2:
            continue
        values, matching_ious = zip(*pairs)
        pearson, spearman = correlation(values, matching_ious)
        logger.info(
            f"\n{name} ({len(values)} images)"
            + f" | pearson: {pearson:+.3f} | spearman: {spearman:+.3f}"
        )
        logger.info(
            f"{'threshold':>12} {'kept':>8} {'IoU kept':>10} {'IoU dropped':>12}"
        )
        # a higher value means a higher confidence, except for these "lower is better" metrics
        descending = name in ("uncertain_ratio", "bottom_gap", "jitter")
        percentiles = [50, 30, 20, 10, 5] if descending else [5, 10, 20, 30, 50]
        for percentile in percentiles:
            threshold = float(
                np.percentile(values, 100 - percentile if descending else percentile)
            )
            passing = [
                (value <= threshold if descending else value >= threshold)
                for value, _ in pairs
            ]
            kept = [iou for (_, iou), keep in zip(pairs, passing) if keep]
            dropped = [iou for (_, iou), keep in zip(pairs, passing) if not keep]
            logger.info(
                f"{threshold:>12.4f} {len(kept) / len(pairs):>7.1%}"
                + f" {np.mean(kept) if kept else float('nan'):>10.4f}"
                + f" {np.mean(dropped) if dropped else float('nan'):>12.4f}"
            )


def main(args):
    logger = simple_logger(__name__, "info")
    base_path = os.path.dirname(__file__)

    if args.crop == "auto":
        crop_coords = "auto"
    elif args.crop == "none":
        crop_coords = None
    else:
        crop_coords = tuple(map(int, args.crop.split(",")))

    thresholds = {
        name: getattr(args, name)
        for name in DEFAULT_THRESHOLDS
        if not np.isnan(getattr(args, name))
    }

    detector = Detector(
        model_path=os.path.join(base_path, "weights", args.model),
        crop_coords=None,  # cropping is handled by the pseudo-labeler
        runtime="pytorch",
        device=args.device,
    )
    ensemble = [
        Detector(
            model_path=os.path.join(base_path, "weights", name),
            crop_coords=None,
            runtime="pytorch",
            device=args.device,
        )
        for name in (args.ensemble or [])
    ]
    labeler = PseudoLabeler(
        detector=detector,
        ensemble=ensemble,
        thresholds=thresholds,
        num_points=args.num_points,
        flip_tta=not args.no_flip_tta,
    )
    # the autocropper is driven here so that both inference passes of an image share
    # the same coordinates, which the flipping consistency check requires
    autocropper = Autocropper(detector.config) if crop_coords == "auto" else None

    ground_truth = None
    if args.calibrate is not None:
        with open(args.calibrate) as f:
            ground_truth = json.load(f)

    images = list_images(args.input, args.recursive, args.limit)
    if ground_truth is not None:
        images = [image for image in images if image in ground_truth]
    if not images:
        raise ValueError(f"No image to process in {args.input}")

    output_path = args.output or os.path.join(args.input, "pseudo_egopath.json")
    if args.visualize is not None:
        os.makedirs(args.visualize, exist_ok=True)

    logger.info(
        f"\nPseudo-labeling {len(images)} images with {args.model}"
        + (f" (ensemble: {', '.join(args.ensemble)})" if args.ensemble else "")
        + (" [calibration mode]" if ground_truth is not None else "")
        + "..."
    )
    progress_bar = simple_logger(f"{__name__}_progress", "info", terminator="\r")

    annotations = {}
    records = []
    for i, name in enumerate(images):
        img = Image.open(os.path.join(args.input, name)).convert("RGB")
        coords = autocropper() if autocropper is not None else crop_coords
        label = labeler.label(img, coords)
        if autocropper is not None:
            autocropper.update(img.size, label["rails"] or label["mask"])

        if label["rails"] is not None and (label["accepted"] or args.keep_rejected):
            annotations[name] = {
                "left_rail": label["rails"][0],
                "right_rail": label["rails"][1],
            }
        record = {
            "name": name,
            "accepted": label["accepted"],
            "reasons": label["reasons"],
            "metrics": label["metrics"],
        }
        if ground_truth is not None:
            annotation = ground_truth[name]
            target = rails_to_mask(
                [annotation["left_rail"], annotation["right_rail"]], img.size
            )
            record["iou"] = compute_iou(label["mask"], target)
        records.append(record)

        if args.visualize is not None:
            status = "accepted" if label["accepted"] else "rejected"
            vis = draw_egopath(img, label["rails"] or label["mask"])
            vis.save(os.path.join(args.visualize, f"{status}_{os.path.basename(name)}"))

        progress_bar.info(
            f"Processed {i + 1:0{len(str(len(images)))}}/{len(images)} images"
            + f" ({(i + 1) / len(images) * 100:.2f}%)"
        )
    logger.info("")

    accepted = [record for record in records if record["accepted"]]
    reasons = {}
    for record in records:
        for reason in record["reasons"]:
            reasons[reason] = reasons.get(reason, 0) + 1
    logger.info(
        f"\nAccepted {len(accepted)}/{len(records)} pseudo-labels"
        + f" ({len(accepted) / len(records) * 100:.2f}%)"
    )
    for reason, count in sorted(reasons.items(), key=lambda item: -item[1]):
        logger.info(f"  rejected on {reason}: {count}")

    if ground_truth is not None:
        report_calibration(records, logger)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(annotations, f)
        logger.info(
            f"\nAnnotations of {len(annotations)} images saved to {output_path}"
        )

    if args.report is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(
                {
                    "teacher": args.model,
                    "ensemble": args.ensemble or [],
                    "thresholds": thresholds,
                    "accepted": len(accepted),
                    "total": len(records),
                    "rejection_reasons": reasons,
                    "images": records,
                },
                f,
                indent=2,
            )
        logger.info(f"Confidence report saved to {args.report}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
