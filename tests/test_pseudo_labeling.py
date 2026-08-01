import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.utils.pseudo_labeling import (
    PseudoLabeler,
    check_confidence,
    extend_rows_to_bottom,
    geometry_metrics,
    mask_to_rows,
    rails_to_rows,
    rows_to_rails,
    softmax,
)

IMG_WIDTH = 240
IMG_HEIGHT = 135
PATH_TOP = 50


def make_mask(top=PATH_TOP, bottom=IMG_HEIGHT - 1, base=(70, 170), apex=(112, 128), extra=None):
    """Builds a trapezoidal ego-path mask, as a segmentation model would output."""
    mask = Image.new("L", (IMG_WIDTH, IMG_HEIGHT), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(
        [(base[0], bottom), (apex[0], top), (apex[1], top), (base[1], bottom)], fill=255
    )
    if extra is not None:
        draw.rectangle(extra, fill=255)
    return mask


class TestMaskToRows:
    def test_extracts_the_path_from_the_bottom_up(self):
        rows = mask_to_rows(make_mask())
        assert rows is not None
        assert rows[0, 0] == IMG_HEIGHT - 1  # the scan starts on the last row
        assert np.all(np.diff(rows[:, 0]) < 0)  # rows go bottom to top
        assert np.all(rows[:, 1] < rows[:, 2])  # left boundary is left of the right one

    def test_path_narrows_with_distance(self):
        rows = mask_to_rows(make_mask())
        widths = rows[:, 2] - rows[:, 1]
        assert np.all(np.diff(widths) <= 0)

    def test_isolated_noise_is_ignored(self):
        # a blob on the side must not widen the extracted path
        noisy = mask_to_rows(make_mask(extra=(10, 100, 30, 120)))
        np.testing.assert_array_equal(noisy, mask_to_rows(make_mask()))

    def test_scan_stops_at_a_discontinuity(self):
        split = np.array(make_mask())
        split[80:88, :] = 0  # horizontal cut across the path
        rows = mask_to_rows(split)
        assert rows is not None
        assert rows[:, 0].min() == 88

    def test_path_not_reaching_the_bottom_is_still_extracted(self):
        # the gap is reported by geometry_metrics rather than aborting the scan
        rows = mask_to_rows(make_mask(bottom=IMG_HEIGHT - 6))
        assert rows is not None
        assert rows[0, 0] == IMG_HEIGHT - 6

    def test_empty_mask_yields_no_path(self):
        assert mask_to_rows(np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)) is None

    def test_single_row_yields_no_path(self):
        assert mask_to_rows(make_mask(top=IMG_HEIGHT - 1)) is None


class TestRowsAndRails:
    def test_point_count_is_capped(self):
        left, right = rows_to_rails(mask_to_rows(make_mask()), num_points=16)
        assert len(left) == len(right) == 16

    def test_extremities_are_always_kept(self):
        rows = mask_to_rows(make_mask())
        left, _ = rows_to_rails(rows, num_points=4)
        assert left[0][1] == rows[0, 0]
        assert left[-1][1] == rows[-1, 0]

    def test_rails_share_their_y_coordinates(self):
        left, right = rows_to_rails(mask_to_rows(make_mask()))
        assert [p[1] for p in left] == [p[1] for p in right]

    def test_points_are_plain_ints(self):
        # the annotations are serialized to JSON, which numpy integers break
        left, right = rows_to_rails(mask_to_rows(make_mask()))
        assert all(isinstance(v, int) for point in left + right for v in point)
        json.dumps({"left_rail": left, "right_rail": right})

    def test_round_trip_through_rails(self):
        rows = mask_to_rows(make_mask())
        np.testing.assert_array_equal(rails_to_rows(rows_to_rails(rows, len(rows))), rows)

    @pytest.mark.parametrize(
        "rails",
        [
            [[[0, 1]], [[2, 1]]],  # a single point per rail
            [[[0, 1], [0, 2]], [[2, 1]]],  # mismatched lengths
        ],
    )
    def test_unusable_rails_are_rejected(self, rails):
        assert rails_to_rows(rails) is None


class TestExtendRowsToBottom:
    def test_extension_reaches_the_last_row(self):
        rows = mask_to_rows(make_mask(bottom=IMG_HEIGHT - 6))
        extended = extend_rows_to_bottom(rows, IMG_HEIGHT)
        assert extended[0, 0] == IMG_HEIGHT - 1
        assert len(extended) == len(rows) + 1

    def test_extension_repeats_the_bottom_boundaries(self):
        rows = mask_to_rows(make_mask(bottom=IMG_HEIGHT - 6))
        extended = extend_rows_to_bottom(rows, IMG_HEIGHT)
        np.testing.assert_array_equal(extended[0, 1:], rows[0, 1:])

    def test_no_op_when_already_at_the_bottom(self):
        rows = mask_to_rows(make_mask())
        np.testing.assert_array_equal(extend_rows_to_bottom(rows, IMG_HEIGHT), rows)


class TestGeometryMetrics:
    def test_metrics_of_a_clean_path(self):
        metrics = geometry_metrics(mask_to_rows(make_mask()), (IMG_WIDTH, IMG_HEIGHT))
        assert metrics["bottom_gap"] == 0
        assert metrics["width_monotonicity"] == 1.0
        assert metrics["jitter"] < 1e-3  # a straight path does not wander
        expected_height = (IMG_HEIGHT - 1 - PATH_TOP) / (IMG_HEIGHT - 1)
        assert abs(metrics["height_ratio"] - expected_height) < 0.05

    def test_gap_to_the_bottom_is_measured(self):
        rows = mask_to_rows(make_mask(bottom=IMG_HEIGHT - 6))
        assert geometry_metrics(rows, (IMG_WIDTH, IMG_HEIGHT))["bottom_gap"] > 0

    def test_short_path_has_a_small_height_ratio(self):
        rows = mask_to_rows(make_mask(top=IMG_HEIGHT - 20))
        assert geometry_metrics(rows, (IMG_WIDTH, IMG_HEIGHT))["height_ratio"] < 0.2


class TestCheckConfidence:
    THRESHOLDS = {"min_flip_iou": 0.85, "max_jitter": 0.01, "min_prob_confidence": 0.9}

    def test_all_criteria_met(self):
        metrics = {"flip_iou": 0.9, "jitter": 0.001, "prob_confidence": 0.95}
        assert check_confidence(metrics, self.THRESHOLDS) == []

    def test_minimum_and_maximum_criteria_are_caught(self):
        assert check_confidence({"flip_iou": 0.5}, self.THRESHOLDS) == ["flip_iou"]
        assert check_confidence({"jitter": 0.5}, self.THRESHOLDS) == ["jitter"]

    def test_every_failure_is_listed(self):
        metrics = {"flip_iou": 0.1, "jitter": 0.5}
        assert sorted(check_confidence(metrics, self.THRESHOLDS)) == ["flip_iou", "jitter"]

    def test_unavailable_metrics_are_skipped(self):
        # a regression teacher exposes no probabilistic confidence
        metrics = {"prob_confidence": None, "flip_iou": 0.9}
        assert check_confidence(metrics, self.THRESHOLDS) == []

    def test_unset_thresholds_are_skipped(self):
        assert check_confidence({"flip_iou": 0.1}, {}) == []


class TestSoftmax:
    def test_rows_sum_to_one(self):
        probs = softmax(np.array([[1.0, 2.0, 3.0]]), axis=1)
        np.testing.assert_allclose(probs.sum(axis=1), 1)

    def test_stable_on_large_logits(self):
        probs = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]]), axis=1)
        np.testing.assert_allclose(probs[0], probs[1])


class StubDetector:
    """Segmentation teacher reading its output from the green channel of the image."""

    def __init__(self, shift=0, temperature=10.0):
        self.runtime = "pytorch"
        self.shift = shift  # breaks the equivariance to horizontal flipping
        self.temperature = temperature  # lowers the confidence of the probabilities
        self.config = {"method": "segmentation", "input_shape": [3, 64, 64]}

    def infer_model_pytorch(self, img):
        resized = np.array(img.resize((64, 64), Image.BILINEAR)).astype(np.float32)
        logits = (resized[:, :, 1] - 128) / 128 * self.temperature
        if self.shift:
            logits = np.roll(logits, self.shift, axis=1)
        return logits[None, None, :, :]


def make_scene(top=PATH_TOP, confidence=1.0):
    """Builds an image whose ego-path region is painted green."""
    img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), (40, 40, 40))
    draw = ImageDraw.Draw(img)
    draw.polygon(
        [(70, IMG_HEIGHT - 1), (112, top), (128, top), (170, IMG_HEIGHT - 1)],
        fill=(0, int(255 * confidence), 0),
    )
    return img


class TestPseudoLabeler:
    def test_confident_prediction_is_accepted(self):
        label = PseudoLabeler(StubDetector()).label(make_scene())
        assert label["accepted"], label["reasons"]
        assert label["rails"][0][0][1] == IMG_HEIGHT - 1  # path starts at the bottom
        assert label["metrics"]["flip_iou"] > 0.95
        assert label["metrics"]["prob_confidence"] > 0.95

    def test_ambiguous_probabilities_are_rejected(self):
        label = PseudoLabeler(StubDetector(temperature=0.15)).label(make_scene(confidence=0.6))
        assert not label["accepted"]
        assert "prob_confidence" in label["reasons"]

    def test_prediction_inconsistent_under_flipping_is_rejected(self):
        # the training pipeline flips at random, so a well-behaved model is equivariant
        label = PseudoLabeler(StubDetector(shift=12)).label(make_scene())
        assert label["reasons"] == ["flip_iou"]

    def test_short_path_is_rejected(self):
        label = PseudoLabeler(StubDetector()).label(make_scene(top=IMG_HEIGHT - 15))
        assert "height_ratio" in label["reasons"]

    def test_empty_prediction_is_rejected(self):
        label = PseudoLabeler(StubDetector()).label(
            Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), (40, 40, 40))
        )
        assert label["reasons"] == ["no_path"]
        assert label["rails"] is None

    def test_cropping_keeps_the_rails_in_the_original_frame(self):
        crop = (40, 20, IMG_WIDTH - 21, IMG_HEIGHT - 1)
        label = PseudoLabeler(StubDetector()).label(make_scene(), crop_coords=crop)
        assert label["accepted"], label["reasons"]
        assert label["rails"][0][0][1] == IMG_HEIGHT - 1
        assert label["metrics"]["flip_iou"] > 0.9  # the crop is mirrored too

    def test_agreeing_ensemble_is_accepted(self):
        labeler = PseudoLabeler(StubDetector(), ensemble=[StubDetector()])
        assert labeler.label(make_scene())["metrics"]["ensemble_iou"] > 0.95

    def test_disagreeing_ensemble_is_rejected(self):
        labeler = PseudoLabeler(StubDetector(), ensemble=[StubDetector(shift=12)])
        assert labeler.label(make_scene())["reasons"] == ["ensemble_iou"]


class TestGeneratedAnnotationsAreTrainable:
    """The generated annotations have to survive the training augmentation.

    PathsDataset.random_crop indexes rails_mask[-1] unconditionally, so a path that
    does not reach the bottom row of the image raises rather than being ignored.
    """

    @pytest.fixture
    def dataset_factory(self, tmp_path, data_config):
        from helpers import make_image

        from src.utils.dataset import PathsDataset

        rails = rows_to_rails(mask_to_rows(make_mask()))
        make_image(IMG_WIDTH, IMG_HEIGHT).save(tmp_path / "generated.png")
        annotations_path = tmp_path / "pseudo.json"
        with open(annotations_path, "w") as f:
            json.dump({"generated.png": {"left_rail": rails[0], "right_rail": rails[1]}}, f)

        def _make(method):
            return PathsDataset(
                imgs_path=str(tmp_path),
                annotations_path=str(annotations_path),
                indices=[0],
                config=data_config,
                method=method,
                img_aug=True,
                to_tensor=True,
            )

        return _make

    @pytest.mark.parametrize("method", ["regression", "classification", "segmentation"])
    def test_targets_are_generated_across_random_crops(self, dataset_factory, data_config, method):
        dataset = dataset_factory(method)
        np.random.seed(0)
        for _ in range(20):  # every draw crops differently
            sample = dataset[0]
        if method == "regression":
            _, traj, ylim = sample
            assert traj.shape == (2, data_config["anchors"])
            assert 0 < ylim.item() <= 1
            assert bool((traj[0] < traj[1]).all())  # left rail stays left of the right one
        elif method == "classification":
            _, target = sample
            assert target.shape == (2, data_config["anchors"])
            valid = (target != data_config["classes"]).all(axis=0).sum()
            assert valid > data_config["anchors"] // 2
        else:
            _, segmentation = sample
            assert 0.02 < segmentation.mean().item() < 0.6
