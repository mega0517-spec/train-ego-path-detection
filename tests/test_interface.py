import numpy as np
import pytest
import torch
from PIL import Image

from src.utils.autocrop import Autocropper
from src.utils.interface import Detector

pytestmark = pytest.mark.slow

IMG_SIZE = (100, 50)  # (W, H)


@pytest.fixture
def image():
    return Image.fromarray(np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8))


def classification_prediction():
    """Logits whose argmax is class 2 on the left rail and class 6 on the right one."""
    pred = np.zeros((2, 4, 9), dtype=np.float32)
    pred[0, :, 2] = 1.0
    pred[1, :, 6] = 1.0
    return pred.reshape(1, -1)


def regression_prediction(ylim_logit=0.0):
    """A constant path at x = 0.2 / x = 0.8, plus the raw y-limit logit."""
    return np.array([[0.2] * 4 + [0.8] * 4 + [ylim_logit]], dtype=np.float32)


def segmentation_prediction():
    pred = np.full((1, 1, 8, 8), -1.0, dtype=np.float32)
    pred[0, 0, 4:, 2:6] = 1.0
    return pred


def stub_inference(detector, monkeypatch, prediction):
    """Replaces the model call by a fixed output, recording the images it was fed.

    Returns:
        list: The sizes of the images handed over to the model, filled in as
            detect() is called.
    """
    seen_sizes = []

    def fake_infer(img):
        seen_sizes.append(img.size)
        return prediction() if callable(prediction) else prediction

    monkeypatch.setattr(detector, "infer_model_pytorch", fake_infer)
    return seen_sizes


class TestInitialization:
    @pytest.mark.parametrize("method", ["classification", "regression", "segmentation"])
    def test_builds_the_model_described_by_the_config(self, model_dir, method):
        path, config = model_dir(method)
        detector = Detector(path, None, "pytorch", "cpu")
        assert detector.config["method"] == method
        assert not detector.model.training  # eval mode

    def test_loads_the_saved_weights(self, model_dir):
        path, _ = model_dir("regression")
        saved = torch.load(f"{path}/best.pt", map_location="cpu")
        loaded = Detector(path, None, "pytorch", "cpu").model.state_dict()
        for key, value in saved.items():
            torch.testing.assert_close(loaded[key], value)

    def test_unknown_runtime_raises(self, model_dir):
        path, _ = model_dir("regression")
        with pytest.raises(ValueError):
            Detector(path, None, "onnx", "cpu")


class TestCropCoords:
    def test_fixed_coordinates_are_used_as_is(self, model_dir):
        path, _ = model_dir("regression")
        detector = Detector(path, (10, 5, 60, 25), "pytorch", "cpu")
        assert detector.get_crop_coords() == (10, 5, 60, 25)

    def test_none_disables_cropping(self, model_dir):
        path, _ = model_dir("regression")
        assert Detector(path, None, "pytorch", "cpu").get_crop_coords() is None

    def test_auto_installs_an_autocropper(self, model_dir):
        path, _ = model_dir("regression")
        detector = Detector(path, "auto", "pytorch", "cpu")
        assert isinstance(detector.crop_coords, Autocropper)
        assert detector.get_crop_coords() is None  # nothing observed yet

    @pytest.mark.parametrize("crop_coords", [[10, 5, 60, 25], (10, 5, 60), "fixed", 42])
    def test_anything_else_silently_disables_cropping(self, model_dir, crop_coords):
        # documents current behaviour: only a 4-tuple or the "auto" string are
        # honoured, a 4-element *list* is quietly ignored
        path, _ = model_dir("regression")
        assert Detector(path, crop_coords, "pytorch", "cpu").get_crop_coords() is None


class TestDetectClassification:
    def test_decodes_logits_into_absolute_rail_points(self, model_dir, image, monkeypatch):
        path, _ = model_dir("classification")
        detector = Detector(path, None, "pytorch", "cpu")
        stub_inference(detector, monkeypatch, classification_prediction())
        left, right = detector.detect(image)
        # x = class / (classes - 1) * (width - 1), y = linspace(1, 0) * (height - 1)
        assert left == [[28, 49], [28, 33], [28, 16], [28, 0]]
        assert right == [[85, 49], [85, 33], [85, 16], [85, 0]]

    def test_background_class_truncates_the_path(self, model_dir, image, monkeypatch):
        path, _ = model_dir("classification")
        detector = Detector(path, None, "pytorch", "cpu")
        pred = classification_prediction().reshape(2, 4, 9)
        pred[1, 2:, 6] = 0.0
        pred[1, 2:, 8] = 1.0  # background class on the last two anchors
        stub_inference(detector, monkeypatch, pred.reshape(1, -1))
        left, right = detector.detect(image)
        assert len(left) == len(right) == 2

    def test_fixed_crop_shifts_the_points_back_into_the_original_frame(
        self, model_dir, image, monkeypatch
    ):
        path, _ = model_dir("classification")
        detector = Detector(path, (10, 5, 60, 25), "pytorch", "cpu")
        seen_sizes = stub_inference(detector, monkeypatch, classification_prediction())
        left, right = detector.detect(image)
        assert seen_sizes == [(51, 21)]  # inclusive coordinates
        assert left[0] == [24, 25]
        assert right[0] == [53, 25]
        assert all(10 <= x <= 60 and 5 <= y <= 25 for x, y in left + right)


class TestDetectRegression:
    def test_decodes_the_trajectory_and_the_ylimit(self, model_dir, image, monkeypatch):
        path, _ = model_dir("regression")
        detector = Detector(path, None, "pytorch", "cpu")
        # a raw logit of 0 becomes a y-limit of 0.5, i.e. half of the 4 anchors
        stub_inference(detector, monkeypatch, regression_prediction())
        left, right = detector.detect(image)
        assert left == [[20, 49], [20, 33]]
        assert right == [[79, 49], [79, 33]]

    def test_higher_ylimit_keeps_more_anchors(self, model_dir, image, monkeypatch):
        path, _ = model_dir("regression")
        detector = Detector(path, None, "pytorch", "cpu")
        stub_inference(detector, monkeypatch, regression_prediction(ylim_logit=10.0))
        left, _ = detector.detect(image)
        assert len(left) == 4

    def test_very_negative_ylimit_yields_an_empty_path(self, model_dir, image, monkeypatch):
        path, _ = model_dir("regression")
        detector = Detector(path, None, "pytorch", "cpu")
        stub_inference(detector, monkeypatch, regression_prediction(ylim_logit=-20.0))
        assert detector.detect(image) == [[], []]


class TestDetectSegmentation:
    def test_thresholds_at_zero_and_rescales_to_the_input_size(
        self, model_dir, image, monkeypatch
    ):
        path, _ = model_dir("segmentation")
        detector = Detector(path, None, "pytorch", "cpu")
        stub_inference(detector, monkeypatch, segmentation_prediction())
        result = detector.detect(image)
        assert isinstance(result, Image.Image)
        assert result.size == IMG_SIZE
        array = np.array(result)
        assert set(np.unique(array)).issubset({0, 255})
        assert array[40, 40] == 255  # inside the predicted region
        assert array[10, 10] == 0  # above it

    def test_empty_prediction_yields_an_empty_mask(self, model_dir, image, monkeypatch):
        path, _ = model_dir("segmentation")
        detector = Detector(path, None, "pytorch", "cpu")
        pred = np.full((1, 1, 8, 8), -1.0, dtype=np.float32)
        stub_inference(detector, monkeypatch, pred)
        assert not np.array(detector.detect(image)).any()


class TestDetectWithAutocrop:
    def test_first_detection_runs_on_the_full_frame(self, model_dir, image, monkeypatch):
        path, _ = model_dir("regression")
        detector = Detector(path, "auto", "pytorch", "cpu")
        seen_sizes = stub_inference(detector, monkeypatch, regression_prediction(10.0))
        detector.detect(image)
        assert seen_sizes == [IMG_SIZE]
        assert detector.crop_coords.n == 1

    def test_the_autocropper_is_updated_from_the_detection(self, model_dir, image, monkeypatch):
        path, _ = model_dir("regression")
        detector = Detector(path, "auto", "pytorch", "cpu")
        stub_inference(detector, monkeypatch, regression_prediction(10.0))
        for _ in range(30):
            detector.detect(image)
        crop = detector.get_crop_coords()
        assert crop is not None
        assert 0 < crop[0] < crop[2] <= IMG_SIZE[0]

    def test_never_reads_past_the_edges_of_the_image(self, model_dir, image, monkeypatch):
        path, _ = model_dir("regression")
        detector = Detector(path, "auto", "pytorch", "cpu")
        seen_sizes = stub_inference(detector, monkeypatch, regression_prediction(10.0))
        for _ in range(30):
            detector.detect(image)
        # detect() crops with (xright + 1, ybottom + 1), and PIL pads with black
        # past the image, so any oversized crop means the coordinates were not
        # the inclusive ones the rest of the pipeline assumes
        assert seen_sizes
        assert all(w <= IMG_SIZE[0] and h <= IMG_SIZE[1] for w, h in seen_sizes)

    def test_detection_stays_inside_the_original_frame(self, model_dir, image, monkeypatch):
        path, _ = model_dir("regression")
        detector = Detector(path, "auto", "pytorch", "cpu")
        stub_inference(detector, monkeypatch, regression_prediction(10.0))
        for _ in range(30):
            left, right = detector.detect(image)
        for x, y in left + right:
            assert 0 <= x < IMG_SIZE[0]
            assert 0 <= y <= IMG_SIZE[1]


class TestInferModelPytorch:
    @pytest.mark.parametrize("input_shape", [[3, 64, 64], [3, 48, 64]])
    def test_resizes_the_input_to_the_configured_shape(self, model_dir, image, input_shape):
        # input_shape is (C, H, W) and must be honoured as such, not transposed
        path, config = model_dir("regression", input_shape=input_shape)
        detector = Detector(path, None, "pytorch", "cpu")
        captured = {}

        original = detector.model.forward

        def spy(tensor):
            captured["shape"] = tuple(tensor.shape)
            return original(tensor)

        detector.model.forward = spy
        detector.infer_model_pytorch(image)
        assert captured["shape"] == (1, *config["input_shape"])

    def test_returns_a_numpy_array(self, model_dir, image):
        path, config = model_dir("regression")
        detector = Detector(path, None, "pytorch", "cpu")
        pred = detector.infer_model_pytorch(image)
        assert isinstance(pred, np.ndarray)
        assert pred.shape == (1, config["anchors"] * 2 + 1)
