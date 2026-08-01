import numpy as np
import pytest
from PIL import Image

from src.utils.evaluate import IoUEvaluator, LatencyEvaluator, compute_iou


def mask(pattern):
    return np.array(pattern, dtype=np.uint8)


class TestComputeIou:
    def test_identical_masks_score_one(self):
        m = mask([[1, 1, 0], [0, 1, 0]])
        assert compute_iou(m, m) == pytest.approx(1.0)

    def test_disjoint_masks_score_zero(self):
        assert compute_iou(mask([[1, 1, 0]]), mask([[0, 0, 1]])) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # intersection 1, union 3
        iou = compute_iou(mask([[1, 1, 0]]), mask([[0, 1, 1]]))
        assert iou == pytest.approx(1 / 3)

    def test_empty_prediction_against_a_real_target_scores_zero(self):
        assert compute_iou(mask([[0, 0, 0]]), mask([[1, 1, 0]])) == pytest.approx(0.0)

    def test_empty_target_is_scored_on_the_negated_masks(self):
        # an empty prediction on an empty target is a perfect match...
        assert compute_iou(mask([[0, 0, 0]]), mask([[0, 0, 0]])) == pytest.approx(1.0)

    def test_false_positives_on_an_empty_target_are_penalised(self):
        # ...while any predicted pixel shrinks the negated intersection
        assert compute_iou(mask([[1, 0, 0]]), mask([[0, 0, 0]])) == pytest.approx(2 / 3)

    def test_fully_wrong_on_an_empty_target_scores_zero(self):
        assert compute_iou(mask([[1, 1, 1]]), mask([[0, 0, 0]])) == pytest.approx(0.0)

    @pytest.mark.parametrize("scale", [1, 255])
    def test_accepts_zero_one_and_zero_255_encodings(self, scale):
        target = mask([[1, 1, 0], [0, 1, 0]])
        assert compute_iou(target * scale, target * scale) == pytest.approx(1.0)

    def test_accepts_boolean_masks(self):
        target = mask([[1, 1, 0], [0, 1, 0]]).astype(bool)
        assert compute_iou(target, target) == pytest.approx(1.0)

    def test_pil_and_numpy_inputs_agree(self):
        prediction = mask([[255, 255, 0], [0, 0, 0]])
        target = mask([[255, 0, 0], [0, 255, 0]])
        numpy_iou = compute_iou(prediction, target)
        pil_iou = compute_iou(Image.fromarray(prediction), Image.fromarray(target))
        assert numpy_iou == pytest.approx(pil_iou)
        assert numpy_iou == pytest.approx(1 / 3)

    def test_mixed_pil_and_numpy_inputs(self):
        prediction = mask([[255, 255, 0]])
        target = mask([[1, 1, 0]])
        assert compute_iou(Image.fromarray(prediction), target) == pytest.approx(1.0)

    def test_returns_a_python_float(self):
        m = mask([[1, 0], [0, 1]])
        assert isinstance(compute_iou(m, m), float)

    def test_is_symmetric_for_non_empty_masks(self):
        a = mask([[1, 1, 0], [0, 1, 1]])
        b = mask([[0, 1, 1], [0, 1, 0]])
        assert compute_iou(a, b) == pytest.approx(compute_iou(b, a))


@pytest.mark.slow
class TestIoUEvaluator:
    # the evaluation set is always a segmentation dataset, whatever the model method
    @pytest.fixture
    def dataset(self, make_dataset):
        return make_dataset(method="segmentation")

    @pytest.mark.parametrize("method", ["classification", "regression", "segmentation"])
    def test_returns_a_score_for_every_method(self, model_dir, dataset, method):
        path, _ = model_dir(method)
        score = IoUEvaluator(dataset, path, "pytorch", "cpu").evaluate()
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_is_reproducible(self, model_dir, dataset):
        path, _ = model_dir("segmentation")
        evaluator = IoUEvaluator(dataset, path, "pytorch", "cpu")
        # evaluate() reseeds from the config, so the augmented test epochs repeat
        assert evaluator.evaluate() == pytest.approx(evaluator.evaluate())

    def test_averages_over_the_configured_number_of_iterations(self, model_dir, dataset):
        path, _ = model_dir("segmentation", test_iterations=3)
        evaluator = IoUEvaluator(dataset, path, "pytorch", "cpu")
        calls = []
        original = evaluator.detector.detect
        evaluator.detector.detect = lambda img: (calls.append(img), original(img))[1]
        evaluator.evaluate()
        assert len(calls) == 3 * len(dataset)

    def test_unknown_runtime_raises(self, model_dir, dataset):
        path, _ = model_dir("segmentation")
        with pytest.raises(ValueError):
            IoUEvaluator(dataset, path, "onnx", "cpu")


@pytest.mark.slow
class TestLatencyEvaluator:
    def test_measures_a_positive_duration(self, model_dir):
        path, _ = model_dir("regression")
        latency = LatencyEvaluator(path, "pytorch", "cpu").evaluate(runs=10)
        assert latency > 0
        assert latency < 10  # seconds per forward pass, sanity bound

    def test_uses_the_configured_input_shape(self, model_dir):
        path, config = model_dir("regression")
        evaluator = LatencyEvaluator(path, "pytorch", "cpu")
        shapes = []
        original = evaluator.detector.model.forward
        evaluator.detector.model.forward = lambda x: (shapes.append(tuple(x.shape)), original(x))[1]
        evaluator.evaluate(runs=10)
        assert set(shapes) == {(1, *config["input_shape"])}

    def test_unknown_runtime_raises(self, model_dir):
        path, _ = model_dir("regression")
        with pytest.raises(ValueError):
            LatencyEvaluator(path, "onnx", "cpu")
