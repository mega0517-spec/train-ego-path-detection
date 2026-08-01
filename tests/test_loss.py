import math

import pytest
import torch

from src.nn.loss import BinaryDiceLoss, CrossEntropyLoss, TrainEgoPathRegressionLoss


def logit(p):
    return math.log(p / (1 - p))


class TestCrossEntropyLoss:
    @staticmethod
    def build(target, confidence):
        """Builds (B, 2 * anchors * (classes + 1)) logits favouring the target class."""
        batch, rails, anchors = target.shape
        logits = torch.zeros(batch, rails, anchors, 5)
        logits.scatter_(3, target.unsqueeze(-1), confidence)
        return logits.reshape(batch, -1)

    def test_confident_and_correct_prediction_gives_no_loss(self):
        target = torch.tensor([[[0, 1, 2], [3, 4, 0]], [[1, 1, 1], [2, 2, 2]]])
        loss = CrossEntropyLoss()(self.build(target, 20.0), target)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_uniform_prediction_gives_the_entropy_of_the_grid(self):
        target = torch.tensor([[[0, 1, 2], [3, 4, 0]]])
        prediction = torch.zeros(1, 2 * 3 * 5)
        loss = CrossEntropyLoss()(prediction, target)
        assert loss.item() == pytest.approx(math.log(5), abs=1e-6)

    def test_wrong_prediction_costs_more_than_a_right_one(self):
        target = torch.tensor([[[0, 1, 2], [3, 4, 0]]])
        right = CrossEntropyLoss()(self.build(target, 5.0), target)
        wrong = CrossEntropyLoss()(self.build((target + 1) % 5, 5.0), target)
        assert wrong.item() > right.item()

    def test_reduces_to_a_scalar(self):
        target = torch.tensor([[[0, 1, 2], [3, 4, 0]], [[1, 1, 1], [2, 2, 2]]])
        assert CrossEntropyLoss()(self.build(target, 3.0), target).ndim == 0

    def test_batch_size_does_not_change_the_loss_of_identical_samples(self):
        single = torch.tensor([[[0, 1, 2], [3, 4, 0]]])
        doubled = single.repeat(2, 1, 1)
        criterion = CrossEntropyLoss()
        assert criterion(self.build(single, 3.0), single).item() == pytest.approx(
            criterion(self.build(doubled, 3.0), doubled).item(), abs=1e-6
        )


class TestTrainEgoPathRegressionLoss:
    @staticmethod
    def build(traj, ylim_target, ylim_pred=None):
        """Packs a trajectory and a y-limit into the (B, 2 * anchors + 1) model output."""
        ylim_pred = ylim_target if ylim_pred is None else ylim_pred
        logits = torch.tensor([[logit(min(max(v, 1e-6), 1 - 1e-6))] for v in ylim_pred])
        return torch.cat([traj.flatten(start_dim=1), logits], dim=1)

    def test_perfect_prediction_gives_no_loss(self):
        traj = torch.tensor([[[0.2, 0.2, 0.2, 0.2], [0.8, 0.8, 0.8, 0.8]]])
        ylim = torch.tensor([0.75])
        criterion = TrainEgoPathRegressionLoss(ylimit_loss_weight=0.5)
        loss = criterion(self.build(traj, [0.75]), (traj, ylim))
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_predictions_beyond_the_ylimit_are_ignored(self):
        traj = torch.tensor([[[0.2, 0.2, 0.2, 0.2], [0.8, 0.8, 0.8, 0.8]]])
        ylim = torch.tensor([0.5])  # covers anchors 0 and 1 only (0.5 * 3 == 1.5)
        criterion = TrainEgoPathRegressionLoss(ylimit_loss_weight=0.5)
        baseline = criterion(self.build(traj, [0.5]), (traj, ylim)).item()
        garbage = traj.clone()
        garbage[:, :, 2:] = torch.tensor([[[-5.0, 5.0], [5.0, -5.0]]]).view(1, 2, 2)
        polluted = criterion(self.build(garbage, [0.5]), (traj, ylim)).item()
        assert polluted == pytest.approx(baseline, abs=1e-6)

    def test_errors_before_the_ylimit_do_count(self):
        traj = torch.tensor([[[0.2, 0.2, 0.2, 0.2], [0.8, 0.8, 0.8, 0.8]]])
        ylim = torch.tensor([0.5])
        criterion = TrainEgoPathRegressionLoss(ylimit_loss_weight=0.5)
        shifted = traj.clone()
        shifted[:, :, 0] += 0.1
        loss = criterion(self.build(shifted, [0.5]), (traj, ylim))
        assert loss.item() > 0.0

    def test_narrow_rails_are_weighted_more_than_wide_ones(self):
        criterion = TrainEgoPathRegressionLoss(ylimit_loss_weight=0.0)
        ylim = torch.tensor([1.0])
        narrow = torch.tensor([[[0.4, 0.4], [0.6, 0.6]]])  # rail width 0.2
        wide = torch.tensor([[[0.2, 0.2], [0.8, 0.8]]])  # rail width 0.6
        narrow_loss = criterion(self.build(narrow + 0.1, [1.0]), (narrow, ylim)).item()
        wide_loss = criterion(self.build(wide + 0.1, [1.0]), (wide, ylim)).item()
        # the weight is 1 / rail_width, so a 3x narrower path costs 3x more
        assert narrow_loss == pytest.approx(3 * wide_loss, rel=1e-5)

    def test_perspective_weight_limit_caps_the_weighting(self):
        ylim = torch.tensor([1.0])
        narrow = torch.tensor([[[0.4, 0.4], [0.6, 0.6]]])  # weight would be 5.0
        prediction = self.build(narrow + 0.1, [1.0])
        unclamped = TrainEgoPathRegressionLoss(0.0)(prediction, (narrow, ylim)).item()
        clamped = TrainEgoPathRegressionLoss(0.0, perspective_weight_limit=2.0)(
            prediction, (narrow, ylim)
        ).item()
        assert clamped == pytest.approx(unclamped * 2.0 / 5.0, rel=1e-5)

    def test_limit_above_the_actual_weight_is_a_no_op(self):
        ylim = torch.tensor([1.0])
        traj = torch.tensor([[[0.4, 0.4], [0.6, 0.6]]])
        prediction = self.build(traj + 0.1, [1.0])
        unclamped = TrainEgoPathRegressionLoss(0.0)(prediction, (traj, ylim)).item()
        clamped = TrainEgoPathRegressionLoss(0.0, perspective_weight_limit=100.0)(
            prediction, (traj, ylim)
        ).item()
        assert clamped == pytest.approx(unclamped, rel=1e-6)

    def test_samples_without_a_path_do_not_contribute_to_the_trajectory_loss(self):
        criterion = TrainEgoPathRegressionLoss(ylimit_loss_weight=0.0)
        traj = torch.tensor([[[0.2, 0.2], [0.8, 0.8]]])
        garbage = torch.tensor([[[-3.0, 4.0], [7.0, -9.0]]])
        empty = criterion(self.build(garbage, [0.0]), (traj, torch.tensor([0.0])))
        assert empty.item() == pytest.approx(0.0, abs=1e-7)

    def test_ylimit_error_is_scaled_by_its_weight(self):
        traj = torch.tensor([[[0.2, 0.2], [0.8, 0.8]]])
        ylim = torch.tensor([0.75])
        prediction = self.build(traj, [0.75], ylim_pred=[0.25])  # trajectory is perfect
        light = TrainEgoPathRegressionLoss(ylimit_loss_weight=0.5)(prediction, (traj, ylim))
        heavy = TrainEgoPathRegressionLoss(ylimit_loss_weight=1.0)(prediction, (traj, ylim))
        assert heavy.item() == pytest.approx(2 * light.item(), rel=1e-5)

    def test_ylimit_loss_is_computed_after_a_sigmoid(self):
        traj = torch.tensor([[[0.2, 0.2], [0.8, 0.8]]])
        ylim = torch.tensor([0.5])
        criterion = TrainEgoPathRegressionLoss(ylimit_loss_weight=1.0)
        # a raw logit of 0 maps to a predicted y-limit of 0.5, i.e. no error at all
        prediction = torch.cat([traj.flatten(start_dim=1), torch.zeros(1, 1)], dim=1)
        assert criterion(prediction, (traj, ylim)).item() == pytest.approx(0.0, abs=1e-7)

    def test_batch_loss_is_the_mean_of_its_samples(self):
        criterion = TrainEgoPathRegressionLoss(ylimit_loss_weight=0.0)
        traj = torch.tensor([[[0.2, 0.2], [0.8, 0.8]], [[0.3, 0.3], [0.7, 0.7]]])
        ylim = torch.tensor([1.0, 1.0])
        prediction = self.build(traj + 0.1, [1.0, 1.0])
        batched = criterion(prediction, (traj, ylim)).item()
        singles = [
            criterion(
                self.build(traj[i : i + 1] + 0.1, [1.0]), (traj[i : i + 1], ylim[i : i + 1])
            ).item()
            for i in range(2)
        ]
        assert batched == pytest.approx(sum(singles) / 2, rel=1e-5)

    def test_loss_is_differentiable(self):
        traj = torch.tensor([[[0.2, 0.2], [0.8, 0.8]]])
        ylim = torch.tensor([0.75])
        prediction = self.build(traj + 0.1, [0.5]).requires_grad_(True)
        TrainEgoPathRegressionLoss(0.5)(prediction, (traj, ylim)).backward()
        assert prediction.grad is not None
        assert torch.isfinite(prediction.grad).all()


class TestBinaryDiceLoss:
    @staticmethod
    def target(pattern):
        return torch.tensor(pattern, dtype=torch.float32).view(1, 1, 2, 4)

    def test_confident_and_correct_prediction_gives_no_loss(self):
        target = self.target([[1, 1, 0, 0], [1, 1, 0, 0]])
        prediction = (target * 2 - 1) * 20  # +20 where 1, -20 where 0
        assert BinaryDiceLoss()(prediction, target).item() == pytest.approx(0.0, abs=1e-6)

    def test_confident_and_wrong_prediction_costs_one(self):
        target = self.target([[1, 1, 0, 0], [1, 1, 0, 0]])
        prediction = (target * 2 - 1) * -20
        assert BinaryDiceLoss()(prediction, target).item() == pytest.approx(1.0, abs=1e-6)

    def test_undecided_prediction_sits_in_between(self):
        target = self.target([[1, 1, 0, 0], [1, 1, 0, 0]])
        loss = BinaryDiceLoss()(torch.zeros_like(target), target)
        assert loss.item() == pytest.approx(0.5, abs=1e-6)

    def test_empty_target_is_scored_on_the_negated_masks(self):
        target = self.target([[0, 0, 0, 0], [0, 0, 0, 0]])
        confident_empty = torch.full_like(target, -20.0)
        assert BinaryDiceLoss()(confident_empty, target).item() == pytest.approx(0.0, abs=1e-6)

    def test_predicting_a_path_on_an_empty_target_costs_one(self):
        target = self.target([[0, 0, 0, 0], [0, 0, 0, 0]])
        confident_full = torch.full_like(target, 20.0)
        assert BinaryDiceLoss()(confident_full, target).item() == pytest.approx(1.0, abs=1e-6)

    def test_partial_overlap_scores_the_dice_coefficient(self):
        target = self.target([[1, 1, 1, 1], [0, 0, 0, 0]])
        prediction = self.target([[1, 1, 0, 0], [1, 1, 0, 0]])
        loss = BinaryDiceLoss()((prediction * 2 - 1) * 20, target).item()
        # |A n B| = 2, |A| + |B| = 8 -> dice = 0.5
        assert loss == pytest.approx(0.5, abs=1e-6)

    def test_reduces_over_the_batch(self):
        target = torch.cat([self.target([[1, 1, 0, 0], [1, 1, 0, 0]])] * 2)
        prediction = torch.cat([(target[:1] * 2 - 1) * 20, (target[1:] * 2 - 1) * -20])
        assert BinaryDiceLoss()(prediction, target).item() == pytest.approx(0.5, abs=1e-6)

    def test_loss_is_differentiable(self):
        target = self.target([[1, 1, 0, 0], [1, 1, 0, 0]])
        prediction = torch.zeros_like(target).requires_grad_(True)
        BinaryDiceLoss()(prediction, target).backward()
        assert prediction.grad is not None
        assert torch.isfinite(prediction.grad).all()

    def test_does_not_modify_the_prediction_it_was_given(self):
        target = self.target([[1, 1, 0, 0], [1, 1, 0, 0]])
        prediction = torch.zeros_like(target)
        BinaryDiceLoss()(prediction, target.clone())
        assert (prediction == 0).all()

    def test_does_not_modify_the_target_on_empty_samples(self):
        # the negation trick must not write through the view returned by flatten(),
        # or the caller's ground truth would be silently flipped
        target = self.target([[0, 0, 0, 0], [0, 0, 0, 0]])
        BinaryDiceLoss()(torch.zeros_like(target), target)
        assert (target == 0).all()

    def test_only_empty_samples_are_negated_within_a_batch(self):
        empty = self.target([[0, 0, 0, 0], [0, 0, 0, 0]])
        filled = self.target([[1, 1, 0, 0], [1, 1, 0, 0]])
        target = torch.cat([empty, filled])
        # a confident empty prediction is perfect on the empty sample and perfectly
        # wrong on the filled one, so the batch mean sits exactly halfway
        prediction = torch.full_like(target, -20.0)
        assert BinaryDiceLoss()(prediction, target).item() == pytest.approx(0.5, abs=1e-6)
        torch.testing.assert_close(target, torch.cat([empty, filled]))

    def test_non_empty_targets_are_left_alone(self):
        target = self.target([[1, 1, 0, 0], [1, 1, 0, 0]])
        original = target.clone()
        BinaryDiceLoss()(torch.zeros_like(target), target)
        torch.testing.assert_close(target, original)
