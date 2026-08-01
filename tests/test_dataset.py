import numpy as np
import pytest
import torch
from PIL import Image

from helpers import make_image, make_rails_mask
from src.utils.common import simple_logger


class TestGenerateRailsMask:
    def test_draws_both_rails_and_trims_above_the_shorter_one(self, make_dataset):
        dataset = make_dataset()
        annotation = {
            "left_rail": [[2, 9], [2, 3]],  # reaches up to row 3
            "right_rail": [[7, 9], [7, 5]],  # reaches up to row 5
        }
        mask = dataset.generate_rails_mask((10, 10), annotation)
        assert mask.shape == (10, 10)
        # rows above the *lower* of the two rail tops are cleared
        assert not mask[:5].any()
        for row in range(5, 10):
            np.testing.assert_array_equal(np.nonzero(mask[row])[0], [2, 7])

    def test_keeps_only_the_outermost_two_points_per_row(self, make_dataset):
        dataset = make_dataset()
        annotation = {
            "left_rail": [[1, 9], [5, 9], [5, 4]],  # horizontal foot then vertical
            "right_rail": [[8, 9], [8, 4]],
        }
        mask = dataset.generate_rails_mask((10, 10), annotation)
        # the horizontal foot puts 6 points on row 9, only the outermost two survive
        np.testing.assert_array_equal(np.nonzero(mask[9])[0], [1, 8])
        for row in range(4, 9):
            np.testing.assert_array_equal(np.nonzero(mask[row])[0], [5, 8])

    def test_mask_is_binary(self, make_dataset):
        dataset = make_dataset()
        annotation = {"left_rail": [[2, 9], [2, 3]], "right_rail": [[7, 9], [7, 3]]}
        mask = dataset.generate_rails_mask((10, 10), annotation)
        assert set(np.unique(mask)).issubset({0, 1})


class TestResizeMask:
    def test_samples_rows_and_rescales_columns(self, make_dataset):
        dataset = make_dataset()
        mask = make_rails_mask()  # 10x10, rails at cols 2 and 7 from row 5 down
        resized = dataset.resize_mask(mask, (4, 10))
        assert resized.shape == (4, 10)
        # source rows sampled are 0, 3, 6, 9 -> only the last two carry rail points
        assert not resized[0].any()
        assert not resized[1].any()
        np.testing.assert_array_equal(np.nonzero(resized[2])[0], [2, 7])
        np.testing.assert_array_equal(np.nonzero(resized[3])[0], [2, 7])

    def test_rescales_columns_to_the_target_width(self, make_dataset):
        dataset = make_dataset()
        resized = dataset.resize_mask(make_rails_mask(), (4, 19))
        # width factor is 18/9 = 2, so columns 2 and 7 land on 4 and 14
        np.testing.assert_array_equal(np.nonzero(resized[3])[0], [4, 14])

    def test_drops_rows_that_do_not_hold_exactly_two_points(self, make_dataset):
        dataset = make_dataset()
        mask = make_rails_mask()
        mask[9, 5] = 1  # a third point on the bottom row
        resized = dataset.resize_mask(mask, (4, 10))
        assert not resized[3].any()  # bottom row dropped entirely
        np.testing.assert_array_equal(np.nonzero(resized[2])[0], [2, 7])

    def test_output_is_binary(self, make_dataset):
        dataset = make_dataset()
        resized = dataset.resize_mask(make_rails_mask(), (4, 10))
        assert set(np.unique(resized)).issubset({0, 1})


class TestGenerateTargetRegression:
    def test_produces_normalized_trajectory_and_ylim(self, make_dataset):
        dataset = make_dataset(method="regression", config={"anchors": 4})
        traj, ylim = dataset.generate_target_regression(make_rails_mask())
        # rows 0-4 are empty, so the path covers the bottom half of the mask
        assert ylim == pytest.approx(0.5)
        assert traj.shape == (2, 4)
        assert traj.dtype == np.float32
        # anchors are filled from the bottom up until a row without 2 points
        np.testing.assert_allclose(traj[0], [2 / 9, 2 / 9, 0.0, 0.0], rtol=1e-6)
        np.testing.assert_allclose(traj[1], [7 / 9, 7 / 9, 1.0, 1.0], rtol=1e-6)

    def test_full_height_path_gives_ylim_one(self, make_dataset):
        dataset = make_dataset(method="regression", config={"anchors": 4})
        _, ylim = dataset.generate_target_regression(make_rails_mask(top=0))
        assert ylim == 1.0

    def test_empty_mask_gives_ylim_zero_and_default_trajectory(self, make_dataset):
        dataset = make_dataset(method="regression", config={"anchors": 4})
        traj, ylim = dataset.generate_target_regression(np.zeros((10, 10), dtype=np.uint8))
        assert ylim == 0.0
        np.testing.assert_allclose(traj[0], np.zeros(4))
        np.testing.assert_allclose(traj[1], np.ones(4))

    def test_defaults_keep_the_rails_ordered(self, make_dataset):
        # unfilled anchors default to 0 / 1, which keeps left < right everywhere
        dataset = make_dataset(method="regression", config={"anchors": 8})
        traj, _ = dataset.generate_target_regression(make_rails_mask())
        assert np.all(traj[0] < traj[1])


class TestGenerateTargetClassification:
    def test_produces_grid_indices_with_background_padding(self, make_dataset):
        dataset = make_dataset(
            method="classification", config={"anchors": 4, "classes": 10}
        )
        target = dataset.generate_target_classification(make_rails_mask())
        assert target.shape == (2, 4)
        # anchors beyond the path are set to the background class (== classes)
        np.testing.assert_array_equal(target[0], [2, 2, 10, 10])
        np.testing.assert_array_equal(target[1], [7, 7, 10, 10])

    def test_empty_mask_is_all_background(self, make_dataset):
        dataset = make_dataset(
            method="classification", config={"anchors": 4, "classes": 10}
        )
        target = dataset.generate_target_classification(np.zeros((10, 10), dtype=np.uint8))
        assert (target == 10).all()

    def test_indices_stay_within_the_class_grid(self, make_dataset):
        dataset = make_dataset(
            method="classification", config={"anchors": 8, "classes": 32}
        )
        target = dataset.generate_target_classification(make_rails_mask(top=0))
        assert target.min() >= 0
        assert target.max() <= 32  # classes itself is the background index


class TestGenerateTargetSegmentation:
    def test_fills_between_the_rails(self, make_dataset):
        dataset = make_dataset(method="segmentation")
        target = np.array(dataset.generate_target_segmentation(make_rails_mask()))
        assert target.shape == (10, 10)
        assert set(np.unique(target)).issubset({0, 255})
        assert not target[:5].any()  # above the rails
        for row in range(5, 10):
            np.testing.assert_array_equal(np.nonzero(target[row])[0], np.arange(2, 8))

    def test_returns_a_pil_image(self, make_dataset):
        dataset = make_dataset(method="segmentation")
        assert isinstance(dataset.generate_target_segmentation(make_rails_mask()), Image.Image)

    def test_stops_at_the_first_row_without_two_points(self, make_dataset):
        dataset = make_dataset(method="segmentation")
        mask = make_rails_mask(top=0)
        mask[5, :] = 0  # break the path halfway
        target = np.array(dataset.generate_target_segmentation(mask))
        assert target[6:].any()  # below the break
        assert not target[:6].any()  # the fill stops there


class TestRandomCrop:
    @staticmethod
    def sample():
        mask = make_rails_mask(height=64, width=64, left=20, right=44, top=10)
        return make_image(64, 64, seed=1), mask

    @pytest.mark.parametrize("seed", range(30))
    def test_image_and_mask_stay_aligned(self, make_dataset, seed):
        dataset = make_dataset()
        np.random.seed(seed)
        img, mask = dataset.random_crop(*self.sample())
        assert (img.width, img.height) == (mask.shape[1], mask.shape[0])

    @pytest.mark.parametrize("seed", range(30))
    def test_never_crops_away_the_bottom_rail_points(self, make_dataset, seed):
        dataset = make_dataset()
        np.random.seed(seed)
        _, mask = dataset.random_crop(*self.sample())
        # the crop is reflected back whenever it would cut into the bottom rails
        assert np.count_nonzero(mask[-1]) == 2

    @pytest.mark.parametrize("seed", range(30))
    def test_keeps_at_least_two_rows(self, make_dataset, seed):
        dataset = make_dataset()
        np.random.seed(seed)
        img, mask = dataset.random_crop(*self.sample())
        assert mask.shape[0] >= 2
        assert img.height >= 2

    def test_is_stochastic(self, make_dataset):
        dataset = make_dataset()
        shapes = set()
        for seed in range(20):
            np.random.seed(seed)
            _, mask = dataset.random_crop(*self.sample())
            shapes.add(mask.shape)
        assert len(shapes) > 1

    def test_requires_rail_points_on_the_bottom_row(self, make_dataset):
        # documents an undocumented assumption on the annotations: a path that does
        # not reach the bottom of the image makes the crop blow up
        dataset = make_dataset()
        mask = make_rails_mask(height=64, width=64, left=20, right=44, top=10)
        mask[-1, :] = 0
        with pytest.raises(IndexError):
            dataset.random_crop(make_image(64, 64), mask)


class TestRandomFlipLr:
    def test_flips_image_and_mask_together(self, make_dataset, monkeypatch):
        dataset = make_dataset()
        monkeypatch.setattr(np.random, "rand", lambda: 0.0)  # always flip
        img, mask = make_image(8, 8, seed=2), make_rails_mask(8, 8, left=1, right=6, top=3)
        flipped_img, flipped_mask = dataset.random_flip_lr(img, mask)
        np.testing.assert_array_equal(np.array(flipped_img), np.array(img)[:, ::-1])
        np.testing.assert_array_equal(np.nonzero(flipped_mask[5])[0], [1, 6])

    def test_leaves_the_sample_untouched_when_not_flipping(self, make_dataset, monkeypatch):
        dataset = make_dataset()
        monkeypatch.setattr(np.random, "rand", lambda: 0.99)  # never flip
        img, mask = make_image(8, 8, seed=2), make_rails_mask(8, 8, left=1, right=6, top=3)
        flipped_img, flipped_mask = dataset.random_flip_lr(img, mask)
        assert flipped_img is img
        assert flipped_mask is mask


class TestGetItem:
    def test_length_follows_the_given_indices(self, make_dataset):
        assert len(make_dataset(indices=[0, 2])) == 2

    def test_regression_sample_without_tensors(self, make_dataset):
        dataset = make_dataset(method="regression", config={"anchors": 8})
        img, traj, ylim = dataset[0]
        assert isinstance(img, Image.Image)
        assert traj.shape == (2, 8)
        assert 0.0 <= ylim <= 1.0

    def test_regression_sample_as_tensors(self, make_dataset):
        dataset = make_dataset(method="regression", config={"anchors": 8}, to_tensor=True)
        img, traj, ylim = dataset[0]
        assert img.shape == (3, 32, 32)
        assert img.dtype == torch.float32
        assert 0.0 <= img.min() and img.max() <= 1.0
        assert traj.shape == (2, 8)
        assert ylim.ndim == 0

    def test_classification_sample_as_tensors(self, make_dataset):
        dataset = make_dataset(
            method="classification", config={"anchors": 8, "classes": 16}, to_tensor=True
        )
        img, target = dataset[0]
        assert img.shape == (3, 32, 32)
        assert target.shape == (2, 8)
        assert target.dtype == torch.int64
        assert target.max() <= 16

    def test_segmentation_sample_as_tensors(self, make_dataset):
        dataset = make_dataset(method="segmentation", to_tensor=True)
        img, segmentation = dataset[0]
        assert img.shape == (3, 32, 32)
        assert segmentation.shape == (1, 32, 32)
        assert set(torch.unique(segmentation).tolist()).issubset({0.0, 1.0})

    def test_segmentation_sample_without_tensors(self, make_dataset):
        dataset = make_dataset(method="segmentation")
        img, segmentation = dataset[0]
        assert isinstance(segmentation, Image.Image)
        assert segmentation.size == img.size

    def test_image_augmentation_changes_the_pixels_only(self, make_dataset):
        plain = make_dataset(method="regression", config={"anchors": 8}, to_tensor=True)
        augmented = make_dataset(
            method="regression", config={"anchors": 8}, to_tensor=True, img_aug=True
        )
        torch.manual_seed(0)
        np.random.seed(0)
        plain_img, plain_traj, _ = plain[0]
        torch.manual_seed(0)
        np.random.seed(0)
        aug_img, aug_traj, _ = augmented[0]
        assert not torch.allclose(plain_img, aug_img)
        torch.testing.assert_close(plain_traj, aug_traj)  # targets are unaffected

    def test_non_square_input_shape_is_transposed(self, make_dataset):
        # documents a latent bug: input_shape is (C, H, W) but the resize is fed
        # (W, H), which is only harmless because the shipped configs are square
        dataset = make_dataset(
            method="regression", config={"input_shape": [3, 24, 48]}, to_tensor=True
        )
        img, _, _ = dataset[0]
        assert img.shape == (3, 48, 24)  # should be (3, 24, 48)


class TestGetPerspectiveWeightLimit:
    def test_returns_a_finite_positive_limit(self, make_dataset):
        dataset = make_dataset(method="regression", config={"anchors": 8}, to_tensor=True)
        logger = simple_logger("test_perspective", "critical")
        limit = dataset.get_perspective_weight_limit(percentile=95, logger=logger)
        assert np.isfinite(limit)
        assert limit > 0

    def test_higher_percentile_gives_a_higher_limit(self, make_dataset):
        dataset = make_dataset(method="regression", config={"anchors": 8}, to_tensor=True)
        logger = simple_logger("test_perspective_2", "critical")
        np.random.seed(0)
        low = dataset.get_perspective_weight_limit(percentile=50, logger=logger)
        np.random.seed(0)
        high = dataset.get_perspective_weight_limit(percentile=95, logger=logger)
        assert high >= low
