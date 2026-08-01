import numpy as np
import pytest
from PIL import Image

from src.utils.postprocessing import (
    classifications_to_rails,
    rails_to_mask,
    regression_to_rails,
    scale_mask,
    scale_rails,
)


class TestClassificationsToRails:
    def test_no_cutoff_keeps_every_anchor(self):
        clf = np.array([[0, 1, 2], [5, 6, 7]])
        rails = classifications_to_rails(clf, classes=8)
        assert rails.shape == (2, 3, 2)
        # x is the class index normalized by (classes - 1)
        np.testing.assert_allclose(rails[0, :, 0], np.array([0, 1, 2]) / 7)
        np.testing.assert_allclose(rails[1, :, 0], np.array([5, 6, 7]) / 7)
        # y goes from 1 (bottom of the image) to 0 (top), one value per anchor
        np.testing.assert_allclose(rails[0, :, 1], [1.0, 0.5, 0.0])
        np.testing.assert_allclose(rails[1, :, 1], [1.0, 0.5, 0.0])

    def test_background_class_cuts_the_path(self):
        # the background class (index == classes) appears on the right rail at anchor 1
        clf = np.array([[0, 1, 2], [5, 8, 7]])
        rails = classifications_to_rails(clf, classes=8)
        assert rails.shape == (2, 1, 2)
        np.testing.assert_allclose(rails[:, 0, 0], [0 / 7, 5 / 7])

    def test_switched_rails_cut_the_path(self):
        # left rail >= right rail from anchor 1 on, no background class involved
        clf = np.array([[0, 5, 2], [5, 3, 7]])
        rails = classifications_to_rails(clf, classes=8)
        assert rails.shape == (2, 1, 2)

    def test_earliest_cutoff_wins(self):
        # background at anchor 2 on the right rail, rails switched at anchor 1
        clf = np.array([[0, 5, 2], [5, 3, 8]])
        rails = classifications_to_rails(clf, classes=8)
        assert rails.shape == (2, 1, 2)

    def test_rails_switched_at_first_anchor_yields_empty_path(self):
        clf = np.array([[5, 1, 2], [5, 6, 7]])
        rails = classifications_to_rails(clf, classes=8)
        assert rails.shape == (2, 0, 2)


class TestRegressionToRails:
    def test_no_cutoff_keeps_every_anchor(self):
        traj = np.array([[0.2, 0.3, 0.4], [0.8, 0.7, 0.6]])
        rails = regression_to_rails(traj, ylim=1.0)
        assert rails.shape == (2, 3, 2)
        np.testing.assert_allclose(rails[0, :, 0], [0.2, 0.3, 0.4])
        np.testing.assert_allclose(rails[1, :, 0], [0.8, 0.7, 0.6])
        np.testing.assert_allclose(rails[0, :, 1], [1.0, 0.5, 0.0])

    def test_ylim_cuts_the_path(self):
        traj = np.array([[0.2, 0.2, 0.2, 0.2], [0.8, 0.8, 0.8, 0.8]])
        rails = regression_to_rails(traj, ylim=0.5)  # round(0.5 * 4) == 2 anchors kept
        assert rails.shape == (2, 2, 2)

    def test_ylim_zero_yields_empty_path(self):
        traj = np.array([[0.2, 0.2], [0.8, 0.8]])
        rails = regression_to_rails(traj, ylim=0.0)
        assert rails.shape == (2, 0, 2)

    def test_switched_rails_cut_before_ylim(self):
        traj = np.array([[0.2, 0.9, 0.2, 0.2], [0.8, 0.5, 0.8, 0.8]])
        rails = regression_to_rails(traj, ylim=1.0)
        assert rails.shape == (2, 1, 2)

    def test_out_of_bounds_predictions_are_clipped(self):
        traj = np.array([[-0.5, -0.2], [1.5, 1.2]])
        rails = regression_to_rails(traj, ylim=1.0)
        np.testing.assert_allclose(rails[0, :, 0], [0.0, 0.0])
        np.testing.assert_allclose(rails[1, :, 0], [1.0, 1.0])


class TestScaleRails:
    def test_without_crop_scales_to_image_extent_minus_one(self):
        rails = np.array([[[0.0, 0.0], [1.0, 1.0]], [[0.5, 0.5], [1.0, 0.0]]])
        scaled = scale_rails(rails.copy(), None, img_shape=(101, 51))
        np.testing.assert_allclose(scaled[0, 0], [0.0, 0.0])
        np.testing.assert_allclose(scaled[0, 1], [100.0, 50.0])
        np.testing.assert_allclose(scaled[1, 0], [50.0, 25.0])

    def test_with_crop_scales_to_crop_extent_and_offsets(self):
        rails = np.array([[[0.0, 0.0], [1.0, 1.0]], [[0.5, 0.5], [1.0, 0.0]]])
        # inclusive crop coordinates, so the region is 51x21 pixels wide/high...
        scaled = scale_rails(rails.copy(), (10, 5, 60, 25), img_shape=(101, 51))
        # ...but the branch scales by (xmax - xmin) and (ymax - ymin), i.e. 50 and 20
        np.testing.assert_allclose(scaled[0, 0], [10.0, 5.0])
        np.testing.assert_allclose(scaled[0, 1], [60.0, 25.0])
        np.testing.assert_allclose(scaled[1, 0], [35.0, 15.0])

    def test_mutates_its_input_in_place(self):
        # documents current behaviour: callers must not reuse the array they passed in
        rails = np.array([[[0.5, 0.5]], [[0.5, 0.5]]])
        scaled = scale_rails(rails, None, img_shape=(11, 11))
        assert scaled is rails
        np.testing.assert_allclose(rails[0, 0], [5.0, 5.0])

    def test_empty_path_is_preserved(self):
        rails = np.zeros((2, 0, 2))
        assert scale_rails(rails, None, img_shape=(10, 10)).shape == (2, 0, 2)


class TestRailsToMask:
    def test_fills_the_polygon_between_the_rails(self):
        left_rail = [[2, 8], [2, 2]]
        right_rail = [[6, 8], [6, 2]]
        mask = rails_to_mask([left_rail, right_rail], mask_shape=(10, 10))
        assert isinstance(mask, Image.Image)
        array = np.array(mask)
        assert array.shape == (10, 10)  # PIL (W, H) -> numpy (H, W)
        assert array[5, 4] == 255  # inside the path
        assert array[5, 8] == 0  # right of the path
        assert array[0, 4] == 0  # above the path

    @pytest.mark.parametrize(
        "rails", [[[], [[6, 8], [6, 2]]], [[[2, 8], [2, 2]], []], [[], []]]
    )
    def test_empty_rail_returns_a_numpy_array_not_an_image(self, rails):
        # documents an inconsistency: the empty branch returns numpy, the normal
        # branch returns PIL, and scale_mask() would fail on the numpy variant
        mask = rails_to_mask(rails, mask_shape=(10, 20))
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (20, 10)
        assert not mask.any()


class TestScaleMask:
    def test_without_crop_resizes_to_the_image_shape(self):
        mask = Image.fromarray(np.full((10, 10), 255, dtype=np.uint8))
        scaled = scale_mask(mask, None, img_shape=(40, 20))
        assert scaled.size == (40, 20)
        assert np.array(scaled).min() == 255

    def test_with_crop_pastes_the_mask_back_at_its_location(self):
        mask = Image.fromarray(np.full((4, 4), 255, dtype=np.uint8))
        scaled = scale_mask(mask, (10, 5, 19, 12), img_shape=(40, 20))
        assert scaled.size == (40, 20)
        array = np.array(scaled)
        # inclusive coordinates: the pasted region spans rows 5..12 and cols 10..19
        assert array[5:13, 10:20].min() == 255
        assert array[4, 10] == 0
        assert array[5, 9] == 0
        assert array[13, 10] == 0
        assert array[5, 20] == 0

    def test_nearest_resampling_keeps_the_mask_binary(self):
        source = np.zeros((10, 10), dtype=np.uint8)
        source[5:, 5:] = 255
        scaled = scale_mask(Image.fromarray(source), None, img_shape=(33, 17))
        assert set(np.unique(np.array(scaled))).issubset({0, 255})
