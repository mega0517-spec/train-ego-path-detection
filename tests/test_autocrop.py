import numpy as np
import pytest
from PIL import Image

from src.utils.autocrop import Autocropper

IMG_SHAPE = (200, 100)  # (W, H)


@pytest.fixture
def config():
    return {"crop_margin_sides": 0.1, "crop_margin_top": 0.1}


def rails_prediction(left, right, top, bottom=99):
    """Builds a classification/regression style prediction: [left_rail, right_rail]."""
    return [
        [[left, bottom], [left, top]],
        [[right, bottom], [right, top]],
    ]


class TestRailsCoords:
    def test_extracts_bounds_from_a_rails_list(self, config):
        cropper = Autocropper(config)
        coords = cropper.rails_coords(rails_prediction(left=40, right=160, top=30))
        assert coords == (40, 30, 160)  # (min left x, min y, max right x)

    def test_extracts_bounds_from_a_mask(self, config):
        cropper = Autocropper(config)
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[30:80, 40:161] = 255
        coords = cropper.rails_coords(Image.fromarray(mask))
        assert coords == (40, 30, 160)

    def test_returns_none_for_an_empty_rails_list(self, config):
        assert Autocropper(config).rails_coords([[], []]) is None

    def test_returns_none_for_an_empty_mask(self, config):
        mask = Image.fromarray(np.zeros((100, 200), dtype=np.uint8))
        assert Autocropper(config).rails_coords(mask) is None


class TestUpdate:
    def test_starts_with_no_crop(self, config):
        assert Autocropper(config)() is None

    def test_empty_prediction_leaves_the_state_untouched(self, config):
        cropper = Autocropper(config)
        cropper.update(IMG_SHAPE, [[], []])
        assert cropper() is None
        assert cropper.n == 0

    def test_first_update_yields_the_full_frame(self, config):
        cropper = Autocropper(config)
        cropper.update(IMG_SHAPE, rails_prediction(left=80, right=120, top=40))
        # the running average is seeded with the full frame, so the first crop
        # never restricts the image whatever the prediction is
        assert cropper() == (0, 0, IMG_SHAPE[0] - 1, IMG_SHAPE[1] - 1)
        assert cropper.n == 1

    def test_converges_towards_the_predicted_region(self, config):
        cropper = Autocropper(config)
        for _ in range(300):
            cropper.update(IMG_SHAPE, rails_prediction(left=80, right=120, top=40))
        xleft, ytop, xright, _ = cropper()
        # the crop settles on the rails plus a 10% margin: 10% of the rails width
        # on the sides, 10% of the height below the rails top above
        assert xleft == 80 - 0.1 * 40
        assert xright == 120 + 0.1 * 40
        assert ytop == 40 - 0.1 * 60

    def test_the_coefficient_only_changes_the_speed_not_the_destination(self, config):
        # the running average is kept in float, so an edge that has to grow reaches
        # its target instead of stalling once its increment drops below a pixel
        loose = Autocropper(config, coeff=0.1)
        tight = Autocropper(config, coeff=0.5)
        for _ in range(300):
            loose.update(IMG_SHAPE, rails_prediction(left=80, right=120, top=40))
            tight.update(IMG_SHAPE, rails_prediction(left=80, right=120, top=40))
        assert loose() == tight() == (76, 34, 124, IMG_SHAPE[1] - 1)

    def test_crop_shrinks_monotonically_for_a_stable_prediction(self, config):
        cropper = Autocropper(config)
        cropper.update(IMG_SHAPE, rails_prediction(left=80, right=120, top=40))
        previous = cropper()
        for _ in range(50):
            cropper.update(IMG_SHAPE, rails_prediction(left=80, right=120, top=40))
            current = cropper()
            assert current[0] >= previous[0]  # left edge moves right
            assert current[1] >= previous[1]  # top edge moves down
            assert current[2] <= previous[2]  # right edge moves left
            previous = current

    def test_crop_stays_inside_the_image(self, config):
        cropper = Autocropper(config)
        rng = np.random.default_rng(0)
        for _ in range(100):
            left = int(rng.integers(0, 90))
            right = int(rng.integers(110, 200))
            top = int(rng.integers(0, 99))
            cropper.update(IMG_SHAPE, rails_prediction(left, right, top))
            xleft, ytop, xright, ybottom = cropper()
            # coordinates are inclusive, so the last valid index is size - 1
            assert 0 <= xleft < xright <= IMG_SHAPE[0] - 1
            assert 0 <= ytop < ybottom <= IMG_SHAPE[1] - 1

    def test_bottom_coordinate_is_the_inclusive_image_bottom(self, config):
        # the ego-path always reaches the bottom of the frame, so ybottom is not
        # tracked -- but it must still be a valid inclusive coordinate
        cropper = Autocropper(config)
        for _ in range(20):
            cropper.update(IMG_SHAPE, rails_prediction(left=80, right=120, top=40))
        assert cropper()[3] == IMG_SHAPE[1] - 1

    def test_widening_prediction_reopens_the_crop(self, config):
        cropper = Autocropper(config)
        for _ in range(100):
            cropper.update(IMG_SHAPE, rails_prediction(left=90, right=110, top=60))
        narrow = cropper()
        for _ in range(100):
            cropper.update(IMG_SHAPE, rails_prediction(left=20, right=180, top=10))
        wide = cropper()
        assert wide[0] < narrow[0]
        assert wide[2] > narrow[2]
        assert wide[1] < narrow[1]

    def test_mask_and_rails_predictions_agree(self, config):
        from_rails = Autocropper(config)
        from_mask = Autocropper(config)
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[40:100, 80:121] = 255
        for _ in range(20):
            from_rails.update(IMG_SHAPE, rails_prediction(left=80, right=120, top=40))
            from_mask.update(IMG_SHAPE, Image.fromarray(mask))
        assert from_rails() == from_mask()

    def test_coefficient_controls_the_convergence_speed(self, config):
        fast = Autocropper(config, coeff=0.5)
        slow = Autocropper(config, coeff=0.05)
        for _ in range(10):
            fast.update(IMG_SHAPE, rails_prediction(left=80, right=120, top=40))
            slow.update(IMG_SHAPE, rails_prediction(left=80, right=120, top=40))
        assert fast()[0] > slow()[0]
