import numpy as np
import pytest
from PIL import Image

from src.utils.visualization import draw_egopath

SIZE = 20  # square test canvas, (W, H)
GREEN = (0, 189, 80)  # draw_egopath default overlay color
RED = (255, 0, 0)  # crop rectangle outline

# a rails pair and a mask covering the exact same region of the canvas
RAILS = [[[2, 2], [2, 16]], [[16, 2], [16, 16]]]
INSIDE = (9, 9)  # (row, col) well within that region
OUTSIDE = (0, 0)


def canvas(level=0):
    return Image.fromarray(np.full((SIZE, SIZE, 3), level, dtype=np.uint8))


def region_mask():
    array = np.zeros((SIZE, SIZE), dtype=np.uint8)
    array[2:17, 2:17] = 255
    return Image.fromarray(array)


def pixel(img, at):
    return tuple(np.array(img)[at])


class TestGeneral:
    @pytest.mark.parametrize("egopath", [RAILS, [[], []]])
    def test_leaves_the_input_image_untouched(self, egopath):
        img = canvas(60)
        before = np.array(img).copy()
        result = draw_egopath(img, egopath, crop_coords=(3, 3, 10, 10))
        assert result is not img
        np.testing.assert_array_equal(np.array(img), before)

    def test_preserves_size_and_mode(self):
        img = canvas(60)
        result = draw_egopath(img, RAILS)
        assert result.size == img.size
        assert result.mode == img.mode


class TestRailsOverlay:
    def test_fills_the_polygon_between_the_rails(self):
        result = draw_egopath(canvas(), RAILS)
        assert pixel(result, INSIDE) == (0, 94, 40)  # 50% of GREEN over black
        assert pixel(result, OUTSIDE) == (0, 0, 0)

    def test_blends_towards_the_color_over_a_non_black_image(self):
        result = draw_egopath(canvas(60), RAILS)
        red, green, blue = pixel(result, INSIDE)
        assert red < 60  # GREEN has no red channel, so it darkens
        assert 60 < green < 189  # pulled towards the overlay without reaching it
        assert 60 < blue < 80

    def test_opacity_zero_leaves_the_image_untouched(self):
        img = canvas(60)
        np.testing.assert_array_equal(
            np.array(draw_egopath(img, RAILS, opacity=0.0)), np.array(img)
        )

    def test_opacity_one_paints_the_solid_color(self):
        result = draw_egopath(canvas(60), RAILS, opacity=1.0)
        assert pixel(result, INSIDE) == GREEN
        assert pixel(result, OUTSIDE) == (60, 60, 60)

    def test_custom_color(self):
        result = draw_egopath(canvas(), RAILS, color=(200, 100, 50), opacity=1.0)
        assert pixel(result, INSIDE) == (200, 100, 50)

    @pytest.mark.parametrize(
        "egopath", [[[], RAILS[1]], [RAILS[0], []], [[], []]], ids=["left", "right", "both"]
    )
    def test_an_empty_rail_draws_nothing(self, egopath):
        img = canvas(60)
        np.testing.assert_array_equal(np.array(draw_egopath(img, egopath)), np.array(img))

    @pytest.mark.parametrize(
        "egopath", [[[], RAILS[1]], [RAILS[0], []], [[], []]], ids=["left", "right", "both"]
    )
    def test_an_empty_rail_still_draws_the_crop_rectangle(self, egopath):
        # the crop box must not flicker on the frames where nothing was detected
        result = draw_egopath(canvas(), egopath, crop_coords=(3, 3, 10, 10))
        array = np.array(result)
        assert tuple(array[3, 3]) == RED
        assert tuple(array[10, 10]) == RED
        assert tuple(array[INSIDE]) == (0, 0, 0)  # but still no overlay


class TestSegmentationOverlay:
    def test_overlays_the_mask_region(self):
        result = draw_egopath(canvas(), region_mask())
        assert pixel(result, INSIDE) == (0, 94, 40)
        assert pixel(result, OUTSIDE) == (0, 0, 0)

    def test_opacity_one_paints_the_solid_color(self):
        result = draw_egopath(canvas(60), region_mask(), opacity=1.0)
        assert pixel(result, INSIDE) == GREEN
        assert pixel(result, OUTSIDE) == (60, 60, 60)

    def test_an_empty_mask_draws_nothing(self):
        img = canvas(60)
        empty = Image.fromarray(np.zeros((SIZE, SIZE), dtype=np.uint8))
        np.testing.assert_array_equal(np.array(draw_egopath(img, empty)), np.array(img))

    def test_agrees_with_the_rails_overlay_on_the_same_region(self):
        from_rails = draw_egopath(canvas(60), RAILS)
        from_mask = draw_egopath(canvas(60), region_mask())
        assert pixel(from_rails, INSIDE) == pixel(from_mask, INSIDE)

    def test_a_numpy_mask_is_accepted_like_a_pil_one(self):
        # the docstring advertises numpy.ndarray, so it must reach an overlay
        # branch instead of silently producing an untouched copy
        from_pil = draw_egopath(canvas(60), region_mask())
        from_numpy = draw_egopath(canvas(60), np.array(region_mask()))
        np.testing.assert_array_equal(np.array(from_numpy), np.array(from_pil))
        assert pixel(from_numpy, INSIDE) != (60, 60, 60)

    def test_a_numpy_mask_honours_opacity_and_color(self):
        result = draw_egopath(
            canvas(), np.array(region_mask()), opacity=1.0, color=(200, 100, 50)
        )
        assert pixel(result, INSIDE) == (200, 100, 50)
        assert pixel(result, OUTSIDE) == (0, 0, 0)


class TestCropRectangle:
    def test_draws_a_red_outline_at_the_inclusive_coordinates(self):
        result = draw_egopath(canvas(), RAILS, crop_coords=(3, 3, 10, 10))
        array = np.array(result)
        # the given coordinates are the outline itself, corners included
        assert tuple(array[3, 3]) == RED
        assert tuple(array[10, 10]) == RED
        assert tuple(array[3, 10]) == RED
        assert tuple(array[10, 3]) == RED
        assert tuple(array[4, 4]) != RED  # one pixel in is already the overlay

    def test_is_not_drawn_without_crop_coordinates(self):
        assert not (np.array(draw_egopath(canvas(), RAILS))[:, :, 0] > 0).any()

    def test_is_drawn_on_top_of_the_overlay(self):
        # the rectangle sits inside the detected region, and must still be visible
        result = draw_egopath(canvas(), RAILS, crop_coords=(5, 5, 12, 12))
        assert tuple(np.array(result)[5, 5]) == RED

    def test_a_full_frame_crop_stays_within_the_image(self):
        # Autocropper hands over inclusive coordinates, so the full frame is
        # (0, 0, width - 1, height - 1) and must not raise or spill over
        result = draw_egopath(canvas(), RAILS, crop_coords=(0, 0, SIZE - 1, SIZE - 1))
        array = np.array(result)
        assert tuple(array[0, 0]) == RED
        assert tuple(array[SIZE - 1, SIZE - 1]) == RED
