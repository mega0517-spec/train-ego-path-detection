"""Small builders shared by the test modules.

Everything here is deliberately dependency-light (numpy + Pillow only) so that
the pure-logic tests can run without torch installed.
"""

import numpy as np
from PIL import Image

# Canonical toy rails mask used across the dataset tests.
#
# 10x10 mask, two perfectly vertical rails at columns 2 and 7, starting at row 5.
# Rows 0-4 are empty (0 rail points), rows 5-9 hold exactly 2 rail points each.
MASK_SIZE = 10
LEFT_COL = 2
RIGHT_COL = 7
TOP_ROW = 5


def make_rails_mask(height=MASK_SIZE, width=MASK_SIZE, left=LEFT_COL, right=RIGHT_COL, top=TOP_ROW):
    """Builds a binary rails mask with two vertical rails.

    Args:
        height (int): Number of rows of the mask.
        width (int): Number of columns of the mask.
        left (int): Column index of the left rail.
        right (int): Column index of the right rail.
        top (int): First row (from the top) where the rails start.

    Returns:
        numpy.ndarray: Mask of shape (height, width) with 1 on the rail pixels.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top:, left] = 1
    mask[top:, right] = 1
    return mask


def make_image(width, height, seed=0):
    """Builds a deterministic RGB image (no dependency on the global RNG state)."""
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(array)


def make_annotation(width, height, left_bottom, left_top, right_bottom, right_top, top_y):
    """Builds a single annotation entry with two straight rails reaching the bottom row."""
    return {
        "left_rail": [[left_bottom, height - 1], [left_top, top_y]],
        "right_rail": [[right_bottom, height - 1], [right_top, top_y]],
    }


def mask_from_rails(rails, shape):
    """Rasterizes rail point lists into a binary mask, mirroring PathsDataset conventions."""
    mask = np.zeros(shape, dtype=np.uint8)
    for rail in rails:
        for x, y in rail:
            mask[int(y), int(x)] = 1
    return mask
