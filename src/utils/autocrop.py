import numpy as np
from PIL import Image


class Autocropper:
    def __init__(self, config, coeff=0.1):
        self.coeff = coeff  # running average coefficient
        self.crop_margin_sides = config["crop_margin_sides"]
        self.crop_margin_top = config["crop_margin_top"]
        self.n = 0
        self.avg = None
        self.crop_coords = None

    def __call__(self):
        if self.crop_coords is None:
            return None
        return tuple(round(coord) for coord in self.crop_coords)

    def rails_coords(self, pred):
        coords = None
        if isinstance(pred, list):
            rails = np.array(pred)
            if rails.size > 0:
                coords = (
                    np.min(rails[0, :, 0]).item(),
                    np.min(rails[:, :, 1]).item(),
                    np.max(rails[1, :, 0]).item(),
                )
        elif isinstance(pred, Image.Image):
            mask = np.array(pred)
            mask = np.nonzero(mask)
            if mask[0].size > 0:
                coords = (
                    np.min(mask[1]).item(),
                    np.min(mask[0]).item(),
                    np.max(mask[1]).item(),
                )
        return coords

    def update(self, img_shape, pred):
        rails_coords = self.rails_coords(pred)
        if rails_coords is None:
            return
        xmax, ymax = img_shape[0] - 1, img_shape[1] - 1  # inclusive image bounds
        if self.n == 0:  # first update
            self.avg = [0, 0, xmax]
            # the ego-path always reaches the bottom of the frame, so ybottom is
            # fixed and only the 3 other coordinates are tracked below
            self.crop_coords = [0, 0, xmax, ymax]
        else:
            for i in range(3):
                self.avg[i] = (self.avg[i] * self.n + rails_coords[i]) / (self.n + 1)
        new_left = min(rails_coords[0], self.avg[0])  # self.avg to prevent collapse
        new_right = max(rails_coords[2], self.avg[2])
        new_top = min(rails_coords[1], self.avg[1])
        margin_sides = self.crop_margin_sides * (new_right - new_left)
        margin_top = self.crop_margin_top * (img_shape[1] - new_top)
        new_coords = (
            max(new_left - margin_sides, 0),
            max(new_top - margin_top, 0),
            min(new_right + margin_sides, xmax),
        )
        for i in range(3):
            # kept in float: rounding here would stall the running average as soon
            # as its per-step increment falls below one pixel
            self.crop_coords[i] = (
                self.crop_coords[i] * (1 - self.coeff) + new_coords[i] * self.coeff
            )
        self.n += 1
