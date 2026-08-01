import json
import os

import pytest

from helpers import make_annotation, make_image

IMG_WIDTH = 64
IMG_HEIGHT = 64
RAIL_TOP_Y = 20

# Rails are straight, steep and always reach the bottom row of the image, which is
# what PathsDataset.random_crop assumes about the annotations.
ANNOTATION_SPECS = {
    "img_000.png": (24, 28, 40, 36),
    "img_001.png": (18, 30, 46, 34),
    "img_002.png": (28, 30, 38, 34),
}


@pytest.fixture(scope="session")
def data_config():
    """Data generation configuration, mirroring the keys used by PathsDataset."""
    return {
        "input_shape": [3, 32, 32],
        "anchors": 8,
        "classes": 16,
        "brightness": 0.5,
        "contrast": 0.5,
        "saturation": 0.5,
        "hue": 0.2,
        "crop_margin_sides": 0.1,
        "crop_margin_top": 0.1,
        "std_dev_factor_sides": 0.3,
        "std_dev_factor_top": 0.1,
    }


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory):
    """Creates a tiny on-disk dataset (images + annotations file).

    Returns:
        tuple: (imgs_path, annotations_path, number of images).
    """
    root = tmp_path_factory.mktemp("dataset")
    imgs_path = root / "images"
    imgs_path.mkdir()
    annotations = {}
    for seed, (name, spec) in enumerate(sorted(ANNOTATION_SPECS.items())):
        make_image(IMG_WIDTH, IMG_HEIGHT, seed=seed).save(imgs_path / name)
        left_bottom, left_top, right_bottom, right_top = spec
        annotations[name] = make_annotation(
            IMG_WIDTH, IMG_HEIGHT, left_bottom, left_top, right_bottom, right_top, RAIL_TOP_Y
        )
    annotations_path = root / "annotations.json"
    with open(annotations_path, "w") as f:
        json.dump(annotations, f)
    return str(imgs_path), str(annotations_path), len(annotations)


@pytest.fixture
def make_dataset(synthetic_dataset, data_config):
    """Factory building a PathsDataset over the synthetic dataset."""
    from src.utils.dataset import PathsDataset

    imgs_path, annotations_path, count = synthetic_dataset

    def _make(method="regression", indices=None, config=None, img_aug=False, to_tensor=False):
        merged = dict(data_config)
        merged.update(config or {})
        return PathsDataset(
            imgs_path=imgs_path,
            annotations_path=annotations_path,
            indices=list(range(count)) if indices is None else indices,
            config=merged,
            method=method,
            img_aug=img_aug,
            to_tensor=to_tensor,
        )

    return _make


@pytest.fixture
def model_dir(tmp_path):
    """Factory writing a trained-model directory (config.yaml + best.pt) for Detector.

    The weights are randomly initialized: these tests care about the plumbing
    (config -> model -> post-processing), not about detection quality.
    """
    import torch
    import yaml

    from src.nn.model import ClassificationNet, RegressionNet, SegmentationNet

    def _make(method, **overrides):
        config = {
            "method": method,
            "backbone": "resnet18",
            "input_shape": [3, 64, 64],
            "anchors": 4,
            "classes": 8,
            "pool_channels": 2,
            "fc_hidden_size": 8,
            "decoder_channels": [16, 8, 4, 4, 4],
            "crop_margin_sides": 0.1,
            "crop_margin_top": 0.1,
            "seed": 42,
            "test_iterations": 1,
        }
        config.update(overrides)
        if method == "classification":
            model = ClassificationNet(
                backbone=config["backbone"],
                input_shape=tuple(config["input_shape"]),
                anchors=config["anchors"],
                classes=config["classes"],
                pool_channels=config["pool_channels"],
                fc_hidden_size=config["fc_hidden_size"],
            )
        elif method == "regression":
            model = RegressionNet(
                backbone=config["backbone"],
                input_shape=tuple(config["input_shape"]),
                anchors=config["anchors"],
                pool_channels=config["pool_channels"],
                fc_hidden_size=config["fc_hidden_size"],
            )
        else:
            model = SegmentationNet(
                backbone=config["backbone"],
                decoder_channels=tuple(config["decoder_channels"]),
            )
        path = tmp_path / method
        path.mkdir(exist_ok=True)
        with open(path / "config.yaml", "w") as f:
            yaml.safe_dump(config, f)
        torch.save(model.state_dict(), os.path.join(path, "best.pt"))
        return str(path), config

    return _make
