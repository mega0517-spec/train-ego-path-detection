import logging

import numpy as np
import pytest
import torch
from PIL import Image

from src.utils.common import set_seeds, set_worker_seeds, simple_logger, split_dataset, to_scaled_tensor


class TestSplitDataset:
    def test_splits_at_the_expected_boundaries(self):
        indices = list(range(10))
        train, val, test = split_dataset(indices, (0.8, 0.1, 0.1))
        assert train == list(range(8))
        assert val == [8]
        assert test == [9]

    def test_splits_are_contiguous_and_disjoint(self):
        indices = list(range(37))
        train, val, test = split_dataset(indices, (0.5, 0.25, 0.25))
        assert train + val + test == indices[: len(train) + len(val) + len(test)]
        assert not set(train) & set(val)
        assert not set(val) & set(test)

    @pytest.mark.parametrize("size", [1, 2, 3, 7, 10, 99, 100])
    @pytest.mark.parametrize(
        "proportions", [(0.8, 0.1, 0.1), (0.5, 0.25, 0.25), (1.0, 0.0, 0.0), (0.34, 0.33, 0.33)]
    )
    def test_never_produces_more_samples_than_available(self, size, proportions):
        indices = list(range(size))
        train, val, test = split_dataset(indices, proportions)
        assert len(train) + len(val) + len(test) <= size

    def test_truncation_can_silently_drop_samples(self):
        # documents current behaviour: int() truncation means proportions summing to
        # less than 1 (or hitting float rounding) leave trailing samples unused
        train, val, test = split_dataset(list(range(10)), (0.3, 0.3, 0.3))
        assert len(train) + len(val) + len(test) < 10

    def test_empty_dataset(self):
        assert split_dataset([], (0.8, 0.1, 0.1)) == ([], [], [])

    def test_preserves_the_given_order(self):
        indices = [5, 3, 9, 1, 7, 2, 8, 4, 6, 0]
        train, val, test = split_dataset(indices, (0.8, 0.1, 0.1))
        assert train == indices[:8]
        assert val == indices[8:9]


class TestSimpleLogger:
    def test_filters_messages_below_the_configured_level(self, capsys):
        logger = simple_logger("test_filters", "warning")
        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        captured = capsys.readouterr().err
        assert "debug message" not in captured
        assert "info message" not in captured
        assert "warning message" in captured

    def test_formats_the_message_without_any_prefix(self, capsys):
        logger = simple_logger("test_format", "info")
        logger.info("bare message")
        assert capsys.readouterr().err == "bare message\n"

    def test_custom_terminator(self, capsys):
        logger = simple_logger("test_terminator", "info", terminator="")
        logger.info("no newline")
        assert capsys.readouterr().err == "no newline"

    def test_does_not_propagate_to_the_root_logger(self):
        logger = simple_logger("test_propagate", "info")
        assert logger.propagate is False

    @pytest.mark.parametrize(
        "level, expected",
        [
            ("debug", logging.DEBUG),
            ("info", logging.INFO),
            ("warning", logging.WARNING),
            ("error", logging.ERROR),
            ("critical", logging.CRITICAL),
        ],
    )
    def test_level_mapping(self, level, expected):
        assert simple_logger(f"test_level_{level}", level).level == expected

    def test_unknown_level_raises(self):
        with pytest.raises(KeyError):
            simple_logger("test_unknown", "verbose")

    def test_repeated_calls_stack_handlers(self, capsys):
        # documents current behaviour: nothing clears existing handlers, so calling
        # simple_logger() twice with the same name duplicates every message
        simple_logger("test_stacking", "info")
        logger = simple_logger("test_stacking", "info")
        logger.info("twice")
        assert capsys.readouterr().err == "twice\ntwice\n"


class TestSeeding:
    def test_set_seeds_makes_every_rng_reproducible(self):
        import random

        set_seeds(1234)
        first = (random.random(), np.random.rand(), torch.rand(1).item())
        set_seeds(1234)
        second = (random.random(), np.random.rand(), torch.rand(1).item())
        assert first == second

    def test_different_seeds_give_different_draws(self):
        set_seeds(1)
        first = np.random.rand()
        set_seeds(2)
        assert np.random.rand() != first

    def test_set_worker_seeds_derives_from_the_torch_seed(self):
        import random

        torch.manual_seed(7)
        set_worker_seeds(worker_id=0)
        first = (random.random(), np.random.rand())
        torch.manual_seed(7)
        set_worker_seeds(worker_id=3)  # worker_id is unused, the torch seed drives it
        assert (random.random(), np.random.rand()) == first


class TestToScaledTensor:
    def test_scales_uint8_images_to_the_unit_range(self):
        img = Image.fromarray(np.array([[0, 128, 255]], dtype=np.uint8))
        tensor = to_scaled_tensor(img)
        assert tensor.dtype == torch.float32
        assert tensor.min().item() == pytest.approx(0.0)
        assert tensor.max().item() == pytest.approx(1.0)
        assert tensor[0, 0, 1].item() == pytest.approx(128 / 255, abs=1e-6)

    def test_moves_channels_first(self):
        img = Image.fromarray(np.zeros((7, 11, 3), dtype=np.uint8))
        assert to_scaled_tensor(img).shape == (3, 7, 11)

    def test_accepts_numpy_arrays(self):
        array = np.zeros((7, 11, 3), dtype=np.uint8)
        assert to_scaled_tensor(array).shape == (3, 7, 11)
