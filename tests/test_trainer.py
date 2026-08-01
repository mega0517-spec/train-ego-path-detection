import os
import sys
import types

import pytest
import torch
import torch.nn as nn

from src.utils.common import simple_logger


@pytest.fixture
def trainer(monkeypatch):
    """Imports src.utils.trainer with wandb stubbed out, and exposes the logged runs."""
    logged = []
    stub = types.ModuleType("wandb")
    stub.log = logged.append
    monkeypatch.setitem(sys.modules, "wandb", stub)
    import src.utils.trainer as module

    monkeypatch.setattr(module, "wandb", stub, raising=False)
    return module, logged


class TinyModel(nn.Module):
    """Smallest thing that can be optimized: a single learnable vector."""

    def __init__(self, size=4):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        return self.weight.expand(x.shape[0], -1)


def single_target_batches(size=4, batches=3):
    return [(torch.zeros(2, 1), torch.ones(2, size)) for _ in range(batches)]


def two_target_batches(anchors=2, batches=3):
    traj = torch.tensor([[[0.2] * anchors, [0.8] * anchors]]).repeat(2, 1, 1)
    ylim = torch.ones(2)
    return [(torch.zeros(2, 1), traj, ylim) for _ in range(batches)]


class TestTrainEpoch:
    def test_returns_the_mean_batch_loss(self, trainer):
        module, _ = trainer
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # frozen weights
        loss = module.train_epoch(
            model, nn.MSELoss(), torch.device("cpu"), single_target_batches(), optimizer
        )
        assert loss == pytest.approx(1.0)  # (0 - 1)^2 on every batch

    def test_updates_the_weights(self, trainer):
        module, _ = trainer
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        module.train_epoch(
            model, nn.MSELoss(), torch.device("cpu"), single_target_batches(), optimizer
        )
        assert model.weight.abs().sum().item() > 0

    def test_leaves_the_model_in_training_mode(self, trainer):
        module, _ = trainer
        model = TinyModel().eval()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        module.train_epoch(
            model, nn.MSELoss(), torch.device("cpu"), single_target_batches(), optimizer
        )
        assert model.training

    def test_supports_multi_target_batches(self, trainer):
        module, _ = trainer
        from src.nn.loss import TrainEgoPathRegressionLoss

        model = TinyModel(size=5)  # 2 rails x 2 anchors + 1 y-limit
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss = module.train_epoch(
            model,
            TrainEgoPathRegressionLoss(0.5),
            torch.device("cpu"),
            two_target_batches(),
            optimizer,
        )
        assert loss > 0
        assert model.weight.abs().sum().item() > 0


class TestValEpoch:
    def test_does_not_update_the_weights(self, trainer):
        module, _ = trainer
        model = TinyModel()
        before = model.weight.detach().clone()
        module.val_epoch(model, nn.MSELoss(), torch.device("cpu"), single_target_batches())
        torch.testing.assert_close(model.weight.detach(), before)

    def test_leaves_the_model_in_eval_mode(self, trainer):
        module, _ = trainer
        model = TinyModel().train()
        module.val_epoch(model, nn.MSELoss(), torch.device("cpu"), single_target_batches())
        assert not model.training

    def test_returns_the_mean_batch_loss(self, trainer):
        module, _ = trainer
        loss = module.val_epoch(
            TinyModel(), nn.MSELoss(), torch.device("cpu"), single_target_batches()
        )
        assert loss == pytest.approx(1.0)


class TestTrain:
    @staticmethod
    def run(module, tmp_path, epochs, val=True, scheduler=None):
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        module.train(
            epochs=epochs,
            dataloaders=(single_target_batches(), single_target_batches() if val else None),
            model=model,
            criterion=nn.MSELoss(),
            optimizer=optimizer,
            scheduler=scheduler,
            save_path=str(tmp_path),
            device=torch.device("cpu"),
            logger=simple_logger(f"test_train_{epochs}_{val}", "critical"),
        )
        return model

    def test_saves_the_best_weights(self, trainer, tmp_path):
        module, _ = trainer
        self.run(module, tmp_path, epochs=20)
        assert os.path.exists(tmp_path / "best.pt")
        assert torch.load(tmp_path / "best.pt")["weight"].abs().sum() > 0

    def test_the_loss_goes_down(self, trainer, tmp_path):
        module, logged = trainer
        self.run(module, tmp_path, epochs=6)
        losses = [entry["train_loss"] for entry in logged if "train_loss" in entry]
        assert len(losses) == 6
        assert losses[-1] < losses[0]

    def test_logs_the_best_validation_loss_once(self, trainer, tmp_path):
        module, logged = trainer
        self.run(module, tmp_path, epochs=4)
        assert sum("best_val_loss" in entry for entry in logged) == 1

    def test_runs_without_a_validation_loader(self, trainer, tmp_path):
        module, logged = trainer
        self.run(module, tmp_path, epochs=20, val=False)
        assert all(entry["val_loss"] == 0 for entry in logged if "val_loss" in entry)
        assert os.path.exists(tmp_path / "best.pt")

    def test_steps_the_scheduler_once_per_epoch(self, trainer, tmp_path):
        module, _ = trainer
        steps = []

        scheduler = types.SimpleNamespace(step=lambda: steps.append(1))
        self.run(module, tmp_path, epochs=3, scheduler=scheduler)
        assert len(steps) == 3

    @pytest.mark.parametrize("epochs, expected_saves", [(1, 0), (4, 0), (10, 1), (20, 2)])
    def test_checkpointing_window(self, trainer, tmp_path, monkeypatch, epochs, expected_saves):
        # documents current behaviour: the guard is `epoch >= epochs * 0.9` on a
        # zero-indexed epoch, so the eligible window is one epoch shorter than the
        # intended last 10% -- and short runs finish with no weights on disk at all
        module, _ = trainer
        saves = []
        original_save = torch.save
        monkeypatch.setattr(
            module.torch, "save", lambda *a, **k: (saves.append(a[1]), original_save(*a, **k))[1]
        )
        self.run(module, tmp_path, epochs=epochs)
        assert len(saves) == expected_saves
        assert os.path.exists(tmp_path / "best.pt") is bool(expected_saves)
