"""Tests for model trainer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from energo.models.parametric import ParametricPriceModel, create_sequences
from energo.models.trainer import TrainConfig, Trainer, TrainResult

if TYPE_CHECKING:
    from pathlib import Path


def _make_synthetic_data(
    n_samples: int = 200,
    input_dim: int = 5,
    seq_len: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create synthetic train/val sequences."""
    rng = torch.Generator().manual_seed(42)
    total = n_samples + seq_len

    features = torch.randn(total, input_dim, generator=rng)
    # Target has a learnable pattern: sum of first 2 features + noise
    targets = features[:, 0] + features[:, 1] + torch.randn(total, generator=rng) * 0.1

    seqs, tgts = create_sequences(features, targets, seq_len=seq_len)

    split = int(len(seqs) * 0.8)
    return seqs[:split], tgts[:split], seqs[split:], tgts[split:]


class TestTrainer:
    """Tests for the training loop."""

    def test_train_completes(self, tmp_path: Path) -> None:
        """Training should run and return results."""
        train_x, train_y, val_x, val_y = _make_synthetic_data()

        model = ParametricPriceModel(
            input_dim=5, hidden_dim=16, num_layers=1, dropout=0.0,
        )
        config = TrainConfig(
            max_epochs=3,
            batch_size=32,
            patience=10,
            checkpoint_dir=tmp_path,
        )

        trainer = Trainer(model=model, config=config)
        result = trainer.train(train_x, train_y, val_x, val_y)

        assert isinstance(result, TrainResult)
        assert len(result.train_losses) == 3
        assert len(result.val_losses) == 3
        assert result.best_epoch >= 1

    def test_loss_decreases(self, tmp_path: Path) -> None:
        """Loss should decrease over training."""
        train_x, train_y, val_x, val_y = _make_synthetic_data()

        model = ParametricPriceModel(
            input_dim=5, hidden_dim=32, num_layers=1, dropout=0.0,
        )
        config = TrainConfig(
            max_epochs=10,
            batch_size=32,
            learning_rate=1e-3,
            patience=20,
            checkpoint_dir=tmp_path,
        )

        trainer = Trainer(model=model, config=config)
        result = trainer.train(train_x, train_y, val_x, val_y)

        # Last train loss should be lower than first
        assert result.train_losses[-1] < result.train_losses[0]

    def test_early_stopping(self, tmp_path: Path) -> None:
        """Training should stop early if val loss doesn't improve."""
        train_x, train_y, val_x, val_y = _make_synthetic_data(n_samples=50)

        model = ParametricPriceModel(
            input_dim=5, hidden_dim=4, num_layers=1, dropout=0.0,
        )
        config = TrainConfig(
            max_epochs=200,
            batch_size=16,
            patience=3,
            checkpoint_dir=tmp_path,
        )

        trainer = Trainer(model=model, config=config)
        result = trainer.train(train_x, train_y, val_x, val_y)

        # Should stop before max_epochs (patience triggers)
        assert len(result.train_losses) < 200

    def test_checkpoint_saved(self, tmp_path: Path) -> None:
        """Best checkpoint should be saved."""
        train_x, train_y, val_x, val_y = _make_synthetic_data()

        model = ParametricPriceModel(
            input_dim=5, hidden_dim=16, num_layers=1,
        )
        config = TrainConfig(max_epochs=3, checkpoint_dir=tmp_path)

        trainer = Trainer(model=model, config=config)
        trainer.train(train_x, train_y, val_x, val_y)

        assert (tmp_path / "best.pt").exists()

    def test_checkpoint_loadable(self, tmp_path: Path) -> None:
        """Saved checkpoint should be loadable."""
        train_x, train_y, val_x, val_y = _make_synthetic_data()

        model = ParametricPriceModel(
            input_dim=5, hidden_dim=16, num_layers=1,
        )
        config = TrainConfig(max_epochs=3, checkpoint_dir=tmp_path)

        trainer = Trainer(model=model, config=config)
        trainer.train(train_x, train_y, val_x, val_y)

        # Load into a new model
        checkpoint = torch.load(
            tmp_path / "best.pt", map_location="cpu", weights_only=True,
        )
        new_model = ParametricPriceModel(
            input_dim=5, hidden_dim=16, num_layers=1,
        )
        new_model.load_state_dict(checkpoint["model_state_dict"])

        # Should produce same output (both on CPU)
        new_model.eval()
        model.cpu().eval()
        test_input = val_x[:4].cpu()
        with torch.no_grad():
            out1 = model(test_input)
            out2 = new_model(test_input)
        torch.testing.assert_close(out1["mu"], out2["mu"])

    def test_student_t_training(self, tmp_path: Path) -> None:
        """Student-t distribution should also train successfully."""
        train_x, train_y, val_x, val_y = _make_synthetic_data()

        model = ParametricPriceModel(
            input_dim=5, hidden_dim=16, num_layers=1,
            distribution="student_t",
        )
        config = TrainConfig(max_epochs=3, checkpoint_dir=tmp_path)

        trainer = Trainer(model=model, config=config)
        result = trainer.train(train_x, train_y, val_x, val_y)

        assert result.best_val_loss < float("inf")
