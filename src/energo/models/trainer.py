"""Training loop for parametric price models.

Handles training, validation, early stopping, checkpointing,
and learning rate scheduling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from energo.models.parametric import ParametricLoss, ParametricPriceModel

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10
    lr_factor: float = 0.5
    lr_patience: int = 5
    min_lr: float = 1e-6
    gradient_clip: float = 1.0
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))


@dataclass
class TrainResult:
    """Training results."""

    train_losses: list[float]
    val_losses: list[float]
    best_epoch: int
    best_val_loss: float


class Trainer:
    """Model trainer with early stopping and LR scheduling."""

    def __init__(
        self,
        model: ParametricPriceModel,
        config: TrainConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or TrainConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self._loss_fn = ParametricLoss(distribution=model.distribution)
        self._optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
        )
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer,
            mode="min",
            factor=self.config.lr_factor,
            patience=self.config.lr_patience,
            min_lr=self.config.min_lr,
        )

    def train(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        val_x: torch.Tensor,
        val_y: torch.Tensor,
    ) -> TrainResult:
        """Run the training loop.

        Args:
            train_x: Training sequences (N, seq_len, features).
            train_y: Training targets (N,).
            val_x: Validation sequences.
            val_y: Validation targets.

        Returns:
            TrainResult with loss history.
        """
        train_loader = self._make_loader(train_x, train_y, shuffle=True)
        val_loader = self._make_loader(val_x, val_y, shuffle=False)

        train_losses: list[float] = []
        val_losses: list[float] = []
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0

        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.config.max_epochs + 1):
            # Training
            train_loss = self._train_epoch(train_loader)
            train_losses.append(train_loss)

            # Validation
            val_loss = self._validate(val_loader)
            val_losses.append(val_loss)

            # LR scheduling
            self._scheduler.step(val_loss)
            current_lr = self._optimizer.param_groups[0]["lr"]

            logger.info(
                "Epoch %3d | train_loss=%.4f | val_loss=%.4f | lr=%.2e",
                epoch,
                train_loss,
                val_loss,
                current_lr,
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                self._save_checkpoint("best.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info(
                        "Early stopping at epoch %d (best=%d, val_loss=%.4f)",
                        epoch,
                        best_epoch,
                        best_val_loss,
                    )
                    break

        # Load best model
        self._load_checkpoint("best.pt")

        return TrainResult(
            train_losses=train_losses,
            val_losses=val_losses,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
        )

    def _train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        count = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self._optimizer.zero_grad()
            params = self.model(batch_x)
            loss = self._loss_fn(params, batch_y)

            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip,
            )
            self._optimizer.step()

            total_loss += loss.item() * len(batch_x)
            count += len(batch_x)

        return total_loss / count if count > 0 else 0.0

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        count = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            params = self.model(batch_x)
            loss = self._loss_fn(params, batch_y)

            total_loss += loss.item() * len(batch_x)
            count += len(batch_x)

        return total_loss / count if count > 0 else 0.0

    def _make_loader(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        shuffle: bool,
    ) -> DataLoader:
        """Create a DataLoader from tensors."""
        dataset = TensorDataset(x, y)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def _save_checkpoint(self, name: str) -> None:
        """Save model checkpoint."""
        path = self.config.checkpoint_dir / name
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "config": {
                    "input_dim": self.model.input_dim,
                    "hidden_dim": self.model.hidden_dim,
                    "distribution": self.model.distribution,
                },
            },
            path,
        )

    def _load_checkpoint(self, name: str) -> None:
        """Load model checkpoint."""
        path = self.config.checkpoint_dir / name
        if not path.exists():
            logger.warning("Checkpoint not found: %s", path)
            return

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded checkpoint: %s", path)
