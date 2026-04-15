"""Tests for model predictor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from energo.models.parametric import ParametricPriceModel
from energo.models.predictor import Predictor, PriceForecast
from energo.models.trainer import TrainConfig, Trainer

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def trained_model(tmp_path: Path) -> tuple[ParametricPriceModel, Path]:
    """Create and train a small model, return model and checkpoint path."""
    rng = torch.Generator().manual_seed(42)
    input_dim = 5
    seq_len = 8

    total = 100
    features = torch.randn(total, input_dim, generator=rng)
    targets = features[:, 0] + torch.randn(total, generator=rng) * 0.1

    from energo.models.parametric import create_sequences
    seqs, tgts = create_sequences(features, targets, seq_len=seq_len)
    split = int(len(seqs) * 0.8)

    model = ParametricPriceModel(
        input_dim=input_dim, hidden_dim=16, num_layers=1, dropout=0.0,
    )
    config = TrainConfig(max_epochs=3, batch_size=16, checkpoint_dir=tmp_path)
    trainer = Trainer(model=model, config=config)
    trainer.train(seqs[:split], tgts[:split], seqs[split:], tgts[split:])

    return model, tmp_path / "best.pt"


class TestPredictor:
    """Tests for the inference predictor."""

    def test_predict_returns_forecasts(
        self, trained_model: tuple[ParametricPriceModel, Path],
    ) -> None:
        model, _ = trained_model
        predictor = Predictor(model=model)

        x = torch.randn(8, 8, 5)
        forecasts = predictor.predict(x)

        assert len(forecasts) == 8
        assert all(isinstance(f, PriceForecast) for f in forecasts)

    def test_forecast_structure(
        self, trained_model: tuple[ParametricPriceModel, Path],
    ) -> None:
        model, _ = trained_model
        predictor = Predictor(model=model)

        x = torch.randn(4, 8, 5)
        forecasts = predictor.predict(x)

        for f in forecasts:
            assert f.std > 0
            assert f.ci_lower_90 < f.mean < f.ci_upper_90
            assert f.ci_lower_95 < f.ci_lower_90
            assert f.ci_upper_95 > f.ci_upper_90
            assert f.ci_width_90 > 0

    def test_from_checkpoint(
        self, trained_model: tuple[ParametricPriceModel, Path],
    ) -> None:
        _, ckpt_path = trained_model
        predictor = Predictor.from_checkpoint(
            checkpoint_path=ckpt_path,
            input_dim=5,
            hidden_dim=16,
            num_layers=1,
        )

        x = torch.randn(4, 8, 5)
        forecasts = predictor.predict(x)
        assert len(forecasts) == 4

    def test_predict_raw(
        self, trained_model: tuple[ParametricPriceModel, Path],
    ) -> None:
        model, _ = trained_model
        predictor = Predictor(model=model)

        x = torch.randn(6, 8, 5)
        raw = predictor.predict_raw(x)

        assert "mu" in raw
        assert "sigma" in raw
        assert isinstance(raw["mu"], np.ndarray)
        assert raw["mu"].shape == (6,)
        assert (raw["sigma"] > 0).all()
