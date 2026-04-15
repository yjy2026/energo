"""Integration test: full pipeline from data to evaluation.

Uses synthetic data to test the complete flow without network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from energo.data.pipeline import DataPipeline
from energo.evaluation.metrics import evaluate
from energo.features.engineering import build_features, get_feature_columns
from energo.features.scaler import FeatureScaler
from energo.models.parametric import ParametricLoss, ParametricPriceModel, create_sequences


def _make_synthetic_dataset(n_days: int = 30) -> pd.DataFrame:
    """Create a realistic synthetic JEPX-like dataset."""
    n_periods = n_days * 48  # 30-min intervals
    timestamps = pd.date_range(
        start="2024-01-01",
        periods=n_periods,
        freq="30min",
        tz="Asia/Tokyo",
    )
    rng = np.random.default_rng(42)

    # Simulate realistic price patterns
    hour_of_day = timestamps.hour + timestamps.minute / 60
    # Peak at 10am and 6pm, low at 3am
    daily_pattern = 3 * np.sin(2 * np.pi * (hour_of_day - 6) / 24) + 1.5 * np.sin(
        2 * np.pi * (hour_of_day - 14) / 24
    )
    # Weekend effect
    is_weekend = timestamps.dayofweek >= 5
    weekend_effect = np.where(is_weekend, -2.0, 0.0)

    base_price = 12.0
    noise = rng.standard_normal(n_periods) * 1.5
    price = base_price + daily_pattern + weekend_effect + noise
    price = np.maximum(price, 0.01)  # Prices can't be negative in spot market

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": price,
            "demand_mw": 30000 + daily_pattern * 3000 + rng.standard_normal(n_periods) * 500,
            "temperature_c": (
                5 + 5 * np.sin(2 * np.pi * np.arange(n_periods) / (48 * 365))
                + rng.standard_normal(n_periods) * 2
            ),
            "solar_radiation_wm2": np.maximum(
                0,
                400 * np.sin(np.pi * (hour_of_day - 6) / 12)
                * (hour_of_day > 6) * (hour_of_day < 18),
            ) + rng.standard_normal(n_periods).clip(-50, 50),
            "wind_speed_ms": np.abs(3 + rng.standard_normal(n_periods) * 1.5),
        }
    )


class TestFullPipeline:
    """Integration test covering data → features → model → evaluation."""

    def test_end_to_end(self) -> None:
        """Test complete pipeline with synthetic data."""
        # Step 1: Build dataset
        df = _make_synthetic_dataset(n_days=30)
        assert len(df) == 30 * 48

        # Step 2: Feature engineering
        featured = build_features(df)
        feature_cols = get_feature_columns(featured)
        assert len(feature_cols) > 10

        # Step 3: Drop NaN rows (from lag features)
        featured_clean = featured.dropna(subset=feature_cols).reset_index(drop=True)
        assert len(featured_clean) > 100

        # Step 4: Scale features
        scaler = FeatureScaler(method="standard")
        scaler.fit(featured_clean, feature_cols)
        scaled = scaler.transform(featured_clean)

        # Step 5: Temporal split
        split = DataPipeline.split_temporal(scaled, train_ratio=0.7, val_ratio=0.15)
        assert split.sizes["train"] > split.sizes["val"]
        assert split.sizes["val"] > 0
        assert split.sizes["test"] > 0

        # Step 6: Create sequences
        def to_tensors(data: pd.DataFrame):
            features = torch.tensor(
                data[feature_cols].values, dtype=torch.float32
            )
            targets = torch.tensor(
                data["price"].values, dtype=torch.float32
            )
            return features, targets

        train_f, train_t = to_tensors(split.train)
        test_f, test_t = to_tensors(split.test)

        seq_len = 24  # 12 hours
        train_x, train_y = create_sequences(train_f, train_t, seq_len=seq_len)
        test_x, test_y = create_sequences(test_f, test_t, seq_len=seq_len)

        assert train_x.shape[1] == seq_len
        assert train_x.shape[2] == len(feature_cols)

        # Step 7: Model forward pass
        model = ParametricPriceModel(
            input_dim=len(feature_cols),
            hidden_dim=32,
            num_layers=1,
            dropout=0.0,
        )

        with torch.no_grad():
            output = model(test_x[:16])

        assert output["mu"].shape == (16,)
        assert (output["sigma"] > 0).all()

        # Step 8: Loss computation
        loss_fn = ParametricLoss(distribution="gaussian")
        loss = loss_fn(output, test_y[:16])
        assert torch.isfinite(loss)

        # Step 9: Quick training (just verify it runs)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _ in range(3):
            optimizer.zero_grad()
            out = model(train_x[:32])
            loss = loss_fn(out, train_y[:32])
            loss.backward()
            optimizer.step()

        # Step 10: Evaluation
        model.eval()
        with torch.no_grad():
            pred = model(test_x)
            mu = pred["mu"].numpy()
            sigma = pred["sigma"].numpy()

        result = evaluate(test_y.numpy(), mu, sigma)

        assert result.mae > 0
        assert result.rmse > 0
        assert result.crps > 0
        assert 0 <= result.coverage_90 <= 1
        assert 0 <= result.coverage_95 <= 1

        # Verify the summary can be generated
        summary = result.summary()
        assert "MAE" in summary

    def test_scaler_save_load(self, tmp_path) -> None:
        """Test scaler serialization round-trip."""
        df = _make_synthetic_dataset(n_days=7)
        featured = build_features(df)
        feature_cols = get_feature_columns(featured)
        featured_clean = featured.dropna(subset=feature_cols)

        scaler = FeatureScaler(method="standard")
        scaler.fit(featured_clean, feature_cols)

        # Save
        save_path = tmp_path / "scaler.json"
        scaler.save(save_path)

        # Load
        loaded = FeatureScaler().load(save_path)

        # Transform should give same results
        original = scaler.transform(featured_clean)
        reloaded = loaded.transform(featured_clean)

        for col in feature_cols:
            if col in original.columns:
                np.testing.assert_array_almost_equal(
                    original[col].values[:10],
                    reloaded[col].values[:10],
                )

    def test_split_temporal_no_leakage(self) -> None:
        """Verify temporal split has no overlap."""
        df = _make_synthetic_dataset(n_days=14)

        split = DataPipeline.split_temporal(df, train_ratio=0.7, val_ratio=0.15)

        train_max = split.train["timestamp"].max()
        val_min = split.val["timestamp"].min()
        val_max = split.val["timestamp"].max()
        test_min = split.test["timestamp"].min()

        assert train_max < val_min, "Train/val overlap!"
        assert val_max < test_min, "Val/test overlap!"
