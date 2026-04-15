"""Tests for feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energo.features.engineering import build_features, get_feature_columns


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Create a sample time-series DataFrame for testing."""
    n_periods = 48 * 7  # 1 week of 30-min data
    timestamps = pd.date_range(
        start="2024-01-01",
        periods=n_periods,
        freq="30min",
        tz="Asia/Tokyo",
    )
    rng = np.random.default_rng(42)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": 10 + rng.standard_normal(n_periods) * 2,
            "demand_mw": 30000 + rng.standard_normal(n_periods) * 3000,
            "temperature_c": 5 + rng.standard_normal(n_periods) * 3,
            "solar_radiation_wm2": np.maximum(
                0, 200 * np.sin(np.linspace(0, 14 * np.pi, n_periods))
            ),
        }
    )


class TestBuildFeatures:
    """Tests for feature engineering pipeline."""

    def test_time_features_created(self, sample_df: pd.DataFrame) -> None:
        result = build_features(sample_df)

        assert "hour" in result.columns
        assert "day_of_week" in result.columns
        assert "month" in result.columns
        assert "period_of_day" in result.columns

    def test_cyclical_encoding(self, sample_df: pd.DataFrame) -> None:
        result = build_features(sample_df)

        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        # Sin/cos should be in [-1, 1]
        assert result["hour_sin"].between(-1, 1).all()
        assert result["hour_cos"].between(-1, 1).all()

    def test_calendar_features(self, sample_df: pd.DataFrame) -> None:
        result = build_features(sample_df)

        assert "is_weekend" in result.columns
        assert "is_holiday" in result.columns
        assert "is_golden_week" in result.columns
        assert "is_obon" in result.columns
        # Binary features
        assert set(result["is_weekend"].unique()).issubset({0, 1})

    def test_lag_features_no_leakage(self, sample_df: pd.DataFrame) -> None:
        """Verify lag features use only past data."""
        result = build_features(sample_df, lag_periods=(1, 48))

        # First lag value should be NaN (no previous data)
        assert pd.isna(result["price_lag_1"].iloc[0])

        # lag_1 at position i should equal price at position i-1
        for i in range(1, min(10, len(result))):
            expected = result["price"].iloc[i - 1]
            actual = result["price_lag_1"].iloc[i]
            assert np.isclose(actual, expected), f"Lag mismatch at position {i}"

    def test_rolling_features_no_leakage(self, sample_df: pd.DataFrame) -> None:
        """Verify rolling stats use shift(1) to avoid leakage."""
        result = build_features(sample_df, rolling_windows=(48,))

        # Rolling mean should exist
        assert "price_ma_48" in result.columns

        # First value should be from shifted data
        assert pd.isna(result["price_ma_48"].iloc[0])

    def test_return_features(self, sample_df: pd.DataFrame) -> None:
        result = build_features(sample_df)

        assert "return_30m" in result.columns
        assert "return_24h" in result.columns

    def test_net_demand(self, sample_df: pd.DataFrame) -> None:
        result = build_features(sample_df)

        assert "net_demand" in result.columns

    def test_temperature_deviation(self, sample_df: pd.DataFrame) -> None:
        result = build_features(sample_df)

        assert "temp_deviation" in result.columns

    def test_output_length_unchanged(self, sample_df: pd.DataFrame) -> None:
        """Feature engineering should not change row count."""
        result = build_features(sample_df)
        assert len(result) == len(sample_df)


class TestGetFeatureColumns:
    """Tests for feature column selection."""

    def test_excludes_metadata(self, sample_df: pd.DataFrame) -> None:
        result = build_features(sample_df)
        feature_cols = get_feature_columns(result)

        assert "timestamp" not in feature_cols
        assert "price" not in feature_cols
        assert "region" not in feature_cols

    def test_includes_engineered(self, sample_df: pd.DataFrame) -> None:
        result = build_features(sample_df)
        feature_cols = get_feature_columns(result)

        assert "hour_sin" in feature_cols
        assert "is_weekend" in feature_cols
        assert len(feature_cols) > 10
