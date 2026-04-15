"""Tests for backtest evaluation."""

from __future__ import annotations

import numpy as np

from energo.evaluation.backtest import (
    BacktestResult,
    compute_economic_value,
    rolling_backtest,
)


def _make_backtest_data(
    n_days: int = 14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic backtest data."""
    rng = np.random.default_rng(42)
    n = n_days * 48

    # Timestamps
    import pandas as pd
    timestamps = pd.date_range(
        "2024-01-01", periods=n, freq="30min", tz="Asia/Tokyo",
    ).values

    # Realistic daily pattern
    hours = np.arange(n) * 0.5 % 24
    base = 10 + 5 * np.sin(2 * np.pi * (hours - 6) / 24)
    actual = base + rng.standard_normal(n) * 1.5
    actual = np.maximum(actual, 1.0)

    # Good predictions
    mu = actual + rng.standard_normal(n) * 0.5
    sigma = np.abs(rng.standard_normal(n)) * 0.5 + 0.3

    return timestamps, actual, mu, sigma


class TestRollingBacktest:
    """Tests for rolling backtest."""

    def test_returns_result(self) -> None:
        timestamps, actual, mu, sigma = _make_backtest_data()
        result = rolling_backtest(timestamps, actual, mu, sigma)

        assert isinstance(result, BacktestResult)
        assert result.overall.mae > 0
        assert result.overall.crps > 0

    def test_by_hour_decomposition(self) -> None:
        timestamps, actual, mu, sigma = _make_backtest_data()
        result = rolling_backtest(timestamps, actual, mu, sigma)

        assert len(result.by_hour) > 0
        # Should have metrics for multiple hours
        assert len(result.by_hour) >= 20

        for hour, metrics in result.by_hour.items():
            assert 0 <= hour <= 23
            assert metrics.mae >= 0

    def test_by_day_of_week(self) -> None:
        timestamps, actual, mu, sigma = _make_backtest_data()
        result = rolling_backtest(timestamps, actual, mu, sigma)

        assert len(result.by_day_of_week) > 0
        for dow, metrics in result.by_day_of_week.items():
            assert 0 <= dow <= 6
            assert metrics.mae >= 0

    def test_predictions_dataframe(self) -> None:
        timestamps, actual, mu, sigma = _make_backtest_data()
        result = rolling_backtest(timestamps, actual, mu, sigma)

        df = result.predictions
        assert "timestamp" in df.columns
        assert "actual" in df.columns
        assert "mu" in df.columns
        assert "sigma" in df.columns
        assert len(df) == len(actual)


class TestEconomicValue:
    """Tests for economic value computation."""

    def test_positive_savings(self) -> None:
        import pandas as pd
        rng = np.random.default_rng(42)
        n = 100

        # Predictions that correctly rank prices
        actual = np.sort(rng.uniform(5, 20, n))
        mu = actual + rng.standard_normal(n) * 0.3  # Close predictions

        pred_df = pd.DataFrame({
            "actual": actual,
            "mu": mu,
            "sigma": np.ones(n),
        })

        result = compute_economic_value(pred_df)

        assert result["savings_pct"] >= 0
        assert result["avg_price_baseline"] > 0
        assert result["avg_price_optimized"] <= result["avg_price_baseline"]

    def test_empty_dataframe(self) -> None:
        import pandas as pd
        result = compute_economic_value(pd.DataFrame())

        assert result["savings_pct"] == 0.0
