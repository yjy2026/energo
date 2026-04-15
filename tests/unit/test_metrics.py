"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from energo.evaluation.metrics import (
    compute_coverage,
    compute_crps_gaussian,
    compute_mae,
    compute_mape,
    compute_rmse,
    evaluate,
)


class TestPointMetrics:
    """Tests for MAE, RMSE, MAPE."""

    def test_mae_perfect(self) -> None:
        actual = np.array([10.0, 20.0, 30.0])
        predicted = np.array([10.0, 20.0, 30.0])
        assert compute_mae(actual, predicted) == pytest.approx(0.0)

    def test_mae_known_value(self) -> None:
        actual = np.array([10.0, 20.0])
        predicted = np.array([12.0, 18.0])
        assert compute_mae(actual, predicted) == pytest.approx(2.0)

    def test_rmse_perfect(self) -> None:
        actual = np.array([10.0, 20.0])
        predicted = np.array([10.0, 20.0])
        assert compute_rmse(actual, predicted) == pytest.approx(0.0)

    def test_rmse_known_value(self) -> None:
        actual = np.array([10.0, 20.0])
        predicted = np.array([13.0, 16.0])
        # MSE = (9 + 16) / 2 = 12.5, RMSE = sqrt(12.5)
        assert compute_rmse(actual, predicted) == pytest.approx(np.sqrt(12.5))

    def test_mape_known_value(self) -> None:
        actual = np.array([100.0, 200.0])
        predicted = np.array([110.0, 180.0])
        # |10/100| + |20/200| = 0.1 + 0.1 = 0.2 / 2 = 0.1
        assert compute_mape(actual, predicted) == pytest.approx(0.1)

    def test_mape_handles_near_zero(self) -> None:
        """MAPE should skip near-zero actual values."""
        actual = np.array([0.0001, 100.0])
        predicted = np.array([50.0, 100.0])
        # Should only compute for the second element
        result = compute_mape(actual, predicted)
        assert np.isfinite(result)


class TestCRPS:
    """Tests for CRPS metric."""

    def test_crps_perfect_prediction(self) -> None:
        """CRPS should be low for good predictions."""
        actual = np.array([10.0, 20.0, 30.0])
        mu = np.array([10.0, 20.0, 30.0])
        sigma = np.array([0.1, 0.1, 0.1])

        crps = compute_crps_gaussian(actual, mu, sigma)
        assert crps < 0.1

    def test_crps_bad_prediction(self) -> None:
        """CRPS should be higher for bad predictions."""
        actual = np.array([10.0, 20.0, 30.0])
        mu_good = np.array([10.0, 20.0, 30.0])
        mu_bad = np.array([20.0, 30.0, 40.0])
        sigma = np.array([1.0, 1.0, 1.0])

        crps_good = compute_crps_gaussian(actual, mu_good, sigma)
        crps_bad = compute_crps_gaussian(actual, mu_bad, sigma)

        assert crps_good < crps_bad

    def test_crps_non_negative(self) -> None:
        """CRPS should always be non-negative."""
        rng = np.random.default_rng(42)
        actual = rng.standard_normal(100)
        mu = rng.standard_normal(100)
        sigma = np.abs(rng.standard_normal(100)) + 0.1

        crps = compute_crps_gaussian(actual, mu, sigma)
        assert crps >= 0


class TestCoverage:
    """Tests for interval coverage."""

    def test_perfect_coverage(self) -> None:
        actual = np.array([10.0, 20.0, 30.0])
        lower = np.array([5.0, 15.0, 25.0])
        upper = np.array([15.0, 25.0, 35.0])

        assert compute_coverage(actual, lower, upper) == pytest.approx(1.0)

    def test_zero_coverage(self) -> None:
        actual = np.array([10.0, 20.0, 30.0])
        lower = np.array([50.0, 50.0, 50.0])
        upper = np.array([60.0, 60.0, 60.0])

        assert compute_coverage(actual, lower, upper) == pytest.approx(0.0)

    def test_partial_coverage(self) -> None:
        actual = np.array([10.0, 20.0, 30.0, 40.0])
        lower = np.array([9.0, 25.0, 29.0, 45.0])
        upper = np.array([11.0, 30.0, 31.0, 50.0])

        # 10 in [9,11]: yes, 20 in [25,30]: no, 30 in [29,31]: yes, 40 in [45,50]: no
        assert compute_coverage(actual, lower, upper) == pytest.approx(0.5)


class TestEvaluate:
    """Tests for the full evaluation suite."""

    def test_evaluate_returns_all_metrics(self) -> None:
        rng = np.random.default_rng(42)
        n = 200
        actual = 10 + rng.standard_normal(n)
        mu = actual + rng.standard_normal(n) * 0.5
        sigma = np.ones(n) * 2.0

        result = evaluate(actual, mu, sigma)

        assert result.mae > 0
        assert result.rmse > 0
        assert 0 <= result.mape <= 10
        assert result.crps > 0
        assert 0 <= result.coverage_90 <= 1
        assert 0 <= result.coverage_95 <= 1
        assert result.mean_sigma > 0

    def test_evaluate_summary_string(self) -> None:
        rng = np.random.default_rng(42)
        actual = rng.standard_normal(50)
        mu = rng.standard_normal(50)
        sigma = np.ones(50)

        result = evaluate(actual, mu, sigma)
        summary = result.summary()

        assert "MAE" in summary
        assert "CRPS" in summary
        assert "Cov 90%" in summary
