"""Rolling-window backtesting for price prediction models.

Tests model performance across time with realistic train/predict cycles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from energo.evaluation.metrics import EvaluationResult, evaluate

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a rolling backtest."""

    overall: EvaluationResult
    by_hour: dict[int, EvaluationResult]
    by_day_of_week: dict[int, EvaluationResult]
    predictions: pd.DataFrame  # timestamp, actual, mu, sigma


def rolling_backtest(
    timestamps: np.ndarray,
    actual: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> BacktestResult:
    """Analyze prediction performance across time dimensions.

    Args:
        timestamps: Array of timestamps.
        actual: Actual prices.
        mu: Predicted means.
        sigma: Predicted standard deviations.

    Returns:
        BacktestResult with overall and sliced metrics.
    """
    # Overall evaluation
    overall = evaluate(actual, mu, sigma)

    # Build predictions DataFrame for slicing
    pred_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "actual": actual,
            "mu": mu,
            "sigma": sigma,
        }
    )
    pred_df["hour"] = pred_df["timestamp"].dt.hour
    pred_df["day_of_week"] = pred_df["timestamp"].dt.dayofweek

    # By hour
    by_hour: dict[int, EvaluationResult] = {}
    for hour in range(24):
        mask = pred_df["hour"] == hour
        if mask.sum() < 10:
            continue
        subset = pred_df[mask]
        by_hour[hour] = evaluate(
            subset["actual"].values,
            subset["mu"].values,
            subset["sigma"].values,
        )

    # By day of week
    by_dow: dict[int, EvaluationResult] = {}
    for dow in range(7):
        mask = pred_df["day_of_week"] == dow
        if mask.sum() < 10:
            continue
        subset = pred_df[mask]
        by_dow[dow] = evaluate(
            subset["actual"].values,
            subset["mu"].values,
            subset["sigma"].values,
        )

    logger.info("Backtest complete: %d predictions", len(pred_df))
    logger.info("Overall:\n%s", overall.summary())

    return BacktestResult(
        overall=overall,
        by_hour=by_hour,
        by_day_of_week=by_dow,
        predictions=pred_df,
    )


def compute_economic_value(
    pred_df: pd.DataFrame,
    peak_threshold_percentile: float = 75.0,
) -> dict[str, float]:
    """Estimate economic value of forecast-based scheduling.

    Simulates a simple strategy: shift workloads from predicted
    expensive periods to predicted cheap periods.

    Args:
        pred_df: DataFrame with timestamp, actual, mu, sigma.
        peak_threshold_percentile: Percentile above which a period
            is considered "expensive".

    Returns:
        Dict with savings estimates.
    """
    if pred_df.empty:
        return {"savings_pct": 0.0, "avg_price_baseline": 0.0, "avg_price_optimized": 0.0}

    # Baseline: uniform consumption
    avg_price_baseline = float(pred_df["actual"].mean())

    # Optimized: avoid top-percentile price periods
    threshold = np.percentile(pred_df["mu"].values, peak_threshold_percentile)
    cheap_mask = pred_df["mu"] <= threshold
    if cheap_mask.sum() == 0:
        avg_price_optimized = avg_price_baseline
    else:
        avg_price_optimized = float(pred_df.loc[cheap_mask, "actual"].mean())

    savings_pct = (
        (avg_price_baseline - avg_price_optimized) / avg_price_baseline
        if avg_price_baseline > 0
        else 0.0
    )

    return {
        "savings_pct": savings_pct,
        "avg_price_baseline": avg_price_baseline,
        "avg_price_optimized": avg_price_optimized,
    }
