"""Evaluation metrics for probabilistic price forecasts.

Includes both probabilistic metrics (CRPS, coverage) and
point-estimate metrics (MAE, RMSE, MAPE).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class EvaluationResult:
    """Complete evaluation results."""

    # Point metrics
    mae: float
    rmse: float
    mape: float

    # Probabilistic metrics
    crps: float
    coverage_90: float
    coverage_95: float

    # Calibration
    mean_sigma: float

    # Spike detection
    spike_recall: float | None = None

    def summary(self) -> str:
        lines = [
            "=== Evaluation Results ===",
            f"  MAE    : {self.mae:.4f}",
            f"  RMSE   : {self.rmse:.4f}",
            f"  MAPE   : {self.mape:.2%}",
            f"  CRPS   : {self.crps:.4f}",
            f"  Cov 90%: {self.coverage_90:.2%} (target: 90%)",
            f"  Cov 95%: {self.coverage_95:.2%} (target: 95%)",
            f"  Mean σ : {self.mean_sigma:.4f}",
        ]
        if self.spike_recall is not None:
            lines.append(f"  Spike↑ : {self.spike_recall:.2%}")
        return "\n".join(lines)


def compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def compute_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error (filters near-zero actuals)."""
    mask = np.abs(actual) > 1e-6
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])))


def compute_crps_gaussian(
    actual: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """Continuous Ranked Probability Score for Gaussian distributions.

    CRPS(N(μ, σ²), y) = σ * [z*(2Φ(z)-1) + 2φ(z) - 1/√π]
    where z = (y - μ) / σ

    Lower is better.
    """
    z = (actual - mu) / sigma

    # Standard normal PDF and CDF
    pdf_z = np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    cdf_z = 0.5 * (1 + _erf_approx(z / math.sqrt(2)))

    crps_values = sigma * (z * (2 * cdf_z - 1) + 2 * pdf_z - 1 / math.sqrt(math.pi))

    return float(np.mean(crps_values))


def compute_coverage(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Fraction of actual values within predicted interval."""
    inside = (actual >= lower) & (actual <= upper)
    return float(np.mean(inside))


def compute_spike_recall(
    actual: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    spike_percentile: float = 95.0,
    sigma_threshold: float = 1.5,
) -> float:
    """Recall for detecting price spikes.

    A spike is defined as actual price above the specified percentile.
    Detection is defined as the model predicting high uncertainty
    (sigma above threshold * mean_sigma) or high mean.

    Args:
        actual: Actual prices.
        mu: Predicted means.
        sigma: Predicted standard deviations.
        spike_percentile: Percentile threshold for "spike".
        sigma_threshold: Multiple of mean sigma for "high uncertainty".

    Returns:
        Spike recall (fraction of spikes detected).
    """
    spike_threshold = np.percentile(actual, spike_percentile)
    is_spike = actual > spike_threshold

    if is_spike.sum() == 0:
        return 1.0

    mean_sigma = np.mean(sigma)
    high_mu = mu > spike_threshold * 0.8
    high_sigma = sigma > sigma_threshold * mean_sigma
    detected = high_mu | high_sigma

    recall = float(np.sum(is_spike & detected) / np.sum(is_spike))
    return recall


def evaluate(
    actual: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> EvaluationResult:
    """Run complete evaluation suite.

    Args:
        actual: Actual prices (N,).
        mu: Predicted means (N,).
        sigma: Predicted standard deviations (N,).

    Returns:
        EvaluationResult with all metrics.
    """
    return EvaluationResult(
        mae=compute_mae(actual, mu),
        rmse=compute_rmse(actual, mu),
        mape=compute_mape(actual, mu),
        crps=compute_crps_gaussian(actual, mu, sigma),
        coverage_90=compute_coverage(
            actual,
            mu - 1.645 * sigma,
            mu + 1.645 * sigma,
        ),
        coverage_95=compute_coverage(
            actual,
            mu - 1.960 * sigma,
            mu + 1.960 * sigma,
        ),
        mean_sigma=float(np.mean(sigma)),
        spike_recall=compute_spike_recall(actual, mu, sigma),
    )


def _erf_approx(x: np.ndarray) -> np.ndarray:
    """Approximation of the error function using numpy."""
    # Abramowitz & Stegun approximation
    a1, a2, a3, a4, a5 = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
    )
    p = 0.3275911

    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

    return sign * y
