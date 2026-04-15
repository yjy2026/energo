"""Cost model using probabilistic price predictions.

Computes expected cost, CVaR (tail risk), and risk-adjusted cost
using the (μ, σ) distribution from the prediction engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from energo.scheduler.workload import SLOT_DURATION_MINUTES, Workload


@dataclass(frozen=True)
class SlotForecast:
    """Price forecast for a single 30-min slot."""

    slot_index: int
    mu: float      # Expected price (JPY/kWh)
    sigma: float   # Price uncertainty (std dev)


@dataclass(frozen=True)
class CostEstimate:
    """Cost estimate for a workload at a specific start time."""

    start_slot: int
    expected_cost: float      # E[cost] using μ
    cvar_cost: float          # CVaR (worst-case tail cost)
    risk_adjusted_cost: float # Blended cost: (1-α)*E + α*CVaR
    cost_std: float           # Standard deviation of cost


def compute_expected_cost(
    workload: Workload,
    start_slot: int,
    forecasts: list[SlotForecast],
) -> float:
    """Compute expected cost for a workload starting at a given slot.

    E[cost] = Σ μ_t × power_kw × slot_duration_hours

    Args:
        workload: The workload to schedule.
        start_slot: Starting slot index.
        forecasts: Price forecasts for available slots.

    Returns:
        Expected cost in JPY.
    """
    hours_per_slot = SLOT_DURATION_MINUTES / 60
    forecast_map = {f.slot_index: f for f in forecasts}

    total = 0.0
    for offset in range(workload.duration_slots):
        slot_idx = start_slot + offset
        forecast = forecast_map.get(slot_idx)
        if forecast is None:
            return float("inf")  # Missing forecast = infeasible
        total += forecast.mu * workload.power_kw * hours_per_slot

    return total


def compute_cost_std(
    workload: Workload,
    start_slot: int,
    forecasts: list[SlotForecast],
) -> float:
    """Compute standard deviation of total cost.

    Assuming independent price uncertainties across slots:
    Var[cost] = Σ (σ_t × power_kw × hours)²
    Std[cost] = √(Var[cost])
    """
    hours_per_slot = SLOT_DURATION_MINUTES / 60
    forecast_map = {f.slot_index: f for f in forecasts}

    variance = 0.0
    for offset in range(workload.duration_slots):
        slot_idx = start_slot + offset
        forecast = forecast_map.get(slot_idx)
        if forecast is None:
            return float("inf")
        slot_cost_std = forecast.sigma * workload.power_kw * hours_per_slot
        variance += slot_cost_std ** 2

    return math.sqrt(variance)


def compute_cvar(
    workload: Workload,
    start_slot: int,
    forecasts: list[SlotForecast],
    quantile: float = 0.95,
) -> float:
    """Compute Conditional Value at Risk (CVaR) for cost.

    For Gaussian: CVaR_α = E[cost] + σ_cost × φ(Φ⁻¹(α)) / (1-α)
    where φ is the standard normal PDF and Φ⁻¹ is the inverse CDF.

    Args:
        workload: The workload.
        start_slot: Starting slot index.
        forecasts: Price forecasts.
        quantile: CVaR confidence level (default 95%).

    Returns:
        CVaR cost estimate in JPY.
    """
    e_cost = compute_expected_cost(workload, start_slot, forecasts)
    if not math.isfinite(e_cost):
        return float("inf")

    std_cost = compute_cost_std(workload, start_slot, forecasts)
    if not math.isfinite(std_cost) or std_cost < 1e-10:
        return e_cost

    # Gaussian CVaR formula
    z = _norm_ppf(quantile)
    phi_z = _norm_pdf(z)
    cvar_multiplier = phi_z / (1 - quantile)

    return e_cost + std_cost * cvar_multiplier


def compute_risk_adjusted_cost(
    workload: Workload,
    start_slot: int,
    forecasts: list[SlotForecast],
    alpha: float = 0.3,
    cvar_quantile: float = 0.95,
) -> CostEstimate:
    """Compute risk-adjusted cost blending expected cost and CVaR.

    risk_adjusted = (1 - α) × E[cost] + α × CVaR[cost]

    Args:
        workload: The workload.
        start_slot: Starting slot index.
        forecasts: Price forecasts.
        alpha: Risk aversion coefficient (0 = aggressive, 1 = conservative).
        cvar_quantile: CVaR confidence level.

    Returns:
        Complete CostEstimate.
    """
    e_cost = compute_expected_cost(workload, start_slot, forecasts)
    cvar = compute_cvar(workload, start_slot, forecasts, cvar_quantile)
    std = compute_cost_std(workload, start_slot, forecasts)

    risk_adj = (1 - alpha) * e_cost + alpha * cvar

    return CostEstimate(
        start_slot=start_slot,
        expected_cost=e_cost,
        cvar_cost=cvar,
        risk_adjusted_cost=risk_adj,
        cost_std=std,
    )


def rank_slots(
    workload: Workload,
    forecasts: list[SlotForecast],
    candidate_slots: list[int],
    alpha: float = 0.3,
) -> list[CostEstimate]:
    """Rank candidate start slots by risk-adjusted cost.

    Args:
        workload: The workload.
        forecasts: All available price forecasts.
        candidate_slots: List of candidate start slot indices.
        alpha: Risk aversion coefficient.

    Returns:
        List of CostEstimate sorted by risk_adjusted_cost (ascending).
    """
    estimates = []
    for slot in candidate_slots:
        est = compute_risk_adjusted_cost(workload, slot, forecasts, alpha)
        if math.isfinite(est.risk_adjusted_cost):
            estimates.append(est)

    estimates.sort(key=lambda e: e.risk_adjusted_cost)
    return estimates


# ── Internal helpers ──────────────────────────────────────


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _norm_ppf(p: float) -> float:
    """Approximate inverse CDF (percent point function) of standard normal.

    Uses the rational approximation from Abramowitz & Stegun.
    Accurate to ~4.5e-4 for 0.0027 < p < 0.9973.
    """
    if p <= 0 or p >= 1:
        msg = f"p must be in (0, 1), got {p}"
        raise ValueError(msg)

    if p < 0.5:
        return -_norm_ppf_inner(1 - p)
    return _norm_ppf_inner(p)


def _norm_ppf_inner(p: float) -> float:
    """Inner approximation for p >= 0.5."""
    # Rational approximation constants
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308

    t = math.sqrt(-2 * math.log(1 - p))
    return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)
