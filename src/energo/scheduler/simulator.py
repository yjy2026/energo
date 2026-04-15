"""Backtesting simulator for scheduling strategies.

Runs scheduling strategies against historical JEPX price data
and measures actual cost savings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from energo.scheduler.constraints import ConstraintSet
from energo.scheduler.cost import SlotForecast
from energo.scheduler.optimizer import GreedyScheduler, UniformScheduler
from energo.scheduler.report import ScheduleReport, generate_report

if TYPE_CHECKING:
    import numpy as np

    from energo.scheduler.workload import Schedule, Workload

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Results from a full backtesting simulation."""

    greedy_report: ScheduleReport
    num_windows: int
    avg_savings_pct: float
    total_savings_jpy: float
    per_window: list[ScheduleReport]


def run_simulation(
    actual_prices: np.ndarray,
    predicted_mu: np.ndarray,
    predicted_sigma: np.ndarray,
    workloads: list[Workload],
    window_size: int = 48,
    step_size: int = 48,
    alpha: float = 0.3,
    max_power_kw: float | None = None,
) -> SimulationResult:
    """Run a rolling-window backtesting simulation.

    Steps through the price data in windows, schedules workloads
    using both greedy and uniform strategies, and compares actual costs.

    Args:
        actual_prices: Historical actual prices (N,).
        predicted_mu: Model-predicted means (N,).
        predicted_sigma: Model-predicted sigmas (N,).
        workloads: Workloads to schedule in each window.
        window_size: Number of slots per scheduling window (default: 48 = 24h).
        step_size: Step between windows (default: 48 = non-overlapping days).
        alpha: Risk aversion coefficient.
        max_power_kw: Optional power constraint.

    Returns:
        SimulationResult with aggregated metrics.
    """
    n = len(actual_prices)
    greedy = GreedyScheduler(alpha=alpha)
    uniform = UniformScheduler(start_slot=0)

    # Build constraints
    constraints = ConstraintSet()
    if max_power_kw is not None:
        from energo.scheduler.constraints import ResourceConstraint
        constraints.add(ResourceConstraint(max_power_kw=max_power_kw))

    per_window: list[ScheduleReport] = []

    for start in range(0, n - window_size, step_size):
        end = start + window_size

        # Build forecasts for this window
        forecasts = [
            SlotForecast(
                slot_index=i - start,
                mu=float(predicted_mu[i]),
                sigma=float(predicted_sigma[i]),
            )
            for i in range(start, end)
        ]

        # Schedule with both strategies
        greedy_schedule = greedy.schedule(workloads, forecasts, constraints)
        uniform_schedule = uniform.schedule(workloads, forecasts, constraints)

        # Fill in actual prices
        _fill_actual_prices(greedy_schedule, actual_prices, start)
        _fill_actual_prices(uniform_schedule, actual_prices, start)

        # Generate report
        report = generate_report(
            greedy_schedule, uniform_schedule, strategy_name="Greedy"
        )
        per_window.append(report)

    # Aggregate
    if per_window:
        total_greedy = sum(r.total_actual_cost for r in per_window)
        total_baseline = sum(r.baseline_cost for r in per_window)
        total_savings = total_baseline - total_greedy
        avg_pct = total_savings / total_baseline if total_baseline > 0 else 0.0
    else:
        total_savings = 0.0
        avg_pct = 0.0

    # Create aggregate greedy report
    agg_report = ScheduleReport(
        strategy_name="Greedy (Aggregate)",
        total_predicted_cost=sum(r.total_predicted_cost for r in per_window),
        total_actual_cost=sum(r.total_actual_cost for r in per_window),
        baseline_cost=sum(r.baseline_cost for r in per_window),
        savings_predicted=sum(r.savings_predicted for r in per_window),
        savings_actual=total_savings,
        savings_pct_predicted=avg_pct,
        savings_pct_actual=avg_pct,
        risk=per_window[0].risk if per_window else None,  # type: ignore[arg-type]
    )

    result = SimulationResult(
        greedy_report=agg_report,
        num_windows=len(per_window),
        avg_savings_pct=avg_pct,
        total_savings_jpy=total_savings,
        per_window=per_window,
    )

    logger.info(
        "Simulation complete: %d windows, avg savings %.1f%%, total ¥%.1f",
        result.num_windows,
        result.avg_savings_pct * 100,
        result.total_savings_jpy,
    )

    return result


def _fill_actual_prices(
    schedule: Schedule,
    actual_prices: np.ndarray,
    window_start: int,
) -> None:
    """Fill actual prices into a schedule for backtesting."""
    for slot in schedule.slots:
        global_idx = window_start + slot.slot_index
        if 0 <= global_idx < len(actual_prices):
            slot.actual_price = float(actual_prices[global_idx])
