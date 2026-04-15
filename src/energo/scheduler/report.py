"""Schedule result reporting.

Generates structured reports comparing scheduling strategies
against baselines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from energo.scheduler.workload import Schedule

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskMetrics:
    """Risk metrics for a schedule."""

    mean_sigma: float          # Average price uncertainty across scheduled slots
    max_sigma: float           # Maximum single-slot uncertainty
    cost_std: float            # Standard deviation of total cost


@dataclass
class ScheduleReport:
    """Comprehensive schedule report."""

    strategy_name: str
    total_predicted_cost: float
    total_actual_cost: float
    baseline_cost: float
    savings_predicted: float  # vs baseline (predicted)
    savings_actual: float     # vs baseline (actual)
    savings_pct_predicted: float
    savings_pct_actual: float
    risk: RiskMetrics
    workload_details: dict[str, WorkloadReport] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"=== Schedule Report: {self.strategy_name} ===",
            f"  Predicted cost : ¥{self.total_predicted_cost:,.1f}",
            f"  Actual cost    : ¥{self.total_actual_cost:,.1f}",
            f"  Baseline cost  : ¥{self.baseline_cost:,.1f}",
            f"  Savings (pred) : ¥{self.savings_predicted:,.1f} ({self.savings_pct_predicted:.1%})",
            f"  Savings (actual): ¥{self.savings_actual:,.1f} ({self.savings_pct_actual:.1%})",
            f"  Risk: avg_σ={self.risk.mean_sigma:.3f}, max_σ={self.risk.max_sigma:.3f}",
        ]
        for wid, wr in self.workload_details.items():
            line = f"  [{wid}] {wr.name}: ¥{wr.actual_cost:,.1f}"
            lines.append(f"{line} @ slots {wr.start_slot}-{wr.end_slot}")
        return "\n".join(lines)


@dataclass(frozen=True)
class WorkloadReport:
    """Report for a single workload."""

    name: str
    start_slot: int
    end_slot: int
    predicted_cost: float
    actual_cost: float


def generate_report(
    schedule: Schedule,
    baseline_schedule: Schedule,
    strategy_name: str = "Greedy",
) -> ScheduleReport:
    """Generate a comparison report between optimized and baseline schedules.

    Args:
        schedule: The optimized schedule.
        baseline_schedule: The baseline (uniform) schedule.
        strategy_name: Name of the optimization strategy.

    Returns:
        ScheduleReport with comparison metrics.
    """
    pred_cost = schedule.total_predicted_cost
    actual_cost = schedule.total_actual_cost
    baseline_cost = baseline_schedule.total_actual_cost

    if baseline_cost == 0:
        baseline_cost = baseline_schedule.total_predicted_cost

    savings_pred = baseline_cost - pred_cost
    savings_actual = baseline_cost - actual_cost
    savings_pct_pred = savings_pred / baseline_cost if baseline_cost > 0 else 0.0
    savings_pct_actual = savings_actual / baseline_cost if baseline_cost > 0 else 0.0

    # Risk metrics
    sigmas = [s.predicted_sigma for s in schedule.slots]
    risk = RiskMetrics(
        mean_sigma=sum(sigmas) / len(sigmas) if sigmas else 0.0,
        max_sigma=max(sigmas) if sigmas else 0.0,
        cost_std=0.0,  # Would need full cost model to compute
    )

    # Per-workload details
    workload_details: dict[str, WorkloadReport] = {}
    for wid, workload in schedule.workloads.items():
        slots = schedule.slots_for_workload(wid)
        if slots:
            slot_indices = [s.slot_index for s in slots]
            workload_details[wid] = WorkloadReport(
                name=workload.name,
                start_slot=min(slot_indices),
                end_slot=max(slot_indices),
                predicted_cost=schedule.cost_for_workload(wid),
                actual_cost=schedule.actual_cost_for_workload(wid),
            )

    return ScheduleReport(
        strategy_name=strategy_name,
        total_predicted_cost=pred_cost,
        total_actual_cost=actual_cost,
        baseline_cost=baseline_cost,
        savings_predicted=savings_pred,
        savings_actual=savings_actual,
        savings_pct_predicted=savings_pct_pred,
        savings_pct_actual=savings_pct_actual,
        risk=risk,
        workload_details=workload_details,
    )
