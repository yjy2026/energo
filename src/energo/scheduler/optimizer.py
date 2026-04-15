"""Scheduling optimization algorithms.

Provides multiple strategies from simple greedy to multi-workload
optimization, all using risk-adjusted cost from the cost model.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from energo.scheduler.cost import SlotForecast, rank_slots
from energo.scheduler.workload import (
    Priority,
    Schedule,
    ScheduledSlot,
    Workload,
)

if TYPE_CHECKING:
    from energo.scheduler.constraints import ConstraintSet

logger = logging.getLogger(__name__)


class SchedulingStrategy(Protocol):
    """Protocol for scheduling strategies."""

    def schedule(
        self,
        workloads: list[Workload],
        forecasts: list[SlotForecast],
        constraints: ConstraintSet,
    ) -> Schedule:
        """Schedule workloads given forecasts and constraints."""
        ...


class GreedyScheduler:
    """Greedy scheduler: assigns each workload to its cheapest feasible slot.

    For each workload (sorted by priority descending), finds the
    feasible start slot with the lowest risk-adjusted cost.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        """Initialize with risk aversion coefficient.

        Args:
            alpha: Risk aversion (0=aggressive, 1=conservative).
        """
        self.alpha = alpha

    def schedule(
        self,
        workloads: list[Workload],
        forecasts: list[SlotForecast],
        constraints: ConstraintSet,
    ) -> Schedule:
        """Schedule all workloads greedily by priority.

        Args:
            workloads: Workloads to schedule.
            forecasts: Price forecasts for available slots.
            constraints: Scheduling constraints.

        Returns:
            Complete schedule.
        """
        result = Schedule()
        total_slots = max(f.slot_index for f in forecasts) + 1 if forecasts else 0

        # Sort by priority (highest first)
        sorted_workloads = sorted(workloads, key=lambda w: w.priority, reverse=True)

        for workload in sorted_workloads:
            result.workloads[workload.id] = workload

            if workload.priority == Priority.CRITICAL:
                # CRITICAL: assign to slot 0 (immediate)
                self._assign_immediate(workload, forecasts, result)
                continue

            # Find feasible slots
            feasible = constraints.get_feasible_slots(
                workload, result, total_slots
            )

            if not feasible:
                logger.warning(
                    "No feasible slot for workload %s (%s). Assigning to first available.",
                    workload.id, workload.name,
                )
                self._assign_immediate(workload, forecasts, result)
                continue

            # Rank by risk-adjusted cost
            ranked = rank_slots(workload, forecasts, feasible, self.alpha)

            if not ranked:
                self._assign_immediate(workload, forecasts, result)
                continue

            # Assign to best slot
            best = ranked[0]
            self._assign_workload(
                workload, best.start_slot, forecasts, result
            )

            logger.info(
                "Scheduled %s at slot %d (cost=%.2f, cvar=%.2f)",
                workload.name, best.start_slot,
                best.expected_cost, best.cvar_cost,
            )

        return result

    @staticmethod
    def _assign_workload(
        workload: Workload,
        start_slot: int,
        forecasts: list[SlotForecast],
        schedule: Schedule,
    ) -> None:
        """Assign a workload to consecutive slots starting at start_slot."""
        forecast_map = {f.slot_index: f for f in forecasts}

        for offset in range(workload.duration_slots):
            slot_idx = start_slot + offset
            forecast = forecast_map.get(slot_idx)
            mu = forecast.mu if forecast else 0.0
            sigma = forecast.sigma if forecast else 0.0

            schedule.slots.append(
                ScheduledSlot(
                    workload_id=workload.id,
                    slot_index=slot_idx,
                    predicted_price=mu,
                    predicted_sigma=sigma,
                )
            )

    @staticmethod
    def _assign_immediate(
        workload: Workload,
        forecasts: list[SlotForecast],
        schedule: Schedule,
    ) -> None:
        """Assign workload starting at slot 0 (or earliest available)."""
        forecast_map = {f.slot_index: f for f in forecasts}

        for offset in range(workload.duration_slots):
            forecast = forecast_map.get(offset)
            mu = forecast.mu if forecast else 0.0
            sigma = forecast.sigma if forecast else 0.0

            schedule.slots.append(
                ScheduledSlot(
                    workload_id=workload.id,
                    slot_index=offset,
                    predicted_price=mu,
                    predicted_sigma=sigma,
                )
            )


class UniformScheduler:
    """Baseline scheduler: assigns workloads to a fixed start time.

    Used as a comparison baseline for measuring optimization value.
    """

    def __init__(self, start_slot: int = 0) -> None:
        self.start_slot = start_slot

    def schedule(
        self,
        workloads: list[Workload],
        forecasts: list[SlotForecast],
        constraints: ConstraintSet,
    ) -> Schedule:
        """Assign all workloads starting at the fixed slot."""
        result = Schedule()
        forecast_map = {f.slot_index: f for f in forecasts}

        current_slot = self.start_slot
        for workload in workloads:
            result.workloads[workload.id] = workload

            for offset in range(workload.duration_slots):
                slot_idx = current_slot + offset
                forecast = forecast_map.get(slot_idx)
                mu = forecast.mu if forecast else 0.0
                sigma = forecast.sigma if forecast else 0.0

                result.slots.append(
                    ScheduledSlot(
                        workload_id=workload.id,
                        slot_index=slot_idx,
                        predicted_price=mu,
                        predicted_sigma=sigma,
                    )
                )

        return result
