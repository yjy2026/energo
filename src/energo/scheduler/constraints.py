"""Schedule constraints.

Defines composable constraint objects that validate whether a
proposed schedule assignment is feasible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from energo.scheduler.workload import Schedule, Workload


class Constraint(Protocol):
    """Protocol for schedule constraints."""

    def is_feasible(
        self,
        workload: Workload,
        start_slot: int,
        current_schedule: Schedule,
        total_slots: int,
    ) -> bool:
        """Check if assigning workload at start_slot is feasible."""
        ...


@dataclass(frozen=True)
class DeadlineConstraint:
    """Ensures workload completes before its deadline."""

    def is_feasible(
        self,
        workload: Workload,
        start_slot: int,
        current_schedule: Schedule,
        total_slots: int,
    ) -> bool:
        end_slot = start_slot + workload.duration_slots
        # Must fit within available slots
        if end_slot > total_slots:
            return False
        # Must meet deadline if specified
        return not (workload.deadline_slot is not None and end_slot > workload.deadline_slot)


@dataclass(frozen=True)
class ResourceConstraint:
    """Limits total simultaneous power consumption."""

    max_power_kw: float

    def is_feasible(
        self,
        workload: Workload,
        start_slot: int,
        current_schedule: Schedule,
        total_slots: int,
    ) -> bool:
        # Check each slot the new workload would occupy
        for offset in range(workload.duration_slots):
            slot_idx = start_slot + offset
            # Sum power of already-scheduled workloads at this slot
            existing_power = sum(
                current_schedule.workloads[s.workload_id].power_kw
                for s in current_schedule.slots
                if s.slot_index == slot_idx
            )
            if existing_power + workload.power_kw > self.max_power_kw:
                return False
        return True


@dataclass(frozen=True)
class BlackoutConstraint:
    """Prevents scheduling during specific time slots."""

    blocked_slots: frozenset[int]

    def is_feasible(
        self,
        workload: Workload,
        start_slot: int,
        current_schedule: Schedule,
        total_slots: int,
    ) -> bool:
        for offset in range(workload.duration_slots):
            if (start_slot + offset) in self.blocked_slots:
                return False
        return True


class ConstraintSet:
    """Composable set of constraints with AND semantics."""

    def __init__(self, constraints: list[Constraint] | None = None) -> None:
        self._constraints: list[Constraint] = constraints or [DeadlineConstraint()]

    def is_feasible(
        self,
        workload: Workload,
        start_slot: int,
        current_schedule: Schedule,
        total_slots: int,
    ) -> bool:
        """Check all constraints. Returns True only if all pass."""
        return all(
            c.is_feasible(workload, start_slot, current_schedule, total_slots)
            for c in self._constraints
        )

    def add(self, constraint: Constraint) -> ConstraintSet:
        """Add a constraint and return self for chaining."""
        self._constraints.append(constraint)
        return self

    def get_feasible_slots(
        self,
        workload: Workload,
        current_schedule: Schedule,
        total_slots: int,
    ) -> list[int]:
        """Get all feasible start slots for a workload."""
        return [
            slot
            for slot in range(total_slots)
            if self.is_feasible(workload, slot, current_schedule, total_slots)
        ]
