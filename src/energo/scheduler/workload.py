"""Workload definition and priority model.

Defines what constitutes a schedulable workload with constraints
like duration, deadline, power consumption, and priority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

SLOT_DURATION_MINUTES = 30


class Priority(IntEnum):
    """Workload priority levels.

    Higher value = higher priority.
    CRITICAL tasks are executed immediately regardless of price.
    """

    LOW = 0       # 72h window, maximum flexibility
    NORMAL = 1    # 24h window
    HIGH = 2      # Deadline-bound, best price within deadline
    CRITICAL = 3  # Immediate execution, ignore price


# Default scheduling windows per priority (in 30-min slots)
PRIORITY_WINDOWS: dict[Priority, int] = {
    Priority.LOW: 144 * 3,     # 72 hours = 432 slots
    Priority.NORMAL: 144,      # 72 slots → wait, should be 48 for 24h. Let me fix: 48 slots = 24h
    Priority.HIGH: 48,         # 24 hours
    Priority.CRITICAL: 1,      # Immediate
}

# Correct: 24h = 48 slots at 30-min intervals
PRIORITY_WINDOWS = {
    Priority.LOW: 48 * 3,      # 72 hours = 144 slots
    Priority.NORMAL: 48,       # 24 hours = 48 slots
    Priority.HIGH: 48,         # Within deadline (up to 24h default)
    Priority.CRITICAL: 1,      # Immediate
}


@dataclass(frozen=True)
class Workload:
    """A schedulable compute workload.

    Attributes:
        id: Unique identifier.
        name: Human-readable name.
        duration_slots: Required number of 30-min slots.
        power_kw: Power consumption in kilowatts.
        deadline_slot: Latest slot index by which the workload must complete.
            None means flexible (uses priority-based default window).
        priority: Scheduling priority level.
        preemptible: Whether the workload can be paused and resumed.
        min_continuous: Minimum consecutive slots per execution chunk.
            Defaults to duration_slots (non-preemptible behavior).
    """

    id: str
    name: str
    duration_slots: int
    power_kw: float
    deadline_slot: int | None = None
    priority: Priority = Priority.NORMAL
    preemptible: bool = False
    min_continuous: int | None = None

    def __post_init__(self) -> None:
        if self.duration_slots <= 0:
            msg = f"duration_slots must be positive, got {self.duration_slots}"
            raise ValueError(msg)
        if self.power_kw <= 0:
            msg = f"power_kw must be positive, got {self.power_kw}"
            raise ValueError(msg)
        if self.min_continuous is not None and self.min_continuous <= 0:
            msg = f"min_continuous must be positive, got {self.min_continuous}"
            raise ValueError(msg)

    @property
    def effective_min_continuous(self) -> int:
        """Minimum consecutive slots (defaults to full duration if not preemptible)."""
        if self.min_continuous is not None:
            return self.min_continuous
        return self.duration_slots if not self.preemptible else 1

    @property
    def duration_hours(self) -> float:
        """Duration in hours."""
        return self.duration_slots * SLOT_DURATION_MINUTES / 60

    @property
    def energy_kwh(self) -> float:
        """Total energy consumption in kWh."""
        return self.power_kw * self.duration_hours


@dataclass
class ScheduledSlot:
    """A workload assigned to a specific time slot.

    Attributes:
        workload_id: ID of the workload.
        slot_index: Index into the forecast array.
        predicted_price: Predicted price (μ) at this slot.
        predicted_sigma: Price uncertainty (σ) at this slot.
        actual_price: Actual price (filled in during backtesting).
    """

    workload_id: str
    slot_index: int
    predicted_price: float
    predicted_sigma: float
    actual_price: float | None = None


@dataclass
class Schedule:
    """Complete schedule for one or more workloads."""

    slots: list[ScheduledSlot] = field(default_factory=list)
    workloads: dict[str, Workload] = field(default_factory=dict)

    def cost_for_workload(self, workload_id: str) -> float:
        """Total predicted cost for a workload (using μ)."""
        workload = self.workloads.get(workload_id)
        if workload is None:
            return 0.0
        return sum(
            s.predicted_price * workload.power_kw * (SLOT_DURATION_MINUTES / 60)
            for s in self.slots
            if s.workload_id == workload_id
        )

    def actual_cost_for_workload(self, workload_id: str) -> float:
        """Total actual cost (filled after backtesting)."""
        workload = self.workloads.get(workload_id)
        if workload is None:
            return 0.0
        return sum(
            (s.actual_price or 0.0) * workload.power_kw * (SLOT_DURATION_MINUTES / 60)
            for s in self.slots
            if s.workload_id == workload_id
        )

    @property
    def total_predicted_cost(self) -> float:
        return sum(self.cost_for_workload(wid) for wid in self.workloads)

    @property
    def total_actual_cost(self) -> float:
        return sum(self.actual_cost_for_workload(wid) for wid in self.workloads)

    def slots_for_workload(self, workload_id: str) -> list[ScheduledSlot]:
        return [s for s in self.slots if s.workload_id == workload_id]
