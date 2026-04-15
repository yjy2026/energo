"""Tests for workload definition."""

from __future__ import annotations

import pytest

from energo.scheduler.workload import (
    Priority,
    Schedule,
    ScheduledSlot,
    Workload,
)


class TestWorkload:
    """Tests for Workload dataclass."""

    def test_basic_creation(self) -> None:
        w = Workload(id="w1", name="Training", duration_slots=4, power_kw=100.0)
        assert w.duration_slots == 4
        assert w.duration_hours == 2.0
        assert w.energy_kwh == 200.0

    def test_priority_default(self) -> None:
        w = Workload(id="w1", name="Test", duration_slots=1, power_kw=10.0)
        assert w.priority == Priority.NORMAL

    def test_invalid_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="duration_slots"):
            Workload(id="w1", name="Bad", duration_slots=0, power_kw=10.0)

    def test_invalid_power_raises(self) -> None:
        with pytest.raises(ValueError, match="power_kw"):
            Workload(id="w1", name="Bad", duration_slots=1, power_kw=-5.0)

    def test_effective_min_continuous_non_preemptible(self) -> None:
        w = Workload(id="w1", name="Test", duration_slots=4, power_kw=10.0, preemptible=False)
        assert w.effective_min_continuous == 4

    def test_effective_min_continuous_preemptible(self) -> None:
        w = Workload(id="w1", name="Test", duration_slots=4, power_kw=10.0, preemptible=True)
        assert w.effective_min_continuous == 1

    def test_effective_min_continuous_override(self) -> None:
        w = Workload(
            id="w1", name="Test", duration_slots=4, power_kw=10.0,
            preemptible=True, min_continuous=2,
        )
        assert w.effective_min_continuous == 2

    def test_priority_ordering(self) -> None:
        assert Priority.CRITICAL > Priority.HIGH
        assert Priority.HIGH > Priority.NORMAL
        assert Priority.NORMAL > Priority.LOW


class TestSchedule:
    """Tests for Schedule."""

    def test_cost_calculation(self) -> None:
        w = Workload(id="w1", name="Test", duration_slots=2, power_kw=100.0)
        schedule = Schedule(
            workloads={"w1": w},
            slots=[
                ScheduledSlot(
                    workload_id="w1", slot_index=0,
                    predicted_price=10.0, predicted_sigma=1.0,
                ),
                ScheduledSlot(
                    workload_id="w1", slot_index=1,
                    predicted_price=12.0, predicted_sigma=1.5,
                ),
            ],
        )
        # cost = (10 * 100 * 0.5) + (12 * 100 * 0.5) = 500 + 600 = 1100
        assert schedule.cost_for_workload("w1") == pytest.approx(1100.0)

    def test_actual_cost(self) -> None:
        w = Workload(id="w1", name="Test", duration_slots=1, power_kw=50.0)
        schedule = Schedule(
            workloads={"w1": w},
            slots=[
                ScheduledSlot(
                    workload_id="w1", slot_index=0,
                    predicted_price=10.0, predicted_sigma=1.0,
                    actual_price=11.0,
                ),
            ],
        )
        # actual cost = 11 * 50 * 0.5 = 275
        assert schedule.actual_cost_for_workload("w1") == pytest.approx(275.0)

    def test_slots_for_workload(self) -> None:
        schedule = Schedule(
            workloads={
                "w1": Workload(id="w1", name="A", duration_slots=1, power_kw=10.0),
                "w2": Workload(id="w2", name="B", duration_slots=1, power_kw=10.0),
            },
            slots=[
                ScheduledSlot(
                    workload_id="w1", slot_index=0,
                    predicted_price=10.0, predicted_sigma=1.0,
                ),
                ScheduledSlot(
                    workload_id="w2", slot_index=1,
                    predicted_price=12.0, predicted_sigma=1.0,
                ),
                ScheduledSlot(
                    workload_id="w1", slot_index=2,
                    predicted_price=8.0, predicted_sigma=1.0,
                ),
            ],
        )
        assert len(schedule.slots_for_workload("w1")) == 2
        assert len(schedule.slots_for_workload("w2")) == 1
