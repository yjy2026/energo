"""Tests for scheduling optimizer."""

from __future__ import annotations

from energo.scheduler.constraints import (
    BlackoutConstraint,
    ConstraintSet,
    DeadlineConstraint,
    ResourceConstraint,
)
from energo.scheduler.cost import SlotForecast
from energo.scheduler.optimizer import GreedyScheduler, UniformScheduler
from energo.scheduler.workload import Priority, Workload


def _make_forecasts(prices: list[float], sigma: float = 1.0) -> list[SlotForecast]:
    """Helper to create forecasts from a simple price list."""
    return [
        SlotForecast(slot_index=i, mu=p, sigma=sigma)
        for i, p in enumerate(prices)
    ]


class TestGreedyScheduler:
    """Tests for the greedy scheduling algorithm."""

    def test_selects_cheapest_slot(self) -> None:
        """Greedy should pick the cheapest feasible slot."""
        prices = [10.0, 15.0, 5.0, 8.0, 12.0, 20.0]
        forecasts = _make_forecasts(prices, sigma=0.1)

        workload = Workload(
            id="w1", name="Test", duration_slots=2, power_kw=100.0,
        )

        scheduler = GreedyScheduler(alpha=0.0)
        constraints = ConstraintSet([DeadlineConstraint()])
        schedule = scheduler.schedule([workload], forecasts, constraints)

        # Should start at slot 2 (prices 5+8=13, cheapest 2-slot window)
        slots = schedule.slots_for_workload("w1")
        start = min(s.slot_index for s in slots)
        assert start == 2

    def test_greedy_cheaper_than_uniform(self) -> None:
        """Greedy should produce cost <= uniform."""
        prices = [20.0, 15.0, 5.0, 8.0, 25.0, 3.0, 4.0, 10.0]
        forecasts = _make_forecasts(prices, sigma=0.1)

        workload = Workload(
            id="w1", name="Test", duration_slots=2, power_kw=100.0,
        )

        constraints = ConstraintSet([DeadlineConstraint()])

        greedy = GreedyScheduler(alpha=0.0)
        uniform = UniformScheduler(start_slot=0)

        g_schedule = greedy.schedule([workload], forecasts, constraints)
        u_schedule = uniform.schedule([workload], forecasts, constraints)

        assert g_schedule.total_predicted_cost <= u_schedule.total_predicted_cost

    def test_critical_workload_immediate(self) -> None:
        """CRITICAL workloads should start at slot 0."""
        prices = [20.0, 15.0, 5.0, 8.0]
        forecasts = _make_forecasts(prices, sigma=1.0)

        workload = Workload(
            id="w1", name="Critical Job", duration_slots=1,
            power_kw=100.0, priority=Priority.CRITICAL,
        )

        scheduler = GreedyScheduler(alpha=0.3)
        constraints = ConstraintSet([DeadlineConstraint()])
        schedule = scheduler.schedule([workload], forecasts, constraints)

        slots = schedule.slots_for_workload("w1")
        assert slots[0].slot_index == 0

    def test_respects_deadline(self) -> None:
        """Workload with deadline should not be scheduled past it."""
        prices = [20.0, 15.0, 5.0, 3.0, 2.0, 1.0]
        forecasts = _make_forecasts(prices, sigma=0.1)

        workload = Workload(
            id="w1", name="Deadline Job", duration_slots=2,
            power_kw=100.0, deadline_slot=4,
        )

        scheduler = GreedyScheduler(alpha=0.0)
        constraints = ConstraintSet([DeadlineConstraint()])
        schedule = scheduler.schedule([workload], forecasts, constraints)

        slots = schedule.slots_for_workload("w1")
        end_slot = max(s.slot_index for s in slots)
        assert end_slot < 4  # Must complete before deadline

    def test_multiple_workloads_priority_order(self) -> None:
        """Higher priority workloads should get better (cheaper) slots."""
        prices = [20.0, 15.0, 5.0, 8.0, 3.0, 4.0]
        forecasts = _make_forecasts(prices, sigma=0.1)

        w_high = Workload(
            id="w_high", name="High", duration_slots=2,
            power_kw=100.0, priority=Priority.HIGH,
        )
        w_low = Workload(
            id="w_low", name="Low", duration_slots=2,
            power_kw=100.0, priority=Priority.LOW,
        )

        scheduler = GreedyScheduler(alpha=0.0)
        constraints = ConstraintSet([DeadlineConstraint()])
        schedule = scheduler.schedule([w_high, w_low], forecasts, constraints)

        high_cost = schedule.cost_for_workload("w_high")
        low_cost = schedule.cost_for_workload("w_low")
        # High priority should get cheaper slot
        assert high_cost <= low_cost


class TestResourceConstraint:
    """Tests for resource constraint."""

    def test_blocks_overload(self) -> None:
        prices = [10.0, 10.0, 10.0, 10.0]
        forecasts = _make_forecasts(prices, sigma=0.1)

        w1 = Workload(id="w1", name="A", duration_slots=2, power_kw=60.0)
        w2 = Workload(id="w2", name="B", duration_slots=2, power_kw=60.0)

        constraints = ConstraintSet([
            DeadlineConstraint(),
            ResourceConstraint(max_power_kw=100.0),
        ])

        scheduler = GreedyScheduler(alpha=0.0)
        schedule = scheduler.schedule([w1, w2], forecasts, constraints)

        # Both workloads should be scheduled but NOT overlapping
        w1_slots = {s.slot_index for s in schedule.slots_for_workload("w1")}
        w2_slots = {s.slot_index for s in schedule.slots_for_workload("w2")}
        assert w1_slots.isdisjoint(w2_slots), "Resource constraint violated!"


class TestBlackoutConstraint:
    """Tests for blackout constraint."""

    def test_avoids_blocked_slots(self) -> None:
        prices = [5.0, 5.0, 20.0, 20.0, 10.0, 10.0]
        forecasts = _make_forecasts(prices, sigma=0.1)

        workload = Workload(id="w1", name="Test", duration_slots=2, power_kw=100.0)

        constraints = ConstraintSet([
            DeadlineConstraint(),
            BlackoutConstraint(blocked_slots=frozenset({0, 1})),
        ])

        scheduler = GreedyScheduler(alpha=0.0)
        schedule = scheduler.schedule([workload], forecasts, constraints)

        slots = schedule.slots_for_workload("w1")
        for s in slots:
            assert s.slot_index not in {0, 1}, "Blackout violated!"
