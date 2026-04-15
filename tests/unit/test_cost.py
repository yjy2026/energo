"""Tests for cost model."""

from __future__ import annotations

import math

import pytest

from energo.scheduler.cost import (
    SlotForecast,
    compute_cost_std,
    compute_cvar,
    compute_expected_cost,
    compute_risk_adjusted_cost,
    rank_slots,
)
from energo.scheduler.workload import Workload


@pytest.fixture()
def simple_workload() -> Workload:
    return Workload(id="w1", name="Test", duration_slots=2, power_kw=100.0)


@pytest.fixture()
def forecasts() -> list[SlotForecast]:
    return [
        SlotForecast(slot_index=0, mu=10.0, sigma=1.0),
        SlotForecast(slot_index=1, mu=12.0, sigma=2.0),
        SlotForecast(slot_index=2, mu=8.0, sigma=0.5),
        SlotForecast(slot_index=3, mu=15.0, sigma=3.0),
        SlotForecast(slot_index=4, mu=6.0, sigma=0.8),
        SlotForecast(slot_index=5, mu=7.0, sigma=1.0),
    ]


class TestExpectedCost:
    """Tests for expected cost computation."""

    def test_basic(self, simple_workload: Workload, forecasts: list[SlotForecast]) -> None:
        # Start at slot 0: uses slots 0,1 → (10+12) * 100 * 0.5 = 1100
        cost = compute_expected_cost(simple_workload, 0, forecasts)
        assert cost == pytest.approx(1100.0)

    def test_cheapest_window(
        self, simple_workload: Workload, forecasts: list[SlotForecast],
    ) -> None:
        # Start at slot 4: uses slots 4,5 → (6+7) * 100 * 0.5 = 650
        cost = compute_expected_cost(simple_workload, 4, forecasts)
        assert cost == pytest.approx(650.0)

    def test_missing_slot_returns_inf(
        self, simple_workload: Workload, forecasts: list[SlotForecast],
    ) -> None:
        cost = compute_expected_cost(simple_workload, 5, forecasts)
        assert cost == float("inf")


class TestCostStd:
    """Tests for cost standard deviation."""

    def test_known_value(self, simple_workload: Workload, forecasts: list[SlotForecast]) -> None:
        # Slots 0,1: σ_cost = sqrt((1*100*0.5)^2 + (2*100*0.5)^2)
        # = sqrt(2500 + 10000) = sqrt(12500)
        std = compute_cost_std(simple_workload, 0, forecasts)
        assert std == pytest.approx(math.sqrt(12500))

    def test_zero_sigma_gives_zero(self) -> None:
        w = Workload(id="w1", name="Test", duration_slots=1, power_kw=100.0)
        forecasts = [SlotForecast(slot_index=0, mu=10.0, sigma=0.0)]
        std = compute_cost_std(w, 0, forecasts)
        assert std == pytest.approx(0.0)


class TestCVaR:
    """Tests for Conditional Value at Risk."""

    def test_cvar_greater_than_expected(
        self, simple_workload: Workload, forecasts: list[SlotForecast]
    ) -> None:
        e_cost = compute_expected_cost(simple_workload, 0, forecasts)
        cvar = compute_cvar(simple_workload, 0, forecasts)
        # CVaR should always be >= E[cost] for risk-averse quantile
        assert cvar >= e_cost

    def test_cvar_with_zero_sigma(self) -> None:
        w = Workload(id="w1", name="Test", duration_slots=1, power_kw=100.0)
        forecasts = [SlotForecast(slot_index=0, mu=10.0, sigma=0.0)]
        # When σ=0, CVaR = E[cost]
        e_cost = compute_expected_cost(w, 0, forecasts)
        cvar = compute_cvar(w, 0, forecasts)
        assert cvar == pytest.approx(e_cost)


class TestRiskAdjustedCost:
    """Tests for risk-adjusted cost."""

    def test_alpha_zero_equals_expected(
        self, simple_workload: Workload, forecasts: list[SlotForecast]
    ) -> None:
        est = compute_risk_adjusted_cost(simple_workload, 0, forecasts, alpha=0.0)
        e_cost = compute_expected_cost(simple_workload, 0, forecasts)
        assert est.risk_adjusted_cost == pytest.approx(e_cost)

    def test_alpha_one_equals_cvar(
        self, simple_workload: Workload, forecasts: list[SlotForecast]
    ) -> None:
        est = compute_risk_adjusted_cost(simple_workload, 0, forecasts, alpha=1.0)
        cvar = compute_cvar(simple_workload, 0, forecasts)
        assert est.risk_adjusted_cost == pytest.approx(cvar)

    def test_estimate_has_all_fields(
        self, simple_workload: Workload, forecasts: list[SlotForecast]
    ) -> None:
        est = compute_risk_adjusted_cost(simple_workload, 0, forecasts, alpha=0.3)
        assert est.start_slot == 0
        assert est.expected_cost > 0
        assert est.cvar_cost >= est.expected_cost
        assert est.cost_std >= 0


class TestRankSlots:
    """Tests for slot ranking."""

    def test_cheapest_first(
        self, simple_workload: Workload, forecasts: list[SlotForecast]
    ) -> None:
        ranked = rank_slots(simple_workload, forecasts, [0, 2, 4], alpha=0.0)
        assert len(ranked) > 0
        # Slot 4 (mu=6,7) should be cheapest
        assert ranked[0].start_slot == 4

    def test_filters_infeasible(
        self, simple_workload: Workload, forecasts: list[SlotForecast]
    ) -> None:
        # Slot 5 is infeasible (only 1 slot left)
        ranked = rank_slots(simple_workload, forecasts, [0, 5], alpha=0.0)
        assert all(r.start_slot != 5 for r in ranked)
