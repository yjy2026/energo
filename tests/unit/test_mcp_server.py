"""Tests for energo MCP server tools.

Uses FastMCP's built-in testing capabilities to validate
tool behavior without starting a real server process.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from energo.mcp.server import (
    compare_schedules,
    estimate_cost,
    get_market_status,
    get_price_forecast,
    schedule_workload,
)
from energo.scheduler.cost import SlotForecast


def _mock_forecasts(n: int = 96) -> list[SlotForecast]:
    """Create mock forecasts with a daily price pattern."""
    rng = np.random.default_rng(42)
    forecasts = []
    for i in range(n):
        hour = (i * 0.5) % 24
        # Higher prices during morning (8-10) and evening (17-20)
        base = 10 + 3 * np.sin(2 * np.pi * (hour - 6) / 24)
        mu = float(base + rng.standard_normal() * 0.5)
        sigma = float(abs(rng.standard_normal()) * 0.5 + 0.3)
        forecasts.append(SlotForecast(slot_index=i, mu=mu, sigma=sigma))
    return forecasts


@pytest.fixture(autouse=True)
def _setup_mock_forecasts():
    """Inject mock forecasts into the base forecast cache for all tests."""
    import energo.mcp.server as srv

    mock = _mock_forecasts(144)  # 3 days
    srv._base_forecasts.clear()
    srv._base_forecasts.extend(mock)
    srv._cache_time = 1e18  # Far future — never expire during tests
    yield
    srv._base_forecasts.clear()
    srv._cache_time = 0.0


class TestGetPriceForecast:
    """Tests for the get_price_forecast tool."""

    def test_returns_forecast_data(self) -> None:
        result = get_price_forecast(hours_ahead=24)

        assert "forecast_horizon_hours" in result
        assert result["forecast_horizon_hours"] == 24
        assert "summary" in result
        assert "slots" in result
        assert len(result["slots"]) == 48  # 24h * 2 slots/h

    def test_summary_statistics(self) -> None:
        result = get_price_forecast(hours_ahead=24)

        summary = result["summary"]
        assert "avg_price" in summary
        assert "min_price" in summary
        assert "max_price" in summary
        assert "cheapest_hour_offset" in summary
        assert summary["min_price"] <= summary["avg_price"] <= summary["max_price"]

    def test_slot_structure(self) -> None:
        result = get_price_forecast(hours_ahead=24)

        slot = result["slots"][0]
        assert "slot" in slot
        assert "price_mean" in slot
        assert "price_std" in slot
        assert "ci_90_lower" in slot
        assert "ci_90_upper" in slot
        assert slot["ci_90_lower"] < slot["price_mean"] < slot["ci_90_upper"]

    def test_clamps_horizon(self) -> None:
        result = get_price_forecast(hours_ahead=200)
        assert result["forecast_horizon_hours"] == 72  # Clamped to max


class TestScheduleWorkload:
    """Tests for the schedule_workload tool."""

    def test_returns_schedule(self) -> None:
        result = schedule_workload(
            name="Test Job",
            duration_hours=2.0,
            power_kw=100.0,
        )

        assert "optimal_schedule" in result
        assert "cost_analysis" in result
        assert "recommendation" in result

    def test_cost_analysis_fields(self) -> None:
        result = schedule_workload(
            name="Test Job",
            duration_hours=1.0,
            power_kw=50.0,
        )

        cost = result["cost_analysis"]
        assert "predicted_cost_jpy" in cost
        assert "baseline_cost_jpy" in cost
        assert "savings_jpy" in cost
        assert "savings_pct" in cost
        assert "cvar_cost_jpy" in cost

    def test_savings_non_negative(self) -> None:
        """Optimized schedule should not be more expensive than baseline."""
        result = schedule_workload(
            name="Test Job",
            duration_hours=2.0,
            power_kw=100.0,
            risk_aversion=0.0,
        )

        assert result["cost_analysis"]["savings_jpy"] >= 0

    def test_critical_priority_immediate(self) -> None:
        result = schedule_workload(
            name="Critical Job",
            duration_hours=1.0,
            power_kw=100.0,
            priority="CRITICAL",
        )

        assert result["optimal_schedule"]["start_slot"] == 0

    def test_risk_aversion_parameter(self) -> None:
        aggressive = schedule_workload(
            name="Job", duration_hours=2.0, power_kw=100.0,
            risk_aversion=0.0,
        )
        conservative = schedule_workload(
            name="Job", duration_hours=2.0, power_kw=100.0,
            risk_aversion=1.0,
        )

        # Both should return valid results
        assert "optimal_schedule" in aggressive
        assert "optimal_schedule" in conservative


class TestEstimateCost:
    """Tests for the estimate_cost tool."""

    def test_returns_cost(self) -> None:
        result = estimate_cost(
            duration_hours=2.0,
            power_kw=100.0,
            start_hours_from_now=0,
        )

        assert "total_cost_jpy" in result
        assert "energy_kwh" in result
        assert result["energy_kwh"] == 200.0  # 100kW * 2h
        assert result["total_cost_jpy"] > 0

    def test_slot_breakdown(self) -> None:
        result = estimate_cost(
            duration_hours=1.0,
            power_kw=50.0,
            start_hours_from_now=0,
        )

        assert "slot_breakdown" in result
        assert len(result["slot_breakdown"]) == 2  # 1h = 2 slots


class TestCompareSchedules:
    """Tests for compare_schedules tool."""

    def test_returns_all_strategies(self) -> None:
        result = compare_schedules(
            name="Test Job",
            duration_hours=2.0,
            power_kw=100.0,
        )

        assert "strategies" in result
        strategies = result["strategies"]
        assert "aggressive" in strategies
        assert "balanced" in strategies
        assert "conservative" in strategies
        assert "max_safety" in strategies

    def test_strategy_fields(self) -> None:
        result = compare_schedules(
            name="Test Job",
            duration_hours=1.0,
            power_kw=50.0,
        )

        for _name, data in result["strategies"].items():
            assert "alpha" in data
            assert "predicted_cost_jpy" in data
            assert "cvar_cost_jpy" in data
            assert "savings_pct" in data


class TestGetMarketStatus:
    """Tests for get_market_status tool."""

    @patch("energo.mcp.server._price_data")
    @patch("energo.mcp.server._model", new="dummy")
    def test_returns_status(self, mock_data) -> None:
        import pandas as pd
        rng = np.random.default_rng(42)
        n = 48 * 7
        mock_data.__bool__ = lambda self: True
        mock_data.tail.return_value = pd.DataFrame({
            "price": 10 + rng.standard_normal(n),
        })
        mock_data.__len__ = lambda self: n

        result = get_market_status()

        # Since we're mocking, verify the structure would work
        assert isinstance(result, dict)
