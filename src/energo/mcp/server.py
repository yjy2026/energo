"""energo MCP Server — Energy-aware AI workload optimization.

Exposes JEPX price forecasting and workload scheduling capabilities
as MCP tools for AI agents to consume.

Usage:
    # stdio transport (for Claude Desktop / Cursor)
    uv run python -m energo.mcp.server

    # SSE transport (for remote clients)
    uv run python -m energo.mcp.server --transport sse --port 8000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastmcp import FastMCP

from energo.features.engineering import build_features, get_feature_columns
from energo.features.scaler import FeatureScaler
from energo.models.parametric import ParametricPriceModel, create_sequences
from energo.scheduler.constraints import (
    ConstraintSet,
    DeadlineConstraint,
    ResourceConstraint,
)
from energo.scheduler.cost import SlotForecast, compute_risk_adjusted_cost
from energo.scheduler.optimizer import GreedyScheduler
from energo.scheduler.workload import Priority, Workload

# Logging — MUST use stderr to avoid corrupting stdio transport
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("energo.mcp")

# ── Global state (loaded once at startup) ────────────────────
_model: ParametricPriceModel | None = None
_scaler: FeatureScaler | None = None
_feature_cols: list[str] = []
_price_data: pd.DataFrame | None = None
_forecasts_cache: dict[str, list[SlotForecast]] = {}
_cache_time: float = 0.0
_CACHE_TTL = 1800.0  # 30 minutes

CHECKPOINT_DIR = Path("checkpoints")
DATA_DIR = Path("data")
SEQ_LEN = 48

# ── MCP Server ───────────────────────────────────────────────
mcp = FastMCP(
    "energo",
    instructions=(
        "energo is an energy-aware AI workload optimization server "
        "for the Japanese electricity market (JEPX). "
        "It provides price forecasting and workload scheduling tools "
        "to minimize compute infrastructure electricity costs."
    ),
)


def _ensure_loaded() -> bool:
    """Lazily load model, scaler, and data on first use."""
    global _model, _scaler, _feature_cols, _price_data

    if _model is not None:
        return True

    # Load model
    model_path = CHECKPOINT_DIR / "best.pt"
    scaler_path = CHECKPOINT_DIR / "scaler.json"

    if not model_path.exists() or not scaler_path.exists():
        logger.error("Model or scaler not found. Run `uv run python scripts/train.py` first.")
        return False

    # Load scaler to get feature columns
    _scaler = FeatureScaler().load(scaler_path)

    # Load data to determine feature count
    csv_path = DATA_DIR / "jepxSpot.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["datetime"]).dt.tz_localize("Asia/Tokyo")
        price_col = [c for c in df.columns if "system" in c.lower() and "price" in c.lower()][0]
        _price_data = pd.DataFrame({
            "timestamp": df["timestamp"],
            "price": pd.to_numeric(df[price_col], errors="coerce"),
        }).dropna().reset_index(drop=True)

        featured = build_features(_price_data.tail(48 * 30))  # Last 30 days
        _feature_cols = get_feature_columns(featured)
    else:
        logger.error("JEPX data not found at %s", csv_path)
        return False

    # Load model
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    _model = ParametricPriceModel(
        input_dim=len(_feature_cols),
        hidden_dim=checkpoint["config"]["hidden_dim"],
        distribution=checkpoint["config"]["distribution"],
    )
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()

    logger.info("Model loaded (%d params)", sum(p.numel() for p in _model.parameters()))

    return True


def _get_forecasts(horizon_slots: int = 48) -> list[SlotForecast]:
    """Generate price forecasts for the next N slots (cached with TTL)."""
    import time

    global _cache_time, _forecasts_cache

    # Invalidate cache if TTL expired
    if _cache_time > 0 and (time.time() - _cache_time) > _CACHE_TTL:
        _forecasts_cache.clear()
        _cache_time = 0.0

    cache_key = f"h{horizon_slots}"
    if cache_key in _forecasts_cache:
        return _forecasts_cache[cache_key]

    if not _ensure_loaded() or _price_data is None or _scaler is None or _model is None:
        return []

    # Use latest data for prediction
    recent = _price_data.tail(SEQ_LEN + horizon_slots + 500).copy().reset_index(drop=True)
    featured = build_features(recent)
    featured_clean = featured.dropna(subset=_feature_cols).reset_index(drop=True)
    scaled = _scaler.transform(featured_clean)

    features_t = torch.tensor(scaled[_feature_cols].values, dtype=torch.float32)
    targets_t = torch.tensor(scaled["price"].values, dtype=torch.float32)

    seqs, _tgts = create_sequences(features_t, targets_t, seq_len=SEQ_LEN)

    with torch.no_grad():
        pred = _model(seqs[-horizon_slots:])
        mu = pred["mu"].numpy()
        sigma = pred["sigma"].numpy()

    forecasts = [
        SlotForecast(slot_index=i, mu=float(mu[i]), sigma=float(sigma[i]))
        for i in range(len(mu))
    ]

    _forecasts_cache[cache_key] = forecasts
    _cache_time = time.time()
    return forecasts


# ── MCP Tools ────────────────────────────────────────────────


@mcp.tool()
def get_price_forecast(hours_ahead: int = 24) -> dict:
    """Get JEPX electricity price forecast for the next N hours.

    Returns predicted price distribution (mean, std, confidence intervals)
    for each 30-minute slot.

    Args:
        hours_ahead: Number of hours to forecast (1-72, default 24).

    Returns:
        Dictionary with forecast data including timestamps, predicted
        prices, uncertainty bands, and summary statistics.
    """
    hours_ahead = max(1, min(72, hours_ahead))
    n_slots = hours_ahead * 2

    forecasts = _get_forecasts(n_slots)
    if not forecasts:
        return {"error": "Model not loaded. Run training first."}

    slots_data = []
    for f in forecasts[:n_slots]:
        slots_data.append({
            "slot": f.slot_index,
            "hour_offset": f.slot_index * 0.5,
            "price_mean": round(f.mu, 2),
            "price_std": round(f.sigma, 2),
            "ci_90_lower": round(f.mu - 1.645 * f.sigma, 2),
            "ci_90_upper": round(f.mu + 1.645 * f.sigma, 2),
        })

    mus = [f.mu for f in forecasts[:n_slots]]
    sigmas = [f.sigma for f in forecasts[:n_slots]]

    return {
        "forecast_horizon_hours": hours_ahead,
        "num_slots": len(slots_data),
        "slot_duration_minutes": 30,
        "currency": "JPY/kWh",
        "summary": {
            "avg_price": round(np.mean(mus), 2),
            "min_price": round(np.min(mus), 2),
            "max_price": round(np.max(mus), 2),
            "avg_uncertainty": round(np.mean(sigmas), 2),
            "cheapest_hour_offset": round(
                float(np.argmin(mus)) * 0.5, 1
            ),
            "most_expensive_hour_offset": round(
                float(np.argmax(mus)) * 0.5, 1
            ),
        },
        "slots": slots_data,
    }


@mcp.tool()
def schedule_workload(
    name: str,
    duration_hours: float,
    power_kw: float,
    priority: str = "NORMAL",
    deadline_hours: float | None = None,
    risk_aversion: float = 0.3,
    max_concurrent_power_kw: float | None = None,
) -> dict:
    """Schedule a compute workload for optimal electricity cost.

    Finds the best time to run the workload based on price predictions
    and risk preferences. Uses CVaR-based risk-adjusted cost optimization.

    Args:
        name: Name of the workload (e.g. "GPU Model Training").
        duration_hours: How long the workload runs (in hours).
        power_kw: Power consumption in kilowatts.
        priority: "CRITICAL", "HIGH", "NORMAL", or "LOW".
        deadline_hours: Must complete within this many hours (optional).
        risk_aversion: 0.0 (aggressive/cheapest) to 1.0 (conservative/safest).
        max_concurrent_power_kw: Max total power at any time slot (optional).

    Returns:
        Optimal schedule with cost estimates and comparison to baseline.
    """
    # Parse priority
    priority_map = {
        "CRITICAL": Priority.CRITICAL,
        "HIGH": Priority.HIGH,
        "NORMAL": Priority.NORMAL,
        "LOW": Priority.LOW,
    }
    pri = priority_map.get(priority.upper(), Priority.NORMAL)

    duration_slots = max(1, int(duration_hours * 2))
    deadline_slot = int(deadline_hours * 2) if deadline_hours else None

    workload = Workload(
        id="mcp_workload",
        name=name,
        duration_slots=duration_slots,
        power_kw=power_kw,
        deadline_slot=deadline_slot,
        priority=pri,
    )

    # Get forecasts
    horizon = max(48, (deadline_slot or 48) + duration_slots)
    forecasts = _get_forecasts(horizon)
    if not forecasts:
        return {"error": "Model not loaded."}

    # Build constraints
    constraints = ConstraintSet([DeadlineConstraint()])
    if max_concurrent_power_kw is not None:
        constraints.add(ResourceConstraint(max_power_kw=max_concurrent_power_kw))

    # Schedule
    scheduler = GreedyScheduler(alpha=risk_aversion)
    schedule = scheduler.schedule([workload], forecasts, constraints)

    slots = schedule.slots_for_workload("mcp_workload")
    if not slots:
        return {"error": "No feasible schedule found."}

    start_slot = min(s.slot_index for s in slots)
    end_slot = max(s.slot_index for s in slots)

    # Cost comparison
    predicted_cost = schedule.cost_for_workload("mcp_workload")

    # Baseline: start now (slot 0)
    baseline_cost = sum(
        f.mu * workload.power_kw * 0.5
        for f in forecasts[:duration_slots]
    )

    savings = baseline_cost - predicted_cost
    savings_pct = savings / baseline_cost if baseline_cost > 0 else 0

    # Risk metrics
    cost_est = compute_risk_adjusted_cost(
        workload, start_slot, forecasts, alpha=risk_aversion,
    )

    return {
        "workload": {
            "name": name,
            "duration_hours": duration_hours,
            "power_kw": power_kw,
            "energy_kwh": workload.energy_kwh,
            "priority": priority.upper(),
        },
        "optimal_schedule": {
            "start_slot": start_slot,
            "end_slot": end_slot,
            "start_hours_from_now": start_slot * 0.5,
            "slot_prices": [
                {"slot": s.slot_index, "price": round(s.predicted_price, 2)}
                for s in slots
            ],
        },
        "cost_analysis": {
            "predicted_cost_jpy": round(predicted_cost, 1),
            "baseline_cost_jpy": round(baseline_cost, 1),
            "savings_jpy": round(savings, 1),
            "savings_pct": round(savings_pct * 100, 1),
            "cvar_cost_jpy": round(cost_est.cvar_cost, 1),
            "risk_adjusted_cost_jpy": round(cost_est.risk_adjusted_cost, 1),
        },
        "recommendation": (
            f"Run '{name}' starting in {start_slot * 0.5:.1f} hours "
            f"to save ¥{savings:.0f} ({savings_pct:.0%}) vs running now."
        ),
    }


@mcp.tool()
def estimate_cost(
    duration_hours: float,
    power_kw: float,
    start_hours_from_now: float = 0,
) -> dict:
    """Estimate electricity cost for running a workload at a specific time.

    Quick cost check without full scheduling optimization.

    Args:
        duration_hours: Workload duration in hours.
        power_kw: Power consumption in kW.
        start_hours_from_now: When to start (hours from now).

    Returns:
        Cost estimate with breakdown by slot.
    """
    start_slot = int(start_hours_from_now * 2)
    duration_slots = max(1, int(duration_hours * 2))

    forecasts = _get_forecasts(start_slot + duration_slots + 10)
    if not forecasts:
        return {"error": "Model not loaded."}

    forecast_map = {f.slot_index: f for f in forecasts}

    total_cost = 0.0
    slot_breakdown = []
    for offset in range(duration_slots):
        idx = start_slot + offset
        f = forecast_map.get(idx)
        if f is None:
            continue
        slot_cost = f.mu * power_kw * 0.5  # 30 min = 0.5h
        total_cost += slot_cost
        slot_breakdown.append({
            "slot": idx,
            "hour_offset": idx * 0.5,
            "price_jpy_kwh": round(f.mu, 2),
            "slot_cost_jpy": round(slot_cost, 1),
        })

    energy_kwh = power_kw * duration_hours

    return {
        "energy_kwh": round(energy_kwh, 1),
        "total_cost_jpy": round(total_cost, 1),
        "avg_price_jpy_kwh": round(
            total_cost / energy_kwh if energy_kwh > 0 else 0, 2
        ),
        "slot_breakdown": slot_breakdown,
    }


@mcp.tool()
def get_market_status() -> dict:
    """Get current JEPX market status and key indicators.

    Returns summary of recent prices, volatility, and trend.
    """
    if not _ensure_loaded() or _price_data is None:
        return {"error": "Data not loaded."}

    recent = _price_data.tail(48 * 7)  # Last 7 days
    prices = recent["price"].values

    today = prices[-48:] if len(prices) >= 48 else prices
    yesterday = prices[-96:-48] if len(prices) >= 96 else prices
    week = prices

    return {
        "market": "JEPX",
        "data_points": len(_price_data),
        "latest_price": round(float(prices[-1]), 2),
        "today": {
            "avg": round(float(np.mean(today)), 2),
            "min": round(float(np.min(today)), 2),
            "max": round(float(np.max(today)), 2),
            "std": round(float(np.std(today)), 2),
        },
        "yesterday": {
            "avg": round(float(np.mean(yesterday)), 2),
            "min": round(float(np.min(yesterday)), 2),
            "max": round(float(np.max(yesterday)), 2),
        },
        "week": {
            "avg": round(float(np.mean(week)), 2),
            "volatility": round(float(np.std(week)), 2),
            "trend": "rising" if np.mean(today) > np.mean(yesterday) else "falling",
        },
    }


@mcp.tool()
def compare_schedules(
    name: str,
    duration_hours: float,
    power_kw: float,
) -> dict:
    """Compare scheduling strategies for a workload.

    Runs both aggressive (α=0) and conservative (α=1) scheduling
    to help decide the risk-cost tradeoff.

    Args:
        name: Workload name.
        duration_hours: Duration in hours.
        power_kw: Power consumption in kW.

    Returns:
        Side-by-side comparison of aggressive, balanced, and conservative schedules.
    """
    strategies = [
        ("aggressive", 0.0),
        ("balanced", 0.3),
        ("conservative", 0.7),
        ("max_safety", 1.0),
    ]

    results = {}
    for strategy_name, alpha in strategies:
        result = schedule_workload(
            name=name,
            duration_hours=duration_hours,
            power_kw=power_kw,
            risk_aversion=alpha,
        )
        if "error" in result:
            return result
        results[strategy_name] = {
            "alpha": alpha,
            "start_hours_from_now": result["optimal_schedule"]["start_hours_from_now"],
            "predicted_cost_jpy": result["cost_analysis"]["predicted_cost_jpy"],
            "cvar_cost_jpy": result["cost_analysis"]["cvar_cost_jpy"],
            "savings_pct": result["cost_analysis"]["savings_pct"],
        }

    return {
        "workload": name,
        "strategies": results,
        "recommendation": (
            "Use 'balanced' (α=0.3) for most workloads. "
            "Use 'aggressive' only if cost is the sole concern. "
            "Use 'conservative' for mission-critical jobs."
        ),
    }


# ── Resources ────────────────────────────────────────────────


@mcp.resource("energo://model/info")
def model_info() -> str:
    """Information about the loaded prediction model."""
    if not _ensure_loaded() or _model is None:
        return "Model not loaded."

    params = sum(p.numel() for p in _model.parameters())
    return (
        f"Model: ParametricPriceModel\n"
        f"Parameters: {params:,}\n"
        f"Input features: {_model.input_dim}\n"
        f"Hidden dim: {_model.hidden_dim}\n"
        f"Distribution: {_model.distribution}\n"
        f"Sequence length: {SEQ_LEN} slots (24h)\n"
        f"Target market: JEPX (Japan)\n"
    )


@mcp.resource("energo://market/regions")
def market_regions() -> str:
    """JEPX market regions and their characteristics."""
    return (
        "JEPX Market Regions:\n"
        "1. Hokkaido - Northern island, wind-rich\n"
        "2. Tohoku - Northeast Honshu\n"
        "3. Tokyo - Largest demand center\n"
        "4. Chubu - Central Japan\n"
        "5. Hokuriku - Japan Sea coast\n"
        "6. Kansai - Osaka/Kyoto area\n"
        "7. Chugoku - Western Honshu\n"
        "8. Shikoku - Smallest island utility\n"
        "9. Kyushu - Southern, solar-rich\n"
    )


# ── Entry point ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="energo MCP Server")
    parser.add_argument(
        "--transport", choices=["stdio", "sse"], default="stdio",
        help="Transport type (default: stdio)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port for SSE transport (default: 8000)",
    )
    args = parser.parse_args()

    logger.info("Starting energo MCP server (transport=%s)", args.transport)

    if args.transport == "sse":
        mcp.run(transport="sse", port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
