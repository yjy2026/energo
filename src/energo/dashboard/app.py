"""energo Dashboard — FastAPI web application.

Provides a real-time visualization of JEPX price predictions
and workload scheduling optimization.

Usage:
    uv run python -m energo.dashboard.app
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from energo.data.refresh import get_data_status, refresh_data
from energo.mcp.server import (
    _ensure_loaded,
    compare_schedules,
    get_market_status,
    get_price_forecast,
    schedule_workload,
)

logger = logging.getLogger(__name__)

DASHBOARD_DIR = Path(__file__).parent
STATIC_DIR = DASHBOARD_DIR / "static"
TEMPLATES_DIR = DASHBOARD_DIR / "templates"
DATA_DIR = Path("data")

app = FastAPI(title="energo Dashboard", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Pydantic models ──────────────────────────────────────────


class ScheduleRequest(BaseModel):
    name: str = "GPU Workload"
    duration_hours: float = 2.0
    power_kw: float = 10.0
    priority: str = "NORMAL"
    risk_aversion: float = 0.3


# ── HTML Page ────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html_path = TEMPLATES_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ── API Endpoints ────────────────────────────────────────────


@app.get("/api/market/status")
async def api_market_status() -> dict:
    return get_market_status()


@app.get("/api/forecast/{hours}")
async def api_forecast(hours: int = 24) -> dict:
    return get_price_forecast(hours_ahead=hours)


@app.post("/api/schedule")
async def api_schedule(req: ScheduleRequest) -> dict:
    return schedule_workload(
        name=req.name,
        duration_hours=req.duration_hours,
        power_kw=req.power_kw,
        priority=req.priority,
        risk_aversion=req.risk_aversion,
    )


@app.get("/api/compare")
async def api_compare(
    name: str = "GPU Workload",
    duration_hours: float = 2.0,
    power_kw: float = 10.0,
) -> dict:
    return compare_schedules(
        name=name,
        duration_hours=duration_hours,
        power_kw=power_kw,
    )


@app.get("/api/data/status")
async def api_data_status() -> dict:
    return get_data_status(DATA_DIR)


@app.post("/api/data/refresh")
async def api_data_refresh() -> dict:
    meta = refresh_data(DATA_DIR, force=True)
    return {
        "status": "ok",
        "total_rows": meta.total_rows,
        "new_rows_added": meta.new_rows_added,
    }


# ── Entry point ──────────────────────────────────────────────


def main() -> None:
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    _ensure_loaded()
    logger.info("Starting energo dashboard at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
