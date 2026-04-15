# ⚡ energo

**Energy-aware workload scheduling for AI infrastructure.**

energo predicts electricity spot prices on the Japanese power exchange (JEPX) and automatically schedules GPU workloads to minimize cost — saving up to **41% on electricity** with zero manual effort.

---

## Key Results

| Metric | Value |
|--------|-------|
| Price forecast MAE | 1.12 JPY/kWh (9.6% error) |
| Spike detection recall | 96.3% |
| Cost reduction (vs. uniform) | **41.6%** |
| Model size | 62,642 parameters |
| Tests | 120 passing · 86% coverage |

## How It Works

```
Historical JEPX data (16 yrs)
        │
        ▼
 ┌──────────────┐
 │ Price Engine  │  LSTM → (μ, σ) probabilistic forecast
 └──────┬───────┘
        │
        ▼
 ┌──────────────┐
 │  Scheduler   │  CVaR-based risk-adjusted optimization
 └──────┬───────┘
        │
        ▼
 ┌──────────────┐
 │  MCP Server  │  5 tools for AI agents
 └──────────────┘
```

The system doesn't just predict the cheapest time slot — it accounts for **price uncertainty** using Conditional Value at Risk (CVaR), letting you tune the aggressiveness of scheduling with a single parameter `α`:

- `α = 0.0` — Minimize expected cost only (aggressive)
- `α = 0.3` — Balance cost and risk (default)
- `α = 1.0` — Minimize worst-case cost (conservative)

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- CUDA-capable GPU (optional, CPU works fine)

### Install & Train

```bash
git clone https://github.com/yjy2026/energo.git
cd energo
uv sync
uv run python scripts/train.py
```

Training takes ~20 seconds on GPU, ~60 seconds on CPU.

### Run the MCP Server

```bash
# stdio transport (Claude Desktop / Cursor / Antigravity)
uv run python -m energo.mcp.server

# SSE transport (remote clients)
uv run python -m energo.mcp.server --transport sse --port 8000
```

### Connect to Claude Desktop

Add to `~/.config/claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "energo": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/energo", "python", "-m", "energo.mcp.server"]
    }
  }
}
```

Then ask: *"When should I run a 4-hour GPU training job to minimize electricity cost?"*

## MCP Tools

### `get_price_forecast`

Returns predicted price distributions (μ, σ, 90% CI) for each 30-minute slot over the next 1–72 hours.

### `schedule_workload`

Finds the optimal time to run a workload given its duration, power draw, priority, and deadline. Returns cost comparison against immediate execution.

### `estimate_cost`

Quick cost estimate for running a workload at a specific time.

### `get_market_status`

Current JEPX market summary — latest price, daily/weekly averages, volatility, and trend direction.

### `compare_schedules`

Side-by-side comparison of aggressive, balanced, and conservative scheduling strategies for a given workload.

## Architecture

```
src/energo/
├── data/              # Data ingestion (JEPX spot prices, Open-Meteo weather)
│   ├── providers/
│   │   ├── base.py    # Country-agnostic DataProvider protocol
│   │   ├── jepx.py    # JEPX spot price & demand parser
│   │   └── weather.py # Open-Meteo historical weather
│   └── pipeline.py    # Merge, align, and split datasets
├── features/          # Feature engineering
│   ├── engineering.py # Cyclical time, lags, rolling stats, calendar
│   └── scaler.py      # Standard/Robust scaling with JSON persistence
├── models/            # Price prediction
│   ├── parametric.py  # LSTM encoder → Gaussian/Student-t (μ, σ) head
│   ├── trainer.py     # AdamW + ReduceLROnPlateau + early stopping
│   └── predictor.py   # Inference with confidence intervals
├── scheduler/         # Workload optimization
│   ├── workload.py    # Workload, Priority, Schedule definitions
│   ├── cost.py        # E[cost], CVaR, risk-adjusted cost
│   ├── constraints.py # Deadline, resource, blackout constraints
│   ├── optimizer.py   # Greedy scheduler with priority ordering
│   ├── simulator.py   # Rolling-window backtester
│   └── report.py      # Structured result reporting
├── evaluation/        # Model evaluation
│   ├── metrics.py     # CRPS, coverage, MAE/RMSE/MAPE, spike recall
│   └── backtest.py    # Time-decomposed performance analysis
└── mcp/               # MCP server
    └── server.py      # FastMCP server with 5 tools + 2 resources
```

## Design Decisions

**Probabilistic, not point forecasts.** The model outputs (μ, σ) instead of a single price, enabling principled risk management through CVaR.

**Country-agnostic protocols.** `DataProvider` and `WeatherProvider` are abstract protocols — adding KPX (Korea) or ERCOT (Texas) requires only a new provider implementation.

**Leakage-free pipeline.** Strict temporal splitting with `shift(1)`-based feature engineering ensures no future information leaks into training.

**Lightweight model.** 62K parameters train in seconds, not hours. The bottleneck should be data quality, not model complexity.

## Testing

```bash
uv run pytest tests/ -v                          # Run all tests
uv run pytest tests/ --cov=energo --cov-report=html  # Coverage report
uv run ruff check src/ tests/                     # Lint
```

## Data Sources

| Source | Data | Cost |
|--------|------|------|
| [japanesepower.org](https://japanesepower.org) | JEPX spot prices, demand, temperature (2010–present) | Free |
| [Open-Meteo](https://open-meteo.com) | Historical weather (temperature, solar radiation, wind) | Free |

## License

MIT
