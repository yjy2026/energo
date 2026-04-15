"""Scheduling simulation with real JEPX data and trained model.

Usage:
    uv run python scripts/simulate.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from energo.features.engineering import build_features, get_feature_columns
from energo.features.scaler import FeatureScaler
from energo.models.parametric import ParametricPriceModel, create_sequences
from energo.scheduler.simulator import run_simulation
from energo.scheduler.workload import Priority, Workload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("energo.simulate")

CHECKPOINT_DIR = Path("checkpoints")
DATA_DIR = Path("data")
SEQ_LEN = 48


def main() -> None:
    logger.info("=" * 60)
    logger.info("energo — Workload Scheduling Simulation")
    logger.info("=" * 60)

    # ── Load and prepare data ─────────────────────────────
    csv_path = DATA_DIR / "jepxSpot.csv"
    if not csv_path.exists():
        logger.error("Data not found: %s", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["datetime"]).dt.tz_localize("Asia/Tokyo")
    price_col = [c for c in df.columns if "system" in c.lower() and "price" in c.lower()][0]
    data = pd.DataFrame({
        "timestamp": df["timestamp"],
        "price": pd.to_numeric(df[price_col], errors="coerce"),
    })
    # Use last 3 months as test set
    data = data[data["timestamp"] >= "2026-01-01"].dropna().reset_index(drop=True)
    logger.info("Test data: %d rows (%s to %s)", len(data), data["timestamp"].min(), data["timestamp"].max())

    # ── Feature engineering ───────────────────────────────
    featured = build_features(data)
    feature_cols = get_feature_columns(featured)
    featured_clean = featured.dropna(subset=feature_cols).reset_index(drop=True)

    # ── Load scaler ───────────────────────────────────────
    scaler_path = CHECKPOINT_DIR / "scaler.json"
    if not scaler_path.exists():
        logger.error("Scaler not found. Run train.py first.")
        sys.exit(1)
    scaler = FeatureScaler().load(scaler_path)
    scaled = scaler.transform(featured_clean)

    # ── Load model ────────────────────────────────────────
    model_path = CHECKPOINT_DIR / "best.pt"
    if not model_path.exists():
        logger.error("Model not found. Run train.py first.")
        sys.exit(1)

    model = ParametricPriceModel(
        input_dim=len(feature_cols),
        hidden_dim=64,
        num_layers=2,
        distribution="gaussian",
    )
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ── Generate predictions ──────────────────────────────
    features_t = torch.tensor(scaled[feature_cols].values, dtype=torch.float32)
    targets_t = torch.tensor(scaled["price"].values, dtype=torch.float32)

    seqs, tgts = create_sequences(features_t, targets_t, seq_len=SEQ_LEN)

    with torch.no_grad():
        pred = model(seqs)
        mu = pred["mu"].numpy()
        sigma = pred["sigma"].numpy()

    actual = tgts.numpy()
    logger.info("Predictions: %d data points", len(mu))

    # ── Define workloads ──────────────────────────────────
    workloads = [
        Workload(
            id="gpu_training",
            name="GPU Model Training (A100×4)",
            duration_slots=8,   # 4 hours
            power_kw=10.0,      # 10 kW (4x A100 + cooling)
            priority=Priority.NORMAL,
        ),
        Workload(
            id="batch_inference",
            name="Batch Inference Pipeline",
            duration_slots=4,   # 2 hours
            power_kw=5.0,       # 5 kW
            priority=Priority.HIGH,
            deadline_slot=24,   # Within 12 hours
        ),
        Workload(
            id="data_etl",
            name="Data ETL & Preprocessing",
            duration_slots=2,   # 1 hour
            power_kw=2.0,       # 2 kW
            priority=Priority.LOW,
        ),
    ]

    # ── Run simulation ────────────────────────────────────
    logger.info("\n=== Running Simulation ===")
    logger.info("Workloads:")
    for w in workloads:
        logger.info(
            "  [%s] %s: %d slots (%.1fh), %.1f kW, priority=%s",
            w.id, w.name, w.duration_slots, w.duration_hours,
            w.power_kw, w.priority.name,
        )

    # Test with different risk aversion levels
    for alpha in [0.0, 0.3, 0.7, 1.0]:
        logger.info("\n--- α = %.1f (%s) ---",
                    alpha,
                    "Aggressive" if alpha == 0 else
                    "Balanced" if alpha == 0.3 else
                    "Conservative" if alpha == 0.7 else
                    "Max Risk Aversion")

        result = run_simulation(
            actual_prices=actual,
            predicted_mu=mu,
            predicted_sigma=sigma,
            workloads=workloads,
            window_size=48,
            step_size=48,
            alpha=alpha,
        )

        logger.info(
            "  Windows: %d  |  Avg savings: %.1f%%  |  Total saved: ¥%.1f",
            result.num_windows,
            result.avg_savings_pct * 100,
            result.total_savings_jpy,
        )

    logger.info("\n" + "=" * 60)
    logger.info("Simulation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
