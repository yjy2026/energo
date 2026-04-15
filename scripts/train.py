"""Training script for energo JEPX price prediction model.

Downloads (or loads cached) JEPX data and trains the Parametric
Return Model on the Tokyo System Price.

Usage:
    uv run python scripts/train.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from energo.evaluation.backtest import compute_economic_value, rolling_backtest
from energo.evaluation.metrics import evaluate
from energo.features.engineering import build_features, get_feature_columns
from energo.features.scaler import FeatureScaler
from energo.models.parametric import ParametricPriceModel, create_sequences
from energo.models.trainer import TrainConfig, Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("energo.train")

# ── Configuration ──────────────────────────────────────
DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("checkpoints")
SCALER_PATH = CHECKPOINT_DIR / "scaler.json"

# Use last 2 years of data for faster iteration
TRAIN_START = "2024-01-01"
REGION = "System"  # System price (national)
SEQ_LEN = 48  # 24 hours of context
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.1
DISTRIBUTION = "gaussian"

TRAIN_CONFIG = TrainConfig(
    learning_rate=5e-4,
    batch_size=128,
    max_epochs=50,
    patience=8,
    lr_factor=0.5,
    lr_patience=4,
    gradient_clip=1.0,
    checkpoint_dir=CHECKPOINT_DIR,
)


def load_jepx_data() -> pd.DataFrame:
    """Load JEPX spot price CSV from local cache."""
    csv_path = DATA_DIR / "jepxSpot.csv"
    if not csv_path.exists():
        logger.error("JEPX data not found at %s. Run download first.", csv_path)
        sys.exit(1)

    logger.info("Loading JEPX data from %s", csv_path)
    df = pd.read_csv(csv_path)

    # Parse the CSV format:
    # datetime, Date, PeriodID, System Price Yen/kWh, Hokkaido, ...
    df["timestamp"] = pd.to_datetime(df["datetime"])
    df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Tokyo")

    # Extract System Price
    price_col = [c for c in df.columns if "system" in c.lower() and "price" in c.lower()]
    if not price_col:
        logger.error("Cannot find System Price column. Columns: %s", df.columns.tolist())
        sys.exit(1)

    result = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "price": pd.to_numeric(df[price_col[0]], errors="coerce"),
        }
    )

    # Filter to training period
    result = result[result["timestamp"] >= pd.Timestamp(TRAIN_START, tz="Asia/Tokyo")]
    result = result.dropna(subset=["price"]).reset_index(drop=True)

    logger.info(
        "Loaded %d rows: %s to %s",
        len(result),
        result["timestamp"].min(),
        result["timestamp"].max(),
    )

    return result


def main() -> None:
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("energo — JEPX Price Prediction Training")
    logger.info("=" * 60)

    # ── Step 1: Load data ───────────────────────────────
    df = load_jepx_data()

    # ── Step 2: Feature engineering ─────────────────────
    logger.info("Building features...")
    featured = build_features(df)
    feature_cols = get_feature_columns(featured)
    logger.info("Generated %d features", len(feature_cols))

    # Drop rows with NaN from lag features
    featured_clean = featured.dropna(subset=feature_cols).reset_index(drop=True)
    logger.info("After NaN drop: %d rows", len(featured_clean))

    # ── Step 3: Scale features ──────────────────────────
    # Temporal split first, then fit scaler on train only
    n = len(featured_clean)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_df = featured_clean.iloc[:train_end].copy()
    val_df = featured_clean.iloc[train_end:val_end].copy()
    test_df = featured_clean.iloc[val_end:].copy()

    logger.info(
        "Split: train=%d, val=%d, test=%d",
        len(train_df), len(val_df), len(test_df),
    )

    scaler = FeatureScaler(method="standard")
    scaler.fit(train_df, feature_cols)

    train_scaled = scaler.transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)

    # Save scaler
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    scaler.save(SCALER_PATH)

    # ── Step 4: Create sequences ─────────────────────────
    def to_sequences(data: pd.DataFrame):
        features = torch.tensor(
            data[feature_cols].values, dtype=torch.float32
        )
        # Target is RAW (unscaled) price
        targets = torch.tensor(
            data["price"].values, dtype=torch.float32
        )
        return create_sequences(features, targets, seq_len=SEQ_LEN)

    train_x, train_y = to_sequences(train_scaled)
    val_x, val_y = to_sequences(val_scaled)
    test_x, test_y = to_sequences(test_scaled)

    logger.info(
        "Sequences: train=%d, val=%d, test=%d (seq_len=%d, features=%d)",
        len(train_x), len(val_x), len(test_x),
        SEQ_LEN, len(feature_cols),
    )

    # ── Step 5: Train model ──────────────────────────────
    model = ParametricPriceModel(
        input_dim=len(feature_cols),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        distribution=DISTRIBUTION,
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d", total_params)

    trainer = Trainer(model=model, config=TRAIN_CONFIG)
    result = trainer.train(train_x, train_y, val_x, val_y)

    logger.info(
        "Training complete: best_epoch=%d, best_val_loss=%.4f",
        result.best_epoch,
        result.best_val_loss,
    )

    # ── Step 6: Evaluate on test set ─────────────────────
    logger.info("Evaluating on test set...")
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        test_pred = model(test_x.to(device))
        mu = test_pred["mu"].cpu().numpy()
        sigma = test_pred["sigma"].cpu().numpy()

    actual = test_y.numpy()

    # Full evaluation
    eval_result = evaluate(actual, mu, sigma)
    logger.info("\n%s", eval_result.summary())

    # ── Step 7: Backtest analysis ────────────────────────
    # Get timestamps for test predictions
    test_timestamps = test_scaled["timestamp"].values[SEQ_LEN:]

    bt_result = rolling_backtest(test_timestamps, actual, mu, sigma)

    # Economic value
    econ = compute_economic_value(bt_result.predictions)
    logger.info("\n=== Economic Value ===")
    logger.info("  Baseline avg price: %.2f JPY/kWh", econ["avg_price_baseline"])
    logger.info("  Optimized avg price: %.2f JPY/kWh", econ["avg_price_optimized"])
    logger.info("  Potential savings: %.1f%%", econ["savings_pct"] * 100)

    # Best/worst hours
    logger.info("\n=== Performance by Hour ===")
    for hour in sorted(bt_result.by_hour.keys()):
        h_result = bt_result.by_hour[hour]
        logger.info(
            "  %02d:00  MAE=%.2f  CRPS=%.4f  Cov90=%.0f%%",
            hour, h_result.mae, h_result.crps, h_result.coverage_90 * 100,
        )

    logger.info("\n" + "=" * 60)
    logger.info("Done! Model saved to %s", CHECKPOINT_DIR / "best.pt")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
