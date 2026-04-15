"""Data refresh CLI script.

Usage:
    uv run python scripts/refresh.py              # Refresh data only
    uv run python scripts/refresh.py --retrain     # Refresh + retrain model
    uv run python scripts/refresh.py --force       # Force refresh (ignore TTL)
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from energo.data.refresh import get_data_status, refresh_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("energo.refresh")

DATA_DIR = Path("data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh JEPX data")
    parser.add_argument(
        "--retrain", action="store_true", help="Retrain model after refresh",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force refresh (ignore TTL)",
    )
    args = parser.parse_args()

    # Show current status
    status = get_data_status(DATA_DIR)
    logger.info("Current data status: %s", status)

    # Refresh
    meta = refresh_data(DATA_DIR, force=args.force)
    logger.info("Refresh complete: %d total rows, %d new", meta.total_rows, meta.new_rows_added)

    # Retrain if requested
    if args.retrain and meta.new_rows_added > 0:
        logger.info("Retraining model...")
        result = subprocess.run(
            [sys.executable, "scripts/train.py"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info("Retrain complete!")
        else:
            logger.error("Retrain failed:\n%s", result.stderr[-500:])
    elif args.retrain and meta.new_rows_added == 0:
        logger.info("No new data — skipping retrain.")


if __name__ == "__main__":
    main()
