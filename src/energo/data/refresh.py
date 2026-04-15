"""Data refresh engine.

Handles incremental data updates from JEPX sources
and cache invalidation for the prediction pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd
import requests

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_BASE_URL = "https://japanesepower.org"
_SPOT_CSV_URL = f"{_BASE_URL}/jepxSpot.csv"
_REQUEST_TIMEOUT = 120
_MIN_REFRESH_INTERVAL = 1800  # 30 minutes


@dataclass
class RefreshMeta:
    """Metadata for data refresh state."""

    last_refresh_utc: str = ""
    total_rows: int = 0
    new_rows_added: int = 0
    source_url: str = _SPOT_CSV_URL

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> RefreshMeta:
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls(**data)


def should_refresh(data_dir: Path) -> bool:
    """Check if data should be refreshed.

    Returns True if:
    - No data exists yet
    - Last refresh was more than 30 minutes ago
    """
    meta_path = data_dir / "refresh_meta.json"
    meta = RefreshMeta.load(meta_path)

    if not meta.last_refresh_utc:
        return True

    last = datetime.fromisoformat(meta.last_refresh_utc)
    elapsed = (datetime.now(UTC) - last).total_seconds()

    return elapsed > _MIN_REFRESH_INTERVAL


def refresh_data(data_dir: Path, force: bool = False) -> RefreshMeta:
    """Download latest JEPX data and merge with existing cache.

    Args:
        data_dir: Directory to store CSV data and metadata.
        force: If True, refresh regardless of time interval.

    Returns:
        RefreshMeta with update statistics.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_path = data_dir / "refresh_meta.json"
    csv_path = data_dir / "jepxSpot.csv"

    if not force and not should_refresh(data_dir):
        meta = RefreshMeta.load(meta_path)
        logger.info(
            "Skipping refresh — last update was recent (%s)",
            meta.last_refresh_utc,
        )
        return meta

    # Download fresh data
    logger.info("Downloading latest JEPX data from %s ...", _SPOT_CSV_URL)
    try:
        response = requests.get(_SPOT_CSV_URL, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException:
        logger.exception("Failed to download JEPX data")
        return RefreshMeta.load(meta_path)

    import io

    new_df = pd.read_csv(io.StringIO(response.text))
    new_rows = len(new_df)

    # Merge with existing data if present
    added = 0
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        old_count = len(existing)

        # Deduplicate by all columns (exact row match)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(keep="first")
        combined.to_csv(csv_path, index=False)

        added = len(combined) - old_count
        total = len(combined)
        logger.info(
            "Merged: %d existing + %d new → %d total (%d added)",
            old_count, new_rows, total, added,
        )
    else:
        new_df.to_csv(csv_path, index=False)
        total = new_rows
        added = new_rows
        logger.info("Saved %d rows (first download)", total)

    # Update metadata
    meta = RefreshMeta(
        last_refresh_utc=datetime.now(UTC).isoformat(),
        total_rows=total,
        new_rows_added=added,
    )
    meta.save(meta_path)

    return meta


def get_data_status(data_dir: Path) -> dict:
    """Get current data status for dashboard/API."""
    meta_path = data_dir / "refresh_meta.json"
    csv_path = data_dir / "jepxSpot.csv"

    meta = RefreshMeta.load(meta_path)

    status = {
        "data_exists": csv_path.exists(),
        "total_rows": meta.total_rows,
        "last_refresh": meta.last_refresh_utc or "never",
        "needs_refresh": should_refresh(data_dir),
    }

    if meta.last_refresh_utc:
        last = datetime.fromisoformat(meta.last_refresh_utc)
        elapsed = (datetime.now(UTC) - last).total_seconds()
        status["seconds_since_refresh"] = int(elapsed)
        status["minutes_since_refresh"] = round(elapsed / 60, 1)

    return status
