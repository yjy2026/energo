"""Feature scaling with save/load support.

Ensures consistent normalization between training and inference.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ScalerState:
    """Persisted scaler parameters."""

    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    method: str = "standard"


class FeatureScaler:
    """Robust feature scaler with serialization.

    Fits on training data and applies the same transformation
    consistently to validation/test/inference data.
    """

    def __init__(self, method: str = "standard") -> None:
        """Initialize scaler.

        Args:
            method: "standard" (z-score) or "robust" (median/IQR).
        """
        if method not in ("standard", "robust"):
            msg = f"Unknown method: {method}. Use 'standard' or 'robust'."
            raise ValueError(msg)
        self._method = method
        self._state: ScalerState | None = None

    @property
    def is_fitted(self) -> bool:
        return self._state is not None

    def fit(self, df: pd.DataFrame, columns: list[str]) -> FeatureScaler:
        """Fit scaler on training data.

        Args:
            df: Training DataFrame.
            columns: Columns to scale.

        Returns:
            Self for chaining.
        """
        state = ScalerState(method=self._method)

        for col in columns:
            if col not in df.columns:
                continue
            values = df[col].dropna()
            if len(values) == 0:
                state.means[col] = 0.0
                state.stds[col] = 1.0
                continue

            if self._method == "standard":
                state.means[col] = float(values.mean())
                state.stds[col] = float(values.std()) or 1.0
            else:  # robust
                state.means[col] = float(values.median())
                q75 = float(values.quantile(0.75))
                q25 = float(values.quantile(0.25))
                state.stds[col] = (q75 - q25) or 1.0

        self._state = state
        logger.info("Fitted scaler (%s) on %d columns", self._method, len(state.means))

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scaling to a DataFrame.

        Args:
            df: DataFrame to transform.

        Returns:
            Transformed DataFrame (copy).
        """
        if self._state is None:
            msg = "Scaler is not fitted. Call fit() first."
            raise RuntimeError(msg)

        result = df.copy()
        for col, mean in self._state.means.items():
            if col in result.columns:
                std = self._state.stds[col]
                result[col] = (result[col] - mean) / std

        return result

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reverse the scaling transformation.

        Args:
            df: Scaled DataFrame.

        Returns:
            Original-scale DataFrame (copy).
        """
        if self._state is None:
            msg = "Scaler is not fitted. Call fit() first."
            raise RuntimeError(msg)

        result = df.copy()
        for col, mean in self._state.means.items():
            if col in result.columns:
                std = self._state.stds[col]
                result[col] = result[col] * std + mean

        return result

    def save(self, path: Path) -> None:
        """Save scaler state to JSON."""
        if self._state is None:
            msg = "Scaler is not fitted."
            raise RuntimeError(msg)

        data = {
            "method": self._state.method,
            "means": self._state.means,
            "stds": self._state.stds,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
        logger.info("Saved scaler to %s", path)

    def load(self, path: Path) -> FeatureScaler:
        """Load scaler state from JSON."""
        data = json.loads(path.read_text())
        self._state = ScalerState(
            method=data["method"],
            means=data["means"],
            stds={k: float(v) for k, v in data["stds"].items()},
        )
        self._method = self._state.method
        logger.info("Loaded scaler from %s (%d columns)", path, len(self._state.means))

        return self
