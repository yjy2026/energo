"""Unified data pipeline.

Merges electricity price, demand, and weather data into a single
ML-ready DataFrame with proper time alignment and train/val/test splits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from energo.data.providers.jepx import JEPXProvider
    from energo.data.providers.weather import OpenMeteoProvider

logger = logging.getLogger(__name__)


@dataclass
class SplitDataset:
    """Time-series aware train/validation/test split."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
        }


class DataPipeline:
    """Unified pipeline that combines price, demand, and weather data.

    Produces a wide-format DataFrame per region suitable for ML feature
    engineering.
    """

    def __init__(
        self,
        jepx_provider: JEPXProvider,
        weather_provider: OpenMeteoProvider,
    ) -> None:
        self._jepx = jepx_provider
        self._weather = weather_provider

    def build_dataset(
        self,
        region: str = "System",
        start_date: str | None = None,
        end_date: str | None = None,
        include_weather: bool = True,
    ) -> pd.DataFrame:
        """Build a unified dataset for a given region.

        Args:
            region: JEPX region name or "System" for system price.
            start_date: Start date filter (YYYY-MM-DD).
            end_date: End date filter (YYYY-MM-DD).
            include_weather: Whether to merge weather data.

        Returns:
            Wide-format DataFrame indexed by timestamp with columns:
                price, demand_mw, temperature_c, solar_radiation_wm2, wind_speed_ms
        """
        # Fetch spot prices for the target region
        prices = self._jepx.fetch_spot_prices(start_date, end_date)
        prices = prices[prices["region"] == region][["timestamp", "price"]].copy()
        prices = prices.set_index("timestamp")

        if prices.empty:
            logger.warning("No price data for region=%s", region)
            return pd.DataFrame()

        # Fetch demand (use Tokyo as proxy for System)
        demand_region = "Tokyo" if region == "System" else region
        demand = self._jepx.fetch_demand(start_date, end_date)
        demand = demand[demand["region"].str.contains(demand_region, case=False, na=False)]
        if not demand.empty:
            demand = demand[["timestamp", "demand_mw"]].set_index("timestamp")
            # Resample demand to 30-min to match prices (forward fill)
            demand = demand.resample("30min").ffill()
            prices = prices.join(demand, how="left")

        # Fetch weather data
        if include_weather and region != "System":
            actual_start = prices.index.min().strftime("%Y-%m-%d")
            actual_end = prices.index.max().strftime("%Y-%m-%d")
            weather = self._weather.fetch_weather_for_region(
                region, actual_start, actual_end
            )
            if not weather.empty:
                weather = weather.set_index("timestamp")
                weather = weather[
                    ["temperature_c", "solar_radiation_wm2", "wind_speed_ms"]
                ]
                # Weather is hourly; resample to 30-min
                weather = weather.resample("30min").interpolate(method="linear")
                prices = prices.join(weather, how="left")
        elif include_weather and region == "System":
            # For System price, use Tokyo weather as proxy
            actual_start = prices.index.min().strftime("%Y-%m-%d")
            actual_end = prices.index.max().strftime("%Y-%m-%d")
            weather = self._weather.fetch_weather_for_region(
                "Tokyo", actual_start, actual_end
            )
            if not weather.empty:
                weather = weather.set_index("timestamp")
                weather = weather[
                    ["temperature_c", "solar_radiation_wm2", "wind_speed_ms"]
                ]
                weather = weather.resample("30min").interpolate(method="linear")
                prices = prices.join(weather, how="left")

        result = prices.reset_index()
        result = result.sort_values("timestamp").reset_index(drop=True)
        result = result.dropna(subset=["price"])

        logger.info(
            "Built dataset: region=%s, rows=%d, date_range=%s to %s",
            region,
            len(result),
            result["timestamp"].min(),
            result["timestamp"].max(),
        )

        return result

    @staticmethod
    def split_temporal(
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> SplitDataset:
        """Split dataset chronologically (no data leakage).

        Args:
            df: DataFrame sorted by timestamp.
            train_ratio: Fraction for training set.
            val_ratio: Fraction for validation set.
                       Test gets the remainder.

        Returns:
            SplitDataset with train/val/test DataFrames.
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        split = SplitDataset(
            train=df.iloc[:train_end].copy().reset_index(drop=True),
            val=df.iloc[train_end:val_end].copy().reset_index(drop=True),
            test=df.iloc[val_end:].copy().reset_index(drop=True),
        )

        logger.info("Split sizes: %s", split.sizes)

        return split
