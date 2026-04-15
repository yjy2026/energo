"""JEPX (Japan Electric Power Exchange) data provider.

Fetches spot prices, demand, and temperature data from japanesepower.org.
Data is provided in 30-minute intervals across 9 Japanese regions.
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

import pandas as pd
import requests

from energo.data.providers.base import DataProvider, Market

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# japanesepower.org CSV endpoints
_BASE_URL = "https://japanesepower.org"
_SPOT_CSV_URL = f"{_BASE_URL}/jepxSpot.csv"
_DEMAND_CSV_URL = f"{_BASE_URL}/demand.csv"
_WEATHER_CSV_URL = f"{_BASE_URL}/weatherData.csv"

# JEPX regions
JEPX_REGIONS = [
    "Hokkaido",
    "Tohoku",
    "Tokyo",
    "Chubu",
    "Hokuriku",
    "Kansai",
    "Chugoku",
    "Shikoku",
    "Kyushu",
]

_REQUEST_TIMEOUT = 60


class JEPXProvider:
    """Data provider for the Japanese electricity market (JEPX).

    Implements the DataProvider protocol using japanesepower.org as the data source.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def market(self) -> Market:
        return Market.JEPX

    def fetch_spot_prices(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch JEPX spot prices from japanesepower.org.

        Returns long-format DataFrame with columns:
            timestamp, region, price
        """
        raw_df = self._fetch_csv(_SPOT_CSV_URL, "jepxSpot.csv")
        df = self._parse_spot_prices(raw_df)

        return self._filter_dates(df, start_date, end_date)

    def fetch_demand(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch electricity demand data."""
        raw_df = self._fetch_csv(_DEMAND_CSV_URL, "demand.csv")
        df = self._parse_demand(raw_df)

        return self._filter_dates(df, start_date, end_date)

    def fetch_temperature(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch temperature data for major Japanese cities."""
        raw_df = self._fetch_csv(_WEATHER_CSV_URL, "weatherData.csv")
        df = self._parse_temperature(raw_df)

        return self._filter_dates(df, start_date, end_date)

    def _fetch_csv(self, url: str, cache_name: str) -> pd.DataFrame:
        """Download CSV or load from cache."""
        if self._cache_dir:
            cache_path = self._cache_dir / cache_name
            if cache_path.exists():
                logger.info("Loading cached data: %s", cache_path)
                return pd.read_csv(cache_path)

        logger.info("Downloading: %s", url)
        response = requests.get(url, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()

        df = pd.read_csv(io.StringIO(response.text))

        if self._cache_dir:
            cache_path = self._cache_dir / cache_name
            df.to_csv(cache_path, index=False)
            logger.info("Cached to: %s", cache_path)

        return df

    def _parse_spot_prices(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Parse raw spot price CSV into standardized long format.

        JEPX spot CSV typically has columns:
        Date, Period(1-48), SystemPrice, Hokkaido, Tohoku, Tokyo, ...
        """
        df = raw_df.copy()

        # Identify date and period columns
        date_col = self._find_column(df, ["Date", "date", "DATE", "日付"])
        period_col = self._find_column(df, ["Period", "period", "PERIOD", "コマ", "時間帯"])

        if date_col is None or period_col is None:
            # Try to infer from first two columns
            date_col = df.columns[0]
            period_col = df.columns[1]

        # Build timestamp from date + period (30-min intervals)
        df["timestamp"] = pd.to_datetime(df[date_col].astype(str))
        df["timestamp"] = df["timestamp"] + pd.to_timedelta(
            (df[period_col].astype(int) - 1) * 30, unit="min"
        )
        df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Tokyo")

        # Identify price columns (regions + system price)
        price_columns = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if "system" in col_lower:
                price_columns[col] = "System"
            else:
                for region in JEPX_REGIONS:
                    if region.lower() in col_lower:
                        price_columns[col] = region
                        break

        if not price_columns:
            # Fallback: use all numeric columns after date/period
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if period_col in numeric_cols:
                numeric_cols.remove(period_col)
            for i, col in enumerate(numeric_cols):
                if i == 0:
                    price_columns[col] = "System"
                elif i - 1 < len(JEPX_REGIONS):
                    price_columns[col] = JEPX_REGIONS[i - 1]

        # Melt to long format
        records = []
        for col, region in price_columns.items():
            temp = df[["timestamp"]].copy()
            temp["region"] = region
            temp["price"] = pd.to_numeric(df[col], errors="coerce")
            records.append(temp)

        result = pd.concat(records, ignore_index=True)
        result = result.dropna(subset=["price"])

        return result.sort_values("timestamp").reset_index(drop=True)

    def _parse_demand(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Parse demand CSV into standardized format."""
        df = raw_df.copy()

        date_col = df.columns[0]
        df["timestamp"] = pd.to_datetime(df[date_col].astype(str))
        df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Tokyo")

        # Melt region columns to long format
        region_cols = [c for c in df.columns if c != date_col and c != "timestamp"]
        records = []
        for col in region_cols:
            temp = df[["timestamp"]].copy()
            # Try to match column name to known region
            region_name = col.strip()
            for region in JEPX_REGIONS:
                if region.lower() in col.lower():
                    region_name = region
                    break
            temp["region"] = region_name
            temp["demand_mw"] = pd.to_numeric(df[col], errors="coerce")
            records.append(temp)

        result = pd.concat(records, ignore_index=True)
        result = result.dropna(subset=["demand_mw"])

        return result.sort_values("timestamp").reset_index(drop=True)

    def _parse_temperature(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Parse temperature CSV into standardized format."""
        df = raw_df.copy()

        date_col = df.columns[0]
        df["timestamp"] = pd.to_datetime(df[date_col].astype(str))
        df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Tokyo")

        city_cols = [c for c in df.columns if c != date_col and c != "timestamp"]
        records = []
        for col in city_cols:
            temp = df[["timestamp"]].copy()
            temp["city"] = col.strip()
            temp["temperature_c"] = pd.to_numeric(df[col], errors="coerce")
            records.append(temp)

        result = pd.concat(records, ignore_index=True)
        result = result.dropna(subset=["temperature_c"])

        return result.sort_values("timestamp").reset_index(drop=True)

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
        """Find a column matching one of the candidate names."""
        for candidate in candidates:
            for col in df.columns:
                if col.strip().lower() == candidate.lower():
                    return col
        return None

    @staticmethod
    def _filter_dates(
        df: pd.DataFrame,
        start_date: str | None,
        end_date: str | None,
    ) -> pd.DataFrame:
        """Filter DataFrame by date range."""
        if start_date:
            start = pd.Timestamp(start_date, tz="Asia/Tokyo")
            df = df[df["timestamp"] >= start]
        if end_date:
            end = pd.Timestamp(end_date, tz="Asia/Tokyo")
            df = df[df["timestamp"] <= end]
        return df.reset_index(drop=True)


# Verify protocol compliance at import time
_: DataProvider = JEPXProvider()  # type: ignore[assignment]
