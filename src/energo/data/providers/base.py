"""Base protocol for data providers.

Country-agnostic interface that enables future expansion
to Korea (KPX), US (ERCOT), etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import pandas as pd


class Market(Enum):
    """Supported electricity markets."""

    JEPX = "jepx"
    KPX = "kpx"  # Future: Korea


@dataclass(frozen=True)
class PriceRecord:
    """A single price observation."""

    timestamp: pd.Timestamp
    region: str
    price: float  # Local currency per kWh
    currency: str


class DataProvider(Protocol):
    """Protocol for electricity market data providers.

    All providers must implement these methods to be used
    with the unified pipeline.
    """

    @property
    def market(self) -> Market:
        """Which market this provider serves."""
        ...

    def fetch_spot_prices(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch spot electricity prices.

        Returns DataFrame with columns:
            - timestamp: pd.Timestamp (timezone-aware)
            - region: str
            - price: float (local currency/kWh)
        """
        ...

    def fetch_demand(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch electricity demand data.

        Returns DataFrame with columns:
            - timestamp: pd.Timestamp (timezone-aware)
            - region: str
            - demand_mw: float
        """
        ...


class WeatherProvider(Protocol):
    """Protocol for weather data providers."""

    def fetch_weather(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch weather data for a location.

        Returns DataFrame with columns:
            - timestamp: pd.Timestamp
            - temperature_c: float
            - solar_radiation_wm2: float
            - wind_speed_ms: float
        """
        ...
