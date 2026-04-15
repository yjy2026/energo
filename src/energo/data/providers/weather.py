"""Open-Meteo weather data provider.

Free API, no API key required. Provides historical weather data
including temperature, solar radiation, and wind speed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
_REQUEST_TIMEOUT = 30


@dataclass(frozen=True)
class Location:
    """Geographic location with name and coordinates."""

    name: str
    latitude: float
    longitude: float


# Major Japanese cities with coordinates
JAPAN_LOCATIONS: dict[str, Location] = {
    "Tokyo": Location("Tokyo", 35.6895, 139.6917),
    "Osaka": Location("Osaka", 34.6937, 135.5023),
    "Nagoya": Location("Nagoya", 35.1815, 136.9066),
    "Sapporo": Location("Sapporo", 43.0618, 141.3545),
    "Fukuoka": Location("Fukuoka", 33.5904, 130.4017),
    "Sendai": Location("Sendai", 38.2682, 140.8694),
    "Hiroshima": Location("Hiroshima", 34.3853, 132.4553),
    "Takamatsu": Location("Takamatsu", 34.3401, 134.0434),
    "Kanazawa": Location("Kanazawa", 36.5613, 136.6562),
}

# Mapping from JEPX regions to representative cities
REGION_TO_CITY: dict[str, str] = {
    "Hokkaido": "Sapporo",
    "Tohoku": "Sendai",
    "Tokyo": "Tokyo",
    "Chubu": "Nagoya",
    "Hokuriku": "Kanazawa",
    "Kansai": "Osaka",
    "Chugoku": "Hiroshima",
    "Shikoku": "Takamatsu",
    "Kyushu": "Fukuoka",
}


class OpenMeteoProvider:
    """Weather data provider using the Open-Meteo Archive API.

    Free for non-commercial use. No API key required.
    """

    def __init__(self, cache_dir: str | None = None) -> None:
        self._session = requests.Session()
        self._cache_dir = cache_dir

    def fetch_weather(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch historical weather data for a single location.

        Args:
            latitude: Location latitude.
            longitude: Location longitude.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            DataFrame with columns:
                timestamp, temperature_c, solar_radiation_wm2, wind_speed_ms
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,shortwave_radiation,wind_speed_10m",
            "timezone": "Asia/Tokyo",
        }

        logger.info(
            "Fetching weather: (%.2f, %.2f) %s to %s",
            latitude,
            longitude,
            start_date,
            end_date,
        )

        response = self._session.get(
            _ARCHIVE_API_URL,
            params=params,
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        hourly = data.get("hourly", {})
        if not hourly or "time" not in hourly:
            logger.warning("No hourly data returned for (%.2f, %.2f)", latitude, longitude)
            return pd.DataFrame(
                columns=["timestamp", "temperature_c", "solar_radiation_wm2", "wind_speed_ms"]
            )

        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(hourly["time"]),
                "temperature_c": hourly.get("temperature_2m"),
                "solar_radiation_wm2": hourly.get("shortwave_radiation"),
                "wind_speed_ms": hourly.get("wind_speed_10m"),
            }
        )
        df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Tokyo")

        return df

    def fetch_weather_for_region(
        self,
        region: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch weather data for a JEPX region using its representative city.

        Args:
            region: JEPX region name (e.g., "Tokyo", "Kansai").
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            DataFrame with weather data and a 'region' column.
        """
        city_name = REGION_TO_CITY.get(region)
        if city_name is None:
            msg = f"Unknown JEPX region: {region}. Known: {list(REGION_TO_CITY.keys())}"
            raise ValueError(msg)

        location = JAPAN_LOCATIONS[city_name]
        df = self.fetch_weather(
            latitude=location.latitude,
            longitude=location.longitude,
            start_date=start_date,
            end_date=end_date,
        )
        df["region"] = region

        return df

    def fetch_all_regions(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch weather data for all 9 JEPX regions.

        Returns:
            Combined DataFrame with weather data for all regions.
        """
        frames = []
        for region in REGION_TO_CITY:
            df = self.fetch_weather_for_region(region, start_date, end_date)
            frames.append(df)

        return pd.concat(frames, ignore_index=True)
