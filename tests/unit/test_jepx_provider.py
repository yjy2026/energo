"""Tests for JEPX data provider."""

from __future__ import annotations

import pandas as pd

from energo.data.providers.jepx import JEPX_REGIONS, JEPXProvider


class TestJEPXProvider:
    """Tests for the JEPX data provider."""

    def test_market_property(self) -> None:
        provider = JEPXProvider()
        assert provider.market.value == "jepx"

    def test_parse_spot_prices_standard_format(self) -> None:
        """Test parsing of standard JEPX spot price CSV format."""
        raw_df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-01", "2024-01-01"],
                "Period": [1, 2, 3],
                "SystemPrice": [10.5, 11.2, 12.0],
                "Tokyo": [10.8, 11.5, 12.3],
                "Kansai": [10.3, 11.0, 11.8],
            }
        )

        provider = JEPXProvider()
        result = provider._parse_spot_prices(raw_df)

        assert "timestamp" in result.columns
        assert "region" in result.columns
        assert "price" in result.columns
        assert len(result) > 0

        # Check regions are detected
        regions = result["region"].unique()
        assert "System" in regions or "Tokyo" in regions

    def test_parse_spot_prices_timestamps(self) -> None:
        """Verify 30-min intervals are correctly computed."""
        raw_df = pd.DataFrame(
            {
                "Date": ["2024-01-01"] * 3,
                "Period": [1, 2, 3],
                "SystemPrice": [10.0, 11.0, 12.0],
            }
        )

        provider = JEPXProvider()
        result = provider._parse_spot_prices(raw_df)

        timestamps = result["timestamp"].unique()
        assert len(timestamps) == 3

        # Period 1 = 00:00, Period 2 = 00:30, Period 3 = 01:00
        hours = sorted([t.hour for t in timestamps])
        minutes = sorted([t.minute for t in timestamps])
        assert 0 in hours
        assert 30 in minutes

    def test_filter_dates(self) -> None:
        """Test date filtering."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2024-01-01", "2024-01-15", "2024-02-01"]
                ).tz_localize("Asia/Tokyo"),
                "price": [10.0, 11.0, 12.0],
            }
        )

        result = JEPXProvider._filter_dates(df, "2024-01-10", "2024-01-20")
        assert len(result) == 1
        assert result.iloc[0]["price"] == 11.0

    def test_filter_dates_no_filter(self) -> None:
        """Test that no filter returns all data."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2024-01-01", "2024-01-15"]
                ).tz_localize("Asia/Tokyo"),
                "price": [10.0, 11.0],
            }
        )

        result = JEPXProvider._filter_dates(df, None, None)
        assert len(result) == 2

    def test_find_column(self) -> None:
        """Test column name matching."""
        df = pd.DataFrame({"Date": [1], "PERIOD": [1], "Price": [10.0]})

        assert JEPXProvider._find_column(df, ["date", "Date"]) == "Date"
        assert JEPXProvider._find_column(df, ["period", "Period"]) == "PERIOD"
        assert JEPXProvider._find_column(df, ["nonexistent"]) is None

    def test_regions_list(self) -> None:
        """Verify all 9 JEPX regions are defined."""
        assert len(JEPX_REGIONS) == 9
        assert "Tokyo" in JEPX_REGIONS
        assert "Kansai" in JEPX_REGIONS
        assert "Hokkaido" in JEPX_REGIONS

    def test_parse_demand(self) -> None:
        """Test demand data parsing."""
        raw_df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02"],
                "Tokyo": [30000, 31000],
                "Kansai": [20000, 21000],
            }
        )

        provider = JEPXProvider()
        result = provider._parse_demand(raw_df)

        assert "timestamp" in result.columns
        assert "region" in result.columns
        assert "demand_mw" in result.columns
        assert len(result) == 4  # 2 dates × 2 regions

    def test_parse_temperature(self) -> None:
        """Test temperature data parsing."""
        raw_df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02"],
                "Tokyo": [5.0, 6.0],
                "Osaka": [7.0, 8.0],
            }
        )

        provider = JEPXProvider()
        result = provider._parse_temperature(raw_df)

        assert "timestamp" in result.columns
        assert "city" in result.columns
        assert "temperature_c" in result.columns
        assert len(result) == 4
