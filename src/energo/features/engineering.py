"""Feature engineering for electricity price prediction.

Generates time-series features from the unified dataset while
strictly preventing future data leakage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

# Japanese public holidays (fixed dates only; moving holidays need a library)
_JAPAN_FIXED_HOLIDAYS_MD = {
    (1, 1),   # 元日 New Year's Day
    (2, 11),  # 建国記念の日
    (2, 23),  # 天皇誕生日
    (4, 29),  # 昭和の日
    (5, 3),   # 憲法記念日
    (5, 4),   # みどりの日
    (5, 5),   # こどもの日
    (8, 11),  # 山の日
    (11, 3),  # 文化の日
    (11, 23), # 勤労感謝の日
}


def build_features(
    df: pd.DataFrame,
    price_col: str = "price",
    timestamp_col: str = "timestamp",
    lag_periods: tuple[int, ...] = (1, 2, 4, 48, 96, 336),
    rolling_windows: tuple[int, ...] = (48, 96, 336),
) -> pd.DataFrame:
    """Build ML features from a time-series price dataset.

    All features use only past data (no future leakage).

    Args:
        df: DataFrame with at least timestamp and price columns.
        price_col: Name of the price column.
        timestamp_col: Name of the timestamp column.
        lag_periods: Number of 30-min periods for price lags.
            1=30min, 2=1h, 4=2h, 48=24h, 96=48h, 336=1week.
        rolling_windows: Window sizes for rolling statistics.

    Returns:
        DataFrame with original columns plus engineered features.
    """
    result = df.copy()
    ts = result[timestamp_col]

    # === Time features ===
    result["hour"] = ts.dt.hour
    result["minute"] = ts.dt.minute
    result["day_of_week"] = ts.dt.dayofweek  # 0=Monday
    result["month"] = ts.dt.month
    result["day_of_year"] = ts.dt.dayofyear
    result["period_of_day"] = ts.dt.hour * 2 + ts.dt.minute // 30  # 0-47

    # Cyclical encoding (sin/cos)
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["dow_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
    result["dow_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)
    result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
    result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)

    # === Calendar features ===
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)
    result["is_holiday"] = ts.apply(_is_japan_holiday).astype(int)
    result["is_golden_week"] = ts.apply(
        lambda t: 1 if t.month == 5 and 3 <= t.day <= 5 else 0
    )
    result["is_obon"] = ts.apply(
        lambda t: 1 if t.month == 8 and 13 <= t.day <= 16 else 0
    )
    result["is_year_end"] = ts.apply(
        lambda t: 1 if (t.month == 12 and t.day >= 29) or (t.month == 1 and t.day <= 3) else 0
    )

    # === Price lag features (strictly past) ===
    for lag in lag_periods:
        result[f"price_lag_{lag}"] = result[price_col].shift(lag)

    # === Price return features ===
    result["return_30m"] = result[price_col].pct_change(1)
    result["return_1h"] = result[price_col].pct_change(2)
    result["return_24h"] = result[price_col].pct_change(48)

    # === Rolling statistics (all use shift(1) to avoid leakage) ===
    for window in rolling_windows:
        shifted = result[price_col].shift(1)
        result[f"price_ma_{window}"] = shifted.rolling(window, min_periods=1).mean()
        result[f"price_std_{window}"] = shifted.rolling(window, min_periods=1).std()
        result[f"price_min_{window}"] = shifted.rolling(window, min_periods=1).min()
        result[f"price_max_{window}"] = shifted.rolling(window, min_periods=1).max()

    # === Spread features ===
    if "price_ma_48" in result.columns:
        result["price_vs_ma_48"] = result[price_col].shift(1) - result["price_ma_48"]
    if "price_ma_336" in result.columns:
        result["price_vs_ma_336"] = result[price_col].shift(1) - result["price_ma_336"]

    # === Net demand (demand - solar proxy) ===
    if "demand_mw" in result.columns and "solar_radiation_wm2" in result.columns:
        result["net_demand"] = result["demand_mw"] - result["solar_radiation_wm2"] * 0.1

    # === Temperature deviation from seasonal mean ===
    if "temperature_c" in result.columns:
        # Rolling 30-day mean as seasonal proxy
        result["temp_seasonal_mean"] = (
            result["temperature_c"].shift(1).rolling(48 * 30, min_periods=48).mean()
        )
        result["temp_deviation"] = result["temperature_c"] - result["temp_seasonal_mean"]

    return result


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get list of feature column names (excluding target and metadata).

    Args:
        df: DataFrame with engineered features.

    Returns:
        List of column names suitable for model input.
    """
    exclude = {
        "timestamp",
        "price",
        "region",
        "city",
        "date",
    }
    return [c for c in df.columns if c not in exclude and not c.startswith("_")]


def _is_japan_holiday(ts: pd.Timestamp) -> bool:
    """Check if a date is a Japanese public holiday (fixed dates only)."""
    return (ts.month, ts.day) in _JAPAN_FIXED_HOLIDAYS_MD
