"""
Regime Detection v2 — returns dual thresholds (primary + meta).
"""

import logging
import numpy as np
import pandas as pd

from config import (
    ADX_TRENDING_THRESHOLD, ADX_RANGING_THRESHOLD,
    ATR_HIGH_VOL_MULTIPLIER, ATR_LOW_VOL_MULTIPLIER,
    ATR_ROLLING_WINDOW, EMA_SLOPE_WINDOW,
    REGIME_THRESHOLDS, PRIMARY_BASE_THRESHOLD, META_BASE_THRESHOLD,
    LOG_LEVEL, LOG_FORMAT,
)

log = logging.getLogger("regime_detection")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

REGIMES = ("Trending", "Ranging", "High_Volatility", "Low_Volatility")


def detect_regime(df: pd.DataFrame) -> str:
    """Classify market regime from tail of DataFrame."""
    if len(df) < ATR_ROLLING_WINDOW:
        return "Ranging"

    latest = df.iloc[-1]
    atr_now = latest["atr_14"]
    adx_now = latest["adx"]
    atr_mean = df["atr_14"].iloc[-ATR_ROLLING_WINDOW:].mean()

    if atr_now > atr_mean * ATR_HIGH_VOL_MULTIPLIER:
        return "High_Volatility"
    if atr_now < atr_mean * ATR_LOW_VOL_MULTIPLIER:
        return "Low_Volatility"

    ema_slice = df["ema_50"].iloc[-EMA_SLOPE_WINDOW:]
    slope = 0.0
    if len(ema_slice) >= 2:
        slope = (ema_slice.iloc[-1] - ema_slice.iloc[0]) / (ema_slice.iloc[0] + 1e-10)

    if adx_now > ADX_TRENDING_THRESHOLD and abs(slope) > 0.0001:
        return "Trending"
    if adx_now < ADX_RANGING_THRESHOLD:
        return "Ranging"

    return "Ranging"


def detect_regime_series(df: pd.DataFrame, lookback: int = 200) -> pd.Series:
    """Detect regime for every row (used by meta model). Returns Series of regime strings."""
    regimes = []
    for i in range(len(df)):
        start = max(0, i - lookback + 1)
        window = df.iloc[start:i + 1]
        if len(window) < 50 or "atr_14" not in window.columns or "adx" not in window.columns:
            regimes.append("Ranging")
        else:
            regimes.append(detect_regime(window))
    return pd.Series(regimes, index=df.index)


def get_regime_thresholds(regime: str) -> dict[str, float]:
    """Return dual thresholds: {'primary': float, 'meta': float}."""
    return REGIME_THRESHOLDS.get(regime, {
        "primary": PRIMARY_BASE_THRESHOLD,
        "meta": META_BASE_THRESHOLD,
    })


def get_volatility_status(df: pd.DataFrame) -> str:
    if len(df) < ATR_ROLLING_WINDOW:
        return "Normal"
    atr_now = df["atr_14"].iloc[-1]
    atr_mean = df["atr_14"].iloc[-ATR_ROLLING_WINDOW:].mean()
    if atr_now > atr_mean * ATR_HIGH_VOL_MULTIPLIER:
        return "High"
    if atr_now < atr_mean * ATR_LOW_VOL_MULTIPLIER:
        return "Low"
    return "Normal"


def regime_stability_score(df: pd.DataFrame) -> float:
    """
    Detect regime transition instability by comparing regimes at multiple lookbacks.
    
    Returns 0.0 (unstable / transitioning) to 1.0 (stable / all lookbacks agree).
    When lookbacks disagree, the market is likely switching regimes — accuracy drops.
    """
    if len(df) < 200:
        return 0.5  # not enough data to assess

    lookbacks = [50, 100, 200]
    regimes = []
    for lb in lookbacks:
        start = max(0, len(df) - lb)
        window = df.iloc[start:]
        if len(window) >= 50 and "atr_14" in window.columns and "adx" in window.columns:
            regimes.append(detect_regime(window))
        else:
            regimes.append("Ranging")

    # Score: 1.0 if all agree, 0.67 if 2/3 agree, 0.33 if all different
    agreements = sum(1 for r in regimes if r == regimes[0])
    if agreements == 3:
        return 1.0
    elif agreements == 2 or regimes[1] == regimes[2]:
        return 0.67
    else:
        return 0.33


def get_session(hour: int) -> str:
    """Identify trading session from UTC hour."""
    from config import SESSION_ASIA, SESSION_LONDON, SESSION_NEW_YORK
    # Check overlap first (London + NY)
    if SESSION_LONDON[0] <= hour < SESSION_NEW_YORK[1] and SESSION_NEW_YORK[0] <= hour:
        return "Overlap"
    if SESSION_LONDON[0] <= hour < SESSION_LONDON[1]:
        return "London"
    if SESSION_NEW_YORK[0] <= hour < SESSION_NEW_YORK[1]:
        return "New_York"
    if SESSION_ASIA[0] <= hour < SESSION_ASIA[1]:
        return "Asian"
    return "Off"
