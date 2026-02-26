"""
Regime Detection — classify current market state.

Single function interface:
    detect_regime(df_row_or_slice) → "Trending" | "Ranging" | "High_Volatility" | "Low_Volatility"

Priority:
    1. High_Volatility  (ATR spike)
    2. Low_Volatility   (ATR drop)
    3. Trending         (Strong ADX + slope)
    4. Ranging          (default)
"""

import logging

import pandas as pd

from config import (
    ADX_TRENDING_THRESHOLD,
    ADX_RANGING_THRESHOLD,
    ATR_HIGH_VOL_MULTIPLIER,
    ATR_LOW_VOL_MULTIPLIER,
    ATR_ROLLING_WINDOW,
    EMA_SLOPE_WINDOW,
    REGIME_THRESHOLDS,
    BASE_THRESHOLD,
    LOG_LEVEL, LOG_FORMAT,
)

log = logging.getLogger("regime_detection")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

REGIMES = ("Trending", "Ranging", "High_Volatility", "Low_Volatility")


def detect_regime(df: pd.DataFrame) -> str:
    """
    Classify market regime using the tail of `df`.

    Required columns in df: atr_14, adx, ema_50, bb_width

    Returns one of: "Trending", "Ranging", "High_Volatility", "Low_Volatility"
    """
    if len(df) < ATR_ROLLING_WINDOW:
        log.warning("Not enough rows (%d) for regime detection, defaulting to Ranging.", len(df))
        return "Ranging"

    latest = df.iloc[-1]
    atr_now = latest["atr_14"]
    adx_now = latest["adx"]

    # ── 1) Volatility regime (highest priority) ─────────────────────────
    atr_mean = df["atr_14"].iloc[-ATR_ROLLING_WINDOW:].mean()

    if atr_now > atr_mean * ATR_HIGH_VOL_MULTIPLIER:
        log.info("Regime: High_Volatility (ATR=%.6f, mean=%.6f)", atr_now, atr_mean)
        return "High_Volatility"

    if atr_now < atr_mean * ATR_LOW_VOL_MULTIPLIER:
        log.info("Regime: Low_Volatility (ATR=%.6f, mean=%.6f)", atr_now, atr_mean)
        return "Low_Volatility"

    # ── 2) Trend vs Range ────────────────────────────────────────────────
    ema_slice = df["ema_50"].iloc[-EMA_SLOPE_WINDOW:]
    if len(ema_slice) >= 2:
        slope = (ema_slice.iloc[-1] - ema_slice.iloc[0]) / (ema_slice.iloc[0] + 1e-10)
    else:
        slope = 0.0

    if adx_now > ADX_TRENDING_THRESHOLD and abs(slope) > 0.0001:
        log.info("Regime: Trending (ADX=%.2f, slope=%.6f)", adx_now, slope)
        return "Trending"

    if adx_now < ADX_RANGING_THRESHOLD:
        log.info("Regime: Ranging (ADX=%.2f)", adx_now)
        return "Ranging"

    # ── 3) Fallback ──────────────────────────────────────────────────────
    log.info("Regime: Ranging (fallback, ADX=%.2f)", adx_now)
    return "Ranging"


def get_regime_threshold(regime: str) -> float:
    """Return probability threshold for the given regime."""
    return REGIME_THRESHOLDS.get(regime, BASE_THRESHOLD)


def get_volatility_status(df: pd.DataFrame) -> str:
    """Return "High", "Low", or "Normal"."""
    if len(df) < ATR_ROLLING_WINDOW:
        return "Normal"
    atr_now = df["atr_14"].iloc[-1]
    atr_mean = df["atr_14"].iloc[-ATR_ROLLING_WINDOW:].mean()
    if atr_now > atr_mean * ATR_HIGH_VOL_MULTIPLIER:
        return "High"
    if atr_now < atr_mean * ATR_LOW_VOL_MULTIPLIER:
        return "Low"
    return "Normal"
