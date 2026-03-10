# -*- coding: utf-8 -*-
"""
Staleness Tracker — v11 Crossover Signal Decay.

Tracks bar age of crossover signals (MACD, SMA) and applies decay multiplier.
Fresh crossovers get full score; stale crossovers are penalized.

On M5, crossover signals must be very recent — old crosses are noise.
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger("staleness_tracker")

# ── Decay config ────────────────────────────────────────────────────────────
MAX_FRESH_BARS = 1      # crossover on current or previous bar = fresh
STALE_AFTER_BARS = 3    # beyond 3 bars, crossover is noise
DECAY_PER_BAR = 0.30    # 30% score loss per bar of staleness


def _find_crossover_age(fast: np.ndarray, slow: np.ndarray, lookback: int = 10) -> int:
    """
    Find how many bars ago a crossover occurred (fast crossing slow).
    Returns 0 if crossover on latest bar, 1 if previous bar, etc.
    Returns -1 if no crossover found within lookback.
    """
    if len(fast) < lookback + 1 or len(slow) < lookback + 1:
        return -1

    for bars_ago in range(lookback):
        idx = -(bars_ago + 1)
        prev_idx = idx - 1
        if abs(prev_idx) > len(fast):
            break
        # Cross above or cross below
        if (fast[prev_idx] <= slow[prev_idx] and fast[idx] > slow[idx]) or \
           (fast[prev_idx] >= slow[prev_idx] and fast[idx] < slow[idx]):
            return bars_ago
    return -1


def compute_staleness_multiplier(bars_ago: int) -> float:
    """
    Compute decay multiplier based on crossover age.

    Returns:
        1.0 for fresh crossover (0-1 bars ago)
        Decayed for stale crossovers
        0.0 for very stale (>= STALE_AFTER_BARS)
    """
    if bars_ago < 0:
        return 0.0  # no crossover found
    if bars_ago <= MAX_FRESH_BARS:
        return 1.0  # fresh — full score
    excess = bars_ago - MAX_FRESH_BARS
    multiplier = max(0.0, 1.0 - excess * DECAY_PER_BAR)
    return round(multiplier, 2)


def track_crossover_staleness(df: pd.DataFrame) -> dict:
    """
    Compute staleness of MACD and SMA crossovers.

    Returns:
        dict with:
            macd_cross_age: int (-1 if none)
            macd_staleness_mult: float (0.0-1.0)
            sma_cross_age: int (-1 if none)
            sma_staleness_mult: float (0.0-1.0)
            overall_freshness: float (average of available multipliers)
            is_fresh: bool (at least one crossover within MAX_FRESH_BARS)
    """
    result = {
        "macd_cross_age": -1,
        "macd_staleness_mult": 0.0,
        "sma_cross_age": -1,
        "sma_staleness_mult": 0.0,
        "overall_freshness": 0.0,
        "is_fresh": False,
    }

    if df is None or len(df) < 30:
        return result

    close = df["close"].values

    # MACD crossover staleness
    try:
        ema12 = pd.Series(close).ewm(span=12, min_periods=6).mean().values
        ema26 = pd.Series(close).ewm(span=26, min_periods=13).mean().values
        macd_line = ema12 - ema26
        signal_line = pd.Series(macd_line).ewm(span=9, min_periods=5).mean().values

        macd_age = _find_crossover_age(macd_line, signal_line, lookback=10)
        result["macd_cross_age"] = macd_age
        result["macd_staleness_mult"] = compute_staleness_multiplier(macd_age)
    except Exception as e:
        log.debug("MACD staleness error: %s", e)

    # SMA crossover staleness
    try:
        sma20 = pd.Series(close).rolling(20, min_periods=10).mean().values
        sma50 = pd.Series(close).rolling(50, min_periods=25).mean().values

        sma_age = _find_crossover_age(sma20, sma50, lookback=10)
        result["sma_cross_age"] = sma_age
        result["sma_staleness_mult"] = compute_staleness_multiplier(sma_age)
    except Exception as e:
        log.debug("SMA staleness error: %s", e)

    # Overall freshness
    mults = []
    if result["macd_cross_age"] >= 0:
        mults.append(result["macd_staleness_mult"])
    if result["sma_cross_age"] >= 0:
        mults.append(result["sma_staleness_mult"])

    if mults:
        result["overall_freshness"] = round(np.mean(mults), 2)
        result["is_fresh"] = any(m >= 0.7 for m in mults)
    else:
        result["overall_freshness"] = 0.0
        result["is_fresh"] = False

    return result
