"""
Risk Filter v2 — 90th percentile spread check, ATR spike, low volatility.
"""

import logging
import numpy as np
import pandas as pd

from config import (
    MIN_ATR_CLOSE_RATIO, SPREAD_PERCENTILE, ATR_SPIKE_MULTIPLIER,
    ATR_ROLLING_WINDOW, LOG_LEVEL, LOG_FORMAT,
)

log = logging.getLogger("risk_filter")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


def check_low_volatility(atr, close):
    ratio = atr / (close + 1e-10)
    if ratio < MIN_ATR_CLOSE_RATIO:
        return True, f"Low volatility: ATR/close={ratio:.6f} (min={MIN_ATR_CLOSE_RATIO})"
    return False, ""


def check_spread_percentile(current_spread, df):
    """Skip if spread > 90th percentile of recent spreads."""
    if "spread" not in df.columns or current_spread <= 0:
        return False, ""
    p90 = np.percentile(df["spread"].dropna().values[-500:], SPREAD_PERCENTILE)
    if current_spread > p90:
        return True, f"Extreme spread: {current_spread} > P{SPREAD_PERCENTILE}={p90:.1f}"
    return False, ""


def check_atr_spike(df):
    if len(df) < ATR_ROLLING_WINDOW:
        return False, ""
    atr_now = df["atr_14"].iloc[-1]
    atr_mean = df["atr_14"].iloc[-ATR_ROLLING_WINDOW:].mean()
    if atr_now > atr_mean * ATR_SPIKE_MULTIPLIER:
        return True, f"ATR spike: {atr_now:.6f} > mean {atr_mean:.6f} x {ATR_SPIKE_MULTIPLIER}"
    return False, ""


def apply_filters(df, current_spread=0):
    """Run all risk filters. Returns (should_skip, reasons)."""
    reasons = []
    last = df.iloc[-1]

    skip, r = check_low_volatility(last["atr_14"], last["close"])
    if skip:
        reasons.append(r)

    skip, r = check_spread_percentile(current_spread, df)
    if skip:
        reasons.append(r)

    skip, r = check_atr_spike(df)
    if skip:
        reasons.append(r)

    should_skip = len(reasons) > 0
    if should_skip:
        log.warning("Risk filters BLOCKED: %s", reasons)
    else:
        log.info("Risk filters passed.")

    return should_skip, reasons
