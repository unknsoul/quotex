"""
Risk Filter v3 — generates warnings only, NEVER blocks predictions.
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


def check_warnings(df, current_spread=0):
    """
    Check risk conditions. Returns list of warning strings.
    These are INFORMATIONAL ONLY — prediction is always generated.
    """
    warnings = []
    last = df.iloc[-1]

    # Low volatility warning
    atr = last["atr_14"]
    close = last["close"]
    ratio = atr / (close + 1e-10)
    if ratio < MIN_ATR_CLOSE_RATIO:
        warnings.append(f"Low volatility: ATR/close={ratio:.6f} (min={MIN_ATR_CLOSE_RATIO})")

    # High spread warning
    if "spread" in df.columns and current_spread > 0:
        p90 = np.percentile(df["spread"].dropna().values[-500:], SPREAD_PERCENTILE)
        if current_spread > p90:
            warnings.append(f"High spread: {current_spread} > P{SPREAD_PERCENTILE}={p90:.1f}")

    # ATR spike warning
    if len(df) >= ATR_ROLLING_WINDOW:
        atr_mean = df["atr_14"].iloc[-ATR_ROLLING_WINDOW:].mean()
        if atr > atr_mean * ATR_SPIKE_MULTIPLIER:
            warnings.append(f"ATR spike: {atr:.6f} > mean {atr_mean:.6f} x {ATR_SPIKE_MULTIPLIER}")

    if warnings:
        log.warning("Risk warnings: %s", warnings)
    else:
        log.info("No risk warnings.")

    return warnings
