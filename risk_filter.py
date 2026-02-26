"""
Risk Filter — guard against bad market conditions before prediction.

Filters:
  1. Low volatility (ATR/close below minimum)
  2. Extreme spread (current spread > N× rolling average)
  3. ATR spike (abnormal volatility burst)
  4. News filter placeholder (manual blackout hours)
"""

import logging
import pandas as pd

from config import (
    MIN_ATR_CLOSE_RATIO,
    MAX_SPREAD_MULTIPLIER,
    ATR_SPIKE_MULTIPLIER,
    ATR_ROLLING_WINDOW,
    LOG_LEVEL, LOG_FORMAT,
)

log = logging.getLogger("risk_filter")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


def check_low_volatility(atr: float, close: float) -> tuple[bool, str]:
    """Skip if ATR/close is below minimum."""
    ratio = atr / (close + 1e-10)
    if ratio < MIN_ATR_CLOSE_RATIO:
        reason = f"Low volatility: ATR/close={ratio:.6f} (min={MIN_ATR_CLOSE_RATIO})"
        return True, reason
    return False, ""


def check_spread(current_spread: float, avg_spread: float) -> tuple[bool, str]:
    """Skip if current spread is abnormally high."""
    if avg_spread <= 0:
        return False, ""
    ratio = current_spread / avg_spread
    if ratio > MAX_SPREAD_MULTIPLIER:
        reason = (f"Extreme spread: {current_spread} vs avg {avg_spread:.1f} "
                  f"({ratio:.1f}x, max={MAX_SPREAD_MULTIPLIER}x)")
        return True, reason
    return False, ""


def check_atr_spike(df: pd.DataFrame) -> tuple[bool, str]:
    """Skip if ATR is spiking far above its rolling mean (crash/flash event)."""
    if len(df) < ATR_ROLLING_WINDOW:
        return False, ""
    atr_now = df["atr_14"].iloc[-1]
    atr_mean = df["atr_14"].iloc[-ATR_ROLLING_WINDOW:].mean()
    if atr_now > atr_mean * ATR_SPIKE_MULTIPLIER:
        reason = (f"ATR spike: {atr_now:.6f} > mean {atr_mean:.6f} × "
                  f"{ATR_SPIKE_MULTIPLIER}")
        return True, reason
    return False, ""


def check_news_placeholder() -> tuple[bool, str]:
    """
    Placeholder for future news-based filtering.
    Can be wired to an external calendar API.
    """
    # Always pass for now
    return False, ""


def apply_filters(
    df: pd.DataFrame,
    current_spread: float = 0,
) -> tuple[bool, list[str]]:
    """
    Run all risk filters.

    Args:
        df: Feature-engineered DataFrame (needs atr_14, close, spread cols).
        current_spread: Live spread from MT5 symbol info.

    Returns:
        (should_skip, list_of_reasons)
    """
    reasons = []
    last = df.iloc[-1]

    skip, r = check_low_volatility(last["atr_14"], last["close"])
    if skip:
        reasons.append(r)

    avg_spread = df["spread"].mean() if "spread" in df.columns else 0
    skip, r = check_spread(current_spread, avg_spread)
    if skip:
        reasons.append(r)

    skip, r = check_atr_spike(df)
    if skip:
        reasons.append(r)

    skip, r = check_news_placeholder()
    if skip:
        reasons.append(r)

    should_skip = len(reasons) > 0
    if should_skip:
        log.warning("Risk filters BLOCKED: %s", reasons)
    else:
        log.info("Risk filters passed.")

    return should_skip, reasons
