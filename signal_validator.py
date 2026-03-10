"""
Signal Validator — Pre-dispatch validation for every signal.

Checks run before a signal is sent to Telegram:
  1. Minimum confidence (72%+)
  2. Duplicate guard (same symbol+direction within window)
  3. Rate limit (max signals per hour)
  4. Multi-timeframe alignment (optional)
  5. Session block check
  6. Circuit breaker check
"""

import logging
from datetime import datetime, timezone

from config import (
    SIGNAL_MIN_CONFIDENCE,
    SIGNAL_DUPLICATE_WINDOW_SEC,
    SIGNAL_MAX_PER_HOUR,
    SIGNAL_REQUIRE_MTF_ALIGNMENT,
)
from signal_db import SignalDB
from session_filter import should_block_session

log = logging.getLogger("signal_validator")

_signal_db = None


def _get_db() -> SignalDB:
    global _signal_db
    if _signal_db is None:
        _signal_db = SignalDB()
    return _signal_db


def validate_signal(pred: dict, symbol: str, circuit_breaker=None) -> tuple:
    """
    Validate a prediction before dispatch.

    Parameters
    ----------
    pred : dict
        Prediction dict from predict_engine.predict().
    symbol : str
        Instrument symbol e.g. "EURUSD".
    circuit_breaker : CircuitBreaker or None
        If provided, checks if the breaker is tripped.

    Returns
    -------
    (is_valid: bool, reasons: list[str])
        is_valid=True means the signal should be dispatched.
        reasons lists every validation failure.
    """
    reasons = []

    # 1. Trade must not be HOLD
    trade = pred.get("suggested_trade", "HOLD")
    if trade == "HOLD":
        reasons.append("HOLD signal")
        return False, reasons

    direction = pred.get("suggested_direction", "HOLD")

    # 2. Minimum confidence check
    confidence = pred.get("final_confidence_percent", 0)
    if confidence < SIGNAL_MIN_CONFIDENCE:
        reasons.append(f"Low confidence ({confidence:.0f}% < {SIGNAL_MIN_CONFIDENCE:.0f}%)")

    # 3. Duplicate guard
    db = _get_db()
    if db.get_duplicate_check(symbol, direction, SIGNAL_DUPLICATE_WINDOW_SEC):
        reasons.append(f"Duplicate ({symbol} {direction} within {SIGNAL_DUPLICATE_WINDOW_SEC}s)")

    # 4. Rate limit
    hourly_count = db.count_signals_last_hour()
    if hourly_count >= SIGNAL_MAX_PER_HOUR:
        reasons.append(f"Rate limit ({hourly_count}/{SIGNAL_MAX_PER_HOUR} per hour)")

    # 5. Multi-timeframe alignment (if enabled)
    if SIGNAL_REQUIRE_MTF_ALIGNMENT:
        h1_trend = pred.get("_meta_row", {}).get("h1_trend_direction", None)
        if h1_trend is not None and h1_trend != 0:
            # h1_trend>0 = bullish, <0 = bearish
            trade_dir = 1 if direction == "UP" else -1
            if (h1_trend > 0 and trade_dir < 0) or (h1_trend < 0 and trade_dir > 0):
                reasons.append("MTF conflict (H1 opposes direction)")

    # 6. Session block — DISABLED v13 (trade all sessions)
    # session = pred.get("session", "Off")
    # if should_block_session(symbol, session):
    #     reasons.append(f"Session blocked ({symbol} in {session})")

    # 7. Circuit breaker
    if circuit_breaker is not None:
        try:
            if not circuit_breaker.can_trade():
                reasons.append("Circuit breaker tripped")
        except Exception:
            pass

    is_valid = len(reasons) == 0
    if not is_valid:
        log.info("Signal REJECTED (%s %s): %s", symbol, direction, "; ".join(reasons))
    return is_valid, reasons
