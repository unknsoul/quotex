"""
Session Filter — v11.1 Strategy-based hour/session confidence scaling.

Instead of blocking hours entirely, every hour gets a data-driven confidence
multiplier so the model can still trade during weaker hours if the signal
is strong enough. This maximizes trade count while preserving accuracy.

Hour multipliers are derived from historical win-rate analysis of 418 outcomes.
"""

import logging
from datetime import datetime, timezone

from config import (
    SESSION_HOURS,
    SESSION_QUALITY_SCORES,
    SESSION_BLOCK_MAP,
    SESSION_CONFIDENCE_MULT,
)

log = logging.getLogger("session_filter")

# ── Per-Hour Confidence Multipliers (data-driven from outcome analysis) ──────
# Instead of blocking bad hours, we scale confidence proportionally.
# Hours with <45% WR get a penalty, hours with >55% WR get a boost.
# This allows strong signals through even in weaker hours.
HOUR_CONFIDENCE_MULT = {
    0: 0.90,   # Asian early — low liquidity
    1: 0.90,
    2: 0.90,
    3: 0.88,
    4: 0.88,   # Pre-London
    5: 1.06,   # 64%+ WR historically — strong
    6: 1.00,
    7: 1.00,
    8: 1.00,   # London open
    9: 0.85,   # 37.5% WR — heavy penalty but NOT blocked
    10: 0.80,  # 23.5% WR — heaviest penalty but NOT blocked
    11: 0.95,
    12: 1.00,
    13: 1.03,  # Overlap start
    14: 1.06,  # 64%+ WR — strong
    15: 1.03,
    16: 1.06,  # 64%+ WR — strong
    17: 1.00,
    18: 0.95,
    19: 0.93,
    20: 0.90,  # Late NY
    21: 0.85,  # Off-hours
    22: 0.85,
    23: 0.85,
}


def map_session(hour: int) -> str:
    """Map a UTC hour (0-23) to the active trading session name."""
    if hour in SESSION_HOURS.get("Overlap", []):
        return "Overlap"
    for name in ("London", "New_York", "Asian"):
        if hour in SESSION_HOURS.get(name, []):
            return name
    return "Off"


def get_session_quality(session: str) -> float:
    """Return a 0-1 quality score for the given session."""
    return SESSION_QUALITY_SCORES.get(session, 0.40)


def get_confidence_multiplier(session: str) -> float:
    """Return the confidence multiplier for a session."""
    return SESSION_CONFIDENCE_MULT.get(session, 1.0)


def get_hour_confidence_multiplier(hour: int) -> float:
    """Return the per-hour confidence multiplier (0.80-1.06)."""
    return HOUR_CONFIDENCE_MULT.get(hour % 24, 0.90)


def get_hour_strategy(hour: int) -> dict:
    """Return strategy info for the given hour.
    
    Returns dict with:
      - multiplier: confidence scaling factor
      - quality: hour quality label
      - require_extra_confirmation: True for weak hours
    """
    mult = get_hour_confidence_multiplier(hour)
    if mult >= 1.03:
        quality = "prime"
        extra = False
    elif mult >= 0.95:
        quality = "normal"
        extra = False
    elif mult >= 0.88:
        quality = "weak"
        extra = True
    else:
        quality = "poor"
        extra = True
    return {
        "multiplier": mult,
        "quality": quality,
        "require_extra_confirmation": extra,
    }


def should_block_session(symbol: str, session: str) -> bool:
    """Return True if trading this symbol is blocked in the current session."""
    blocked = SESSION_BLOCK_MAP.get(symbol, [])
    return session in blocked


def get_current_session() -> str:
    """Return the current trading session based on UTC time."""
    return map_session(datetime.now(timezone.utc).hour)


def filter_symbols_by_session(symbols: list, session: str = None) -> list:
    """Return symbols that are NOT blocked in the given session."""
    if session is None:
        session = get_current_session()
    return [s for s in symbols if not should_block_session(s, session)]


def get_session_summary() -> dict:
    """Return a dict of all sessions with their quality scores and hours."""
    return {
        name: {
            "hours": SESSION_HOURS.get(name, []),
            "quality": SESSION_QUALITY_SCORES.get(name, 0),
        }
        for name in SESSION_HOURS
    }
