"""
Session Filter — Dedicated module for trading session logic.

Maps UTC hour to session, assigns quality scores, and determines
whether a symbol should be blocked in the current session.
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


def map_session(hour: int) -> str:
    """Map a UTC hour (0-23) to the active trading session name."""
    # Check overlap first (it's a subset of London + New_York)
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
