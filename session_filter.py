"""
Session Filter — v12 Adaptive hour/session confidence scaling.

Instead of blocking hours entirely, every hour gets a confidence
multiplier. v12 upgrade: multipliers now self-adapt from outcome data
while keeping the hardcoded defaults as priors.
"""

import json
import logging
import os
from datetime import datetime, timezone

from config import (
    SESSION_HOURS,
    SESSION_QUALITY_SCORES,
    SESSION_BLOCK_MAP,
    SESSION_CONFIDENCE_MULT,
)

log = logging.getLogger("session_filter")

# ── Per-Hour Confidence Multipliers (data-driven from outcome analysis) ──────
# v12: These are now the PRIOR values. The adaptive system will learn from
# live outcomes and blend with these defaults.
# v13: All hour multipliers set to 1.0 — no time-based confidence penalties
HOUR_CONFIDENCE_MULT = {h: 1.0 for h in range(24)}

# ── v12: Adaptive Hour Multipliers (learned from outcomes) ────────────────────
ADAPTIVE_STATE_PATH = os.path.join(os.path.dirname(__file__), "logs", "session_adaptive.json")
_hour_outcomes = {}  # {hour: {"wins": int, "total": int}}


def _load_hour_outcomes():
    global _hour_outcomes
    try:
        if os.path.exists(ADAPTIVE_STATE_PATH):
            with open(ADAPTIVE_STATE_PATH, "r") as f:
                _hour_outcomes = json.load(f)
    except Exception:
        _hour_outcomes = {}


def _save_hour_outcomes():
    try:
        os.makedirs(os.path.dirname(ADAPTIVE_STATE_PATH), exist_ok=True)
        with open(ADAPTIVE_STATE_PATH, "w") as f:
            json.dump(_hour_outcomes, f)
    except Exception:
        pass


def update_hour_outcome(hour: int, was_correct: bool):
    """Record an outcome for adaptive hour multiplier learning."""
    key = str(hour % 24)
    if key not in _hour_outcomes:
        _hour_outcomes[key] = {"wins": 0, "total": 0}
    _hour_outcomes[key]["total"] += 1
    if was_correct:
        _hour_outcomes[key]["wins"] += 1
    # Rolling window: decay after 100 observations per hour
    if _hour_outcomes[key]["total"] > 100:
        _hour_outcomes[key]["wins"] = round(_hour_outcomes[key]["wins"] * 0.8)
        _hour_outcomes[key]["total"] = round(_hour_outcomes[key]["total"] * 0.8)
    if _hour_outcomes[key]["total"] % 5 == 0:
        _save_hour_outcomes()


_load_hour_outcomes()


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
    return 1.0  # v13: All sessions treated equally


def get_confidence_multiplier(session: str) -> float:
    """Return the confidence multiplier for a session."""
    return 1.0  # v13: No session-based confidence penalty


def get_hour_confidence_multiplier(hour: int) -> float:
    """Return the per-hour confidence multiplier.
    
    v13: Always returns 1.0 — no hour-based penalty.
    """
    return 1.0


def get_hour_strategy(hour: int) -> dict:
    """Return strategy info for the given hour.
    
    v13: All hours treated as normal — no penalties or extra confirmation.
    """
    return {
        "multiplier": 1.0,
        "quality": "normal",
        "require_extra_confirmation": False,
    }


def should_block_session(symbol: str, session: str) -> bool:
    """Return True if trading this symbol is blocked in the current session."""
    return False  # v13: Never block any session


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
