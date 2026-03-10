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
    return SESSION_QUALITY_SCORES.get(session, 0.40)


def get_confidence_multiplier(session: str) -> float:
    """Return the confidence multiplier for a session."""
    return SESSION_CONFIDENCE_MULT.get(session, 1.0)


def get_hour_confidence_multiplier(hour: int) -> float:
    """Return the per-hour confidence multiplier (0.80-1.10).
    
    v12: Blends hardcoded prior with adaptive learned multiplier.
    Adaptive multiplier is derived from observed win rate at this hour.
    """
    prior = HOUR_CONFIDENCE_MULT.get(hour % 24, 0.90)
    
    # Check if we have enough adaptive data
    key = str(hour % 24)
    data = _hour_outcomes.get(key)
    if not data or data.get("total", 0) < 10:
        return prior
    
    # Compute adaptive multiplier from win rate
    wr = data["wins"] / max(data["total"], 1)
    # Map win rate to multiplier: 50% → 1.0, 60% → 1.06, 40% → 0.85
    adaptive = 1.0 + (wr - 0.50) * 0.60  # e.g., 55% → 1.03, 45% → 0.97
    adaptive = max(0.75, min(1.12, adaptive))
    
    # Blend: give more weight to adaptive as data grows
    n = data["total"]
    adaptive_weight = min(0.60, n / 100.0)  # max 60% adaptive, 40% prior
    blended = (1.0 - adaptive_weight) * prior + adaptive_weight * adaptive
    
    return round(blended, 3)


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
