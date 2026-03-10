# -*- coding: utf-8 -*-
"""Candle Pattern Confirmation - v11 (12-pattern overlay).

Detects bullish and bearish candlestick patterns on the last few bars.
A signal is stronger when the ML prediction aligns with a recognized pattern.

Patterns detected (12 total):
  Bullish: hammer, bullish engulfing, morning star, piercing line,
           inverted hammer, bullish marubozu
  Bearish: shooting star, bearish engulfing, evening star, dark cloud cover,
           hanging man, bearish marubozu

v11 additions:
  - 4 new patterns (inverted hammer, bullish marubozu, hanging man, bearish marubozu)
  - Same-candle pattern suppression (doji cluster, same-color streak, identical body)
  - Score capping at ±20 for multiple same-direction patterns
  - Pattern score adjustment (+10 confirm, -15 contradict)
"""

import logging
import numpy as np

log = logging.getLogger("candle_patterns")


def _body(row):
    return abs(row["close"] - row["open"])

def _range(row):
    return row["high"] - row["low"] if row["high"] > row["low"] else 1e-10

def _is_green(row):
    return row["close"] > row["open"]

def _upper_wick(row):
    return row["high"] - max(row["close"], row["open"])

def _lower_wick(row):
    return min(row["close"], row["open"]) - row["low"]


def detect_patterns(df, lookback=3) -> dict:
    """
    Detect candlestick patterns on the last `lookback` bars.

    Returns:
        dict with:
            bullish_patterns: list of pattern names detected
            bearish_patterns: list of pattern names detected
            bullish_score: 0-3 (how many bullish patterns found)
            bearish_score: 0-3
            net_bias: 'BULLISH', 'BEARISH', or 'NEUTRAL'
    """
    if df is None or len(df) < lookback + 1:
        return {
            "bullish_patterns": [], "bearish_patterns": [],
            "bullish_score": 0, "bearish_score": 0,
            "net_bias": "NEUTRAL",
        }

    bullish = []
    bearish = []

    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3] if len(df) >= 3 else None

    body_last = _body(last)
    range_last = _range(last)
    body_prev = _body(prev)

    # Average body size for context
    avg_body = df["close"].tail(20).diff().abs().mean()
    if avg_body < 1e-10:
        avg_body = 1e-10

    # --- BULLISH PATTERNS ---

    # 1. Hammer: small body at top, long lower wick (>2x body)
    lower_wick = _lower_wick(last)
    upper_wick = _upper_wick(last)
    if body_last > 0 and lower_wick > 2 * body_last and upper_wick < body_last * 0.5:
        if not _is_green(prev):  # after a red candle = reversal signal
            bullish.append("Hammer")

    # 2. Bullish Engulfing: green candle fully engulfs previous red
    if (_is_green(last) and not _is_green(prev) and
            last["close"] > prev["open"] and last["open"] < prev["close"] and
            body_last > body_prev * 1.1):
        bullish.append("Bullish Engulfing")

    # 3. Morning Star (3-candle): red, small body, green
    if prev2 is not None:
        if (not _is_green(prev2) and _body(prev) < avg_body * 0.5 and
                _is_green(last) and last["close"] > (prev2["open"] + prev2["close"]) / 2):
            bullish.append("Morning Star")

    # 4. Piercing Line: gap down open, close above midpoint of prev red
    if (not _is_green(prev) and _is_green(last) and
            last["open"] < prev["close"] and
            last["close"] > (prev["open"] + prev["close"]) / 2):
        bullish.append("Piercing Line")

    # --- BEARISH PATTERNS ---

    # 1. Shooting Star: small body at bottom, long upper wick
    if body_last > 0 and upper_wick > 2 * body_last and lower_wick < body_last * 0.5:
        if _is_green(prev):  # after a green candle = reversal
            bearish.append("Shooting Star")

    # 2. Bearish Engulfing: red candle fully engulfs previous green
    if (not _is_green(last) and _is_green(prev) and
            last["open"] > prev["close"] and last["close"] < prev["open"] and
            body_last > body_prev * 1.1):
        bearish.append("Bearish Engulfing")

    # 3. Evening Star (3-candle): green, small body, red
    if prev2 is not None:
        if (_is_green(prev2) and _body(prev) < avg_body * 0.5 and
                not _is_green(last) and last["close"] < (prev2["open"] + prev2["close"]) / 2):
            bearish.append("Evening Star")

    # 4. Dark Cloud Cover: gap up open, close below midpoint of prev green
    if (_is_green(prev) and not _is_green(last) and
            last["open"] > prev["close"] and
            last["close"] < (prev["open"] + prev["close"]) / 2):
        bearish.append("Dark Cloud Cover")

    # --- NEW v11 PATTERNS ---

    # 5. Inverted Hammer (bullish): small body, long upper wick > 2x body, after downtrend
    if body_last > 0 and upper_wick > 2 * body_last and lower_wick < body_last * 0.5:
        if not _is_green(prev):  # after a red candle
            bullish.append("Inverted Hammer")

    # 6. Bullish Marubozu: large green body, minimal wicks (< 5% of range)
    if (_is_green(last) and body_last > avg_body * 1.5 and range_last > 0 and
            upper_wick < range_last * 0.05 and lower_wick < range_last * 0.05):
        bullish.append("Bullish Marubozu")

    # 5b. Hanging Man (bearish): small body at top, long lower wick, after uptrend
    if body_last > 0 and lower_wick > 2 * body_last and upper_wick < body_last * 0.5:
        if _is_green(prev):  # after a green candle = reversal warning
            bearish.append("Hanging Man")

    # 6b. Bearish Marubozu: large red body, minimal wicks
    if (not _is_green(last) and body_last > avg_body * 1.5 and range_last > 0 and
            upper_wick < range_last * 0.05 and lower_wick < range_last * 0.05):
        bearish.append("Bearish Marubozu")

    # Net bias
    b_score = len(bullish)
    s_score = len(bearish)
    if b_score > s_score:
        bias = "BULLISH"
    elif s_score > b_score:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    return {
        "bullish_patterns": bullish,
        "bearish_patterns": bearish,
        "bullish_score": b_score,
        "bearish_score": s_score,
        "net_bias": bias,
    }


def compute_pattern_score_adjustment(df, predicted_direction: str) -> dict:
    """
    Compute score adjustment from candlestick patterns.
    v11 overlay: +10 for confirm, -15 for contradict, cap at ±20.

    Also applies same-candle suppression rules:
    - Consecutive doji: skip bonus entirely
    - 5+ same color: flag MOMENTUM_EXHAUSTION
    - 3+ identical body: reduce weight 50%

    Returns:
        dict with:
            adjustment: int score change
            patterns: list of pattern names
            suppressed: bool (doji cluster suppresses patterns)
            momentum_exhaustion: bool
            details: str
    """
    result = detect_patterns(df)
    adjustment = 0
    suppressed = False
    momentum_exhaustion = False
    body_penalty = False

    if df is None or len(df) < 6:
        return {
            "adjustment": 0, "patterns": [], "suppressed": False,
            "momentum_exhaustion": False, "details": "insufficient data",
        }

    # Same-candle suppression checks
    try:
        bodies = abs(df["close"].values - df["open"].values)
        avg_body = max(np.mean(bodies[-20:]), 1e-10)

        # Consecutive doji check (3+)
        doji_count = 0
        for b in reversed(bodies[-6:]):
            if b < avg_body * 0.10:
                doji_count += 1
            else:
                break
        if doji_count >= 3:
            suppressed = True

        # Consecutive same color 5+ → momentum exhaustion
        colors = (df["close"].values > df["open"].values)[-6:]
        if len(colors) >= 5:
            if all(colors[-5:]) or all(~colors[-5:]):
                momentum_exhaustion = True

        # 3+ identical body size (within 10%)
        recent_bodies = bodies[-5:]
        if len(recent_bodies) >= 3:
            ref = recent_bodies[-1]
            if ref > 1e-10:
                same_count = sum(1 for b in recent_bodies[-3:]
                                 if abs(b - ref) / ref < 0.10)
                if same_count >= 3:
                    body_penalty = True
    except Exception:
        pass

    if suppressed:
        return {
            "adjustment": 0, "patterns": [], "suppressed": True,
            "momentum_exhaustion": momentum_exhaustion,
            "details": "doji cluster — patterns suppressed",
        }

    # Compute adjustment
    all_patterns = result["bullish_patterns"] + result["bearish_patterns"]

    if predicted_direction == "UP":
        if result["bullish_score"] > 0:
            adjustment += min(result["bullish_score"] * 10, 20)
        if result["bearish_score"] > 0:
            adjustment -= min(result["bearish_score"] * 15, 20)
    elif predicted_direction == "DOWN":
        if result["bearish_score"] > 0:
            adjustment += min(result["bearish_score"] * 10, 20)
        if result["bullish_score"] > 0:
            adjustment -= min(result["bullish_score"] * 15, 20)

    # Body penalty: reduce by 50%
    if body_penalty:
        adjustment = adjustment // 2

    details_parts = []
    if all_patterns:
        details_parts.append(", ".join(all_patterns))
    if body_penalty:
        details_parts.append("body_penalty")
    if momentum_exhaustion:
        details_parts.append("momentum_exhaustion")

    return {
        "adjustment": adjustment,
        "patterns": all_patterns,
        "suppressed": suppressed,
        "momentum_exhaustion": momentum_exhaustion,
        "details": " | ".join(details_parts) if details_parts else "no patterns",
    }


def pattern_confirms(df, predicted_direction: str) -> dict:
    """
    Check if candle patterns confirm the predicted direction.

    Returns:
        dict with:
            confirmed: bool
            patterns: list of confirming patterns
            reason: str
    """
    result = detect_patterns(df)

    if predicted_direction == "UP":
        if result["bullish_score"] > 0:
            return {
                "confirmed": True,
                "patterns": result["bullish_patterns"],
                "reason": f"Bullish patterns: {', '.join(result['bullish_patterns'])}",
            }
        elif result["bearish_score"] > 0:
            return {
                "confirmed": False,
                "patterns": result["bearish_patterns"],
                "reason": f"Bearish patterns contradict UP: {', '.join(result['bearish_patterns'])}",
            }
    elif predicted_direction == "DOWN":
        if result["bearish_score"] > 0:
            return {
                "confirmed": True,
                "patterns": result["bearish_patterns"],
                "reason": f"Bearish patterns: {', '.join(result['bearish_patterns'])}",
            }
        elif result["bullish_score"] > 0:
            return {
                "confirmed": False,
                "patterns": result["bullish_patterns"],
                "reason": f"Bullish patterns contradict DOWN: {', '.join(result['bullish_patterns'])}",
            }

    # No patterns detected — neutral, allow signal
    return {
        "confirmed": True,
        "patterns": [],
        "reason": "No contradicting patterns",
    }
