"""
Candle Pattern Confirmation — Upgrade 2.

Detects bullish and bearish candlestick patterns on the last few bars.
A signal is stronger when the ML prediction aligns with a recognized pattern.

Patterns detected:
  Bullish: hammer, bullish engulfing, morning star, piercing line
  Bearish: shooting star, bearish engulfing, evening star, dark cloud cover
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
