"""
Candle Quality Filter — v11: Detects stale/repetitive candles in long price moves.

Problem: During sustained trends, consecutive candles look nearly identical
(same body size, same direction, same wick ratios). The model keeps predicting
the same direction with declining accuracy because there's no NEW information.

Solution: Score each candle's "freshness" — how different it is from recent candles.
Low freshness → reduce confidence or skip the signal entirely.
"""

import numpy as np
import pandas as pd
import logging

log = logging.getLogger("candle_quality")

# ─── Configuration ───────────────────────────────────────────────────────────
FRESHNESS_LOOKBACK = 5          # Compare current candle to last N candles
SIMILARITY_THRESHOLD = 0.85     # Above this → candle is "stale" (0-1 scale)
MIN_BODY_RATIO = 0.15           # Minimum body/range ratio to consider meaningful
DOJI_THRESHOLD = 0.05           # Body < 5% of range = doji (skip)
EXHAUSTION_RSI_HIGH = 75        # RSI above this in uptrend = exhaustion risk
EXHAUSTION_RSI_LOW = 25         # RSI below this in downtrend = exhaustion risk


def candle_freshness_score(df, lookback=FRESHNESS_LOOKBACK):
    """
    Compute how "fresh" (different from recent candles) the latest candle is.
    
    Returns float 0.0 (very stale/repetitive) to 1.0 (very fresh/unique).
    
    Compares: body_size, direction, wick_ratios, range relative to ATR.
    """
    if len(df) < lookback + 1:
        return 1.0  # Not enough history → assume fresh
    
    recent = df.iloc[-(lookback + 1):]
    current = recent.iloc[-1]
    previous = recent.iloc[:-1]
    
    o, h, l, c = current["open"], current["high"], current["low"], current["close"]
    bar_range = h - l
    if bar_range < 1e-10:
        return 0.0  # Zero-range candle → completely stale
    
    body = abs(c - o)
    direction = 1 if c >= o else -1
    body_ratio = body / bar_range
    upper_wick = (h - max(o, c)) / bar_range
    lower_wick = (min(o, c) - l) / bar_range
    
    # Compare to each previous candle
    similarities = []
    for _, prev in previous.iterrows():
        po, ph, pl, pc = prev["open"], prev["high"], prev["low"], prev["close"]
        prev_range = ph - pl
        if prev_range < 1e-10:
            similarities.append(0.5)
            continue
        
        prev_body = abs(pc - po)
        prev_dir = 1 if pc >= po else -1
        prev_body_ratio = prev_body / prev_range
        prev_upper = (ph - max(po, pc)) / prev_range
        prev_lower = (min(po, pc) - pl) / prev_range
        
        # Direction similarity (0 or 1)
        dir_sim = 1.0 if direction == prev_dir else 0.0
        
        # Body ratio similarity (closer = more similar)
        body_sim = 1.0 - abs(body_ratio - prev_body_ratio)
        
        # Wick similarity
        wick_sim = 1.0 - 0.5 * (abs(upper_wick - prev_upper) + abs(lower_wick - prev_lower))
        
        # Range similarity (relative to ATR)
        range_sim = 1.0 - min(abs(bar_range - prev_range) / max(bar_range, prev_range, 1e-10), 1.0)
        
        # Weighted similarity score
        sim = 0.35 * dir_sim + 0.25 * body_sim + 0.20 * wick_sim + 0.20 * range_sim
        similarities.append(max(0.0, min(1.0, sim)))
    
    avg_similarity = np.mean(similarities)
    
    # Freshness = inverse of similarity
    freshness = 1.0 - avg_similarity
    return round(float(freshness), 4)


def is_stale_candle(df, threshold=SIMILARITY_THRESHOLD):
    """Check if the latest candle is too similar to recent ones."""
    freshness = candle_freshness_score(df)
    return freshness < (1.0 - threshold), freshness


def candle_quality_score(df):
    """
    Multi-factor candle quality assessment for the latest bar.
    
    Returns dict with:
      - quality: float 0-100 (overall quality)
      - freshness: float 0-1 (how different from recent)
      - body_quality: float 0-1 (meaningful body vs doji)
      - momentum_quality: float 0-1 (strong move vs indecision)
      - skip_reason: str or None
    """
    if len(df) < 10:
        return {"quality": 50.0, "freshness": 1.0, "body_quality": 0.5,
                "momentum_quality": 0.5, "skip_reason": None}
    
    current = df.iloc[-1]
    o, h, l, c = current["open"], current["high"], current["low"], current["close"]
    bar_range = h - l
    
    # 1. Freshness
    freshness = candle_freshness_score(df)
    
    # 2. Body quality (penalize dojis and tiny bodies)
    body_quality = 0.5
    if bar_range > 1e-10:
        body_ratio = abs(c - o) / bar_range
        if body_ratio < DOJI_THRESHOLD:
            body_quality = 0.1  # Doji → very low quality
        elif body_ratio < MIN_BODY_RATIO:
            body_quality = 0.3  # Tiny body
        elif body_ratio > 0.7:
            body_quality = 1.0  # Strong body
        else:
            body_quality = 0.3 + 0.7 * (body_ratio - MIN_BODY_RATIO) / (0.7 - MIN_BODY_RATIO)
    
    # 3. Momentum quality (compare range to recent average)
    recent_ranges = (df["high"] - df["low"]).iloc[-10:-1]
    avg_range = recent_ranges.mean()
    if avg_range > 1e-10:
        range_ratio = bar_range / avg_range
        momentum_quality = min(range_ratio / 1.5, 1.0)  # Normalized: 1.5x avg = perfect
    else:
        momentum_quality = 0.5
    
    # 4. Skip reason
    skip_reason = None
    if freshness < 0.1:
        skip_reason = f"Stale candle (freshness={freshness:.2f})"
    elif body_quality < 0.15:
        skip_reason = f"Doji/indecision candle"
    
    # Weighted quality score (0-100)
    quality = (
        40 * freshness +
        30 * body_quality +
        30 * momentum_quality
    )
    
    return {
        "quality": round(quality, 1),
        "freshness": freshness,
        "body_quality": round(body_quality, 3),
        "momentum_quality": round(momentum_quality, 3),
        "skip_reason": skip_reason,
    }


def streak_direction_count(df, lookback=20):
    """
    Count consecutive same-direction candles at the end of df.
    
    Returns (count, direction) where direction is 1 (bullish) or -1 (bearish).
    """
    if len(df) < 2:
        return 0, 0
    
    recent = df.iloc[-lookback:] if len(df) >= lookback else df
    directions = np.sign(recent["close"].values - recent["open"].values)
    
    # Count consecutive from the end
    current_dir = directions[-1]
    if current_dir == 0:
        return 0, 0
    
    count = 0
    for d in reversed(directions):
        if d == current_dir:
            count += 1
        else:
            break
    
    return count, int(current_dir)


def compute_candle_features(df):
    """
    Compute v11 candle quality features for the latest bar.
    These are added to the feature vector for training/prediction.
    
    Returns dict of features:
      - candle_freshness: 0-1 freshness score
      - candle_body_quality: 0-1 body quality
      - streak_length: consecutive same-direction candles
      - candle_range_vs_avg: current range / avg range
    """
    freshness = candle_freshness_score(df)
    
    current = df.iloc[-1]
    o, h, l, c = current["open"], current["high"], current["low"], current["close"]
    bar_range = h - l
    
    # Body quality
    body_quality = 0.5
    if bar_range > 1e-10:
        body_ratio = abs(c - o) / bar_range
        body_quality = min(body_ratio / 0.7, 1.0)
    
    # Streak length
    streak_len, _ = streak_direction_count(df)
    
    # Range vs average
    if len(df) >= 10:
        avg_range = (df["high"] - df["low"]).iloc[-10:-1].mean()
        range_vs_avg = bar_range / max(avg_range, 1e-10)
    else:
        range_vs_avg = 1.0
    
    return {
        "candle_freshness": freshness,
        "candle_body_quality": round(body_quality, 4),
        "streak_length": min(streak_len, 20),  # cap at 20
        "candle_range_vs_avg": round(min(range_vs_avg, 3.0), 4),  # cap at 3x
    }
