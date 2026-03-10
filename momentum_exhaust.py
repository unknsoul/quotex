"""
Momentum Exhaustion Detector — v11: Detects trend exhaustion and reversals.

Problem: Model keeps predicting same direction during long price moves,
but accuracy degrades because the trend is running out of steam.

Solution: Multi-factor exhaustion scoring using RSI extremes, declining
momentum, increasing wicks, and volume divergence to detect when a
trend is likely to reverse or stall.
"""

import numpy as np
import pandas as pd
import logging

log = logging.getLogger("momentum_exhaust")

# ─── Configuration ───────────────────────────────────────────────────────────
RSI_OVERBOUGHT = 72
RSI_OVERSOLD = 28
MOMENTUM_DECAY_WINDOW = 5       # Compare last N bars momentum to prior N
WICK_REJECTION_THRESHOLD = 0.4  # Wick > 40% of range = rejection
EXHAUSTION_SCORE_SKIP = 75      # Skip signal if exhaustion > 75%


def compute_exhaustion_score(df, predicted_direction="DOWN"):
    """
    Compute trend exhaustion score for the latest bar.
    
    Returns dict:
      - exhaustion_score: 0-100 (0=fresh trend, 100=extremely exhausted)
      - factors: dict of individual factor scores
      - should_skip: bool (True if exhaustion is too high for this direction)
      - reason: str
    """
    if len(df) < 20:
        return {"exhaustion_score": 0, "factors": {}, "should_skip": False, "reason": ""}
    
    current = df.iloc[-1]
    
    # ─── Factor 1: RSI Extreme (0-100) ───────────────────────────────────
    rsi = float(current.get("rsi_14", 50))
    rsi_score = 0
    if predicted_direction == "UP" and rsi > RSI_OVERBOUGHT:
        # Predicting UP but RSI already overbought → exhaustion
        rsi_score = min((rsi - RSI_OVERBOUGHT) * 3, 100)
    elif predicted_direction == "DOWN" and rsi < RSI_OVERSOLD:
        # Predicting DOWN but RSI already oversold → exhaustion
        rsi_score = min((RSI_OVERSOLD - rsi) * 3, 100)
    
    # ─── Factor 2: Momentum Decay (0-100) ────────────────────────────────
    # Compare recent momentum to prior momentum
    closes = df["close"].values
    if len(closes) >= 2 * MOMENTUM_DECAY_WINDOW:
        recent_returns = np.abs(np.diff(closes[-MOMENTUM_DECAY_WINDOW:]))
        prior_returns = np.abs(np.diff(closes[-2*MOMENTUM_DECAY_WINDOW:-MOMENTUM_DECAY_WINDOW]))
        
        recent_avg = np.mean(recent_returns) if len(recent_returns) > 0 else 0
        prior_avg = np.mean(prior_returns) if len(prior_returns) > 0 else 1e-10
        
        if prior_avg > 1e-10:
            decay_ratio = recent_avg / prior_avg
            # If recent momentum < prior momentum → trend losing steam
            if decay_ratio < 1.0:
                momentum_score = min((1.0 - decay_ratio) * 200, 100)
            else:
                momentum_score = 0
        else:
            momentum_score = 0
    else:
        momentum_score = 0
    
    # ─── Factor 3: Wick Rejection (0-100) ────────────────────────────────
    # Large wicks opposing the predicted direction = rejection
    o, h, l, c = current["open"], current["high"], current["low"], current["close"]
    bar_range = h - l
    wick_score = 0
    if bar_range > 1e-10:
        if predicted_direction == "UP":
            upper_wick = (h - max(o, c)) / bar_range
            if upper_wick > WICK_REJECTION_THRESHOLD:
                wick_score = min(upper_wick * 150, 100)
        else:  # DOWN
            lower_wick = (min(o, c) - l) / bar_range
            if lower_wick > WICK_REJECTION_THRESHOLD:
                wick_score = min(lower_wick * 150, 100)
    
    # ─── Factor 4: ADX Decline (0-100) ──────────────────────────────────
    # Declining ADX = weakening trend
    adx_score = 0
    if "adx" in df.columns and len(df) >= 5:
        adx_recent = df["adx"].iloc[-3:].mean()
        adx_prior = df["adx"].iloc[-8:-3].mean()
        if adx_prior > 1e-10 and adx_recent < adx_prior:
            decline = (adx_prior - adx_recent) / adx_prior
            adx_score = min(decline * 300, 100)
    
    # ─── Factor 5: Price Extension (0-100) ───────────────────────────────
    # How far price has moved from recent mean
    extension_score = 0
    if len(df) >= 20:
        mean_20 = df["close"].iloc[-20:].mean()
        std_20 = df["close"].iloc[-20:].std()
        if std_20 > 1e-10:
            z_score = abs(current["close"] - mean_20) / std_20
            if z_score > 1.5:
                extension_score = min((z_score - 1.5) * 60, 100)
    
    # ─── Composite Score ─────────────────────────────────────────────────
    weights = {
        "rsi_extreme": 0.25,
        "momentum_decay": 0.25,
        "wick_rejection": 0.20,
        "adx_decline": 0.15,
        "price_extension": 0.15,
    }
    
    factors = {
        "rsi_extreme": round(rsi_score, 1),
        "momentum_decay": round(momentum_score, 1),
        "wick_rejection": round(wick_score, 1),
        "adx_decline": round(adx_score, 1),
        "price_extension": round(extension_score, 1),
    }
    
    exhaustion = sum(weights[k] * factors[k] for k in weights)
    should_skip = exhaustion > EXHAUSTION_SCORE_SKIP
    
    reason = ""
    if should_skip:
        top_factor = max(factors, key=factors.get)
        reason = f"Trend exhaustion ({exhaustion:.0f}%): {top_factor}={factors[top_factor]:.0f}"
        log.info(reason)
    
    return {
        "exhaustion_score": round(exhaustion, 1),
        "factors": factors,
        "should_skip": should_skip,
        "reason": reason,
    }


def compute_exhaustion_features(df):
    """
    Compute exhaustion-related features for the feature vector.
    
    Returns dict of features to add to the model.
    """
    if len(df) < 20:
        return {
            "trend_exhaustion_composite": 0.0,
            "momentum_decay_ratio": 1.0,
            "wick_rejection_strength": 0.0,
        }
    
    closes = df["close"].values
    current = df.iloc[-1]
    
    # Momentum decay ratio
    if len(closes) >= 10:
        recent = np.abs(np.diff(closes[-5:]))
        prior = np.abs(np.diff(closes[-10:-5]))
        r_avg = np.mean(recent) if len(recent) > 0 else 0
        p_avg = np.mean(prior) if len(prior) > 0 else 1e-10
        decay_ratio = min(r_avg / max(p_avg, 1e-10), 3.0)
    else:
        decay_ratio = 1.0
    
    # Wick rejection strength (direction-agnostic)
    o, h, l, c = current["open"], current["high"], current["low"], current["close"]
    bar_range = h - l
    if bar_range > 1e-10:
        max_wick = max((h - max(o, c)), (min(o, c) - l)) / bar_range
    else:
        max_wick = 0
    
    # Composite exhaustion (simplified for feature)
    rsi = float(current.get("rsi_14", 50))
    rsi_extreme = max(rsi - 70, 30 - rsi, 0) / 30  # 0-1 scale
    
    composite = (rsi_extreme * 0.4 + (1 - min(decay_ratio, 1)) * 0.3 + max_wick * 0.3)
    
    return {
        "trend_exhaustion_composite": round(float(min(composite, 1.0)), 4),
        "momentum_decay_ratio": round(float(decay_ratio), 4),
        "wick_rejection_strength": round(float(max_wick), 4),
    }
