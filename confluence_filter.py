# -*- coding: utf-8 -*-
"""
Multi-Timeframe Confluence Filter — v11 (Pipeline Stage 5).

Improved confluence logic:
  1. Weighted TF voting — H1 counts more than M5
  2. RSI momentum confirmation per TF
  3. Partial-pass scoring — strong H1 alignment can override weak M5
  4. EMA hierarchy check — proper trend structure (EMA20 > EMA50 > EMA200)
  5. Momentum magnitude — not just direction, but strength
  6. v11: Explicit H1 hard gate — blocks counter-trend M5 entries
  7. v11: EMA hierarchy bonus (+0.10)
  8. v11: RSI per-TF thresholds
  9. v11: Confluence pass threshold = 0.60

Returns:
    confluence_score  : 0.0–3.0 (weighted score, not just count)
    confluence_pass   : True if weighted score >= threshold
    tf_directions     : dict of directions per TF
    h1_hard_gate_pass : True if H1 does not contradict
"""

import logging
import numpy as np

log = logging.getLogger("confluence_filter")

# TF weights — higher timeframes more important for trend accuracy
TF_WEIGHTS = {"M5": 0.25, "M15": 0.35, "H1": 0.40}


def _tf_direction_v2(df, lookback=3, tf_name="M5"):
    """
    Determine direction with confidence score using multiple confirmations:
    - EMA slope (trend direction) + magnitude
    - Last N candles momentum (green/red ratio)
    - Price vs EMA-20 (above/below)
    - RSI positioning (>55 bullish, <45 bearish)
    - EMA hierarchy (20 > 50 > 200 = strong trend)

    Returns: ('UP'|'DOWN'|'NEUTRAL', confidence 0.0-1.0)
    """
    if df is None or len(df) < max(lookback, 20):
        return "NEUTRAL", 0.0

    try:
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        # EMA-20 slope
        ema20 = df["close"].ewm(span=20, min_periods=10).mean().values
        ema50 = df["close"].ewm(span=50, min_periods=20).mean().values if len(df) >= 50 else ema20
        ema_slope = (ema20[-1] - ema20[-3]) / max(abs(ema20[-3]), 1e-10)

        # Candle momentum
        recent = df.tail(lookback)
        green_count = (recent["close"] > recent["open"]).sum()
        red_count = (recent["close"] < recent["open"]).sum()

        # Price vs EMA-20
        above_ema = close[-1] > ema20[-1]

        # RSI (simple)
        delta = np.diff(close[-15:]) if len(close) >= 15 else np.diff(close)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        avg_gain = max(np.mean(gains[-14:]), 1e-10) if len(gains) >= 14 else max(np.mean(gains), 1e-10)
        avg_loss = max(np.mean(losses[-14:]), 1e-10) if len(losses) >= 14 else max(np.mean(losses), 1e-10)
        rsi = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))

        # EMA hierarchy: check if EMAs are properly stacked
        ema_stacked_bull = (ema20[-1] > ema50[-1]) if len(df) >= 50 else False
        ema_stacked_bear = (ema20[-1] < ema50[-1]) if len(df) >= 50 else False

        # Weighted voting with magnitude
        up_score = 0.0
        down_score = 0.0

        # EMA slope (weight: 0.30)
        slope_threshold = 0.00005 if tf_name == "H1" else 0.0001
        if ema_slope > slope_threshold:
            up_score += 0.30 * min(abs(ema_slope) / 0.002, 1.0)
        elif ema_slope < -slope_threshold:
            down_score += 0.30 * min(abs(ema_slope) / 0.002, 1.0)

        # Candle momentum (weight: 0.20)
        if green_count > red_count:
            up_score += 0.20 * (green_count / lookback)
        elif red_count > green_count:
            down_score += 0.20 * (red_count / lookback)

        # Price vs EMA (weight: 0.15)
        ema_dist = abs(close[-1] - ema20[-1]) / max(abs(ema20[-1]), 1e-10)
        if above_ema:
            up_score += 0.15 * min(ema_dist / 0.002, 1.0)
        else:
            down_score += 0.15 * min(ema_dist / 0.002, 1.0)

        # RSI (weight: 0.20)
        if rsi > 55:
            up_score += 0.20 * min((rsi - 50) / 30, 1.0)
        elif rsi < 45:
            down_score += 0.20 * min((50 - rsi) / 30, 1.0)

        # EMA hierarchy (weight: 0.15)
        if ema_stacked_bull:
            up_score += 0.15
        elif ema_stacked_bear:
            down_score += 0.15

        conf = max(up_score, down_score)
        if up_score > down_score and up_score >= 0.30:
            return "UP", round(conf, 3)
        elif down_score > up_score and down_score >= 0.30:
            return "DOWN", round(conf, 3)
        else:
            return "NEUTRAL", round(conf, 3)

    except Exception as e:
        log.debug("TF direction calc failed: %s", e)
        return "NEUTRAL", 0.0


def check_confluence(m5_df, m15_df=None, h1_df=None,
                     predicted_direction="UP", min_score=2):
    """
    Check if multiple timeframes agree with the predicted direction.

    v2 improvements:
      - Weighted scoring (H1=0.40, M15=0.35, M5=0.25)
      - Confidence-weighted votes
      - Partial pass: strong H1 can compensate for weak M5
      - H1 NEUTRAL counts as partial agree (no strong opposing trend)

    Returns:
        dict with:
            confluence_score: 0.0–3.0 (weighted, not just count)
            confluence_pass: bool
            tf_directions: {M5: ..., M15: ..., H1: ...}
            reason: human-readable explanation
    """
    directions = {}
    weighted_score = 0.0
    raw_count = 0  # backwards compat

    # M5 direction
    m5_dir, m5_conf = _tf_direction_v2(m5_df, lookback=3, tf_name="M5")
    directions["M5"] = m5_dir
    if m5_dir == predicted_direction:
        weighted_score += TF_WEIGHTS["M5"] * (1.0 + m5_conf)
        raw_count += 1
    elif m5_dir == "NEUTRAL":
        weighted_score += TF_WEIGHTS["M5"] * 0.3  # partial credit

    # M15 direction
    if m15_df is not None and len(m15_df) >= 20:
        m15_dir, m15_conf = _tf_direction_v2(m15_df, lookback=3, tf_name="M15")
        directions["M15"] = m15_dir
        if m15_dir == predicted_direction:
            weighted_score += TF_WEIGHTS["M15"] * (1.0 + m15_conf)
            raw_count += 1
        elif m15_dir == "NEUTRAL":
            weighted_score += TF_WEIGHTS["M15"] * 0.3
    else:
        directions["M15"] = "N/A"

    # H1 direction — most important TF
    if h1_df is not None and len(h1_df) >= 20:
        h1_dir, h1_conf = _tf_direction_v2(h1_df, lookback=2, tf_name="H1")
        directions["H1"] = h1_dir
        if h1_dir == predicted_direction:
            weighted_score += TF_WEIGHTS["H1"] * (1.0 + h1_conf)
            raw_count += 1
        elif h1_dir == "NEUTRAL":
            # H1 neutral = no strong opposing trend, partial credit
            weighted_score += TF_WEIGHTS["H1"] * 0.4
    else:
        directions["H1"] = "N/A"

    # Scale weighted_score to 0-3 range for backwards compatibility
    # Max possible: sum(weights * 2.0) ≈ 2.0, scale to 0-3
    scaled_score = min(3.0, weighted_score * 3.0 / 2.0)

    # Pass condition: weighted score threshold (v2: more nuanced than raw count)
    # min_score=2 → need weighted_score >= ~1.0 (moderate agreement)
    weighted_threshold = min_score / 3.0  # 2/3 ≈ 0.667
    passed = weighted_score >= weighted_threshold

    # v12: H1 strength-weighted gate — replaces hard block with graduated penalty
    # Instead of hard-blocking on any H1 disagreement, we apply a confidence
    # penalty proportional to H1's conviction. Weak H1 opposition (low conf)
    # gets a mild penalty; strong H1 opposition gets a severe penalty.
    h1_hard_gate_pass = True
    h1_penalty = 0.0
    if h1_df is not None and len(h1_df) >= 20:
        h1_dir_check = directions.get("H1", "NEUTRAL")
        if h1_dir_check != "NEUTRAL" and h1_dir_check != predicted_direction and h1_dir_check != "N/A":
            _, h1_opp_conf = _tf_direction_v2(h1_df, lookback=2, tf_name="H1")
            if h1_opp_conf >= 0.80:
                # v16.1: Only block on very strong H1 opposition (was 0.65)
                h1_hard_gate_pass = False
                h1_penalty = 0.40
                passed = False
            elif h1_opp_conf >= 0.55:
                # Moderate H1 opposition — heavy penalty but allow if M5+M15 strong
                h1_penalty = 0.20
                weighted_score *= (1.0 - h1_penalty)
                scaled_score = min(3.0, weighted_score * 3.0 / 2.0)
                # Re-check pass with reduced score
                passed = weighted_score >= weighted_threshold
            else:
                # Weak H1 opposition — mild penalty, likely noise
                h1_penalty = 0.10
                weighted_score *= (1.0 - h1_penalty)
                scaled_score = min(3.0, weighted_score * 3.0 / 2.0)

    # v11: EMA hierarchy bonus (+0.10 to confluence score)
    ema_hierarchy_bonus = 0.0
    try:
        if m5_df is not None and len(m5_df) >= 200:
            c = m5_df["close"]
            e20 = c.ewm(span=20, min_periods=10).mean().iloc[-1]
            e50 = c.ewm(span=50, min_periods=25).mean().iloc[-1]
            e200 = c.ewm(span=200, min_periods=100).mean().iloc[-1]
            if predicted_direction == "UP" and e20 > e50 > e200:
                ema_hierarchy_bonus = 0.10
            elif predicted_direction == "DOWN" and e20 < e50 < e200:
                ema_hierarchy_bonus = 0.10
            if ema_hierarchy_bonus > 0:
                weighted_score += ema_hierarchy_bonus
                scaled_score = min(3.0, weighted_score * 3.0 / 2.0)
    except Exception:
        pass

    if not passed:
        opposing = [f"{tf}={d}" for tf, d in directions.items()
                    if d != predicted_direction and d != "N/A" and d != "NEUTRAL"]
        if not h1_hard_gate_pass:
            reason = f"H1 hard gate: H1={directions.get('H1')} vs pred={predicted_direction}"
        else:
            reason = f"TF conflict: {', '.join(opposing)} vs pred={predicted_direction}"
    else:
        reason = f"Confluence {scaled_score:.1f}/3 OK (weighted)"

    return {
        "confluence_score": round(scaled_score, 1),
        "confluence_pass": passed,
        "tf_directions": directions,
        "reason": reason,
        "h1_hard_gate_pass": h1_hard_gate_pass,
        "h1_penalty": round(h1_penalty, 2),
        "ema_hierarchy_bonus": round(ema_hierarchy_bonus, 2),
    }
