# -*- coding: utf-8 -*-
"""
Same-Candle / Stale Candle Detector — v11 SC-1 to SC-5.

Detects stale, frozen, or repetitive candles that produce false signals.
Major source of losses on M5 Quotex feeds.

Checks:
  SC-1: Identical OHLC Clone Check
  SC-2: Zero-Body Doji Streak
  SC-3: Duplicate Close Sequence
  SC-4: Volume Anomaly (if available)
  SC-5: Alternating Color Chop
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger("same_candle_detector")

# ── Thresholds ──────────────────────────────────────────────────────────────
SC1_TOLERANCE_PCT = 0.001       # OHLCV match tolerance (0.001%)
SC2_CONSECUTIVE_DOJI = 3        # consecutive doji limit
SC2_BODY_RATIO = 0.10           # body < 10% of avg = doji
SC3_WINDOW = 4                  # duplicate close window
SC3_PIP_TOLERANCE = 0.5         # within 0.5 pips
SC4_VOLUME_RATIO = 0.20         # tick vol < 20% of avg = low volume
SC4_CONFIDENCE_PENALTY = 0.25   # 25% confidence reduction
SC5_CHOP_WINDOW = 6             # alternating color check window


def detect_same_candle(df: pd.DataFrame) -> dict:
    """
    Run all 5 same-candle checks on the DataFrame.

    Returns:
        dict with:
            stale: bool — SC-1 identical candle detected
            doji_cluster: bool — SC-2 consecutive doji streak
            flat_market: bool — SC-3 duplicate closes
            low_volume: bool — SC-4 volume anomaly
            volume_penalty: float — 0.0–0.25 confidence penalty
            chop_detected: bool — SC-5 alternating color pattern
            flags: list of str — human-readable flag names
            should_skip: bool — any hard skip triggered
            confidence_penalty: float — total penalty to apply (0.0–1.0 = multiplier)
            details: str — summary string
    """
    result = {
        "stale": False, "doji_cluster": False, "flat_market": False,
        "low_volume": False, "volume_penalty": 0.0,
        "chop_detected": False, "flags": [],
        "should_skip": False, "confidence_penalty": 1.0, "details": "",
    }

    if df is None or len(df) < SC5_CHOP_WINDOW + 1:
        return result

    flags = []
    penalty = 1.0

    # ── SC-1: Identical OHLC Clone Check ────────────────────────────────
    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        tol = SC1_TOLERANCE_PCT / 100.0
        ohlc_cols = ["open", "high", "low", "close"]
        all_match = True
        for col in ohlc_cols:
            if abs(last[col] - prev[col]) > abs(prev[col]) * tol + 1e-10:
                all_match = False
                break
        if all_match:
            result["stale"] = True
            flags.append("SC1_STALE")
    except Exception as e:
        log.debug("SC-1 error: %s", e)

    # ── SC-2: Zero-Body Doji Streak ─────────────────────────────────────
    try:
        bodies = abs(df["close"].values - df["open"].values)
        avg_body = np.mean(bodies[-20:]) if len(bodies) >= 20 else np.mean(bodies)
        if avg_body < 1e-10:
            avg_body = 1e-10
        threshold = avg_body * SC2_BODY_RATIO
        consecutive = 0
        for i in range(len(bodies) - 1, max(len(bodies) - 10, -1), -1):
            if bodies[i] < threshold:
                consecutive += 1
            else:
                break
        if consecutive >= SC2_CONSECUTIVE_DOJI:
            result["doji_cluster"] = True
            flags.append(f"SC2_DOJI_x{consecutive}")
    except Exception as e:
        log.debug("SC-2 error: %s", e)

    # ── SC-3: Duplicate Close Sequence ──────────────────────────────────
    try:
        closes = df["close"].values[-SC3_WINDOW:]
        if len(closes) >= SC3_WINDOW:
            # Estimate pip size from price level
            price = closes[-1]
            if price > 50:       # gold, indices
                pip = 0.1
            elif price > 1:      # most forex pairs
                pip = 0.0001
            else:                # exotic
                pip = 0.00001
            spread = np.max(closes) - np.min(closes)
            if spread <= SC3_PIP_TOLERANCE * pip:
                result["flat_market"] = True
                flags.append("SC3_FLAT")
    except Exception as e:
        log.debug("SC-3 error: %s", e)

    # ── SC-4: Volume Anomaly ────────────────────────────────────────────
    try:
        if "tick_volume" in df.columns:
            vols = df["tick_volume"].values
            current_vol = vols[-1]
            avg_vol = np.mean(vols[-20:]) if len(vols) >= 20 else np.mean(vols)
            if avg_vol > 0 and current_vol < avg_vol * SC4_VOLUME_RATIO:
                result["low_volume"] = True
                result["volume_penalty"] = SC4_CONFIDENCE_PENALTY
                penalty *= (1.0 - SC4_CONFIDENCE_PENALTY)
                flags.append("SC4_LOW_VOL")
    except Exception as e:
        log.debug("SC-4 error: %s", e)

    # ── SC-5: Alternating Color Chop ────────────────────────────────────
    try:
        recent = df.tail(SC5_CHOP_WINDOW)
        colors = (recent["close"] > recent["open"]).values  # True=green, False=red
        if len(colors) >= SC5_CHOP_WINDOW:
            alternating = True
            for i in range(1, len(colors)):
                if colors[i] == colors[i - 1]:
                    alternating = False
                    break
            if alternating:
                result["chop_detected"] = True
                flags.append("SC5_CHOP")
    except Exception as e:
        log.debug("SC-5 error: %s", e)

    # ── Derive composite results ────────────────────────────────────────
    result["flags"] = flags
    skip = result["stale"] or result["doji_cluster"] or result["flat_market"]
    result["should_skip"] = skip
    result["confidence_penalty"] = penalty
    details_parts = []
    if result["stale"]:
        details_parts.append("identical OHLC")
    if result["doji_cluster"]:
        details_parts.append("doji cluster")
    if result["flat_market"]:
        details_parts.append("flat market")
    if result["low_volume"]:
        details_parts.append("low volume")
    if result["chop_detected"]:
        details_parts.append("chop pattern")
    result["details"] = ", ".join(details_parts) if details_parts else "clear"

    return result
