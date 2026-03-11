# -*- coding: utf-8 -*-
"""
Signal Consensus Engine v18 — 55+ indicator-based directional signals.

Each signal reads the last row of computed features and votes BUY (+1) or SELL (-1).
The consensus score is used alongside the ML ensemble to determine final direction.
Validation checks stay active but never block signals — they adjust confidence only.
"""

import logging
import numpy as np

log = logging.getLogger("signal_consensus")


def _safe_get(row, col, default=0.0):
    """Safely get a feature value from the last row."""
    try:
        v = row.get(col, default) if hasattr(row, 'get') else getattr(row, col, default)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return float(v)
    except Exception:
        return default


def compute_all_signals(row):
    """
    Compute 55+ directional signals from the feature row.
    Each signal returns +1 (BUY) or -1 (SELL).
    Returns dict with signal names, votes, and summary.
    """
    signals = {}

    # ─── TREND SIGNALS (EMA/SMA crossovers & position) ───

    # 1. EMA 5 vs EMA 20 crossover
    ema5 = _safe_get(row, "ema_5", 0.0)
    ema20 = _safe_get(row, "ema_20", 0.0)
    signals["ema5_vs_ema20"] = 1 if ema5 > 0 and ema20 < ema5 else (-1 if ema5 < 0 else 0)

    # 2. EMA 8 vs EMA 13 (fast scalp cross)
    ema8 = _safe_get(row, "ema_8", 0.0)
    ema13 = _safe_get(row, "ema_13", 0.0)
    signals["ema8_vs_ema13"] = 1 if ema8 > ema13 else -1

    # 3. EMA 20 vs EMA 50
    ema50_dist = _safe_get(row, "ema_50", 0.0)
    signals["ema20_vs_ema50"] = 1 if ema20 > ema50_dist else -1

    # 4. EMA 50 vs EMA 200 (golden/death cross)
    ema200 = _safe_get(row, "ema_200", 0.0)
    signals["ema50_vs_ema200"] = 1 if ema50_dist > ema200 else -1

    # 5. Price above EMA 20 (close > ema20 means ema_20 feature is positive)
    signals["price_above_ema20"] = 1 if ema20 > 0 else -1

    # 6. Price above EMA 50
    signals["price_above_ema50"] = 1 if ema50_dist > 0 else -1

    # 7. Price above EMA 200
    signals["price_above_ema200"] = 1 if ema200 > 0 else -1

    # 8. SMA 10 vs SMA 30
    sma10 = _safe_get(row, "sma_10", 0.0)
    sma30 = _safe_get(row, "sma_30", 0.0)
    signals["sma10_vs_sma30"] = 1 if sma10 > sma30 else -1

    # 9. DEMA 20 direction
    dema20 = _safe_get(row, "dema_20", 0.0)
    signals["dema20_direction"] = 1 if dema20 > 0 else -1

    # 10. TEMA 20 direction
    tema20 = _safe_get(row, "tema_20", 0.0)
    signals["tema20_direction"] = 1 if tema20 > 0 else -1

    # 11. WMA 20 direction
    wma20 = _safe_get(row, "wma_20", 0.0)
    signals["wma20_direction"] = 1 if wma20 > 0 else -1

    # 12. EMA slope (trend direction)
    ema_slope = _safe_get(row, "ema_slope", 0.0)
    signals["ema_slope_direction"] = 1 if ema_slope > 0 else -1

    # 13. Linear regression slope
    linreg_slope = _safe_get(row, "linreg_slope_20", 0.0)
    signals["linreg_slope"] = 1 if linreg_slope > 0 else -1

    # ─── MOMENTUM SIGNALS ───

    # 14. RSI 14 mean reversion
    rsi14 = _safe_get(row, "rsi_14", 50.0)
    if rsi14 < 30:
        signals["rsi14_signal"] = 1   # Oversold → BUY
    elif rsi14 > 70:
        signals["rsi14_signal"] = -1  # Overbought → SELL
    else:
        signals["rsi14_signal"] = 1 if rsi14 > 50 else -1

    # 15. RSI 7 (fast)
    rsi7 = _safe_get(row, "rsi_7", 50.0)
    if rsi7 < 25:
        signals["rsi7_signal"] = 1
    elif rsi7 > 75:
        signals["rsi7_signal"] = -1
    else:
        signals["rsi7_signal"] = 1 if rsi7 > 50 else -1

    # 16. RSI 21 (slow)
    rsi21 = _safe_get(row, "rsi_21", 50.0)
    signals["rsi21_signal"] = 1 if rsi21 > 50 else -1

    # 17. Stochastic K vs D
    stoch_k = _safe_get(row, "stoch_k", 50.0)
    stoch_d = _safe_get(row, "stoch_d", 50.0)
    signals["stoch_crossover"] = 1 if stoch_k > stoch_d else -1

    # 18. Stochastic oversold/overbought
    if stoch_k < 20:
        signals["stoch_extreme"] = 1   # Oversold
    elif stoch_k > 80:
        signals["stoch_extreme"] = -1  # Overbought
    else:
        signals["stoch_extreme"] = 1 if stoch_k > 50 else -1

    # 19. MACD above signal
    macd = _safe_get(row, "macd", 0.0)
    macd_signal = _safe_get(row, "macd_signal", 0.0)
    signals["macd_crossover"] = 1 if macd > macd_signal else -1

    # 20. MACD histogram
    macd_hist = _safe_get(row, "macd_histogram", 0.0)
    signals["macd_histogram"] = 1 if macd_hist > 0 else -1

    # 21. MACD histogram acceleration
    macd_hist_accel = _safe_get(row, "macd_histogram_accel", 0.0)
    signals["macd_hist_accel"] = 1 if macd_hist_accel > 0 else -1

    # 22. ROC 10 (rate of change)
    roc10 = _safe_get(row, "roc_10", 0.0)
    signals["roc10_direction"] = 1 if roc10 > 0 else -1

    # 23. ROC 20
    roc20 = _safe_get(row, "roc_20", 0.0)
    signals["roc20_direction"] = 1 if roc20 > 0 else -1

    # 24. Momentum acceleration
    mom_accel = _safe_get(row, "momentum_acceleration", 0.0)
    signals["momentum_accel"] = 1 if mom_accel > 0 else -1

    # 25. RSI velocity (rising or falling)
    rsi_vel = _safe_get(row, "rsi_velocity", 0.0)
    signals["rsi_velocity"] = 1 if rsi_vel > 0 else -1

    # 26. Stochastic RSI
    stoch_rsi = _safe_get(row, "stoch_rsi", 0.5)
    signals["stoch_rsi_signal"] = 1 if stoch_rsi > 0.5 else -1

    # ─── OSCILLATOR SIGNALS ───

    # 27. CCI 20
    cci = _safe_get(row, "cci_20", 0.0)
    if cci < -100:
        signals["cci_signal"] = 1   # Oversold
    elif cci > 100:
        signals["cci_signal"] = -1  # Overbought
    else:
        signals["cci_signal"] = 1 if cci > 0 else -1

    # 28. Williams %R
    williams = _safe_get(row, "williams_r", -50.0)
    if williams < -80:
        signals["williams_r_signal"] = 1   # Oversold
    elif williams > -20:
        signals["williams_r_signal"] = -1  # Overbought
    else:
        signals["williams_r_signal"] = 1 if williams > -50 else -1

    # 29. Ultimate Oscillator
    ult_osc = _safe_get(row, "ultimate_oscillator", 0.5)
    signals["ultimate_osc_signal"] = 1 if ult_osc > 0.5 else -1

    # 30. TSI (True Strength Index)
    tsi = _safe_get(row, "tsi", 0.0)
    signals["tsi_signal"] = 1 if tsi > 0 else -1

    # 31. Chande Momentum Oscillator
    chande = _safe_get(row, "chande_momentum", 0.0)
    signals["chande_signal"] = 1 if chande > 0 else -1

    # 32. PPO (Percentage Price Oscillator)
    ppo = _safe_get(row, "ppo", 0.0)
    signals["ppo_signal"] = 1 if ppo > 0 else -1

    # 33. Awesome Oscillator
    ao = _safe_get(row, "awesome_oscillator", 0.0)
    signals["awesome_osc_signal"] = 1 if ao > 0 else -1

    # 34. RSI Bollinger
    rsi_bb = _safe_get(row, "rsi_bollinger", 0.5)
    signals["rsi_bollinger_signal"] = 1 if rsi_bb > 0.5 else -1

    # ─── DIRECTIONAL / TREND STRENGTH SIGNALS ───

    # 35. +DI vs -DI
    plus_di = _safe_get(row, "plus_di", 0.0)
    minus_di = _safe_get(row, "minus_di", 0.0)
    signals["di_crossover"] = 1 if plus_di > minus_di else -1

    # 36. ADX + DI direction (strong trend with direction)
    adx = _safe_get(row, "adx", 0.0)
    if adx > 25:
        signals["adx_trend_dir"] = 1 if plus_di > minus_di else -1
    else:
        signals["adx_trend_dir"] = 0  # No clear trend

    # 37. Parabolic SAR direction
    sar_dir = _safe_get(row, "parabolic_sar_dir", 0.5)
    signals["parabolic_sar"] = 1 if sar_dir > 0.5 else -1

    # 38. SuperTrend direction
    supertrend = _safe_get(row, "supertrend_dir", 0.5)
    signals["supertrend"] = 1 if supertrend > 0.5 else -1

    # 39. Aroon oscillator
    aroon_osc = _safe_get(row, "aroon_osc", 0.0)
    signals["aroon_signal"] = 1 if aroon_osc > 0 else -1

    # 40. Vortex indicator
    vortex = _safe_get(row, "vortex_diff", 0.0)
    signals["vortex_signal"] = 1 if vortex > 0 else -1

    # 41. Ichimoku TK cross
    ichimoku_tkx = _safe_get(row, "ichimoku_tkx", 0.0)
    signals["ichimoku_tk_cross"] = 1 if ichimoku_tkx > 0 else -1

    # 42. Ichimoku cloud position
    ichimoku_cloud = _safe_get(row, "ichimoku_cloud_dist", 0.0)
    signals["ichimoku_cloud_dir"] = 1 if ichimoku_cloud > 0 else -1

    # ─── VOLUME / FLOW SIGNALS ───

    # 43. MFI (Money Flow Index)
    mfi = _safe_get(row, "mfi_14", 50.0)
    if mfi < 20:
        signals["mfi_signal"] = 1   # Oversold
    elif mfi > 80:
        signals["mfi_signal"] = -1  # Overbought
    else:
        signals["mfi_signal"] = 1 if mfi > 50 else -1

    # 44. Chaikin Money Flow
    cmf = _safe_get(row, "cmf_20", 0.0)
    signals["cmf_signal"] = 1 if cmf > 0 else -1

    # 45. Force Index
    force = _safe_get(row, "force_index_13", 0.0)
    signals["force_index_signal"] = 1 if force > 0 else -1

    # 46. OBV slope
    obv_slope = _safe_get(row, "obv_slope_10", 0.0)
    signals["obv_slope_signal"] = 1 if obv_slope > 0 else -1

    # 47. Cumulative delta slope
    cum_delta_slope = _safe_get(row, "cumulative_delta_slope", 0.0)
    signals["cum_delta_slope"] = 1 if cum_delta_slope > 0 else -1

    # 48. Buy pressure ratio
    buy_pressure = _safe_get(row, "buy_pressure_ratio", 0.5)
    signals["buy_pressure"] = 1 if buy_pressure > 0.5 else -1

    # 49. Volume momentum confirm
    vol_mom = _safe_get(row, "volume_momentum_confirm", 0.0)
    signals["volume_momentum"] = 1 if vol_mom > 0 else -1

    # ─── CHANNEL / BAND SIGNALS ───

    # 50. Bollinger Band position
    bb_pos = _safe_get(row, "bb_position", 0.5)
    if bb_pos < 0.2:
        signals["bb_position_signal"] = 1   # Near lower band → BUY
    elif bb_pos > 0.8:
        signals["bb_position_signal"] = -1  # Near upper band → SELL
    else:
        signals["bb_position_signal"] = 1 if bb_pos > 0.5 else -1

    # 51. Keltner Channel position
    keltner = _safe_get(row, "keltner_position", 0.5)
    signals["keltner_signal"] = 1 if keltner > 0.5 else -1

    # 52. Donchian Channel position
    donchian = _safe_get(row, "donchian_position", 0.5)
    signals["donchian_signal"] = 1 if donchian > 0.5 else -1

    # ─── CANDLE PATTERN SIGNALS ───

    # 53. Candle direction
    candle_dir = _safe_get(row, "candle_direction", 0.0)
    signals["candle_direction"] = 1 if candle_dir > 0 else -1

    # 54. Three candle momentum
    three_cm = _safe_get(row, "three_candle_momentum", 0.0)
    signals["three_candle_mom"] = 1 if three_cm > 0 else -1

    # 55. Heiken Ashi direction
    ha_dir = _safe_get(row, "ha_direction", 0.0)
    signals["heiken_ashi_dir"] = 1 if ha_dir > 0 else -1

    # 56. Engulfing pattern
    engulfing = _safe_get(row, "engulfing_score", 0.0)
    signals["engulfing_signal"] = 1 if engulfing > 0 else (-1 if engulfing < 0 else 0)

    # 57. Pin bar score
    pin_bar = _safe_get(row, "pin_bar_score", 0.0)
    signals["pin_bar_signal"] = 1 if pin_bar > 0 else (-1 if pin_bar < 0 else 0)

    # ─── STRUCTURE / PRESSURE SIGNALS ───

    # 58. Swing structure
    swing = _safe_get(row, "swing_structure", 0.0)
    signals["swing_structure"] = 1 if swing > 0 else -1

    # 59. Directional pressure
    dir_pressure = _safe_get(row, "directional_pressure", 0.0)
    signals["directional_pressure"] = 1 if dir_pressure > 0 else -1

    # 60. Bull power vs Bear power (Elder Ray)
    bull_pwr = _safe_get(row, "bull_power", 0.0)
    bear_pwr = _safe_get(row, "bear_power", 0.0)
    signals["elder_ray"] = 1 if bull_pwr > abs(bear_pwr) else -1

    # 61. Micro trend 3
    micro = _safe_get(row, "micro_trend_3", 0.0)
    signals["micro_trend"] = 1 if micro > 0 else -1

    # 62. Imbalance 10
    imbalance = _safe_get(row, "imbalance_10", 0.0)
    signals["imbalance_signal"] = 1 if imbalance > 0 else -1

    # ─── MULTI-TIMEFRAME SIGNALS ───

    # 63. H1 trend direction
    h1_trend = _safe_get(row, "h1_trend_direction", 0.0)
    signals["h1_trend"] = 1 if h1_trend > 0 else -1

    # 64. H1 EMA alignment
    h1_ema = _safe_get(row, "h1_ema_alignment", 0.0)
    signals["h1_ema_align"] = 1 if h1_ema > 0 else -1

    # 65. M15 momentum
    m15_mom = _safe_get(row, "m15_momentum", 0.0)
    signals["m15_momentum"] = 1 if m15_mom > 0 else -1

    # 66. TF agreement
    tf_agree = _safe_get(row, "tf_agreement", 0.0)
    signals["tf_agreement"] = 1 if tf_agree > 0 else -1

    # Filter out zero votes (neutral/unavailable)
    active = {k: v for k, v in signals.items() if v != 0}
    total = len(active)
    buy_count = sum(1 for v in active.values() if v > 0)
    sell_count = sum(1 for v in active.values() if v < 0)
    net_score = sum(active.values())  # positive = BUY bias, negative = SELL bias

    consensus_dir = "BUY" if net_score >= 0 else "SELL"
    strength = abs(net_score) / max(total, 1)  # 0.0 to 1.0

    return {
        "signals": signals,
        "active_count": total,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "net_score": net_score,
        "consensus_direction": consensus_dir,
        "consensus_strength": round(strength, 4),
    }


def get_consensus_vote(row, ml_direction="BUY", ml_green_p=0.5):
    """
    Combine indicator consensus with ML prediction.
    
    Strategy:
    - ML ensemble gets 40% weight (it's trained on all features)
    - Indicator consensus gets 60% weight (direct signal reading)
    - If both agree → high confidence in that direction
    - If they disagree → go with the majority indicator vote
    - Always returns BUY or SELL, never HOLD
    
    Returns: (direction, confidence_boost, consensus_info)
    """
    consensus = compute_all_signals(row)

    ml_vote = 1 if ml_direction == "BUY" else -1
    consensus_vote = 1 if consensus["consensus_direction"] == "BUY" else -1

    # Weighted combination: ML=40%, Indicators=60%
    ml_weight = 0.40
    indicator_weight = 0.60

    # ML score: distance from 0.5 scaled to -1..+1
    ml_score = (ml_green_p - 0.5) * 2.0  # -1 to +1

    # Indicator score: net normalized
    total = max(consensus["active_count"], 1)
    indicator_score = consensus["net_score"] / total  # -1 to +1

    combined = ml_weight * ml_score + indicator_weight * indicator_score

    final_dir = "BUY" if combined >= 0 else "SELL"

    # Agreement bonus: if ML and indicators agree, boost confidence
    agree = (ml_vote == consensus_vote)
    if agree:
        confidence_boost = 1.0 + consensus["consensus_strength"] * 0.10  # up to 10% boost
    else:
        confidence_boost = 1.0 - consensus["consensus_strength"] * 0.05  # up to 5% reduction

    return final_dir, confidence_boost, consensus
