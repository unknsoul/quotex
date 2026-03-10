# -*- coding: utf-8 -*-
# regime_classifier.py — Market regime detection for model routing
# Called at the start of every candle prediction cycle.
# Output used by: dynamic_ensemble, Gate 4, predict_engine model selector.

import numpy as np
import pandas as pd
from dataclasses import dataclass

# ── Config ──────────────────────────────────────────────────────────────────
ADX_TREND_THRESHOLD  = 25.0   # ADX > 25 = trending
ADX_RANGE_THRESHOLD  = 20.0   # ADX < 20 = ranging
ADX_CHOPPY_THRESHOLD = 15.0   # ADX < 15 + alternating candles = choppy
ATR_HIGH_VOL_PCT     = 80.0   # ATR percentile > 80 = high volatility
ATR_LOW_VOL_PCT      = 30.0   # ATR percentile < 30 = low volatility
REGIME_LOOKBACK      = 500    # bars for percentile computation
TRANSITION_LOOKBACK  = 3      # if regime changed in last N bars = TRANSITION

@dataclass
class RegimeResult:
    regime:     str    # TREND | RANGE | HIGH_VOL | CHOPPY | TRANSITION
    confidence: float  # 0.0–1.0 (used in Gate 4)
    adx:        float
    atr_pct:    float  # ATR percentile vs lookback
    bb_width:   float  # Bollinger band width percentile
    return_autocorr: float


def compute_adx(high, low, close, period=14) -> float:
    """True Strength Index-based ADX."""
    n = len(close)
    if n < period + 2: return 20.0
    tr   = np.maximum(high[1:]-low[1:], np.maximum(abs(high[1:]-close[:-1]), abs(low[1:]-close[:-1])))
    plus = np.maximum(high[1:]-high[:-1], 0)
    minus= np.maximum(low[:-1]-low[1:],  0)
    smooth = lambda x: pd.Series(x).ewm(span=period, adjust=False).mean().values
    dip   = smooth(plus)  / (smooth(tr) + 1e-10)
    dim   = smooth(minus) / (smooth(tr) + 1e-10)
    adx_v = pd.Series(abs(dip-dim)/(dip+dim+1e-10)).ewm(span=period, adjust=False).mean().values
    return float(adx_v[-1] * 100)


def _detect_chop(df: pd.DataFrame, window: int = 6) -> bool:
    """
    Detect choppy market: alternating candle colors (BGBGBG or GBGBGB).
    Returns True if recent candles alternate direction frequently.
    """
    if df is None or len(df) < window:
        return False
    try:
        recent = df.tail(window)
        colors = (recent["close"].values > recent["open"].values)
        alternations = sum(1 for i in range(1, len(colors)) if colors[i] != colors[i - 1])
        # If 80%+ of transitions are alternating, it's choppy
        return alternations >= (window - 2)
    except Exception:
        return False


def classify_regime(df: pd.DataFrame) -> RegimeResult:
    """
    Classify current market regime from last N bars of M5 data.
    df: DataFrame with columns open, high, low, close
    Returns RegimeResult.
    """
    n = min(len(df), REGIME_LOOKBACK)
    h = df["high"].values[-n:]
    l = df["low"].values[-n:]
    c = df["close"].values[-n:]

    # ADX
    adx = compute_adx(h, l, c)

    # ATR percentile
    atr_series = pd.Series(h - l).rolling(14).mean().dropna().values
    if len(atr_series) == 0:
        return RegimeResult(regime="TRANSITION", confidence=0.45,
                            adx=adx, atr_pct=50.0, bb_width=50.0,
                            return_autocorr=0.0)
    current_atr = float(atr_series[-1])
    atr_pct_rank = float(np.mean(atr_series <= current_atr) * 100)

    # Bollinger band width percentile
    sma = pd.Series(c).rolling(20).mean()
    std = pd.Series(c).rolling(20).std()
    bb_width_series = (std * 4 / sma).dropna().values
    if len(bb_width_series) == 0:
        bb_pct = 50.0
    else:
        bb_pct = float(np.mean(bb_width_series <= bb_width_series[-1]) * 100)

    # Return autocorrelation (momentum persistence)
    returns = np.diff(c[-100:])
    autocorr = float(pd.Series(returns).autocorr(lag=1)) if len(returns) >= 20 else 0.0
    if np.isnan(autocorr):
        autocorr = 0.0

    # Classification logic
    if atr_pct_rank >= ATR_HIGH_VOL_PCT:
        regime = "HIGH_VOL"
        confidence = min(1.0, (atr_pct_rank - ATR_HIGH_VOL_PCT) / 20 + 0.6)
    elif adx >= ADX_TREND_THRESHOLD and autocorr > 0.05:
        regime = "TREND"
        confidence = min(1.0, (adx - ADX_TREND_THRESHOLD) / 20 + 0.65)
    elif adx <= ADX_RANGE_THRESHOLD and atr_pct_rank < 50:
        regime = "RANGE"
        confidence = min(1.0, (ADX_RANGE_THRESHOLD - adx) / 10 + 0.60)
    elif adx < ADX_CHOPPY_THRESHOLD and _detect_chop(df):
        regime = "CHOPPY"
        confidence = min(1.0, (ADX_CHOPPY_THRESHOLD - adx) / 10 + 0.55)
    else:
        regime = "TRANSITION"
        confidence = 0.45

    return RegimeResult(
        regime=regime, confidence=confidence,
        adx=adx, atr_pct=atr_pct_rank,
        bb_width=bb_pct, return_autocorr=autocorr
    )


def classify_regime_series(df: pd.DataFrame, lookback=500) -> list:
    """Classify regime for every bar in the DataFrame."""
    regimes = []
    for i in range(len(df)):
        start = max(0, i - lookback)
        if i < 50:
            regimes.append(RegimeResult(
                regime="TRANSITION", confidence=0.45,
                adx=20.0, atr_pct=50.0, bb_width=50.0,
                return_autocorr=0.0
            ))
        else:
            regimes.append(classify_regime(df.iloc[start:i+1]))
    return regimes
