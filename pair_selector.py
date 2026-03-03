"""
Pair Selector — V3 Layer 14: Monthly 6-metric pair scoring.

Scores currency pairs on 6 criteria (5 points each, max 30):
  1. Hurst exponent (mean-reversion vs trending)
  2. Spread-to-ATR ratio (tradability)
  3. ATR stability (consistent volatility)
  4. Momentum persistence (autocorrelation)
  5. Liquidity score (tick volume consistency)
  6. Structure quality (BOS/pivot clarity)

Pairs scoring >= 22/30 are recommended for trading.
Re-run monthly to adapt to changing market conditions.
"""

import numpy as np
import pandas as pd
import logging

log = logging.getLogger("pair_selector")

MIN_SCORE = 22  # Minimum score to recommend pair


def compute_hurst(series, max_lag=100):
    """Compute Hurst exponent using R/S analysis."""
    n = len(series)
    if n < max_lag * 2:
        return 0.5
    
    lags = range(10, min(max_lag, n // 4))
    rs = []
    
    for lag in lags:
        subseries = [series[i:i + lag] for i in range(0, n - lag, lag)]
        rs_vals = []
        for s in subseries:
            if len(s) < 2:
                continue
            mean = np.mean(s)
            cumdev = np.cumsum(s - mean)
            r = np.max(cumdev) - np.min(cumdev)
            std = np.std(s)
            if std > 0:
                rs_vals.append(r / std)
        if rs_vals:
            rs.append((np.log(lag), np.log(np.mean(rs_vals))))
    
    if len(rs) < 3:
        return 0.5
    
    x = np.array([r[0] for r in rs])
    y = np.array([r[1] for r in rs])
    slope = np.polyfit(x, y, 1)[0]
    return float(np.clip(slope, 0, 1))


def score_pair(df, symbol="UNKNOWN"):
    """
    Score a pair on 6 metrics (0-5 each, max 30).
    
    Args:
        df: DataFrame with OHLCV + atr_14 + tick_volume columns
    
    Returns:
        dict with individual scores and total
    """
    close = df["close"].values
    returns = np.diff(close) / close[:-1]
    
    scores = {}
    
    # 1. Hurst exponent (0.4-0.6 = random, <0.4 = mean-reverting, >0.6 = trending)
    hurst = compute_hurst(returns)
    if hurst > 0.55:  # Trending = good for momentum
        scores["hurst"] = min(5, int((hurst - 0.5) * 50))
    elif hurst < 0.45:  # Mean-reverting = good for range
        scores["hurst"] = min(5, int((0.5 - hurst) * 50))
    else:
        scores["hurst"] = 2  # Random = harder
    
    # 2. Spread-to-ATR ratio (lower = better tradability)
    if "atr_14" in df.columns:
        atr_mean = df["atr_14"].mean()
        spread_atr = 0.0002 / (atr_mean + 1e-10)  # Approximate spread
        if spread_atr < 0.1:
            scores["spread_ratio"] = 5
        elif spread_atr < 0.2:
            scores["spread_ratio"] = 4
        elif spread_atr < 0.3:
            scores["spread_ratio"] = 3
        else:
            scores["spread_ratio"] = 2
    else:
        scores["spread_ratio"] = 3
    
    # 3. ATR stability (coefficient of variation)
    if "atr_14" in df.columns:
        atr_cv = df["atr_14"].std() / (df["atr_14"].mean() + 1e-10)
        if atr_cv < 0.3:
            scores["atr_stability"] = 5
        elif atr_cv < 0.5:
            scores["atr_stability"] = 4
        elif atr_cv < 0.7:
            scores["atr_stability"] = 3
        else:
            scores["atr_stability"] = 2
    else:
        scores["atr_stability"] = 3
    
    # 4. Momentum persistence (lag-1 return autocorrelation)
    autocorr = float(pd.Series(returns).autocorr(lag=1))
    if np.isnan(autocorr):
        autocorr = 0
    persistence = abs(autocorr)
    scores["momentum"] = min(5, int(persistence * 25) + 2)
    
    # 5. Liquidity score (tick volume consistency)
    if "tick_volume" in df.columns:
        tv = df["tick_volume"].values
        tv_cv = np.std(tv) / (np.mean(tv) + 1e-10)
        if tv_cv < 0.5:
            scores["liquidity"] = 5
        elif tv_cv < 1.0:
            scores["liquidity"] = 4
        else:
            scores["liquidity"] = 3
    else:
        scores["liquidity"] = 3
    
    # 6. Structure quality (range-to-body ratio)
    if "body_size" in df.columns:
        avg_body = df["body_size"].mean()
        scores["structure"] = min(5, int(avg_body * 10) + 2)
    else:
        scores["structure"] = 3
    
    total = sum(scores.values())
    recommended = total >= MIN_SCORE
    
    log.info("Pair %s: score=%d/30 [%s] %s",
             symbol, total, "RECOMMENDED" if recommended else "SKIP",
             scores)
    
    return {
        "symbol": symbol,
        "scores": scores,
        "total": total,
        "max": 30,
        "recommended": recommended,
        "hurst": hurst,
    }
