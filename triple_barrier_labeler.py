"""
Triple Barrier Labeler — V3 Layer 4: Industry-standard labeling with sample weights.

For each bar, checks which barrier is hit first:
  - Take-profit: high[t+k] >= close[t] + tp_mult * ATR → label = 1
  - Stop-loss:   low[t+k]  <= close[t] - sl_mult * ATR → label = 0
  - Time barrier: k > max_bars → label based on return sign (not NaN)

Key difference from V2: time barrier assigns label based on return direction
(not NaN), and produces tb_weight for sample weighting (higher weight for
clear TP/SL hits, lower for time barrier exits).
"""

import numpy as np
import pandas as pd
import logging

log = logging.getLogger("triple_barrier")

# Defaults from V3 spec
DEFAULT_TP_MULT = 1.5
DEFAULT_SL_MULT = 1.0
DEFAULT_MAX_BARS = 4


def label_triple_barrier(df, tp_mult=DEFAULT_TP_MULT, sl_mult=DEFAULT_SL_MULT,
                          max_bars=DEFAULT_MAX_BARS):
    """
    Apply Triple Barrier labeling to DataFrame.
    
    Returns DataFrame with added columns:
      - target: 1 (TP hit) or 0 (SL hit or negative return at timeout)
      - tb_weight: sample weight (1.0 for TP/SL, 0.5 for time barrier)
      - tb_type: 'TP', 'SL', or 'TIME' (which barrier was hit)
    """
    df = df.copy()
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    
    # Use raw ATR (not normalized) for barrier levels
    if "atr_14" in df.columns:
        atr = df["atr_14"].values
        # Check if ATR is normalized (very small values < 0.01 means it's pct of price)
        if np.nanmedian(atr) < 0.01:
            atr = atr * close  # Convert back to absolute ATR
    else:
        # Compute ATR manually if not available
        tr = np.maximum(high - low,
                        np.maximum(np.abs(high - np.roll(close, 1)),
                                   np.abs(low - np.roll(close, 1))))
        atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    
    n = len(df)
    targets = np.full(n, np.nan)
    weights = np.full(n, 0.0)
    barrier_types = [""] * n
    
    for i in range(n - max_bars):
        entry = close[i]
        barrier_atr = atr[i]
        if np.isnan(barrier_atr) or barrier_atr <= 0:
            continue
        
        tp_level = entry + tp_mult * barrier_atr
        sl_level = entry - sl_mult * barrier_atr
        
        hit = False
        for k in range(1, max_bars + 1):
            j = i + k
            if j >= n:
                break
            
            # Check TP hit first (bullish)
            if high[j] >= tp_level:
                targets[i] = 1.0
                weights[i] = 1.0
                barrier_types[i] = "TP"
                hit = True
                break
            
            # Check SL hit (bearish)
            if low[j] <= sl_level:
                targets[i] = 0.0
                weights[i] = 1.0
                barrier_types[i] = "SL"
                hit = True
                break
        
        # Time barrier — assign label based on return direction
        if not hit:
            end_idx = min(i + max_bars, n - 1)
            ret = close[end_idx] - entry
            if ret > 0:
                targets[i] = 1.0
            else:
                targets[i] = 0.0
            weights[i] = 0.5  # Lower weight for ambiguous exits
            barrier_types[i] = "TIME"
    
    df["target"] = targets
    df["tb_weight"] = weights
    df["tb_type"] = barrier_types
    
    valid = np.sum(~np.isnan(targets))
    tp_count = int(np.nansum(targets[~np.isnan(targets)] == 1))
    sl_count = int(np.nansum(targets[~np.isnan(targets)] == 0))
    time_count = sum(1 for t in barrier_types if t == "TIME")
    
    log.info("Triple Barrier: %d labeled (TP=%d, SL=%d, TIME=%d), "
             "tp=%.1fxATR, sl=%.1fxATR, max=%d bars",
             valid, tp_count, sl_count, time_count, tp_mult, sl_mult, max_bars)
    
    return df
