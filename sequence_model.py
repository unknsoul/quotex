"""
Sequence Model — temporal pattern features for meta-model.

Extracts rolling sequence statistics from feature time series,
capturing patterns that tree-based models miss (e.g., momentum shifts,
volatility regime transitions, trend accelerations).

Phase 4 upgrade: adds 8 sequence-aware features to meta-model input.
"""

import numpy as np
import pandas as pd
import logging

log = logging.getLogger("sequence_model")

# Lookback window for sequence features
SEQ_WINDOW = 20


def compute_sequence_features(df, window=SEQ_WINDOW):
    """
    Compute temporal sequence features on the last `window` bars.
    These capture patterns invisible to row-by-row tree models.

    Returns dict of sequence features (for a single prediction).
    """
    if len(df) < window:
        return _empty_features()

    tail = df.iloc[-window:]

    features = {}

    # 1. Trend strength: linear regression slope of close prices
    close = tail["close"].values
    x = np.arange(len(close))
    if np.std(close) > 0:
        slope = np.polyfit(x, close, 1)[0]
        features["seq_trend_slope"] = slope / (np.std(close) + 1e-10)
    else:
        features["seq_trend_slope"] = 0.0

    # 2. Momentum change: acceleration of returns
    if "return_last_5" in tail.columns:
        rets = tail["return_last_5"].values
        features["seq_momentum_accel"] = float(rets[-5:].mean() - rets[:5].mean())
    else:
        features["seq_momentum_accel"] = 0.0

    # 3. Volatility regime: is volatility expanding or contracting?
    if "atr_14" in tail.columns:
        atr = tail["atr_14"].values
        features["seq_vol_trend"] = float(atr[-5:].mean() / (atr[:5].mean() + 1e-10) - 1.0)
    else:
        features["seq_vol_trend"] = 0.0

    # 4. Candle pattern consistency: are recent candles consistent direction?
    if "candle_direction" in tail.columns:
        dirs = tail["candle_direction"].values[-10:]
        features["seq_direction_consistency"] = float(abs(dirs.mean() - 0.5) * 2)
    else:
        features["seq_direction_consistency"] = 0.0

    # 5. RSI momentum shift
    if "rsi_14" in tail.columns:
        rsi = tail["rsi_14"].values
        features["seq_rsi_shift"] = float(rsi[-5:].mean() - rsi[:5].mean()) / 100
    else:
        features["seq_rsi_shift"] = 0.0

    # 6. Volume pattern: is volume increasing with trend?
    if "tick_volume_zscore" in tail.columns:
        vol_z = tail["tick_volume_zscore"].values[-5:]
        features["seq_volume_trend"] = float(vol_z.mean())
    else:
        features["seq_volume_trend"] = 0.0

    # 7. EMA convergence/divergence speed
    if "ema_20" in tail.columns and "ema_50" in tail.columns:
        gap = (tail["ema_20"] - tail["ema_50"]).values
        features["seq_ema_convergence"] = float(gap[-1] - gap[0]) / (abs(gap[0]) + 1e-10)
    else:
        features["seq_ema_convergence"] = 0.0

    # 8. Support/resistance proximity
    if "range_position" in tail.columns:
        rp = tail["range_position"].values[-1]
        # Near extremes = more predictable
        features["seq_range_extremity"] = float(abs(rp - 0.5) * 2)
    else:
        features["seq_range_extremity"] = 0.0

    return features


def _empty_features():
    """Return zero-valued sequence features."""
    return {
        "seq_trend_slope": 0.0,
        "seq_momentum_accel": 0.0,
        "seq_vol_trend": 0.0,
        "seq_direction_consistency": 0.0,
        "seq_rsi_shift": 0.0,
        "seq_volume_trend": 0.0,
        "seq_ema_convergence": 0.0,
        "seq_range_extremity": 0.0,
    }


SEQUENCE_FEATURE_NAMES = list(_empty_features().keys())


def compute_sequence_features_batch(df, window=SEQ_WINDOW):
    """
    Compute sequence features for all rows in a DataFrame (for training).
    Returns a DataFrame with sequence features aligned to input index.
    """
    records = []
    for i in range(len(df)):
        if i < window:
            records.append(_empty_features())
        else:
            records.append(compute_sequence_features(df.iloc[:i + 1], window))

    seq_df = pd.DataFrame(records, index=df.index)
    return seq_df
