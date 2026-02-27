"""
Feature Engineering v7 — volume imbalance + TF agreement + pattern sequences.

v7 adds: volume imbalance, multi-TF agreement, 5-bar patterns,
regime acceleration, doji count, consecutive wicks.
Total: 66 features.
"""

import numpy as np
import pandas as pd
import logging

from config import (
    EMA_20, EMA_50, EMA_100, EMA_200, EMA_SLOPE_WINDOW,
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    STOCH_K_PERIOD, STOCH_D_PERIOD, BB_PERIOD, BB_STD,
    ATR_PERIOD, ADX_PERIOD, ROLLING_STD_PERIOD,
    VOLATILITY_ROLLING, MOMENTUM_ROLLING, RETURN_LOOKBACK,
    RANGE_POSITION_WINDOW, LIQUIDITY_SWEEP_WINDOW,
    VOLATILITY_ZSCORE_WINDOW, ATR_PERCENTILE_WINDOW,
    SESSION_ASIA, SESSION_LONDON, SESSION_NEW_YORK,
    LOG_LEVEL, LOG_FORMAT,
)

log = logging.getLogger("feature_engineering")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

FEATURE_COLUMNS = [
    # Trend (5)
    "ema_20", "ema_50", "ema_200", "ema_slope", "adx",
    # Momentum (5)
    "rsi_14", "macd", "macd_signal", "stoch_k", "stoch_d",
    # Volatility (4)
    "atr_14", "bb_width", "rolling_std_20", "volatility_zscore",
    # Candle structure (7)
    "body_size", "upper_wick_ratio", "lower_wick_ratio",
    "range_position", "liquidity_sweep",
    "candle_direction", "three_candle_momentum",
    # Context (5)
    "session_flag", "hour_sin", "hour_cos",
    "return_last_5", "volatility_rolling_10",
    "momentum_rolling_5",
    # Multi-timeframe (4)
    "h1_trend_direction", "h1_ema_alignment", "h1_atr", "m15_momentum",
    # Continuous regime (v4) (3)
    "adx_normalized", "ema_slope_magnitude", "atr_percentile_rank",
    # ---- v5: Structural features (12) ----
    # Liquidity sweep detection
    "sweep_high", "sweep_low",
    # Compression breakout
    "compression_ratio",
    # Return distribution shape
    "rolling_skew_20", "rolling_kurt_20",
    # Directional imbalance
    "bullish_count_10", "bearish_count_10", "imbalance_10",
    # Regime transition momentum
    "delta_adx_5", "delta_atr_5",
    # Multi-horizon confirmation
    "return_2bar_momentum", "return_3bar_trend",
    # ---- v6: Microstructure features (8) ----
    # Order flow proxy
    "wick_absorption",
    # Momentum exhaustion
    "momentum_decay",
    # Volume features
    "tick_volume_zscore", "volume_price_divergence",
    # Session interaction
    "session_momentum_interaction", "session_volatility_interaction",
    # Trend × regime
    "ema_adx_interaction",
    # Compression acceleration
    "range_compression_speed",
    # ---- v7: Accuracy upgrades (11) ----
    # Volume imbalance (order flow)
    "volume_imbalance_5", "volume_imbalance_10",
    # Multi-TF agreement
    "tf_agreement", "tf_momentum_strength",
    # Candle pattern sequences
    "five_candle_pattern", "doji_count_5", "consecutive_same_wicks",
    # Regime acceleration
    "delta_adx_accel", "delta_atr_accel",
    # Spread-relative move size
    "move_vs_spread",
    # Hour granularity
    "minute_sin",
]


# =============================================================================
#  Indicator Functions
# =============================================================================

def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close, period):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_g = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_l = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_g / (avg_l + 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(high, low, close, period):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _adx(high, low, close, period):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / (atr_val + 1e-10))
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / (atr_val + 1e-10))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    return dx.ewm(span=period, adjust=False).mean()


def _stochastic(high, low, close, k_period, d_period):
    low_min = low.rolling(k_period).min()
    high_max = high.rolling(k_period).max()
    k = 100 * (close - low_min) / (high_max - low_min + 1e-10)
    d = k.rolling(d_period).mean()
    return k, d


def _engulfing(o, h, l, c):
    po, pc = o.shift(1), c.shift(1)
    bull = (pc < po) & (c > o) & (o <= pc) & (c >= po)
    bear = (pc > po) & (c < o) & (o >= pc) & (c <= po)
    result = pd.Series(0, index=o.index)
    result[bull] = 1
    result[bear] = -1
    return result


def _range_position(close, high, low, window):
    roll_h = high.rolling(window).max()
    roll_l = low.rolling(window).min()
    return (close - roll_l) / (roll_h - roll_l + 1e-10)


def _liquidity_sweep(high, low, close, o, window):
    prev_h = high.shift(1).rolling(window).max()
    prev_l = low.shift(1).rolling(window).min()
    bull = (low < prev_l) & (close > o)
    bear = (high > prev_h) & (close < o)
    result = pd.Series(0, index=close.index)
    result[bull] = -1
    result[bear] = 1
    return result


def _volatility_zscore(atr, window):
    mean = atr.rolling(window).mean()
    std = atr.rolling(window).std()
    return (atr - mean) / (std + 1e-10)


def _atr_percentile_rank(atr, window):
    """Rolling percentile rank of ATR (0-1). Continuous regime strength."""
    return atr.rolling(window).rank(pct=True)


def _session(hour):
    if SESSION_NEW_YORK[0] <= hour < SESSION_NEW_YORK[1]:
        return 2
    if SESSION_LONDON[0] <= hour < SESSION_LONDON[1]:
        return 1
    return 0


# =============================================================================
#  Multi-Timeframe Features
# =============================================================================

def compute_htf_features(m5_df, m15_df=None, h1_df=None):
    df = m5_df.copy()

    if h1_df is not None and len(h1_df) > 0:
        h1 = h1_df.copy()
        h1_c = h1["close"]
        h1["h1_ema20"] = _ema(h1_c, EMA_20)
        h1["h1_ema50"] = _ema(h1_c, EMA_50)
        h1["h1_ema200"] = _ema(h1_c, EMA_200)

        h1["h1_trend_direction"] = 0
        h1.loc[h1["h1_ema20"] > h1["h1_ema50"], "h1_trend_direction"] = 1
        h1.loc[h1["h1_ema20"] < h1["h1_ema50"], "h1_trend_direction"] = -1

        h1["h1_ema_alignment"] = 0
        bull = (h1["h1_ema20"] > h1["h1_ema50"]) & (h1["h1_ema50"] > h1["h1_ema200"])
        bear = (h1["h1_ema20"] < h1["h1_ema50"]) & (h1["h1_ema50"] < h1["h1_ema200"])
        h1.loc[bull, "h1_ema_alignment"] = 1
        h1.loc[bear, "h1_ema_alignment"] = -1

        h1["h1_atr"] = _atr(h1["high"], h1["low"], h1["close"], ATR_PERIOD)

        h1_merge = h1[["time", "h1_trend_direction", "h1_ema_alignment", "h1_atr"]].copy()
        h1_merge = h1_merge.sort_values("time")
        df = df.sort_values("time")
        df = pd.merge_asof(df, h1_merge, on="time", direction="backward")
    else:
        df["h1_trend_direction"] = 0
        df["h1_ema_alignment"] = 0
        df["h1_atr"] = 0.0

    if m15_df is not None and len(m15_df) > 0:
        m15 = m15_df.copy()
        m15_c = m15["close"]
        m15_rsi = _rsi(m15_c, RSI_PERIOD)
        m15_macd = _ema(m15_c, MACD_FAST) - _ema(m15_c, MACD_SLOW)
        m15_macd_sig = _ema(m15_macd, MACD_SIGNAL)
        m15["m15_momentum"] = ((m15_rsi - 50) / 50) + (m15_macd > m15_macd_sig).astype(float)

        m15_merge = m15[["time", "m15_momentum"]].copy()
        m15_merge = m15_merge.sort_values("time")
        df = pd.merge_asof(df, m15_merge, on="time", direction="backward")
    else:
        df["m15_momentum"] = 0.0

    df["h1_trend_direction"] = df["h1_trend_direction"].fillna(0).astype(int)
    df["h1_ema_alignment"] = df["h1_ema_alignment"].fillna(0).astype(int)
    df["h1_atr"] = df["h1_atr"].fillna(0.0)
    df["m15_momentum"] = df["m15_momentum"].fillna(0.0)
    return df


# =============================================================================
#  Main API
# =============================================================================

def compute_features(df, m15_df=None, h1_df=None):
    df = df.copy()
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    # Trend
    df["ema_20"] = _ema(c, EMA_20)
    df["ema_50"] = _ema(c, EMA_50)
    df["ema_200"] = _ema(c, EMA_200)
    df["ema_slope"] = (df["ema_50"] - df["ema_50"].shift(EMA_SLOPE_WINDOW)) / (
        df["ema_50"].shift(EMA_SLOPE_WINDOW) + 1e-10)
    df["adx"] = _adx(h, l, c, ADX_PERIOD)

    # Momentum
    df["rsi_14"] = _rsi(c, RSI_PERIOD)
    df["macd"] = _ema(c, MACD_FAST) - _ema(c, MACD_SLOW)
    df["macd_signal"] = _ema(df["macd"], MACD_SIGNAL)
    sk, sd = _stochastic(h, l, c, STOCH_K_PERIOD, STOCH_D_PERIOD)
    df["stoch_k"] = sk
    df["stoch_d"] = sd

    # Volatility
    df["atr_14"] = _atr(h, l, c, ATR_PERIOD)
    bb_mid = c.rolling(BB_PERIOD).mean()
    bb_std = c.rolling(BB_PERIOD).std()
    df["bb_width"] = ((bb_mid + BB_STD * bb_std) - (bb_mid - BB_STD * bb_std)) / (bb_mid + 1e-10)
    df["rolling_std_20"] = c.rolling(ROLLING_STD_PERIOD).std()
    df["volatility_zscore"] = _volatility_zscore(df["atr_14"], VOLATILITY_ZSCORE_WINDOW)

    # Candle structure
    candle_range = h - l + 1e-10
    df["body_size"] = (c - o).abs() / candle_range
    df["upper_wick_ratio"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / candle_range
    df["lower_wick_ratio"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / candle_range
    df["range_position"] = _range_position(c, h, l, RANGE_POSITION_WINDOW)
    df["liquidity_sweep"] = _liquidity_sweep(h, l, c, o, LIQUIDITY_SWEEP_WINDOW)
    df["candle_direction"] = (c > o).astype(int)
    d0 = df["candle_direction"]
    d1 = d0.shift(1).fillna(0).astype(int)
    d2 = d0.shift(2).fillna(0).astype(int)
    df["three_candle_momentum"] = d2 * 4 + d1 * 2 + d0

    # Context
    hour = df["time"].dt.hour
    df["session_flag"] = hour.apply(_session)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["return_last_5"] = c.pct_change(RETURN_LOOKBACK)
    df["volatility_rolling_10"] = df["atr_14"].rolling(VOLATILITY_ROLLING).mean() / (c + 1e-10)
    df["momentum_rolling_5"] = c.pct_change(1).rolling(MOMENTUM_ROLLING).mean()

    # Multi-timeframe
    df = compute_htf_features(df, m15_df, h1_df)

    # v4: Continuous regime features (soft, no abrupt switches)
    df["adx_normalized"] = df["adx"] / 100.0
    df["ema_slope_magnitude"] = df["ema_slope"].abs()
    df["atr_percentile_rank"] = _atr_percentile_rank(df["atr_14"], ATR_PERCENTILE_WINDOW)

    # ---- v5: Structural features ----
    returns = c.pct_change()

    # Liquidity sweep detection (separate high/low sweeps)
    roll_high_20 = h.rolling(20).max().shift(1)
    roll_low_20 = l.rolling(20).min().shift(1)
    df["sweep_high"] = ((h > roll_high_20) & (c < h)).astype(int)  # fake breakout up
    df["sweep_low"] = ((l < roll_low_20) & (c > l)).astype(int)    # fake breakout down

    # Compression breakout probability
    range_width = h.rolling(20).max() - l.rolling(20).min()
    range_mean_100 = range_width.rolling(100, min_periods=20).mean()
    df["compression_ratio"] = range_width / (range_mean_100 + 1e-10)

    # Return distribution shape
    df["rolling_skew_20"] = returns.rolling(20, min_periods=10).skew()
    df["rolling_kurt_20"] = returns.rolling(20, min_periods=10).kurt()

    # Directional imbalance
    is_bull = (c > o).astype(int)
    is_bear = (c < o).astype(int)
    df["bullish_count_10"] = is_bull.rolling(10, min_periods=1).sum()
    df["bearish_count_10"] = is_bear.rolling(10, min_periods=1).sum()
    df["imbalance_10"] = df["bullish_count_10"] - df["bearish_count_10"]

    # Regime transition momentum
    df["delta_adx_5"] = df["adx"] - df["adx"].shift(5)
    df["delta_atr_5"] = df["atr_14"] - df["atr_14"].shift(5)

    # Multi-horizon confirmation
    df["return_2bar_momentum"] = returns.rolling(2, min_periods=1).sum()
    df["return_3bar_trend"] = c.pct_change(3)

    # ---- v6: Microstructure features ----

    # Wick absorption ratio (order flow proxy)
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
    df["wick_absorption"] = lower_wick / (upper_wick + lower_wick + 1e-10)

    # Momentum decay rate
    mom_1 = returns.fillna(0)
    mom_3_avg = returns.rolling(3, min_periods=1).mean()
    df["momentum_decay"] = mom_1 / (mom_3_avg + 1e-10)
    df["momentum_decay"] = df["momentum_decay"].clip(-10, 10)  # cap extremes

    # Tick volume features
    if "tick_volume" in df.columns:
        tv = df["tick_volume"].astype(float)
        tv_mean = tv.rolling(20, min_periods=5).mean()
        tv_std = tv.rolling(20, min_periods=5).std()
        df["tick_volume_zscore"] = (tv - tv_mean) / (tv_std + 1e-10)
        # Volume-price divergence: price up + volume down = weakness
        price_dir = (returns > 0).astype(float) * 2 - 1  # +1 or -1
        vol_dir = (tv > tv_mean).astype(float) * 2 - 1    # +1 or -1
        df["volume_price_divergence"] = price_dir * vol_dir  # +1=confirm, -1=diverge
    else:
        df["tick_volume_zscore"] = 0.0
        df["volume_price_divergence"] = 0.0

    # Session interaction features
    session = df["session_flag"]
    df["session_momentum_interaction"] = session * df["momentum_rolling_5"]
    df["session_volatility_interaction"] = session * df["volatility_zscore"]

    # Trend × regime interaction
    df["ema_adx_interaction"] = df["ema_slope"] * df["adx_normalized"]

    # Compression acceleration (how fast is range narrowing)
    df["range_compression_speed"] = df["compression_ratio"] - df["compression_ratio"].shift(5)

    # ---- v7: Accuracy upgrades ----

    # 1. Tick volume imbalance bars (order flow proxy)
    if "tick_volume" in df.columns:
        tv = df["tick_volume"].astype(float)
        buy_vol = tv * (c > o).astype(float)
        sell_vol = tv * (c < o).astype(float)
        buy_sum_5 = buy_vol.rolling(5, min_periods=1).sum()
        sell_sum_5 = sell_vol.rolling(5, min_periods=1).sum()
        df["volume_imbalance_5"] = buy_sum_5 / (sell_sum_5 + 1e-10)
        buy_sum_10 = buy_vol.rolling(10, min_periods=1).sum()
        sell_sum_10 = sell_vol.rolling(10, min_periods=1).sum()
        df["volume_imbalance_10"] = buy_sum_10 / (sell_sum_10 + 1e-10)
    else:
        df["volume_imbalance_5"] = 1.0
        df["volume_imbalance_10"] = 1.0

    # 2. Multi-TF target agreement
    m5_mom = df["momentum_rolling_5"]
    m15_mom = df.get("m15_momentum", pd.Series(0, index=df.index))
    h1_trend = df.get("h1_trend_direction", pd.Series(0, index=df.index))
    m5_sign = (m5_mom > 0).astype(int) * 2 - 1
    m15_sign = (m15_mom > 0).astype(int) * 2 - 1
    # Agreement: +3 = all bullish, -3 = all bearish, mixed = near 0
    df["tf_agreement"] = m5_sign + m15_sign + h1_trend
    df["tf_momentum_strength"] = m5_mom.abs() + m15_mom.abs()

    # 5. Candle pattern sequences
    d0 = df["candle_direction"]
    d1 = d0.shift(1).fillna(0).astype(int)
    d2 = d0.shift(2).fillna(0).astype(int)
    d3 = d0.shift(3).fillna(0).astype(int)
    d4 = d0.shift(4).fillna(0).astype(int)
    df["five_candle_pattern"] = d4 * 16 + d3 * 8 + d2 * 4 + d1 * 2 + d0

    # Doji count (small body candles)
    body_ratio = (c - o).abs() / candle_range
    is_doji = (body_ratio < 0.3).astype(int)
    df["doji_count_5"] = is_doji.rolling(5, min_periods=1).sum()

    # Consecutive wicks in same direction
    wick_dir = (lower_wick > upper_wick).astype(int)  # 1=demand, 0=supply
    wick_same = (wick_dir == wick_dir.shift(1)).astype(int)
    df["consecutive_same_wicks"] = wick_same.rolling(5, min_periods=1).sum()

    # 6. Regime transition acceleration
    df["delta_adx_accel"] = df["delta_adx_5"] - df["delta_adx_5"].shift(5)
    df["delta_atr_accel"] = df["delta_atr_5"] - df["delta_atr_5"].shift(5)

    # 4. Spread-relative move size (tradeable move filter feature)
    if "spread" in df.columns:
        spread_price = df["spread"].astype(float) * 0.00001  # convert points to price
        avg_move = returns.abs().rolling(20, min_periods=5).mean() * c
        df["move_vs_spread"] = avg_move / (spread_price + 1e-10)
    else:
        df["move_vs_spread"] = 10.0  # default: moves are large relative to spread

    # 3. Hour granularity (minute-level cyclical encoding)
    if "time" in df.columns:
        minute_of_day = df["time"].dt.hour * 60 + df["time"].dt.minute
        df["minute_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    else:
        df["minute_sin"] = 0.0

    # Drop warmup
    warmup = max(EMA_200, BB_PERIOD, ATR_PERIOD, RSI_PERIOD, MACD_SLOW,
                 ADX_PERIOD, STOCH_K_PERIOD, ROLLING_STD_PERIOD,
                 EMA_SLOPE_WINDOW, VOLATILITY_ROLLING, MOMENTUM_ROLLING,
                 RANGE_POSITION_WINDOW, LIQUIDITY_SWEEP_WINDOW,
                 VOLATILITY_ZSCORE_WINDOW, ATR_PERCENTILE_WINDOW) + 10
    df = df.iloc[warmup:].reset_index(drop=True)

    log.info("Computed %d features on %d rows.", len(FEATURE_COLUMNS), len(df))
    return df


def add_target(df):
    """Simple binary target (used at prediction time)."""
    df = df.copy()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(float)
    df.loc[df.index[-1], "target"] = np.nan
    return df


def add_target_atr_filtered(df, threshold=None):
    """
    ATR-threshold target (v5): only significant moves count.
    Moves within ±threshold×ATR are marked NaN (dropped from training).
    """
    from config import TARGET_ATR_THRESHOLD
    if threshold is None:
        threshold = TARGET_ATR_THRESHOLD

    df = df.copy()
    move = df["close"].shift(-1) - df["close"]
    atr = df["atr_14"]
    min_move = threshold * atr

    df["target"] = np.nan
    df.loc[move > min_move, "target"] = 1.0
    df.loc[move < -min_move, "target"] = 0.0
    # Last row always NaN
    df.loc[df.index[-1], "target"] = np.nan

    valid = df["target"].notna().sum()
    total = len(df) - 1  # exclude last
    log.info("ATR-filtered target: %d/%d rows (%.1f%% kept, threshold=%.2f)",
             valid, total, valid / total * 100 if total else 0, threshold)
    return df

