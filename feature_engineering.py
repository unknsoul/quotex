"""
Feature Engineering v2 — multi-timeframe features + structural upgrades.

New features: RangePosition, LiquiditySweepFlag, VolatilityZScore,
H1 trend direction, H1 EMA alignment, M15 momentum.

Input:  Raw OHLC DataFrames (M5 + optional M15/H1)
Output: Feature DataFrame with ~35 columns + target for training
"""

import numpy as np
import pandas as pd
import logging

from config import (
    EMA_20, EMA_50, EMA_100, EMA_200, EMA_SLOPE_WINDOW,
    RSI_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    STOCH_K_PERIOD, STOCH_D_PERIOD,
    BB_PERIOD, BB_STD,
    ATR_PERIOD, ADX_PERIOD,
    ROLLING_STD_PERIOD, VOLATILITY_ROLLING, MOMENTUM_ROLLING,
    RETURN_LOOKBACK,
    RANGE_POSITION_WINDOW, LIQUIDITY_SWEEP_WINDOW, VOLATILITY_ZSCORE_WINDOW,
    SESSION_ASIA, SESSION_LONDON, SESSION_NEW_YORK,
    LOG_LEVEL, LOG_FORMAT,
)

log = logging.getLogger("feature_engineering")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


# --- Ordered feature list (saved with primary model) -------------------------
FEATURE_COLUMNS = [
    # Trend (5)
    "ema_20", "ema_50", "ema_200",
    "ema_slope", "adx",
    # Momentum (5)
    "rsi_14",
    "macd", "macd_signal",
    "stoch_k", "stoch_d",
    # Volatility (4)
    "atr_14", "bb_width", "rolling_std_20",
    "volatility_zscore",
    # Candle structure (8)
    "body_size", "upper_wick_ratio", "lower_wick_ratio",
    "range_position", "liquidity_sweep",
    "candle_direction", "three_candle_momentum",
    "engulfing",
    # Context (5)
    "session_flag", "hour_sin", "hour_cos",
    "return_last_5",
    "volatility_rolling_10",
    "momentum_rolling_5",
    # Multi-timeframe (4)
    "h1_trend_direction",
    "h1_ema_alignment",
    "h1_atr",
    "m15_momentum",
]


# =============================================================================
#  Internal Indicator Functions
# =============================================================================

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
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


# =============================================================================
#  Candle Pattern Detectors
# =============================================================================

def _engulfing(o, h, l, c):
    po, pc = o.shift(1), c.shift(1)
    bull = (pc < po) & (c > o) & (o <= pc) & (c >= po)
    bear = (pc > po) & (c < o) & (o >= pc) & (c <= po)
    result = pd.Series(0, index=o.index)
    result[bull] = 1
    result[bear] = -1
    return result


def _range_position(close, high, low, window):
    """Where close sits within rolling high-low range. 0 = at low, 1 = at high."""
    roll_high = high.rolling(window).max()
    roll_low = low.rolling(window).min()
    return (close - roll_low) / (roll_high - roll_low + 1e-10)


def _liquidity_sweep(high, low, close, o, window):
    """
    Detects liquidity sweep: wick exceeds prior N-bar extreme but close reverses.
    +1 = bearish sweep (high exceeded then close below open)
    -1 = bullish sweep (low exceeded then close above open)
    """
    prev_high = high.shift(1).rolling(window).max()
    prev_low = low.shift(1).rolling(window).min()
    bull_sweep = (low < prev_low) & (close > o)   # swept lows, reversed up
    bear_sweep = (high > prev_high) & (close < o)  # swept highs, reversed down
    result = pd.Series(0, index=close.index)
    result[bull_sweep] = -1
    result[bear_sweep] = 1
    return result


def _volatility_zscore(atr, window):
    """Z-score of ATR: (ATR - rolling_mean) / rolling_std."""
    mean = atr.rolling(window).mean()
    std = atr.rolling(window).std()
    return (atr - mean) / (std + 1e-10)


def _session(hour):
    if SESSION_NEW_YORK[0] <= hour < SESSION_NEW_YORK[1]:
        return 2
    if SESSION_LONDON[0] <= hour < SESSION_LONDON[1]:
        return 1
    return 0


# =============================================================================
#  Multi-Timeframe Feature Computation
# =============================================================================

def compute_htf_features(m5_df: pd.DataFrame,
                         m15_df: pd.DataFrame = None,
                         h1_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compute H1 and M15 features and merge onto M5 DataFrame.
    Uses forward-fill from last completed higher-TF bar (no leakage).
    """
    df = m5_df.copy()

    # -- H1 features --
    if h1_df is not None and len(h1_df) > 0:
        h1 = h1_df.copy()
        h1_c = h1["close"]
        h1["h1_ema20"] = _ema(h1_c, EMA_20)
        h1["h1_ema50"] = _ema(h1_c, EMA_50)
        h1["h1_ema200"] = _ema(h1_c, EMA_200)

        # H1 trend direction: 1 if EMA20 > EMA50, -1 if below
        h1["h1_trend_direction"] = 0
        h1.loc[h1["h1_ema20"] > h1["h1_ema50"], "h1_trend_direction"] = 1
        h1.loc[h1["h1_ema20"] < h1["h1_ema50"], "h1_trend_direction"] = -1

        # H1 EMA alignment: 1 if 20>50>200 (bull), -1 if 20<50<200 (bear)
        h1["h1_ema_alignment"] = 0
        bull = (h1["h1_ema20"] > h1["h1_ema50"]) & (h1["h1_ema50"] > h1["h1_ema200"])
        bear = (h1["h1_ema20"] < h1["h1_ema50"]) & (h1["h1_ema50"] < h1["h1_ema200"])
        h1.loc[bull, "h1_ema_alignment"] = 1
        h1.loc[bear, "h1_ema_alignment"] = -1

        # H1 ATR
        h1["h1_atr"] = _atr(h1["high"], h1["low"], h1["close"], ATR_PERIOD)

        # Merge onto M5 using asof join (last completed H1 bar)
        h1_merge = h1[["time", "h1_trend_direction", "h1_ema_alignment", "h1_atr"]].copy()
        h1_merge = h1_merge.sort_values("time")
        df = df.sort_values("time")
        df = pd.merge_asof(df, h1_merge, on="time", direction="backward")
    else:
        df["h1_trend_direction"] = 0
        df["h1_ema_alignment"] = 0
        df["h1_atr"] = 0.0

    # -- M15 features --
    if m15_df is not None and len(m15_df) > 0:
        m15 = m15_df.copy()
        m15_c = m15["close"]
        m15_rsi = _rsi(m15_c, RSI_PERIOD)
        m15_macd = _ema(m15_c, MACD_FAST) - _ema(m15_c, MACD_SLOW)
        m15_macd_sig = _ema(m15_macd, MACD_SIGNAL)

        # M15 momentum: normalized RSI direction + MACD sign
        m15["m15_momentum"] = 0.0
        m15["m15_momentum"] = ((m15_rsi - 50) / 50) + (m15_macd > m15_macd_sig).astype(float)

        m15_merge = m15[["time", "m15_momentum"]].copy()
        m15_merge = m15_merge.sort_values("time")
        df = pd.merge_asof(df, m15_merge, on="time", direction="backward")
    else:
        df["m15_momentum"] = 0.0

    # Fill any NaN from merge
    df["h1_trend_direction"] = df["h1_trend_direction"].fillna(0).astype(int)
    df["h1_ema_alignment"] = df["h1_ema_alignment"].fillna(0).astype(int)
    df["h1_atr"] = df["h1_atr"].fillna(0.0)
    df["m15_momentum"] = df["m15_momentum"].fillna(0.0)

    return df


# =============================================================================
#  Main Public API
# =============================================================================

def compute_features(df: pd.DataFrame,
                     m15_df: pd.DataFrame = None,
                     h1_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compute all features from OHLC DataFrames.

    Args:
        df: M5 OHLC DataFrame (required)
        m15_df: M15 OHLC DataFrame (optional, for multi-TF features)
        h1_df: H1 OHLC DataFrame (optional, for multi-TF features)

    Returns DataFrame with all feature columns.
    """
    df = df.copy()
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    # -- Trend --
    df["ema_20"] = _ema(c, EMA_20)
    df["ema_50"] = _ema(c, EMA_50)
    df["ema_200"] = _ema(c, EMA_200)
    df["ema_slope"] = (df["ema_50"] - df["ema_50"].shift(EMA_SLOPE_WINDOW)) / (
        df["ema_50"].shift(EMA_SLOPE_WINDOW) + 1e-10
    )
    df["adx"] = _adx(h, l, c, ADX_PERIOD)

    # -- Momentum --
    df["rsi_14"] = _rsi(c, RSI_PERIOD)
    ema_fast = _ema(c, MACD_FAST)
    ema_slow = _ema(c, MACD_SLOW)
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = _ema(df["macd"], MACD_SIGNAL)
    sk, sd = _stochastic(h, l, c, STOCH_K_PERIOD, STOCH_D_PERIOD)
    df["stoch_k"] = sk
    df["stoch_d"] = sd

    # -- Volatility --
    df["atr_14"] = _atr(h, l, c, ATR_PERIOD)
    bb_mid = c.rolling(BB_PERIOD).mean()
    bb_std = c.rolling(BB_PERIOD).std()
    bb_upper = bb_mid + BB_STD * bb_std
    bb_lower = bb_mid - BB_STD * bb_std
    df["bb_width"] = (bb_upper - bb_lower) / (bb_mid + 1e-10)
    df["_bb_upper"] = bb_upper
    df["_bb_lower"] = bb_lower
    df["rolling_std_20"] = c.rolling(ROLLING_STD_PERIOD).std()
    df["volatility_zscore"] = _volatility_zscore(df["atr_14"], VOLATILITY_ZSCORE_WINDOW)

    # -- Candle structure --
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

    df["engulfing"] = _engulfing(o, h, l, c)

    # -- Context --
    hour = df["time"].dt.hour
    df["session_flag"] = hour.apply(_session)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["return_last_5"] = c.pct_change(RETURN_LOOKBACK)
    df["volatility_rolling_10"] = df["atr_14"].rolling(VOLATILITY_ROLLING).mean() / (c + 1e-10)
    df["momentum_rolling_5"] = c.pct_change(1).rolling(MOMENTUM_ROLLING).mean()

    # -- Multi-timeframe --
    df = compute_htf_features(df, m15_df, h1_df)

    # -- Drop warm-up rows --
    warmup = max(EMA_200, BB_PERIOD, ATR_PERIOD, RSI_PERIOD,
                 MACD_SLOW, ADX_PERIOD, STOCH_K_PERIOD,
                 ROLLING_STD_PERIOD, EMA_SLOPE_WINDOW,
                 VOLATILITY_ROLLING, MOMENTUM_ROLLING,
                 RANGE_POSITION_WINDOW, LIQUIDITY_SWEEP_WINDOW,
                 VOLATILITY_ZSCORE_WINDOW) + 10
    df = df.iloc[warmup:].reset_index(drop=True)

    log.info("Computed %d features on %d rows.", len(FEATURE_COLUMNS), len(df))
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Target: 1 if next_close > current_close (bullish), else 0."""
    df = df.copy()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(float)
    df.loc[df.index[-1], "target"] = np.nan
    return df
