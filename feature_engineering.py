"""
Feature Engineering — generate all technical indicators and structural features.

Input:  Raw OHLC DataFrame (from data_collector CSV or live fetch)
Output: Feature DataFrame + shifted target column (for training only)

Zero data leakage. Every feature uses only past/current candles.
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
    SESSION_ASIA, SESSION_LONDON, SESSION_NEW_YORK,
    LOG_LEVEL, LOG_FORMAT,
)

log = logging.getLogger("feature_engineering")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


# ─── Ordered feature list — this is saved with the model ────────────────────
FEATURE_COLUMNS = [
    # Trend (6)
    "ema_20", "ema_50", "ema_100", "ema_200",
    "ema_slope", "adx",
    # Momentum (5)
    "rsi_14",
    "macd", "macd_signal",
    "stoch_k", "stoch_d",
    # Volatility (3)
    "atr_14", "bb_width", "rolling_std_20",
    # Candle structure (9)
    "body_size", "upper_wick_ratio", "lower_wick_ratio",
    "candle_direction", "three_candle_momentum",
    "engulfing", "doji", "hammer", "shooting_star",
    # Context (5)
    "session_flag", "hour_sin", "hour_cos",
    "return_last_5",
    "volatility_rolling_10",
    "momentum_rolling_5",
]


# ═══════════════════════════════════════════════════════════════════════════
#  Internal Indicator Functions
# ═══════════════════════════════════════════════════════════════════════════

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


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k_period: int, d_period: int) -> tuple[pd.Series, pd.Series]:
    low_min = low.rolling(k_period).min()
    high_max = high.rolling(k_period).max()
    k = 100 * (close - low_min) / (high_max - low_min + 1e-10)
    d = k.rolling(d_period).mean()
    return k, d


# ═══════════════════════════════════════════════════════════════════════════
#  Candle Pattern Detectors
# ═══════════════════════════════════════════════════════════════════════════

def _engulfing(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    """Bullish engulfing = +1, bearish engulfing = -1, none = 0."""
    po, pc = o.shift(1), c.shift(1)
    bull = (pc < po) & (c > o) & (o <= pc) & (c >= po)
    bear = (pc > po) & (c < o) & (o >= pc) & (c <= po)
    result = pd.Series(0, index=o.index)
    result[bull] = 1
    result[bear] = -1
    return result


def _doji(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series,
          threshold: float = 0.05) -> pd.Series:
    body = (c - o).abs()
    rng = h - l + 1e-10
    return (body / rng < threshold).astype(int)


def _hammer(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series,
            body_pct: float = 0.3, wick_mult: float = 2.0) -> pd.Series:
    body = (c - o).abs()
    rng = h - l + 1e-10
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    return ((body / rng < body_pct) & (lower_wick > body * wick_mult) & (upper_wick < body)).astype(int)


def _shooting_star(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series,
                   body_pct: float = 0.3, wick_mult: float = 2.0) -> pd.Series:
    body = (c - o).abs()
    rng = h - l + 1e-10
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
    return ((body / rng < body_pct) & (upper_wick > body * wick_mult) & (lower_wick < body)).astype(int)


def _session(hour: int) -> int:
    if SESSION_NEW_YORK[0] <= hour < SESSION_NEW_YORK[1]:
        return 2
    if SESSION_LONDON[0] <= hour < SESSION_LONDON[1]:
        return 1
    return 0  # Asia / off-hours


# ═══════════════════════════════════════════════════════════════════════════
#  Main Public API
# ═══════════════════════════════════════════════════════════════════════════

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features from an OHLC DataFrame.

    Required columns: time, open, high, low, close
    Optional columns: tick_volume, spread (used if present)

    Returns a copy with feature columns appended.
    Warm-up rows are dropped so that every row has valid values.
    Internal columns (prefixed '_') used by regime detection are kept.
    """
    df = df.copy()
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    # ── Trend indicators ─────────────────────────────────────────────────
    df["ema_20"] = _ema(c, EMA_20)
    df["ema_50"] = _ema(c, EMA_50)
    df["ema_100"] = _ema(c, EMA_100)
    df["ema_200"] = _ema(c, EMA_200)
    df["ema_slope"] = (df["ema_50"] - df["ema_50"].shift(EMA_SLOPE_WINDOW)) / (
        df["ema_50"].shift(EMA_SLOPE_WINDOW) + 1e-10
    )
    df["adx"] = _adx(h, l, c, ADX_PERIOD)

    # ── Momentum indicators ──────────────────────────────────────────────
    df["rsi_14"] = _rsi(c, RSI_PERIOD)

    ema_fast = _ema(c, MACD_FAST)
    ema_slow = _ema(c, MACD_SLOW)
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = _ema(df["macd"], MACD_SIGNAL)

    sk, sd = _stochastic(h, l, c, STOCH_K_PERIOD, STOCH_D_PERIOD)
    df["stoch_k"] = sk
    df["stoch_d"] = sd

    # ── Volatility indicators ────────────────────────────────────────────
    df["atr_14"] = _atr(h, l, c, ATR_PERIOD)

    bb_mid = c.rolling(BB_PERIOD).mean()
    bb_std = c.rolling(BB_PERIOD).std()
    bb_upper = bb_mid + BB_STD * bb_std
    bb_lower = bb_mid - BB_STD * bb_std
    df["bb_width"] = (bb_upper - bb_lower) / (bb_mid + 1e-10)
    df["_bb_upper"] = bb_upper   # kept for regime / breakout checks
    df["_bb_lower"] = bb_lower

    df["rolling_std_20"] = c.rolling(ROLLING_STD_PERIOD).std()

    # ── Candle structure features ────────────────────────────────────────
    candle_range = h - l + 1e-10
    df["body_size"] = (c - o).abs() / candle_range
    df["upper_wick_ratio"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / candle_range
    df["lower_wick_ratio"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / candle_range
    df["candle_direction"] = (c > o).astype(int)

    # Three-candle momentum: encode last 3 directions as 0-7
    d0 = df["candle_direction"]
    d1 = d0.shift(1).fillna(0).astype(int)
    d2 = d0.shift(2).fillna(0).astype(int)
    df["three_candle_momentum"] = d2 * 4 + d1 * 2 + d0

    df["engulfing"] = _engulfing(o, h, l, c)
    df["doji"] = _doji(o, h, l, c)
    df["hammer"] = _hammer(o, h, l, c)
    df["shooting_star"] = _shooting_star(o, h, l, c)

    # ── Market context ───────────────────────────────────────────────────
    hour = df["time"].dt.hour
    df["session_flag"] = hour.apply(_session)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["return_last_5"] = c.pct_change(RETURN_LOOKBACK)
    df["volatility_rolling_10"] = df["atr_14"].rolling(VOLATILITY_ROLLING).mean() / (c + 1e-10)
    df["momentum_rolling_5"] = c.pct_change(1).rolling(MOMENTUM_ROLLING).mean()

    # ── Drop warm-up rows ────────────────────────────────────────────────
    warmup = max(EMA_200, BB_PERIOD, ATR_PERIOD, RSI_PERIOD,
                 MACD_SLOW, ADX_PERIOD, STOCH_K_PERIOD,
                 ROLLING_STD_PERIOD, EMA_SLOPE_WINDOW,
                 VOLATILITY_ROLLING, MOMENTUM_ROLLING) + 10
    df = df.iloc[warmup:].reset_index(drop=True)

    log.info("Computed %d features on %d rows.", len(FEATURE_COLUMNS), len(df))
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary target column for training.
    Target: 1 if next_close > current_close (bullish), else 0.
    Last row gets NaN and must be dropped before training.
    """
    df = df.copy()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(float)
    df.loc[df.index[-1], "target"] = np.nan
    return df
