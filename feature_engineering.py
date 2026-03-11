# -*- coding: utf-8 -*-
"""
Feature Engineering v17 — Ichimoku, CCI, Williams %R, +DI/-DI, deduplication.

v17 adds: Ichimoku cloud features (4), CCI, Williams %R, +DI/-DI directional ratio,
          removes redundant features (dollar_strength_proxy, five_candle_pattern as int).
Total: ~72 features.
"""

import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
import logging

from config import (
    EMA_20, EMA_50, EMA_100, EMA_200, EMA_SLOPE_WINDOW,
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    STOCH_K_PERIOD, STOCH_D_PERIOD, BB_PERIOD, BB_STD,
    ATR_PERIOD, ADX_PERIOD, ROLLING_STD_PERIOD,
    VOLATILITY_ROLLING, MOMENTUM_ROLLING, RETURN_LOOKBACK,
    RANGE_POSITION_WINDOW, LIQUIDITY_SWEEP_WINDOW,
    VOLATILITY_ZSCORE_WINDOW, ATR_PERCENTILE_WINDOW,
    TARGET_LOOKAHEAD,
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
    # v17: Directional indicators (+DI/-DI from ADX)
    "plus_di", "minus_di", "di_ratio",
    # v17: Ichimoku Cloud features
    "ichimoku_tkx",  # Tenkan-Kijun cross signal
    "ichimoku_cloud_dist",  # price distance from cloud
    "ichimoku_cloud_width",  # cloud thickness (trend strength)
    "ichimoku_chikou_pos",  # Chikou span position
    # v17: CCI and Williams %R
    "cci_20",
    "williams_r",
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
    # Phase 5 (1-Candle Expiry) Binary Options Microstructure
    "upper_wick_percent", "lower_wick_percent", "body_percent",
    "rsi_velocity", "atr_acceleration", "close_to_high_dist", "close_to_low_dist",
    "micro_trend_3", "price_action_score",
    # Phase 7 (60%+ Accuracy Upgrades)
    "dist_to_resistance", "dist_to_support", 
    "bullish_divergence", "bearish_divergence",
    "tick_vol_accel_1", "tick_vol_accel_2", "dollar_strength_proxy",
    # Phase 8 (Deep Microstructure M1)
    "trade_intensity", "cumulative_delta", "m1_absorption",
    # Phase 9 (Accuracy Maximization)
    "rsi_mean_reversion", "vol_contraction",
    "momentum_quality_5", "momentum_quality_10",
    "vwap_distance", "rejection_ratio",
    "trend_exhaustion_10", "range_expansion", "bb_position",
    # Phase 10 (v10+ upgrades: HA, OBV, pin bar, HH/HL, pattern scoring)
    "ha_body_ratio", "ha_direction", "ha_streak",
    "obv_slope_10", "obv_divergence",
    "pin_bar_score", "hammer_score", "engulfing_score",
    "hh_hl_count_10", "ll_lh_count_10", "swing_structure",
    # Phase 11 (Accuracy v2 Upgrades)
    "rsi_stoch_confluence",        # RSI + Stochastic agreement
    "ema_ribbon_width",            # EMA fan width (trend strength)
    "momentum_acceleration",       # 2nd derivative of momentum
    "candle_strength_ratio",       # body-to-wick quality
    "volume_momentum_confirm",     # volume supporting price direction
    "price_rejection_level",       # rejection at key levels
    "multi_bar_trend_quality",     # consistency of recent trend
    "intrabar_volatility_ratio",   # current bar vs recent bars
    "directional_pressure",        # net buying/selling pressure
    "smart_money_divergence",      # OBV vs price divergence (enhanced)
    # v11: Candle Quality & Freshness Features
    "candle_freshness",            # How different from recent candles (0=stale, 1=fresh)
    "candle_body_quality",         # Body/range ratio (0=doji, 1=marubozu)
    "streak_length",               # Consecutive same-direction candle count
    "candle_range_vs_avg",         # Current range / avg range (abnormality)
    # v11.1: Hour-Quality Features (lets model learn hour effects natively)
    "hour_quality_score",          # Data-driven hour quality (0.80-1.06)
    "hour_volatility_interaction", # Hour quality × volatility
    "hour_momentum_interaction",   # Hour quality × momentum
    # v11 Advanced: Strategy & Pattern Features
    "same_candle_score",           # Same-candle detector composite (0=stale, 1=clear)
    "crossover_staleness",         # Crossover signal freshness (0=none, 1=fresh)
    "pattern_score",               # Candlestick pattern score (-20 to +20 scaled to 0-1)
    "strategy_agreement",          # Fraction of 8 strategies agreeing (0-1)
    "strategy_composite",          # Composite strategy score (0-100 scaled to 0-1)
    # ═══════ v15: Advanced Features (30 new) ═══════
    # Cross-timeframe features
    "m15_rsi",                     # M15 RSI value
    "h1_rsi",                      # H1 RSI value
    "m15_bb_position",             # M15 Bollinger Band position
    "h1_adx",                      # H1 ADX strength
    "tf_rsi_agreement",            # Agreement of RSI across M5/M15/H1
    # Fractal / Market structure
    "fractal_dimension",           # Fractal dimension (Higuchi method proxy)
    "hurst_exponent",              # Hurst exponent (trending vs mean-reverting)
    "efficiency_ratio",            # Kaufman efficiency ratio
    # Wavelet / Frequency domain energy
    "high_freq_energy",            # High-frequency component energy
    "low_freq_energy",             # Low-frequency (trend) energy ratio
    # Advanced order flow
    "buy_pressure_ratio",          # Buy volume / total volume (rolling)
    "aggressive_ratio",            # Large moves / all moves
    "absorption_rate_5",           # Wick absorption momentum (5-bar)
    # Regime quantification
    "regime_persistence",          # How long current regime has lasted
    "regime_transition_prob",      # Probability of regime change
    "volatility_of_volatility",    # Vol of vol (VVIX proxy)
    # Advanced candle features
    "inside_bar",                  # Inside bar pattern (1/0)
    "outside_bar",                 # Outside bar pattern (1/0)
    "three_bar_reversal",          # Three-bar reversal pattern
    "morning_star_score",          # Morning/evening star pattern
    # Momentum higher-order
    "macd_histogram_accel",        # MACD histogram acceleration
    "rsi_bollinger",               # RSI position within its own Bollinger Band
    "stoch_rsi",                   # Stochastic RSI
    # Volume microstructure
    "volume_climax_score",         # Volume spike + reversal candle
    "cumulative_delta_slope",      # Slope of cumulative delta proxy
    # Session / Time features
    "day_of_week_sin",             # Day of week cyclical
    "day_of_week_cos",             # Day of week cyclical
    "minutes_to_session_end",      # Minutes until session close
    # ═══════ v18: 35+ Classic TA Indicators (Binary Options Optimized) ═══════
    # Fast EMAs (short-term scalping)
    "ema_5",                       # EMA 5 distance from close
    "ema_8",                       # EMA 8 distance from close
    "ema_13",                      # EMA 13 distance from close
    # Moving average variants
    "sma_10",                      # SMA 10 distance from close
    "sma_30",                      # SMA 30 distance from close
    "dema_20",                     # Double EMA 20 distance
    "tema_20",                     # Triple EMA 20 distance
    "wma_20",                      # Weighted MA 20 distance
    # Multi-period RSI
    "rsi_7",                       # Fast RSI (7-period)
    "rsi_21",                      # Slow RSI (21-period)
    # Rate of Change (momentum)
    "roc_10",                      # Rate of Change 10-period
    "roc_20",                      # Rate of Change 20-period
    # Trend indicators
    "parabolic_sar_dir",           # Parabolic SAR direction (1=bull, 0=bear)
    "supertrend_dir",              # Supertrend direction (1=bull, 0=bear)
    "aroon_up",                    # Aroon Up (0-100 normalized)
    "aroon_down",                  # Aroon Down (0-100 normalized)
    "aroon_osc",                   # Aroon Oscillator (-100 to +100)
    "vortex_diff",                 # Vortex indicator (+VI - -VI)
    # Volatility channels
    "keltner_position",            # Price position within Keltner Channel (0-1)
    "keltner_width",               # Keltner Channel width (normalized)
    "donchian_position",           # Donchian Channel position (0-1)
    "donchian_width",              # Donchian Channel width (normalized)
    # Volume indicators
    "mfi_14",                      # Money Flow Index (0-100 normalized)
    "cmf_20",                      # Chaikin Money Flow (-1 to +1)
    "force_index_13",              # Force Index (normalized)
    # Advanced momentum oscillators
    "ultimate_oscillator",         # Ultimate Oscillator (0-1)
    "tsi",                         # True Strength Index (-1 to +1)
    "chande_momentum",             # Chande Momentum Oscillator (-1 to +1)
    "ppo",                         # Percentage Price Oscillator
    "awesome_oscillator",          # Awesome Oscillator (normalized)
    # Elder Ray
    "bull_power",                  # Elder Ray Bull Power (normalized)
    "bear_power",                  # Elder Ray Bear Power (normalized)
    # Pivots & regression
    "pivot_distance",              # Distance from classic pivot point
    "linreg_slope_20",             # Linear regression slope (20-bar)
    "linreg_r2_20",                # Linear regression R² (trend quality)
    # MACD histogram raw
    "macd_histogram",              # Raw MACD histogram
    # Mass Index (squeeze detection)
    "mass_index",                  # Mass Index (squeeze indicator)
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
    # Standard DM: keep the larger positive move, zero out the other
    both_neg = (plus_dm <= 0) & (minus_dm <= 0)
    plus_bigger = plus_dm > minus_dm
    plus_dm = plus_dm.where(plus_bigger & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where(~plus_bigger & (minus_dm > 0), 0.0)
    plus_dm[both_neg] = 0.0
    minus_dm[both_neg] = 0.0
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / (atr_val + 1e-10))
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / (atr_val + 1e-10))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    return dx.ewm(span=period, adjust=False).mean()


def _adx_full(high, low, close, period):
    """Return (adx, +DI, -DI) for directional analysis."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    both_neg = (plus_dm <= 0) & (minus_dm <= 0)
    plus_bigger = plus_dm > minus_dm
    plus_dm = plus_dm.where(plus_bigger & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where(~plus_bigger & (minus_dm > 0), 0.0)
    plus_dm[both_neg] = 0.0
    minus_dm[both_neg] = 0.0
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / (atr_val + 1e-10))
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / (atr_val + 1e-10))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx, plus_di, minus_di


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

    # Force time columns to datetime64[ns, UTC] for merge_asof compatibility.
    # MT5 returns datetime64[s, UTC], but pd.Timedelta ops produce [us, UTC].
    # merge_asof requires exact dtype match, so we standardize to [ns, UTC].
    def _normalize_time(frame):
        if "time" in frame.columns:
            frame["time"] = pd.to_datetime(frame["time"], utc=True).astype("datetime64[ns, UTC]")
        return frame

    df = _normalize_time(df)

    if h1_df is not None and len(h1_df) > 0:
        h1 = h1_df.copy()
        h1 = _normalize_time(h1)
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
        h1["h1_rsi"] = _rsi(h1["close"], RSI_PERIOD)
        h1["h1_adx"] = _adx(h1["high"], h1["low"], h1["close"], ADX_PERIOD)

        h1_merge = h1[["time", "h1_trend_direction", "h1_ema_alignment", "h1_atr",
                        "h1_rsi", "h1_adx"]].copy()
        h1_merge = h1_merge.sort_values("time")
        # merge_asof direction="backward" already ensures we only use
        # H1 bars whose timestamp <= M5 timestamp (i.e. completed bars).
        # No time shift needed — shifting forward BREAKS the alignment.
        h1_merge = _normalize_time(h1_merge)
        df = _normalize_time(df)
        df = df.sort_values("time")
        df = pd.merge_asof(df, h1_merge, on="time", direction="backward")
    else:
        df["h1_trend_direction"] = 0
        df["h1_ema_alignment"] = 0
        df["h1_atr"] = 0.0
        df["h1_rsi"] = 50.0
        df["h1_adx"] = 25.0

    if m15_df is not None and len(m15_df) > 0:
        m15 = m15_df.copy()
        m15 = _normalize_time(m15)
        m15_c = m15["close"]
        m15_rsi = _rsi(m15_c, RSI_PERIOD)
        m15_macd = _ema(m15_c, MACD_FAST) - _ema(m15_c, MACD_SLOW)
        m15_macd_sig = _ema(m15_macd, MACD_SIGNAL)
        m15["m15_momentum"] = ((m15_rsi - 50) / 50) + (m15_macd > m15_macd_sig).astype(float)
        m15["m15_rsi"] = m15_rsi
        # M15 Bollinger Band position
        m15_bb_mid = m15_c.rolling(BB_PERIOD).mean()
        m15_bb_std = m15_c.rolling(BB_PERIOD).std()
        m15_bb_upper = m15_bb_mid + BB_STD * m15_bb_std
        m15_bb_lower = m15_bb_mid - BB_STD * m15_bb_std
        m15["m15_bb_position"] = ((m15_c - m15_bb_lower) / (m15_bb_upper - m15_bb_lower + 1e-10)) * 2 - 1

        m15_merge = m15[["time", "m15_momentum", "m15_rsi", "m15_bb_position"]].copy()
        m15_merge = m15_merge.sort_values("time")
        # merge_asof direction="backward" already uses only completed M15 bars.
        m15_merge = _normalize_time(m15_merge)
        df = _normalize_time(df)
        df = pd.merge_asof(df, m15_merge, on="time", direction="backward")
    else:
        df["m15_momentum"] = 0.0
        df["m15_rsi"] = 50.0
        df["m15_bb_position"] = 0.0

    df["h1_trend_direction"] = df["h1_trend_direction"].fillna(0).astype(int)
    df["h1_ema_alignment"] = df["h1_ema_alignment"].fillna(0).astype(int)
    df["h1_atr"] = df["h1_atr"].fillna(0.0)
    df["h1_rsi"] = df["h1_rsi"].fillna(50.0)
    df["h1_adx"] = df["h1_adx"].fillna(25.0)
    df["m15_momentum"] = df["m15_momentum"].fillna(0.0)
    df["m15_rsi"] = df["m15_rsi"].fillna(50.0)
    df["m15_bb_position"] = df["m15_bb_position"].fillna(0.0)
    return df


# =============================================================================
#  Main API
# =============================================================================

def compute_features(df, m15_df=None, h1_df=None, m1_df=None):
    df = df.copy()
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    # Trend — NORMALIZED to distance from close (scale-invariant)
    ema20 = _ema(c, EMA_20)
    ema50 = _ema(c, EMA_50)
    ema200 = _ema(c, EMA_200)
    df["ema_20"] = (c - ema20) / (c + 1e-10)     # distance from EMA20 as %
    df["ema_50"] = (c - ema50) / (c + 1e-10)     # distance from EMA50 as %
    df["ema_200"] = (c - ema200) / (c + 1e-10)   # distance from EMA200 as %
    df["ema_slope"] = (ema50 - ema50.shift(EMA_SLOPE_WINDOW)) / (
        ema50.shift(EMA_SLOPE_WINDOW) + 1e-10)
    df["adx"] = _adx(h, l, c, ADX_PERIOD)

    # v17: +DI / -DI directional indicators
    _, pdi, mdi = _adx_full(h, l, c, ADX_PERIOD)
    df["plus_di"] = pdi / 100.0  # normalize to 0-1
    df["minus_di"] = mdi / 100.0
    df["di_ratio"] = pdi / (pdi + mdi + 1e-10)  # 0.5 = balanced, >0.5 = bullish

    # v17: Ichimoku Cloud features (9/26/52 standard settings)
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (h.rolling(52).max() + l.rolling(52).min()) / 2
    # Tenkan-Kijun cross: positive = bullish cross
    df["ichimoku_tkx"] = (tenkan - kijun) / (c + 1e-10)
    # Price distance from cloud (positive = above cloud = bullish)
    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
    df["ichimoku_cloud_dist"] = (c - (cloud_top + cloud_bot) / 2) / (c + 1e-10)
    # Cloud width (trend strength)
    df["ichimoku_cloud_width"] = (cloud_top - cloud_bot) / (c + 1e-10)
    # Chikou span position (close vs close 26 bars ago)
    df["ichimoku_chikou_pos"] = (c - c.shift(26)) / (c + 1e-10)

    # v17: CCI (Commodity Channel Index) — 20-period
    tp = (h + l + c) / 3  # typical price
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["cci_20"] = ((tp - tp_sma) / (0.015 * tp_mad + 1e-10)).clip(-300, 300) / 300.0  # normalized

    # v17: Williams %R — 14-period
    highest_14 = h.rolling(14).max()
    lowest_14 = l.rolling(14).min()
    df["williams_r"] = ((highest_14 - c) / (highest_14 - lowest_14 + 1e-10))  # 0=overbought, 1=oversold

    # Momentum
    df["rsi_14"] = _rsi(c, RSI_PERIOD)
    df["macd"] = _ema(c, MACD_FAST) - _ema(c, MACD_SLOW)
    df["macd_signal"] = _ema(df["macd"], MACD_SIGNAL)
    sk, sd = _stochastic(h, l, c, STOCH_K_PERIOD, STOCH_D_PERIOD)
    df["stoch_k"] = sk
    df["stoch_d"] = sd

    # Volatility
    atr_raw = _atr(h, l, c, ATR_PERIOD)
    df["atr_14"] = atr_raw / (c + 1e-10)  # ATR as % of price (scale-invariant)
    bb_mid = c.rolling(BB_PERIOD).mean()
    bb_std = c.rolling(BB_PERIOD).std()
    df["bb_width"] = ((bb_mid + BB_STD * bb_std) - (bb_mid - BB_STD * bb_std)) / (bb_mid + 1e-10)
    df["rolling_std_20"] = c.rolling(ROLLING_STD_PERIOD).std() / (c + 1e-10)  # as % of price
    df["volatility_zscore"] = _volatility_zscore(atr_raw, VOLATILITY_ZSCORE_WINDOW)

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

    # Phase 1: Contextualize momentum inside trend wrapper
    ema_slope_dir = (df["ema_slope"] > 0).astype(int) * 2 - 1  # 1 for up, -1 for down
    adx_trend = (df["adx"] > 25).astype(int)
    
    # 1 if momentum aligns with strong trend, -1 if against strong trend, 0 if ranging/weak
    df["momentum_in_trend"] = 0
    df.loc[(df["candle_direction"] == 1) & (ema_slope_dir == 1) & (adx_trend == 1), "momentum_in_trend"] = 1
    df.loc[(df["candle_direction"] == 0) & (ema_slope_dir == -1) & (adx_trend == 1), "momentum_in_trend"] = -1
    
    # EMA distance interaction with momentum
    df["momentum_vs_ema"] = df["three_candle_momentum"] * (c - df["ema_20"]) / (df["ema_20"] + 1e-10)

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
    df["atr_percentile_rank"] = _atr_percentile_rank(atr_raw, ATR_PERCENTILE_WINDOW)

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
    df["delta_atr_5"] = atr_raw.pct_change(5)  # ATR change as %

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

    # --- Phase 5 (1-Candle Expiry) Proprietary Features ---
    # Define variables explicitly
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    # 1. Wick & Body Proportions (crucial for next-candle momentum)
    candle_range = (high - low).replace(0, 1e-10)
    df["upper_wick_percent"] = (high - df[["open", "close"]].max(axis=1)) / candle_range
    df["lower_wick_percent"] = (df[["open", "close"]].min(axis=1) - low) / candle_range
    df["body_percent"] = df["body_size"] / candle_range
    
    # 2. RSI Velocity (How fast is momentum shifting right now)
    df["rsi_velocity"] = df["rsi_14"].diff(1).fillna(0)
    
    # 3. ATR Acceleration (Volatility expansion inside the 5m window)
    df["atr_acceleration"] = (df["atr_14"].diff(1) / df["atr_14"].shift(1).replace(0, 1e-10)).fillna(0)
    
    # 4. Intra-candle closing pressure
    df["close_to_high_dist"] = (high - close) / candle_range
    df["close_to_low_dist"] = (close - low) / candle_range
    
    # 5. Micro-trend (3-bar immediate direction)
    df["micro_trend_3"] = (close > close.shift(3)).astype(float).fillna(0.5)
    
    # 6. Synthesized Price Action Score (Combines wicks + body direction)
    # A strong green body closing near the high = 1.0, strong red closing near low = -1.0
    pa_score = pd.Series(0.0, index=df.index)
    bull_mask = close > df["open"]
    bear_mask = close < df["open"]
    
    # Bulls get points for big body, small upper wick (closing near high)
    pa_score.loc[bull_mask] = df.loc[bull_mask, "body_percent"] - df.loc[bull_mask, "upper_wick_percent"]
    # Bears get points for big body, small lower wick (closing near low)
    pa_score.loc[bear_mask] = -(df.loc[bear_mask, "body_percent"] - df.loc[bear_mask, "lower_wick_percent"])
    df["price_action_score"] = pa_score.fillna(0)

    # --- Phase 7 (60%+ Accuracy Upgrades) ---
    
    # [Component 1] Live Support/Resistance Proximity (The Brick Wall Guard)
    # Scan last 50 candles for local swing highs/lows
    local_max = df["high"].rolling(window=50, min_periods=5).max()
    local_min = df["low"].rolling(window=50, min_periods=5).min()
    # Distance to resistance/support relative to current price ATR
    atr_val = df["atr_14"].replace(0, 1e-10)
    df["dist_to_resistance"] = ((local_max - close) / atr_val).fillna(5.0)
    df["dist_to_support"] = ((close - local_min) / atr_val).fillna(5.0)
    
    # [Component 2] RSI Divergence Detection (The Reversal Cheat-Code)
    # Simple 2-peak trailing comparison (Lookback 10 candles for a pivot)
    rsi_shifted = df["rsi_14"].shift(3) # Proxy for previous peak
    price_shifted = close.shift(3)
    
    # Bearish Divergence: Price making Higher High, RSI making Lower High
    df["bearish_divergence"] = ((close > price_shifted) & (df["rsi_14"] < rsi_shifted)).astype(float)
    # Bullish Divergence: Price making Lower Low, RSI making Higher Low
    df["bullish_divergence"] = ((close < price_shifted) & (df["rsi_14"] > rsi_shifted)).astype(float)
    
    # [Component 3] Tick-Volume Acceleration
    if "tick_volume" in df.columns:
        tv = df["tick_volume"].astype(float).replace(0, 1e-10)
        df["tick_vol_accel_1"] = (tv.diff(1) / tv.shift(1)).fillna(0)
        df["tick_vol_accel_2"] = (tv.diff(2) / tv.shift(2)).fillna(0)
    else:
        df["tick_vol_accel_1"] = 0.0
        df["tick_vol_accel_2"] = 0.0
        
    # [Component 4] Multi-Pair Dollar Liquidity Proxy
    # Since we don't have multi-symbol streams natively here, we use EUR/USD 
    # inverted trend as a proxy for raw Dollar strength
    df["dollar_strength_proxy"] = -df["ema_slope"].fillna(0)

    # --- Phase 8 (Deep Microstructure M1) ---
    if m1_df is not None and not m1_df.empty and "tick_volume" in m1_df.columns:
        m1 = m1_df.copy()
        if "time" in m1.columns:
            m1["time"] = pd.to_datetime(m1["time"], utc=True).astype("datetime64[ns, UTC]")
            df["time"] = pd.to_datetime(df["time"], utc=True).astype("datetime64[ns, UTC]")
            
            # Simple tick direction proxy: Close > Open = Buy volume, Close < Open = Sell volume
            m1_c, m1_o = m1["close"], m1["open"]
            tv = m1["tick_volume"].astype(float)
            m1_buy_vol = tv * (m1_c > m1_o).astype(float)
            m1_sell_vol = tv * (m1_c < m1_o).astype(float)
            
            m1["delta"] = m1_buy_vol - m1_sell_vol
            m1["intensity"] = tv
            
            # Absorption: High volume but tiny body compared to the full range
            m1_body = (m1_c - m1_o).abs()
            m1_range = (m1["high"] - m1["low"]).replace(0, 1e-10)
            m1["absorption"] = tv / (m1_body / m1_range + 1e-10)
            
            # Roll them up to 5-minute blocks
            m1["cumulative_delta"] = m1["delta"].rolling(5, min_periods=1).sum()
            m1["trade_intensity"] = m1["intensity"].rolling(5, min_periods=1).sum()
            m1["m1_absorption"] = m1["absorption"].rolling(5, min_periods=1).mean()
            
            m1_merge = m1[["time", "cumulative_delta", "trade_intensity", "m1_absorption"]].copy()
            m1_merge = m1_merge.sort_values("time")
            
            df = df.sort_values("time")
            df = pd.merge_asof(df, m1_merge, on="time", direction="backward")
        else:
            df["cumulative_delta"] = 0.0
            df["trade_intensity"] = 0.0
            df["m1_absorption"] = 0.0
    else:
        df["cumulative_delta"] = 0.0
        df["trade_intensity"] = 0.0
        df["m1_absorption"] = 0.0


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
    candle_range_5 = (df["high"] - df["low"]).replace(0, 1e-10)
    body_ratio = (df["close"] - df["open"]).abs() / candle_range_5
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

    # --- Phase 9 (Accuracy Maximization Features) ---
    
    # 1. RSI mean-reversion signal: distance from 50 combined with recent change
    rsi_dist = (df["rsi_14"] - 50) / 50  # normalized -1 to +1
    rsi_chg = df["rsi_14"].diff(3).fillna(0)
    df["rsi_mean_reversion"] = -rsi_dist * np.sign(rsi_chg)  # positive when reverting
    
    # 2. Volatility contraction/expansion ratio (narrow range = breakout pending)
    atr_fast = _atr(df["high"], df["low"], df["close"], 5)
    atr_slow = _atr(df["high"], df["low"], df["close"], 20)
    df["vol_contraction"] = atr_fast / (atr_slow + 1e-10)
    
    # 3. Momentum quality: consistency of returns direction in window
    ret_sign = (returns > 0).astype(float)
    df["momentum_quality_5"] = ret_sign.rolling(5, min_periods=2).mean()  # 1.0 = all green
    df["momentum_quality_10"] = ret_sign.rolling(10, min_periods=3).mean()
    
    # 4. Price distance from VWAP proxy (tick_volume weighted average)
    if "tick_volume" in df.columns:
        tv = df["tick_volume"].astype(float).replace(0, 1)
        vwap = (c * tv).rolling(20, min_periods=5).sum() / tv.rolling(20, min_periods=5).sum()
        df["vwap_distance"] = (c - vwap) / (atr_raw + 1e-10)
    else:
        df["vwap_distance"] = 0.0
    
    # 5. Candle rejection ratio: wick vs body tells conviction
    total_wick = (h - pd.concat([o, c], axis=1).max(axis=1)) + (pd.concat([o, c], axis=1).min(axis=1) - l)
    body = (c - o).abs()
    df["rejection_ratio"] = total_wick / (body + total_wick + 1e-10)
    
    # 6. Trend exhaustion: how far has price moved relative to ATR in the window
    df["trend_exhaustion_10"] = c.pct_change(10).abs() / (df["atr_14"] + 1e-10)
    
    # 7. High-low range relative to recent average (expanding/contracting range)
    bar_range = h - l
    avg_range = bar_range.rolling(20, min_periods=5).mean()
    df["range_expansion"] = bar_range / (avg_range + 1e-10)
    
    # 8. Close position within Bollinger Bands (normalized -1 to +1)
    bb_upper = bb_mid + BB_STD * bb_std
    bb_lower = bb_mid - BB_STD * bb_std
    df["bb_position"] = (c - bb_lower) / (bb_upper - bb_lower + 1e-10) * 2 - 1

    # ---- Phase 10 (v10+ upgrades): Heikin-Ashi, OBV, pin bar, HH/HL, pattern scoring ----

    # Heikin-Ashi candles
    ha_close = (o + h + l + c) / 4
    ha_open = pd.Series(o.values.copy(), index=o.index)
    for i in range(1, len(ha_open)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
    ha_body = ha_close - ha_open
    ha_range = h - l + 1e-10
    df["ha_body_ratio"] = ha_body.abs() / ha_range
    df["ha_direction"] = (ha_close > ha_open).astype(float)
    # Streak of consecutive same-direction HA candles
    ha_dir_int = df["ha_direction"].astype(int)
    ha_change = ha_dir_int.ne(ha_dir_int.shift(1)).astype(int)
    ha_group = ha_change.cumsum()
    df["ha_streak"] = ha_dir_int.groupby(ha_group).cumcount() + 1

    # On-Balance Volume (OBV)
    if "tick_volume" in df.columns:
        tv_raw = df["tick_volume"].astype(float)
        obv_dir = pd.Series(0.0, index=df.index)
        obv_dir[c > c.shift(1)] = 1.0
        obv_dir[c < c.shift(1)] = -1.0
        obv = (tv_raw * obv_dir).cumsum()
        obv_ema = _ema(obv, 10)
        df["obv_slope_10"] = (obv_ema - obv_ema.shift(3)) / (obv_ema.shift(3).abs() + 1e-10)
        # OBV divergence: price up but OBV down, or vice versa
        price_up = (c > c.shift(3)).astype(float) * 2 - 1
        obv_up = (obv_ema > obv_ema.shift(3)).astype(float) * 2 - 1
        df["obv_divergence"] = (price_up != obv_up).astype(float)
    else:
        df["obv_slope_10"] = 0.0
        df["obv_divergence"] = 0.0

    # Pin bar detection (long wick + small body)
    _body = (c - o).abs()
    _lower = pd.concat([o, c], axis=1).min(axis=1) - l
    _upper = h - pd.concat([o, c], axis=1).max(axis=1)
    _range = h - l + 1e-10
    body_pct = _body / _range
    # Pin bar: body < 30% range AND one wick > 60% range
    long_lower = (_lower / _range > 0.6) & (body_pct < 0.3)
    long_upper = (_upper / _range > 0.6) & (body_pct < 0.3)
    df["pin_bar_score"] = (long_lower.astype(float) + long_upper.astype(float)).clip(0, 1)

    # Hammer: small body near top + long lower wick (bullish reversal)
    df["hammer_score"] = (_lower / _range > 0.55).astype(float) * (body_pct < 0.35).astype(float)

    # Engulfing scoring: uses existing _engulfing fn but as float score
    eng = _engulfing(o, h, l, c)
    df["engulfing_score"] = eng.astype(float)

    # Higher-Highs / Higher-Lows & Lower-Lows / Lower-Highs structure
    hh = (h > h.shift(1)).astype(float)
    hl = (l > l.shift(1)).astype(float)
    ll = (l < l.shift(1)).astype(float)
    lh = (h < h.shift(1)).astype(float)
    df["hh_hl_count_10"] = (hh * hl).rolling(10, min_periods=1).sum()  # bullish structure
    df["ll_lh_count_10"] = (ll * lh).rolling(10, min_periods=1).sum()  # bearish structure
    # Net swing structure: positive = bullish, negative = bearish
    df["swing_structure"] = df["hh_hl_count_10"] - df["ll_lh_count_10"]

    # ---- Phase 11 (Accuracy v2 Upgrades) ----

    # 1. RSI + Stochastic confluence: both overbought/oversold = strong signal
    rsi_bull = (df["rsi_14"] > 55).astype(float)
    rsi_bear = (df["rsi_14"] < 45).astype(float)
    stoch_bull = (df["stoch_k"] > 55).astype(float)
    stoch_bear = (df["stoch_k"] < 45).astype(float)
    df["rsi_stoch_confluence"] = (rsi_bull * stoch_bull) - (rsi_bear * stoch_bear)

    # 2. EMA ribbon width: distance between fastest and slowest EMA
    # Wide ribbon = strong trend, narrow = consolidation
    df["ema_ribbon_width"] = (ema20 - ema200).abs() / (c + 1e-10)

    # 3. Momentum acceleration: 2nd derivative of price change
    mom_1 = c.pct_change(1)
    mom_3 = c.pct_change(3)
    df["momentum_acceleration"] = (mom_1 - mom_1.shift(1)).fillna(0)

    # 4. Candle strength: ratio of body to total range, combined with direction
    # High body ratio close near high/low = strong conviction
    body_abs = (c - o).abs()
    full_range = h - l + 1e-10
    direction_sign = (c > o).astype(float) * 2 - 1  # +1 for green, -1 for red
    # Close near high when green (buying strength) or near low when red (selling strength)
    close_quality = np.where(
        c > o,
        (c - l) / full_range,  # green: close near high = good
        (h - c) / full_range   # red: close near low = good
    )
    df["candle_strength_ratio"] = (body_abs / full_range) * close_quality * direction_sign

    # 5. Volume confirming momentum: ticks increase with price direction
    if "tick_volume" in df.columns:
        tv = df["tick_volume"].astype(float)
        tv_change = tv.pct_change(1).fillna(0)
        price_dir = (c > c.shift(1)).astype(float) * 2 - 1
        # Positive when volume increases WITH price direction
        df["volume_momentum_confirm"] = (tv_change * price_dir).rolling(3, min_periods=1).mean()
    else:
        df["volume_momentum_confirm"] = 0.0

    # 6. Price rejection at key levels: wicks touching recent highs/lows
    recent_high_20 = h.rolling(20).max()
    recent_low_20 = l.rolling(20).min()
    # Near resistance rejection (wick at top, closes below)
    near_resist = (h >= recent_high_20 * 0.999) & (c < h - full_range * 0.3)
    near_support = (l <= recent_low_20 * 1.001) & (c > l + full_range * 0.3)
    df["price_rejection_level"] = near_support.astype(float) - near_resist.astype(float)

    # 7. Multi-bar trend quality: how consistent is the last 5 bars direction
    green_bars = (c > o).astype(float)
    df["multi_bar_trend_quality"] = green_bars.rolling(5, min_periods=2).mean() * 2 - 1
    # +1 = all green (strong uptrend), -1 = all red (strong downtrend)

    # 8. Intra-bar volatility ratio: current bar range vs average recent
    current_range = h - l
    avg_recent_range = current_range.rolling(10, min_periods=3).mean()
    df["intrabar_volatility_ratio"] = current_range / (avg_recent_range + 1e-10)

    # 9. Directional pressure: net of upper vs lower wicks across recent bars
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_wick_v2 = pd.concat([o, c], axis=1).min(axis=1) - l
    # Positive = more buying pressure (larger lower wicks = demand), negative = supply
    pressure = (lower_wick_v2 - upper_wick) / (full_range + 1e-10)
    df["directional_pressure"] = pressure.rolling(5, min_periods=1).mean()

    # 10. Smart money divergence: enhanced OBV vs price
    if "tick_volume" in df.columns:
        tv = df["tick_volume"].astype(float)
        obv_dir = pd.Series(0.0, index=df.index)
        obv_dir[c > c.shift(1)] = 1.0
        obv_dir[c < c.shift(1)] = -1.0
        obv_raw = (tv * obv_dir).cumsum()
        # 5-bar comparison: price direction vs OBV direction
        price_chg_5 = c.pct_change(5).fillna(0)
        obv_chg_5 = obv_raw.diff(5).fillna(0)
        # divergence: price up + OBV down (-1) or price down + OBV up (+1)
        df["smart_money_divergence"] = np.where(
            (price_chg_5 > 0) & (obv_chg_5 < 0), -1.0,
            np.where((price_chg_5 < 0) & (obv_chg_5 > 0), 1.0, 0.0)
        )
    else:
        df["smart_money_divergence"] = 0.0

    # ═══════════════════════════════════════════════════════════════
    # v11: Candle Freshness & Quality Features
    # ═══════════════════════════════════════════════════════════════
    
    # Candle freshness: how similar is each candle to its recent predecessors (vectorized)
    candle_dir = np.where(c.values >= o.values, 1.0, -1.0)
    candle_range = (h - l).values
    candle_range_safe = np.where(candle_range < 1e-10, 1e-10, candle_range)
    body_ratio = np.abs(c.values - o.values) / candle_range_safe

    freshness_vals = np.ones(len(df))
    for lag in range(1, 6):
        dir_prev = np.roll(candle_dir, lag)
        br_prev = np.roll(body_ratio, lag)
        rng_prev = np.roll(candle_range, lag)
        rng_prev_safe = np.where(rng_prev < 1e-10, 1e-10, rng_prev)

        dir_match = np.where(candle_dir == dir_prev, 1.0, 0.0)
        br_diff = 1.0 - np.abs(body_ratio - br_prev)
        rng_max = np.maximum(candle_range_safe, rng_prev_safe)
        rng_sim = 1.0 - np.minimum(np.abs(candle_range - rng_prev) / rng_max, 1.0)

        sim = 0.35 * dir_match + 0.25 * br_diff + 0.40 * rng_sim
        sim = np.clip(sim, 0, 1)
        freshness_vals -= sim / 5.0

    freshness_vals[:5] = 1.0  # not enough history
    freshness_vals[candle_range < 1e-10] = 0.0
    df["candle_freshness"] = pd.Series(freshness_vals, index=df.index).clip(0, 1)
    
    # Body quality: body/range ratio (0=doji, 1=marubozu)
    full_range = h - l
    df["candle_body_quality"] = (abs(c - o) / (full_range + 1e-10)).clip(0, 1)
    
    # Consecutive same-direction streak length  
    dirs = np.sign(c.values - o.values)
    streak = np.ones(len(dirs))
    for i in range(1, len(dirs)):
        if dirs[i] == dirs[i-1] and dirs[i] != 0:
            streak[i] = streak[i-1] + 1
    df["streak_length"] = pd.Series(streak, index=df.index).clip(0, 20)
    
    # Range vs recent average (detects abnormal candles)
    avg_range = full_range.rolling(10, min_periods=1).mean()
    df["candle_range_vs_avg"] = (full_range / (avg_range + 1e-10)).clip(0, 3)

    # v11.1: Hour-quality features (lets model learn hour effects natively)
    from session_filter import HOUR_CONFIDENCE_MULT
    hour_q = hour.map(HOUR_CONFIDENCE_MULT).fillna(0.90)
    df["hour_quality_score"] = hour_q
    df["hour_volatility_interaction"] = hour_q * df["volatility_zscore"]
    df["hour_momentum_interaction"] = hour_q * df["momentum_rolling_5"]

    # v11 Advanced: Strategy & Pattern Features (computed statistically)
    # Same-candle score: composite staleness indicator
    # Uses candle_freshness + candle_body_quality + range_vs_avg
    sc_fresh = df["candle_freshness"].fillna(0.5)
    sc_body_q = df["candle_body_quality"].fillna(0.5)
    sc_range_norm = df["candle_range_vs_avg"].clip(0.2, 3.0) / 3.0
    df["same_candle_score"] = (sc_fresh * 0.4 + sc_body_q * 0.3 + sc_range_norm * 0.3).clip(0, 1)

    # Crossover staleness: detect if MACD/SMA crossovers are fresh
    macd_vals = df["macd"].values
    macd_sig = df["macd_signal"].values
    sma_20 = close.rolling(20, min_periods=10).mean().values
    sma_50 = close.rolling(50, min_periods=25).mean().values
    # Simple proxy: abs change in MACD cross gap (larger change = fresher cross)
    macd_gap = np.abs(macd_vals - macd_sig)
    macd_gap_change = pd.Series(macd_gap).diff().abs().fillna(0).values
    sma_gap = np.abs(sma_20 - sma_50)
    sma_gap_change = pd.Series(sma_gap).diff().abs().fillna(0).values
    # Normalize to 0-1 range using percentile rank within rolling window
    mg_series = pd.Series(macd_gap_change)
    sg_series = pd.Series(sma_gap_change)
    mg_rank = mg_series.rolling(50, min_periods=10).apply(lambda x: (x[-1] >= x).mean(), raw=True).fillna(0.5)
    sg_rank = sg_series.rolling(50, min_periods=10).apply(lambda x: (x[-1] >= x).mean(), raw=True).fillna(0.5)
    df["crossover_staleness"] = ((mg_rank + sg_rank) / 2).clip(0, 1).values

    # Pattern score: use existing pattern-related features as proxy
    # Combine pin_bar_score, hammer_score, engulfing_score into normalized pattern signal
    ps = df.get("pin_bar_score", pd.Series(0, index=df.index)).fillna(0)
    hs = df.get("hammer_score", pd.Series(0, index=df.index)).fillna(0)
    es = df.get("engulfing_score", pd.Series(0, index=df.index)).fillna(0)
    raw_pattern = (ps + hs + es).clip(-3, 3)
    df["pattern_score"] = ((raw_pattern + 3) / 6).clip(0, 1)  # scale -3..+3 to 0..1

    # Strategy agreement: proxy from multi-indicator agreement
    # RSI direction, MACD direction, EMA slope direction agreement
    rsi_bull = (df["rsi_14"] < 50).astype(float)
    macd_bull = (df["macd"] > df["macd_signal"]).astype(float)
    ema_bull = (df["ema_slope"] > 0).astype(float)
    stoch_bull = (df["stoch_k"] < 50).astype(float)
    mom_bull = (df["momentum_rolling_5"] > 0).astype(float)
    agreement_raw = (rsi_bull + macd_bull + ema_bull + stoch_bull + mom_bull) / 5.0
    # Convert to agreement score: 0.5 = no agreement, 0/1 = full agreement
    df["strategy_agreement"] = (2 * (agreement_raw - 0.5).abs()).clip(0, 1)

    # Strategy composite: weighted agreement × confidence factors
    df["strategy_composite"] = (
        df["strategy_agreement"] * 0.4 +
        df["same_candle_score"] * 0.2 +
        df["crossover_staleness"] * 0.2 +
        df["pattern_score"] * 0.2
    ).clip(0, 1)

    # ═══════════════════════════════════════════════════════════════════════
    # v15: Advanced Features (30 new)
    # ═══════════════════════════════════════════════════════════════════════

    # --- Cross-timeframe RSI / BB / ADX ---
    # These are computed in compute_htf_features and merged via merge_asof
    # If not present (no HTF data), fall back to 0 / neutral
    if "m15_rsi" not in df.columns:
        df["m15_rsi"] = 50.0
    if "h1_rsi" not in df.columns:
        df["h1_rsi"] = 50.0
    if "m15_bb_position" not in df.columns:
        df["m15_bb_position"] = 0.0
    if "h1_adx" not in df.columns:
        df["h1_adx"] = 25.0

    # TF RSI agreement: how aligned are RSI values across timeframes
    rsi_m5 = df["rsi_14"].fillna(50)
    rsi_m15 = df["m15_rsi"].fillna(50)
    rsi_h1 = df["h1_rsi"].fillna(50)
    rsi_mean = (rsi_m5 + rsi_m15 + rsi_h1) / 3
    rsi_std = pd.concat([rsi_m5, rsi_m15, rsi_h1], axis=1).std(axis=1)
    df["tf_rsi_agreement"] = 1.0 - (rsi_std / 50.0).clip(0, 1)

    # --- Fractal / Market structure ---
    # Efficiency ratio (Kaufman): net price change / sum of all changes
    price_change_abs = c.diff().abs()
    net_change_10 = (c - c.shift(10)).abs()
    path_length_10 = price_change_abs.rolling(10, min_periods=2).sum()
    df["efficiency_ratio"] = (net_change_10 / (path_length_10 + 1e-10)).clip(0, 1)

    # Fractal dimension proxy (box-counting approximation via range/returns)
    log_range = np.log((h - l).clip(lower=1e-10))
    log_range_mean = log_range.rolling(20, min_periods=5).mean()
    log_range_std = log_range.rolling(20, min_periods=5).std()
    df["fractal_dimension"] = (log_range_std / (log_range_mean.abs() + 1e-10)).clip(0, 2)

    # Hurst exponent proxy: R/S analysis simplified
    # H > 0.5 = trending, H < 0.5 = mean-reverting
    ret_series = returns.fillna(0)
    def _rolling_hurst(vals, w=50):
        result = np.full(len(vals), 0.5)
        for i in range(w, len(vals)):
            window = vals[i-w:i]
            if np.std(window) < 1e-10:
                continue
            cumdev = np.cumsum(window - np.mean(window))
            r = np.max(cumdev) - np.min(cumdev)
            s = np.std(window)
            if s > 1e-10 and r > 0:
                result[i] = np.log(r / s + 1e-10) / np.log(w)
        return result
    df["hurst_exponent"] = pd.Series(
        _rolling_hurst(ret_series.values, 50), index=df.index
    ).clip(0, 1).fillna(0.5)

    # --- Wavelet / Frequency energy proxy ---
    # High-frequency energy: variance of first differences (noise)
    ret_diff = ret_series.diff().fillna(0)
    df["high_freq_energy"] = ret_diff.rolling(20, min_periods=5).var().fillna(0)
    # Low-frequency energy: variance of smoothed returns (trend)
    smooth_ret = ret_series.rolling(10, min_periods=3).mean()
    df["low_freq_energy"] = (smooth_ret.rolling(20, min_periods=5).var() /
                              (df["high_freq_energy"] + 1e-10)).clip(0, 10)

    # --- Advanced order flow ---
    if "tick_volume" in df.columns:
        tv = df["tick_volume"].astype(float).clip(lower=1)
        buy_vol = tv * (c > o).astype(float)
        sell_vol = tv * (c <= o).astype(float)
        total_vol_10 = tv.rolling(10, min_periods=1).sum()
        df["buy_pressure_ratio"] = buy_vol.rolling(10, min_periods=1).sum() / (total_vol_10 + 1e-10)
    else:
        df["buy_pressure_ratio"] = 0.5

    # Aggressive ratio: fraction of bars with >1 ATR move
    big_moves = (returns.abs() > df["atr_14"]).astype(float)
    df["aggressive_ratio"] = big_moves.rolling(20, min_periods=5).mean().fillna(0)

    # Absorption rate over 5 bars
    df["absorption_rate_5"] = df["wick_absorption"].rolling(5, min_periods=1).mean()

    # --- Regime quantification ---
    # Regime persistence: how many consecutive bars in same ADX regime
    trending = (df["adx"] > 25).astype(int)
    trend_change = trending.ne(trending.shift(1)).astype(int)
    trend_group = trend_change.cumsum()
    df["regime_persistence"] = trending.groupby(trend_group).cumcount() + 1

    # Regime transition probability: ADX near threshold = high transition prob
    df["regime_transition_prob"] = 1.0 - ((df["adx"] - 25).abs() / 25).clip(0, 1)

    # Volatility of volatility
    atr_returns = df["atr_14"].pct_change().fillna(0)
    df["volatility_of_volatility"] = atr_returns.rolling(20, min_periods=5).std().fillna(0)

    # --- Advanced candle patterns ---
    prev_h = h.shift(1)
    prev_l = l.shift(1)
    df["inside_bar"] = ((h <= prev_h) & (l >= prev_l)).astype(float)
    df["outside_bar"] = ((h > prev_h) & (l < prev_l)).astype(float)

    # Three-bar reversal: down-doji-up or up-doji-down
    body_1 = (c.shift(2) - o.shift(2))
    body_2_abs = (c.shift(1) - o.shift(1)).abs()
    range_2 = (h.shift(1) - l.shift(1)).clip(lower=1e-10)
    body_3 = (c - o)
    is_doji_mid = (body_2_abs / range_2) < 0.3
    bull_reversal_3 = (body_1 < 0) & is_doji_mid & (body_3 > 0)
    bear_reversal_3 = (body_1 > 0) & is_doji_mid & (body_3 < 0)
    df["three_bar_reversal"] = bull_reversal_3.astype(float) - bear_reversal_3.astype(float)

    # Morning/evening star (simplified): big down, small body, big up
    big_body_thresh = full_range.rolling(20, min_periods=5).mean() * 0.6
    big_down_1 = (o.shift(2) - c.shift(2)) > big_body_thresh.shift(2)
    small_body_2 = body_2_abs < big_body_thresh.shift(1) * 0.5
    big_up_3 = (c - o) > big_body_thresh
    morning = big_down_1 & small_body_2 & big_up_3
    big_up_1 = (c.shift(2) - o.shift(2)) > big_body_thresh.shift(2)
    big_down_3 = (o - c) > big_body_thresh
    evening = big_up_1 & small_body_2 & big_down_3
    df["morning_star_score"] = morning.astype(float) - evening.astype(float)

    # --- Momentum higher-order ---
    macd_hist = df["macd"] - df["macd_signal"]
    df["macd_histogram_accel"] = macd_hist.diff(1).fillna(0)

    # RSI Bollinger: RSI position within its own rolling BB
    rsi_vals = df["rsi_14"]
    rsi_ma = rsi_vals.rolling(20, min_periods=10).mean()
    rsi_std = rsi_vals.rolling(20, min_periods=10).std()
    rsi_upper = rsi_ma + 2 * rsi_std
    rsi_lower = rsi_ma - 2 * rsi_std
    df["rsi_bollinger"] = ((rsi_vals - rsi_lower) / (rsi_upper - rsi_lower + 1e-10)).clip(0, 1)

    # Stochastic RSI
    rsi_min_14 = rsi_vals.rolling(14, min_periods=5).min()
    rsi_max_14 = rsi_vals.rolling(14, min_periods=5).max()
    df["stoch_rsi"] = ((rsi_vals - rsi_min_14) / (rsi_max_14 - rsi_min_14 + 1e-10)).clip(0, 1)

    # --- Volume microstructure ---
    if "tick_volume" in df.columns:
        tv = df["tick_volume"].astype(float)
        tv_mean_20 = tv.rolling(20, min_periods=5).mean()
        tv_spike = tv > (tv_mean_20 * 2)  # volume > 2x average
        reversal_candle = (c.shift(1) > o.shift(1)) != (c > o)  # direction change
        df["volume_climax_score"] = (tv_spike & reversal_candle).astype(float)

        # Cumulative delta slope
        if "cumulative_delta" in df.columns:
            cd = df["cumulative_delta"]
            df["cumulative_delta_slope"] = (cd - cd.shift(5)) / (cd.shift(5).abs() + 1e-10)
            df["cumulative_delta_slope"] = df["cumulative_delta_slope"].clip(-10, 10).fillna(0)
        else:
            df["cumulative_delta_slope"] = 0.0
    else:
        df["volume_climax_score"] = 0.0
        df["cumulative_delta_slope"] = 0.0

    # --- Session / Time features ---
    if "time" in df.columns:
        dow = df["time"].dt.dayofweek  # 0=Monday
        df["day_of_week_sin"] = np.sin(2 * np.pi * dow / 5)
        df["day_of_week_cos"] = np.cos(2 * np.pi * dow / 5)

        # Minutes to session end (London=16:00, NY=21:00)
        hour_now = df["time"].dt.hour
        minute_now = df["time"].dt.minute
        total_min = hour_now * 60 + minute_now
        # Use closest major session close
        london_close = 16 * 60
        ny_close = 21 * 60
        dist_london = (london_close - total_min).clip(lower=0)
        dist_ny = (ny_close - total_min).clip(lower=0)
        min_dist = pd.concat([dist_london, dist_ny], axis=1).min(axis=1)
        df["minutes_to_session_end"] = (min_dist / 60.0).clip(0, 8)  # normalize to hours
    else:
        df["day_of_week_sin"] = 0.0
        df["day_of_week_cos"] = 0.0
        df["minutes_to_session_end"] = 4.0

    # ═══════════════════════════════════════════════════════════════════════
    # v18: 35+ Classic TA Indicators (Binary Options Optimized)
    # ═══════════════════════════════════════════════════════════════════════

    # --- Fast EMAs (scalping) ---
    ema5 = _ema(c, 5)
    ema8 = _ema(c, 8)
    ema13 = _ema(c, 13)
    df["ema_5"] = (c - ema5) / (c + 1e-10)
    df["ema_8"] = (c - ema8) / (c + 1e-10)
    df["ema_13"] = (c - ema13) / (c + 1e-10)

    # --- Moving average variants ---
    sma10 = c.rolling(10, min_periods=5).mean()
    sma30 = c.rolling(30, min_periods=15).mean()
    df["sma_10"] = (c - sma10) / (c + 1e-10)
    df["sma_30"] = (c - sma30) / (c + 1e-10)

    # DEMA(20) = 2*EMA(20) - EMA(EMA(20), 20)
    dema20 = 2 * ema20 - _ema(ema20, EMA_20)
    df["dema_20"] = (c - dema20) / (c + 1e-10)

    # TEMA(20) = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
    ema_ema20 = _ema(ema20, EMA_20)
    tema20 = 3 * ema20 - 3 * ema_ema20 + _ema(ema_ema20, EMA_20)
    df["tema_20"] = (c - tema20) / (c + 1e-10)

    # WMA(20): linearly weighted moving average
    weights_20 = np.arange(1, 21, dtype=float)
    df["wma_20"] = (c.rolling(20, min_periods=10).apply(
        lambda x: np.dot(x, weights_20[:len(x)]) / weights_20[:len(x)].sum(), raw=True
    ) - c) / (c + 1e-10) * -1  # distance from close

    # --- Multi-period RSI ---
    df["rsi_7"] = _rsi(c, 7)
    df["rsi_21"] = _rsi(c, 21)

    # --- Rate of Change ---
    df["roc_10"] = c.pct_change(10).fillna(0).clip(-0.1, 0.1)
    df["roc_20"] = c.pct_change(20).fillna(0).clip(-0.2, 0.2)

    # --- Parabolic SAR direction ---
    # Simplified: compare close to exponentially-weighted trailing stop
    sar_af_init, sar_af_max = 0.02, 0.20
    sar_dir = pd.Series(1.0, index=df.index)  # 1=bull, 0=bear
    sar_val = l.rolling(5, min_periods=1).min().values.copy()
    ep = h.values.copy()
    af = np.full(len(df), sar_af_init)
    c_vals, h_vals, l_vals = c.values, h.values, l.values
    for i in range(1, len(df)):
        if sar_dir.iloc[i-1] == 1:  # bullish
            sar_val[i] = sar_val[i-1] + af[i-1] * (ep[i-1] - sar_val[i-1])
            sar_val[i] = min(sar_val[i], l_vals[i-1], l_vals[max(0, i-2)])
            if h_vals[i] > ep[i-1]:
                ep[i] = h_vals[i]
                af[i] = min(af[i-1] + sar_af_init, sar_af_max)
            else:
                ep[i] = ep[i-1]
                af[i] = af[i-1]
            if l_vals[i] < sar_val[i]:
                sar_dir.iloc[i] = 0
                sar_val[i] = ep[i-1]
                ep[i] = l_vals[i]
                af[i] = sar_af_init
            else:
                sar_dir.iloc[i] = 1
        else:  # bearish
            sar_val[i] = sar_val[i-1] - af[i-1] * (sar_val[i-1] - ep[i-1])
            sar_val[i] = max(sar_val[i], h_vals[i-1], h_vals[max(0, i-2)])
            if l_vals[i] < ep[i-1]:
                ep[i] = l_vals[i]
                af[i] = min(af[i-1] + sar_af_init, sar_af_max)
            else:
                ep[i] = ep[i-1]
                af[i] = af[i-1]
            if h_vals[i] > sar_val[i]:
                sar_dir.iloc[i] = 1
                sar_val[i] = ep[i-1]
                ep[i] = h_vals[i]
                af[i] = sar_af_init
            else:
                sar_dir.iloc[i] = 0
    df["parabolic_sar_dir"] = sar_dir.values

    # --- Supertrend direction ---
    st_mult = 3.0
    st_period = 10
    hl2 = (h + l) / 2
    st_atr = _atr(h, l, c, st_period)
    upper_band = hl2 + st_mult * st_atr
    lower_band = hl2 - st_mult * st_atr
    supertrend = pd.Series(0.0, index=df.index)
    direction = pd.Series(1.0, index=df.index)
    final_ub = upper_band.copy()
    final_lb = lower_band.copy()
    for i in range(1, len(df)):
        if upper_band.iloc[i] < final_ub.iloc[i-1] or c.iloc[i-1] > final_ub.iloc[i-1]:
            final_ub.iloc[i] = upper_band.iloc[i]
        else:
            final_ub.iloc[i] = final_ub.iloc[i-1]
        if lower_band.iloc[i] > final_lb.iloc[i-1] or c.iloc[i-1] < final_lb.iloc[i-1]:
            final_lb.iloc[i] = lower_band.iloc[i]
        else:
            final_lb.iloc[i] = final_lb.iloc[i-1]
        if direction.iloc[i-1] == 1:
            if c.iloc[i] < final_lb.iloc[i]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = 1
        else:
            if c.iloc[i] > final_ub.iloc[i]:
                direction.iloc[i] = 1
            else:
                direction.iloc[i] = -1
    df["supertrend_dir"] = ((direction + 1) / 2).values  # 0=bear, 1=bull

    # --- Aroon ---
    aroon_period = 25
    df["aroon_up"] = h.rolling(aroon_period + 1, min_periods=aroon_period).apply(
        lambda x: x.argmax() / aroon_period * 100, raw=True).fillna(50) / 100.0
    df["aroon_down"] = l.rolling(aroon_period + 1, min_periods=aroon_period).apply(
        lambda x: x.argmin() / aroon_period * 100, raw=True).fillna(50) / 100.0
    df["aroon_osc"] = df["aroon_up"] - df["aroon_down"]

    # --- Vortex Indicator ---
    vm_plus = (h - l.shift(1)).abs()
    vm_minus = (l - h.shift(1)).abs()
    tr_v = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    vi_plus = vm_plus.rolling(14, min_periods=7).sum() / (tr_v.rolling(14, min_periods=7).sum() + 1e-10)
    vi_minus = vm_minus.rolling(14, min_periods=7).sum() / (tr_v.rolling(14, min_periods=7).sum() + 1e-10)
    df["vortex_diff"] = vi_plus - vi_minus

    # --- Keltner Channel ---
    keltner_ema = _ema(c, 20)
    keltner_atr = _atr(h, l, c, 10)
    keltner_upper = keltner_ema + 2 * keltner_atr
    keltner_lower = keltner_ema - 2 * keltner_atr
    df["keltner_position"] = ((c - keltner_lower) / (keltner_upper - keltner_lower + 1e-10)).clip(0, 1)
    df["keltner_width"] = (keltner_upper - keltner_lower) / (c + 1e-10)

    # --- Donchian Channel ---
    donchian_high = h.rolling(20, min_periods=10).max()
    donchian_low = l.rolling(20, min_periods=10).min()
    df["donchian_position"] = ((c - donchian_low) / (donchian_high - donchian_low + 1e-10)).clip(0, 1)
    df["donchian_width"] = (donchian_high - donchian_low) / (c + 1e-10)

    # --- MFI (Money Flow Index) ---
    if "tick_volume" in df.columns:
        tv_mfi = df["tick_volume"].astype(float).clip(lower=1)
        typical_price = (h + l + c) / 3
        raw_money_flow = typical_price * tv_mfi
        pos_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        neg_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
        pos_sum = pos_flow.rolling(14, min_periods=7).sum()
        neg_sum = neg_flow.rolling(14, min_periods=7).sum()
        mfi_ratio = pos_sum / (neg_sum + 1e-10)
        df["mfi_14"] = (100 - 100 / (1 + mfi_ratio)) / 100.0  # 0-1
    else:
        df["mfi_14"] = 0.5

    # --- CMF (Chaikin Money Flow) ---
    if "tick_volume" in df.columns:
        clv = ((c - l) - (h - c)) / (h - l + 1e-10)
        tv_cmf = df["tick_volume"].astype(float).clip(lower=1)
        df["cmf_20"] = (clv * tv_cmf).rolling(20, min_periods=10).sum() / (
            tv_cmf.rolling(20, min_periods=10).sum() + 1e-10)
    else:
        df["cmf_20"] = 0.0

    # --- Force Index ---
    if "tick_volume" in df.columns:
        fi = c.diff() * df["tick_volume"].astype(float)
        fi_ema = _ema(fi.fillna(0), 13)
        fi_std = fi_ema.rolling(50, min_periods=10).std()
        df["force_index_13"] = (fi_ema / (fi_std + 1e-10)).clip(-3, 3) / 3.0
    else:
        df["force_index_13"] = 0.0

    # --- Ultimate Oscillator ---
    bp = c - pd.concat([l, c.shift(1)], axis=1).min(axis=1)
    tr_uo = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    avg7 = bp.rolling(7, min_periods=4).sum() / (tr_uo.rolling(7, min_periods=4).sum() + 1e-10)
    avg14 = bp.rolling(14, min_periods=7).sum() / (tr_uo.rolling(14, min_periods=7).sum() + 1e-10)
    avg28 = bp.rolling(28, min_periods=14).sum() / (tr_uo.rolling(28, min_periods=14).sum() + 1e-10)
    df["ultimate_oscillator"] = ((4 * avg7 + 2 * avg14 + avg28) / 7.0).clip(0, 1)

    # --- TSI (True Strength Index) ---
    price_change = c.diff()
    double_smooth_pc = _ema(_ema(price_change.fillna(0), 25), 13)
    double_smooth_apc = _ema(_ema(price_change.abs().fillna(0), 25), 13)
    df["tsi"] = (double_smooth_pc / (double_smooth_apc + 1e-10)).clip(-1, 1)

    # --- Chande Momentum Oscillator ---
    gains_1 = price_change.clip(lower=0)
    losses_1 = (-price_change).clip(lower=0)
    sum_gains = gains_1.rolling(14, min_periods=7).sum()
    sum_losses = losses_1.rolling(14, min_periods=7).sum()
    df["chande_momentum"] = ((sum_gains - sum_losses) / (sum_gains + sum_losses + 1e-10)).clip(-1, 1)

    # --- PPO (Percentage Price Oscillator) ---
    ppo_fast = _ema(c, 12)
    ppo_slow = _ema(c, 26)
    df["ppo"] = ((ppo_fast - ppo_slow) / (ppo_slow + 1e-10) * 100).clip(-5, 5) / 5.0

    # --- Awesome Oscillator ---
    ao_fast = (h + l).rolling(5, min_periods=3).mean() / 2
    ao_slow = (h + l).rolling(34, min_periods=17).mean() / 2
    ao_raw = ao_fast - ao_slow
    ao_std = ao_raw.rolling(50, min_periods=10).std()
    df["awesome_oscillator"] = (ao_raw / (ao_std + 1e-10)).clip(-3, 3) / 3.0

    # --- Elder Ray Bull/Bear Power ---
    elder_ema = _ema(c, 13)
    bull_pwr = h - elder_ema
    bear_pwr = l - elder_ema
    bp_std = bull_pwr.rolling(50, min_periods=10).std()
    df["bull_power"] = (bull_pwr / (bp_std + 1e-10)).clip(-3, 3) / 3.0
    brp_std = bear_pwr.rolling(50, min_periods=10).std()
    df["bear_power"] = (bear_pwr / (brp_std + 1e-10)).clip(-3, 3) / 3.0

    # --- Pivot distance ---
    prev_h_piv = h.shift(1)
    prev_l_piv = l.shift(1)
    prev_c_piv = c.shift(1)
    pivot = (prev_h_piv + prev_l_piv + prev_c_piv) / 3
    df["pivot_distance"] = ((c - pivot) / (atr_raw + 1e-10)).clip(-5, 5) / 5.0

    # --- Linear regression slope & R² (20-bar) ---
    def _linreg_features(series, window=20):
        slope = pd.Series(np.nan, index=series.index)
        r2 = pd.Series(np.nan, index=series.index)
        x = np.arange(window, dtype=float)
        x_mean = x.mean()
        ss_xx = ((x - x_mean) ** 2).sum()
        vals = series.values
        for i in range(window, len(vals)):
            y = vals[i-window:i]
            if np.isnan(y).any():
                continue
            y_mean = y.mean()
            ss_xy = ((x - x_mean) * (y - y_mean)).sum()
            ss_yy = ((y - y_mean) ** 2).sum()
            b = ss_xy / (ss_xx + 1e-10)
            slope.iloc[i] = b
            r2.iloc[i] = (ss_xy ** 2) / (ss_xx * ss_yy + 1e-10)
        return slope.fillna(0), r2.fillna(0).clip(0, 1)
    lr_slope, lr_r2 = _linreg_features(c, 20)
    slope_std = lr_slope.rolling(50, min_periods=10).std()
    df["linreg_slope_20"] = (lr_slope / (slope_std + 1e-10)).clip(-3, 3) / 3.0
    df["linreg_r2_20"] = lr_r2

    # --- MACD Histogram (raw) ---
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    # --- Mass Index (squeeze detection) ---
    ema9_hl = _ema(h - l, 9)
    ema9_ema9_hl = _ema(ema9_hl, 9)
    ema_ratio = ema9_hl / (ema9_ema9_hl + 1e-10)
    df["mass_index"] = ema_ratio.rolling(25, min_periods=10).sum() / 25.0  # normalized around 1.0

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


def add_target_smoothed(df, lookahead=3):
    """
    Multi-bar smoothed target — majority direction over next `lookahead` bars.
    
    Instead of noisy single-bar close-to-close, uses the overall trend
    over the next 3 bars. This dramatically reduces M5 noise while
    preserving genuine directional signal.
    
    Label = 1 if majority of next `lookahead` bars are green
    Label = 0 if majority are red
    Tie = based on cumulative return sign
    """
    df = df.copy()
    close = df["close"]
    
    # Count green bars in next `lookahead` bars
    green_counts = pd.Series(0.0, index=df.index)
    for k in range(1, lookahead + 1):
        green_counts += (close.shift(-k) > close.shift(-k + 1)).astype(float)
    
    # Majority vote
    majority = lookahead / 2.0
    target = pd.Series(np.nan, index=df.index)
    target[green_counts > majority] = 1.0
    target[green_counts < majority] = 0.0
    
    # Tie-breaking: use cumulative return over lookahead
    tie_mask = green_counts == majority
    cum_return = close.shift(-lookahead) - close
    target[tie_mask & (cum_return > 0)] = 1.0
    target[tie_mask & (cum_return <= 0)] = 0.0
    
    # Last `lookahead` bars are NaN
    target.iloc[-lookahead:] = np.nan
    
    df["target"] = target
    valid = target.notna().sum()
    ones = (target == 1.0).sum()
    zeros = (target == 0.0).sum()
    log.info("Smoothed target (lookahead=%d): %d valid (%d green, %d red, %.1f%% green)",
             lookahead, valid, ones, zeros, ones / valid * 100 if valid else 0)
    return df


def add_primary_training_target(df):
    """
    Canonical primary target used across train/meta/weight/threshold stages.

    Keeping a single label definition prevents silent OOF index drift between
    the primary model and second-stage models.

    v10: Uses triple barrier labeling when enabled (TP=1.5×ATR, SL=1.0×ATR, max_bars=4).
    Falls back to smoothed target when disabled.
    """
    from config import TRIPLE_BARRIER_ENABLED
    if TRIPLE_BARRIER_ENABLED:
        return add_target_triple_barrier(df)
    return add_target_smoothed(df, lookahead=TARGET_LOOKAHEAD)


def add_target_atr_filtered(df, threshold=None):
    """
    ATR-threshold target (v5): only significant moves count.
    Moves within +/-threshold*ATR are marked NaN (dropped from training).
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


def add_target_triple_barrier(df, tp_mult=None, sl_mult=None, max_bars=None):
    """
    Triple Barrier Method (Phase 4) — industry-standard labeling.
    
    For each bar, check which barrier is hit first:
      - Take-profit: high[t+k] >= close[t] + tp_mult * ATR  -> label = 1
      - Stop-loss:   low[t+k]  <= close[t] - sl_mult * ATR  -> label = 0
      - Time barrier: k > max_bars -> label = NaN (ambiguous, dropped)
    
    This produces MUCH cleaner labels than close-to-close comparison.
    Used at DE Shaw, Citadel, Two Sigma.
    """
    from config import TRIPLE_BARRIER_TP, TRIPLE_BARRIER_SL, TRIPLE_BARRIER_MAX_BARS
    if tp_mult is None:
        tp_mult = TRIPLE_BARRIER_TP
    if sl_mult is None:
        sl_mult = TRIPLE_BARRIER_SL
    if max_bars is None:
        max_bars = TRIPLE_BARRIER_MAX_BARS

    df = df.copy()
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df["atr_14"].values
    n = len(df)

    targets = np.full(n, np.nan)

    for i in range(n - max_bars):
        entry = close[i]
        barrier_atr = atr[i]
        if np.isnan(barrier_atr) or barrier_atr <= 0:
            continue

        tp_level = entry + tp_mult * barrier_atr
        sl_level = entry - sl_mult * barrier_atr

        for k in range(1, max_bars + 1):
            j = i + k
            if j >= n:
                break
            # Check TP hit (bullish)
            if high[j] >= tp_level:
                targets[i] = 1.0
                break
            # Check SL hit (bearish)
            if low[j] <= sl_level:
                targets[i] = 0.0
                break
            # Neither hit — if last bar in window, mark NaN (ambiguous)

    df["target"] = targets
    valid = np.sum(~np.isnan(targets))
    total = n - max_bars
    tp_count = int(np.nansum(targets))
    sl_count = valid - tp_count
    log.info("Triple Barrier target: %d/%d rows (%.1f%% labeled, TP=%d, SL=%d, "
             "tp=%.1fxATR, sl=%.1fxATR, max_bars=%d)",
             valid, total, valid / total * 100 if total else 0,
             tp_count, sl_count, tp_mult, sl_mult, max_bars)
    return df

