# -*- coding: utf-8 -*-
"""
Strategy Scorer — v11 Nine-Strategy Signal Engine.

Runs 8 directional strategies + 1 risk management (Martingale) in parallel
on every M5 close. Each produces direction (BUY/SELL/NEUTRAL) and score (0-100).
Scores are weighted and summed into a composite signal.

Strategies:
  1. Triple Confluence Momentum (RSI + MACD + BB)
  2. Oversold Bounce with Trend (RSI + EMA + H1)
  3. Moving Average Crossover (SMA20 × SMA50)
  4. Bollinger Band Squeeze Breakout
  5. RSI Divergence
  6. MACD Histogram Reversal
  7. Support and Resistance Bounce
  8. Trend Continuation Pullback
  9. Martingale Money Management (stake sizing only)
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger("strategy_scorer")

# ── Strategy weights ────────────────────────────────────────────────────────
STRATEGY_WEIGHTS = {
    1: 0.18,  # Triple Confluence Momentum
    2: 0.12,  # Oversold Bounce with Trend
    3: 0.10,  # Moving Average Crossover
    4: 0.13,  # Bollinger Band Squeeze Breakout
    5: 0.11,  # RSI Divergence
    6: 0.10,  # MACD Histogram Reversal
    7: 0.11,  # Support and Resistance Bounce
    8: 0.10,  # Trend Continuation Pullback
    9: 0.05,  # Martingale (applied to sizing, not direction)
}

# ── Minimum thresholds ──────────────────────────────────────────────────────
MIN_STRATEGIES_AGREEING = 4
MIN_COMPOSITE_SCORE = 62


def _ema(series, period):
    """Compute EMA as numpy array."""
    return pd.Series(series).ewm(span=period, min_periods=max(period // 2, 1)).mean().values


def _sma(series, period):
    """Compute SMA as numpy array."""
    return pd.Series(series).rolling(period, min_periods=max(period // 2, 1)).mean().values


def _rsi(close, period=14):
    """Compute RSI value from close prices."""
    if len(close) < period + 1:
        return 50.0
    delta = np.diff(close[-(period + 1):])
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10
    rs = avg_gain / max(avg_loss, 1e-10)
    return float(100 - (100 / (1 + rs)))


def _macd(close, fast=12, slow=26, signal=9):
    """Compute MACD line, signal line, histogram."""
    if len(close) < slow + signal:
        return 0.0, 0.0, 0.0
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return float(macd_line[-1]), float(signal_line[-1]), float(hist[-1])


def _bollinger_bands(close, period=20, std_dev=2.0):
    """Compute BB upper, middle, lower, width."""
    sma = _sma(close, period)
    std = pd.Series(close).rolling(period, min_periods=period // 2).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    width = upper - lower
    return upper, sma, lower, width


def _atr(high, low, close, period=14):
    """Compute ATR."""
    if len(close) < period + 1:
        return float(np.mean(high[-5:] - low[-5:])) if len(high) >= 5 else 0.0001
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1]))
    )
    return float(pd.Series(tr).rolling(period).mean().iloc[-1])


def _crosses_above(fast, slow, lookback=2):
    """Check if fast crossed above slow within last `lookback` bars."""
    if len(fast) < lookback + 1 or len(slow) < lookback + 1:
        return False
    for i in range(-lookback, 0):
        if fast[i - 1] <= slow[i - 1] and fast[i] > slow[i]:
            return True
    return False


def _crosses_below(fast, slow, lookback=2):
    """Check if fast crossed below slow within last `lookback` bars."""
    if len(fast) < lookback + 1 or len(slow) < lookback + 1:
        return False
    for i in range(-lookback, 0):
        if fast[i - 1] >= slow[i - 1] and fast[i] < slow[i]:
            return True
    return False


def _swing_high(high, lookback=20):
    """Find swing high level from last N bars."""
    h = high[-lookback:]
    if len(h) < 5:
        return float(high[-1])
    return float(np.max(h))


def _swing_low(low, lookback=20):
    """Find swing low level from last N bars."""
    lo = low[-lookback:]
    if len(lo) < 5:
        return float(low[-1])
    return float(np.min(lo))


# ═══════════════════════════════════════════════════════════════════════════
#  Strategy 1: Triple Confluence Momentum (RSI + MACD + BB)
# ═══════════════════════════════════════════════════════════════════════════

def _strategy_1(close, high, low, **kw):
    """Triple Confluence Momentum."""
    rsi = _rsi(close, 14)
    macd_line, sig_line, hist = _macd(close)
    upper, mid, lower, width = _bollinger_bands(close)

    # Full MACD arrays for crossover detection
    ema_fast = _ema(close, 12)
    ema_slow = _ema(close, 26)
    macd_arr = ema_fast - ema_slow
    sig_arr = _ema(macd_arr, 9)

    buy_conds = 0
    sell_conds = 0

    # RSI
    if rsi < 25:
        buy_conds += 1
    elif rsi > 75:
        sell_conds += 1

    # MACD crossover (fresh — within 2 bars)
    if _crosses_above(macd_arr, sig_arr, 2):
        buy_conds += 1
    elif _crosses_below(macd_arr, sig_arr, 2):
        sell_conds += 1

    # Bollinger Bands
    if close[-1] <= lower[-1]:
        buy_conds += 1
    elif close[-1] >= upper[-1]:
        sell_conds += 1

    # Score: 3/3 = 100, 2/3 = 60, 1/3 = 0
    if buy_conds >= 3:
        return "BUY", 100
    elif buy_conds == 2:
        return "BUY", 60
    elif sell_conds >= 3:
        return "SELL", 100
    elif sell_conds == 2:
        return "SELL", 60
    return "NEUTRAL", 0


# ═══════════════════════════════════════════════════════════════════════════
#  Strategy 2: Oversold Bounce with Trend
# ═══════════════════════════════════════════════════════════════════════════

def _strategy_2(close, high, low, h1_direction="NEUTRAL", **kw):
    """Oversold Bounce with H1 trend gate."""
    rsi = _rsi(close, 14)
    ema20 = _ema(close, 20)

    buy = 0
    sell = 0

    if rsi < 30:
        buy += 1
    elif rsi > 70:
        sell += 1

    if close[-1] > ema20[-1]:
        buy += 1
    elif close[-1] < ema20[-1]:
        sell += 1

    if h1_direction == "UP":
        buy += 1
    elif h1_direction == "DOWN":
        sell += 1

    if buy >= 3:
        return "BUY", 100
    elif buy == 2:
        return "BUY", 50
    elif sell >= 3:
        return "SELL", 100
    elif sell == 2:
        return "SELL", 50
    return "NEUTRAL", 0


# ═══════════════════════════════════════════════════════════════════════════
#  Strategy 3: Moving Average Crossover
# ═══════════════════════════════════════════════════════════════════════════

def _strategy_3(close, high, low, **kw):
    """SMA 20/50 crossover with staleness decay."""
    if len(close) < 52:
        return "NEUTRAL", 0

    sma20 = _sma(close, 20)
    sma50 = _sma(close, 50)

    # Check for crossover in last 3 bars
    cross_age = None
    for bars_ago in range(1, 4):
        idx = -bars_ago
        prev_idx = idx - 1
        if len(sma20) >= abs(prev_idx) + 1:
            if sma20[prev_idx] <= sma50[prev_idx] and sma20[idx] > sma50[idx]:
                cross_age = bars_ago
                direction = "BUY"
                break
            elif sma20[prev_idx] >= sma50[prev_idx] and sma20[idx] < sma50[idx]:
                cross_age = bars_ago
                direction = "SELL"
                break

    if cross_age is None:
        return "NEUTRAL", 0

    # Staleness decay: fresh=100, 2 bars=60, 3 bars=30
    if cross_age == 1:
        score = 100
    elif cross_age == 2:
        score = 60
    else:
        score = 30

    # Price must be above/below SMA20
    if direction == "BUY" and close[-1] <= sma20[-1]:
        score = max(score - 40, 0)
    elif direction == "SELL" and close[-1] >= sma20[-1]:
        score = max(score - 40, 0)

    return direction, score


# ═══════════════════════════════════════════════════════════════════════════
#  Strategy 4: Bollinger Band Squeeze Breakout
# ═══════════════════════════════════════════════════════════════════════════

def _strategy_4(close, high, low, **kw):
    """BB squeeze + breakout + ATR expansion."""
    if len(close) < 25:
        return "NEUTRAL", 0

    upper, mid, lower, width = _bollinger_bands(close)
    atr_val = _atr(high, low, close)

    # Check if band width is at 20-bar minimum (squeeze)
    w20 = width[-20:]
    if len(w20) < 20:
        return "NEUTRAL", 0
    current_width = w20[-1]
    squeeze = current_width <= np.min(w20[:-1]) * 1.05  # within 5% of minimum

    # ATR rising (expansion)
    atr_series = pd.Series(
        np.maximum(high[1:] - low[1:],
                   np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    ).rolling(14).mean().values
    atr_rising = len(atr_series) >= 3 and atr_series[-1] > atr_series[-3]

    # Breakout detection
    if close[-1] > upper[-1]:
        direction = "BUY"
    elif close[-1] < lower[-1]:
        direction = "SELL"
    else:
        direction = "NEUTRAL"

    if direction == "NEUTRAL":
        return "NEUTRAL", 0

    conds = int(squeeze) + int(direction != "NEUTRAL") + int(atr_rising)
    if conds >= 3:
        return direction, 100
    elif conds == 2:
        return direction, 75
    return "NEUTRAL", 0


# ═══════════════════════════════════════════════════════════════════════════
#  Strategy 5: RSI Divergence
# ═══════════════════════════════════════════════════════════════════════════

def _strategy_5(close, high, low, **kw):
    """RSI divergence detection (5-10 bar window)."""
    if len(close) < 12:
        return "NEUTRAL", 0

    rsi_val = _rsi(close, 14)

    # Compute RSI series for last 12 bars
    rsi_series = []
    for i in range(max(len(close) - 12, 14), len(close)):
        rsi_series.append(_rsi(close[:i + 1], 14))
    if len(rsi_series) < 8:
        return "NEUTRAL", 0

    prices = close[-len(rsi_series):]
    rsis = np.array(rsi_series)

    # Find swing lows in price (bullish divergence)
    # Price lower low but RSI higher low
    p_min_idx = np.argmin(prices[-10:-1])
    if p_min_idx > 0:
        ref_price = prices[-10 + p_min_idx]
        ref_rsi = rsis[-10 + p_min_idx] if len(rsis) >= 10 else rsis[p_min_idx]
        current_price = prices[-1]
        current_rsi = rsis[-1]

        if current_price < ref_price and current_rsi > ref_rsi and rsi_val < 60:
            return "BUY", 100

    # Find swing highs in price (bearish divergence)
    p_max_idx = np.argmax(prices[-10:-1])
    if p_max_idx > 0:
        ref_price = prices[-10 + p_max_idx]
        ref_rsi = rsis[-10 + p_max_idx] if len(rsis) >= 10 else rsis[p_max_idx]
        current_price = prices[-1]
        current_rsi = rsis[-1]

        if current_price > ref_price and current_rsi < ref_rsi and rsi_val > 40:
            return "SELL", 100

    return "NEUTRAL", 0


# ═══════════════════════════════════════════════════════════════════════════
#  Strategy 6: MACD Histogram Reversal
# ═══════════════════════════════════════════════════════════════════════════

def _strategy_6(close, high, low, **kw):
    """MACD histogram momentum shift."""
    if len(close) < 30:
        return "NEUTRAL", 0

    ema_fast = _ema(close, 12)
    ema_slow = _ema(close, 26)
    macd_arr = ema_fast - ema_slow
    sig_arr = _ema(macd_arr, 9)
    hist = macd_arr - sig_arr

    if len(hist) < 4:
        return "NEUTRAL", 0

    current_hist = hist[-1]
    avg_magnitude = np.mean(np.abs(hist[-20:])) if len(hist) >= 20 else np.mean(np.abs(hist[-5:]))

    # BUY: histogram negative, increasing for 2+ bars, slope > threshold
    if current_hist < 0 and hist[-1] > hist[-2] > hist[-3]:
        slope = hist[-1] - hist[-2]
        if abs(slope) > 0.1 * avg_magnitude:
            return "BUY", 100
        return "BUY", 55

    # SELL: histogram positive, decreasing for 2+ bars
    if current_hist > 0 and hist[-1] < hist[-2] < hist[-3]:
        slope = hist[-2] - hist[-1]
        if abs(slope) > 0.1 * avg_magnitude:
            return "SELL", 100
        return "SELL", 55

    return "NEUTRAL", 0


# ═══════════════════════════════════════════════════════════════════════════
#  Strategy 7: Support and Resistance Bounce
# ═══════════════════════════════════════════════════════════════════════════

def _strategy_7(close, high, low, **kw):
    """S/R bounce with wick rejection + RSI confirmation."""
    if len(close) < 22:
        return "NEUTRAL", 0

    rsi = _rsi(close, 14)
    open_prices = kw.get("open", close)  # fallback to close if no open

    support = _swing_low(low, 20)
    resistance = _swing_high(high, 20)

    # Pip size estimation
    price = close[-1]
    if price > 50:
        pip = 0.1
    elif price > 1:
        pip = 0.0001
    else:
        pip = 0.00001
    zone_tol = 3 * pip

    body = abs(close[-1] - open_prices[-1])
    lower_wick = min(close[-1], open_prices[-1]) - low[-1]
    upper_wick = high[-1] - max(close[-1], open_prices[-1])

    buy_conds = 0
    sell_conds = 0

    # BUY: near support with bullish wick
    if abs(close[-1] - support) <= zone_tol:
        buy_conds += 1
    if body > 0 and lower_wick > 1.5 * body:
        buy_conds += 1
    if rsi < 55:
        buy_conds += 1

    # SELL: near resistance with bearish wick
    if abs(close[-1] - resistance) <= zone_tol:
        sell_conds += 1
    if body > 0 and upper_wick > 1.5 * body:
        sell_conds += 1
    if rsi > 45:
        sell_conds += 1

    if buy_conds >= 3:
        return "BUY", 100
    elif buy_conds == 2:
        return "BUY", 65
    elif sell_conds >= 3:
        return "SELL", 100
    elif sell_conds == 2:
        return "SELL", 65
    return "NEUTRAL", 0


# ═══════════════════════════════════════════════════════════════════════════
#  Strategy 8: Trend Continuation Pullback
# ═══════════════════════════════════════════════════════════════════════════

def _strategy_8(close, high, low, **kw):
    """EMA-200 trend + EMA-50 pullback + bounce confirmation."""
    if len(close) < 210:
        return "NEUTRAL", 0

    ema50 = _ema(close, 50)
    ema200 = _ema(close, 200)
    rsi = _rsi(close, 14)

    # Pip tolerance
    price = close[-1]
    if price > 50:
        pip = 0.1
    elif price > 1:
        pip = 0.0001
    else:
        pip = 0.00001
    pullback_tol = 5 * pip

    buy = 0
    sell = 0

    # BUY conditions
    if close[-1] > ema200[-1]:
        buy += 1
    if abs(close[-1] - ema50[-1]) <= pullback_tol:
        buy += 1
    if close[-1] > close[-2]:
        buy += 1
    if 35 <= rsi <= 60:
        buy += 1

    # SELL conditions
    if close[-1] < ema200[-1]:
        sell += 1
    if abs(close[-1] - ema50[-1]) <= pullback_tol:
        sell += 1
    if close[-1] < close[-2]:
        sell += 1
    if 40 <= rsi <= 65:
        sell += 1

    if buy >= 4:
        return "BUY", 100
    elif buy == 3:
        return "BUY", 60
    elif sell >= 4:
        return "SELL", 100
    elif sell == 3:
        return "SELL", 60
    return "NEUTRAL", 0


# ═══════════════════════════════════════════════════════════════════════════
#  Strategy 9: Martingale Money Management (not directional)
# ═══════════════════════════════════════════════════════════════════════════

# Martingale state — persisted across calls
_martingale_step = 0
_martingale_paused = False

MARTINGALE_BASE_STAKE_PCT = 1.0
MARTINGALE_MAX_STEP = 3
MARTINGALE_LADDER = [1.0, 2.0, 4.0, 8.0]
MARTINGALE_MIN_SCORE = 70  # only activate on high-confidence signals


def martingale_update(was_win: bool, daily_drawdown_pct: float = 0.0):
    """Update martingale state after trade result."""
    global _martingale_step, _martingale_paused
    if was_win:
        _martingale_step = 0
        _martingale_paused = False
    else:
        _martingale_step = min(_martingale_step + 1, MARTINGALE_MAX_STEP)
        if _martingale_step >= MARTINGALE_MAX_STEP:
            _martingale_paused = True
    if daily_drawdown_pct > 8.0:
        _martingale_paused = True


def martingale_stake(composite_score: float) -> float:
    """Get stake multiplier. Martingale only activates on high-confidence."""
    if _martingale_paused or composite_score < MARTINGALE_MIN_SCORE:
        return MARTINGALE_BASE_STAKE_PCT
    idx = min(_martingale_step, len(MARTINGALE_LADDER) - 1)
    return MARTINGALE_BASE_STAKE_PCT * MARTINGALE_LADDER[idx]


def get_martingale_state() -> dict:
    """Return current martingale state."""
    return {
        "step": _martingale_step,
        "paused": _martingale_paused,
        "stake_multiplier": MARTINGALE_LADDER[min(_martingale_step, len(MARTINGALE_LADDER) - 1)],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Composite Scorer — Run all strategies
# ═══════════════════════════════════════════════════════════════════════════

_STRATEGIES = {
    1: _strategy_1,
    2: _strategy_2,
    3: _strategy_3,
    4: _strategy_4,
    5: _strategy_5,
    6: _strategy_6,
    7: _strategy_7,
    8: _strategy_8,
}

STRATEGY_NAMES = {
    1: "Triple Confluence",
    2: "Oversold Bounce",
    3: "MA Crossover",
    4: "BB Squeeze",
    5: "RSI Divergence",
    6: "MACD Histogram",
    7: "S/R Bounce",
    8: "Trend Pullback",
    9: "Martingale (sizing)",
}


def score_strategies(df: pd.DataFrame, h1_direction: str = "NEUTRAL",
                     regime: str = "TRANSITION",
                     regime_weights: dict = None) -> dict:
    """
    Run all 8 directional strategies + martingale on provided DataFrame.

    Args:
        df: M5 DataFrame with OHLCV columns
        h1_direction: H1 trend direction for strategy gating
        regime: Current market regime for weight overrides
        regime_weights: Optional {strategy_id: new_weight} overrides

    Returns:
        dict with:
            per_strategy: {id: {name, direction, score, weight}}
            composite_score: 0-100 weighted
            composite_direction: BUY/SELL/NEUTRAL
            strategies_agreeing: count of strategies agreeing with composite
            min_agreement_met: bool (>= 4 strategies agree)
            min_score_met: bool (>= 62 composite)
            signal_valid: bool (both minimums met)
            martingale: {step, paused, stake_multiplier}
            details: human-readable summary
    """
    if df is None or len(df) < 30:
        return _empty_result()

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    open_prices = df["open"].values if "open" in df.columns else close

    # Active weights (regime overrides)
    weights = dict(STRATEGY_WEIGHTS)
    if regime_weights:
        for sid, w in regime_weights.items():
            if sid in weights:
                weights[sid] = w

    # Run each strategy
    per_strategy = {}
    buy_score = 0.0
    sell_score = 0.0
    buy_count = 0
    sell_count = 0

    for sid, func in _STRATEGIES.items():
        try:
            kwargs = {"open": open_prices, "h1_direction": h1_direction}
            direction, score = func(close, high, low, **kwargs)
        except Exception as e:
            log.debug("Strategy %d error: %s", sid, e)
            direction, score = "NEUTRAL", 0

        w = weights.get(sid, 0.10)
        per_strategy[sid] = {
            "name": STRATEGY_NAMES.get(sid, f"Strategy {sid}"),
            "direction": direction,
            "score": score,
            "weight": w,
        }

        if direction == "BUY":
            buy_score += score * w
            buy_count += 1
        elif direction == "SELL":
            sell_score += score * w
            sell_count += 1

    # Determine composite direction
    if buy_score > sell_score and buy_count > 0:
        composite_direction = "BUY"
        raw_score = buy_score
        agreeing = buy_count
    elif sell_score > buy_score and sell_count > 0:
        composite_direction = "SELL"
        raw_score = sell_score
        agreeing = sell_count
    else:
        composite_direction = "NEUTRAL"
        raw_score = 0
        agreeing = 0

    # Normalize composite to 0-100
    max_possible = sum(100 * w for w in weights.values() if w > 0)
    composite = min(100, raw_score / max_possible * 100) if max_possible > 0 else 0

    min_agree = agreeing >= MIN_STRATEGIES_AGREEING
    min_score = composite >= MIN_COMPOSITE_SCORE
    valid = min_agree and min_score and composite_direction != "NEUTRAL"

    # Strategy 9: Martingale state
    mart = get_martingale_state()
    per_strategy[9] = {
        "name": STRATEGY_NAMES[9],
        "direction": "N/A",
        "score": 0,
        "weight": STRATEGY_WEIGHTS[9],
    }

    # Build details
    strat_votes = []
    for sid in sorted(per_strategy):
        s = per_strategy[sid]
        if s["direction"] not in ("NEUTRAL", "N/A"):
            strat_votes.append(f"S{sid}:{s['direction'][0]}({s['score']})")

    return {
        "per_strategy": per_strategy,
        "composite_score": round(composite, 1),
        "composite_direction": composite_direction,
        "strategies_agreeing": agreeing,
        "min_agreement_met": min_agree,
        "min_score_met": min_score,
        "signal_valid": valid,
        "martingale": mart,
        "details": " | ".join(strat_votes) if strat_votes else "No signals",
    }


def _empty_result():
    """Return empty result when insufficient data."""
    return {
        "per_strategy": {},
        "composite_score": 0,
        "composite_direction": "NEUTRAL",
        "strategies_agreeing": 0,
        "min_agreement_met": False,
        "min_score_met": False,
        "signal_valid": False,
        "martingale": get_martingale_state(),
        "details": "Insufficient data",
    }


def get_regime_weight_overrides(regime: str) -> dict:
    """
    Return strategy weight overrides based on current market regime.
    Boosts strategies suited to regime, suppresses inappropriate ones.
    """
    if regime == "TREND":
        return {2: 0.20, 3: 0.18, 8: 0.18, 4: 0.05, 5: 0.05}
    elif regime == "RANGE":
        return {1: 0.22, 5: 0.18, 7: 0.18, 3: 0.05, 8: 0.05}
    elif regime == "HIGH_VOL":
        return {4: 0.25, 6: 0.18, 7: 0.05}
    elif regime == "CHOPPY":
        return {}  # all suspended — handled by caller
    return {}  # TRANSITION: default weights
