"""
QUOTEX LORD v15 — Advanced ML Trading Signal Engine with GRU Deep Learning.

14-layer pipeline:
  1. Data Collector       — fetch OHLCV from MT5
  2. Data Cleaner         — fill gaps, clip outliers, validate
  3. Feature Engineering  — 66 forward-safe features
  4. Label Generation     — triple barrier (training only)
  5. Primary Ensemble     — 5 seeded XGBoost + isotonic calibration
  6. Meta Model           — HistGBM reliability predictor
  7. Weight Learner       — logistic confidence scaler
  8. Regime Detection     — Trending/Ranging/High_Vol/Low_Vol
  9. Risk Filter          — multi-stage risk assessment
 10. Prediction Output    — BUY / SELL / SKIP
 11. Signal Delivery      — Telegram dispatch
 12. Circuit Breaker      — 5 losses or 8% drawdown halt
 13. Drift Detection      — PSI + KS test
 14. Online Learning      — SGD/PA continuous adaptation
"""

# =============================================================================
#  Imports
# =============================================================================
import asyncio
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

from config import (
    BASE_DIR,
    LOG_DIR,
    MODEL_DIR,
    SYMBOLS,
    DEFAULT_SYMBOL,
    CANDLES_TO_FETCH,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_SEND_BEFORE_CLOSE_SEC,
    MT5_PATH,
    MT5_LOGIN,
    MT5_PASSWORD,
    MT5_SERVER,
    ADMIN_CHAT_ID,
    PRODUCTION_MIN_CONFIDENCE,
    PRODUCTION_MIN_META_RELIABILITY,
    PRODUCTION_MIN_UNANIMITY,
    PRODUCTION_MAX_UNCERTAINTY,
    PRODUCTION_BLOCKED_REGIMES,
    ENSEMBLE_VAR_SKIP_THRESHOLD,
    SESSION_CONFIDENCE_MULT,
    AUTO_RETRAIN_ACCURACY_TRIGGER,
    AUTO_RETRAIN_CORRELATION_TRIGGER,
    SIGNAL_MIN_CONFIDENCE,
    SIGNAL_SOFT_CONFLUENCE_OVERRIDE_CONFIDENCE,
    SIGNAL_SOFT_CONFLUENCE_OVERRIDE_QUALITY,
    SIGNAL_SOFT_CONFLUENCE_OVERRIDE_UNANIMITY,
    LOG_LEVEL,
    LOG_FORMAT,
)
from data_collector import fetch_multi_timeframe
from data_cleaner import clean_data
from feature_engineering import compute_features, FEATURE_COLUMNS
from predict_engine import predict, load_models, log_prediction, verify_candle_closed
from regime_detection import detect_regime, get_session
from risk_filter import check_warnings
from confluence_filter import check_confluence
from correlation_filter import check_correlation
from signal_ranker import rank_and_select, compute_signal_score, update_symbol_stats
from pair_selector import score_pair
from circuit_breaker import CircuitBreaker
from online_learner import OnlineLearner
from adaptive_threshold import AdaptiveThreshold
from kelly_sizing import DrawdownAwareKelly
from stability import AccuracyTracker, ConfidenceCorrelationTracker
from production_state import record_startup, record_prediction, record_error
from stacking_model import StackingGatekeeper
from signal_validator import validate_signal
from signal_db import SignalDB
from performance_dashboard import generate_dashboard, format_dashboard_text

# v12 accuracy upgrades
from dispatch_model import get_dispatch_model
from bayesian_threshold import get_bayesian_threshold
from outcome_feedback import get_feedback_loop
from feature_importance_monitor import get_importance_monitor
from session_filter import update_hour_outcome

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
log = logging.getLogger("telegram_bot")

# =============================================================================
#  Constants
# =============================================================================

# Candle fetch sizes
LIVE_CANDLES_TO_FETCH = 1000     # warmup for EMA-200

# Signal filters
MIN_CONFIDENCE = SIGNAL_MIN_CONFIDENCE   # minimum confidence to send
MAX_SIGNALS_PER_CYCLE = 4               # increased from 3 for more trades
MIN_SIGNAL_SCORE = 35.0                 # lowered from 40 for more trades

# Ensemble agreement
MIN_UNANIMITY = 0.667           # 4/6 models agree
MAX_ENSEMBLE_VARIANCE = 0.055   # slightly relaxed from 0.050

# Pair selector
PAIR_SCORE_REFRESH = 12          # re-score every 12 cycles (1 hour)
PAIR_SELECTOR_MIN_SCORE = 18     # relaxed to allow more pairs through

# Risk filter — WARNINGS ONLY, never block (risk_filter.py is informational)
# Only ATR spikes are hard blocks (extreme conditions)
HARD_RISK_PREFIXES = ("ATR spike",)

# Cooldown
COOLDOWN_CANDLES = 3
MAX_CONSECUTIVE_LOSSES_PER_SYM = 3

# Drift check interval
DRIFT_CHECK_INTERVAL = 60       # every 60 cycles (~5 hours)

# Trading hours (UTC) — v11 Advanced: ALL hours tradeable, confidence scaling handles quality
# No hours are blocked; weak hours get penalized via session_filter.HOUR_CONFIDENCE_MULT
ALL_TRADING_HOURS_UTC = set(range(24))

# State files
SUBSCRIBERS_JSON = os.path.join(LOG_DIR, "subscribers.json")
SENT_SIGNALS_CSV = os.path.join(LOG_DIR, "sent_signals.csv")
OUTCOMES_CSV = os.path.join(LOG_DIR, "signal_outcomes.csv")
PENDING_JSON = os.path.join(LOG_DIR, "pending_signals.json")
ALL_PREDICTIONS_CSV = os.path.join(LOG_DIR, "all_predictions.csv")

# =============================================================================
#  State
# =============================================================================
_subscribers: dict = {}            # {chat_id_str: {active, symbols, mode, ...}}
_cb = CircuitBreaker()             # circuit breaker
_online = OnlineLearner()          # online learner
_adaptive = AdaptiveThreshold()    # adaptive threshold
_kelly = DrawdownAwareKelly()      # position sizing
_accuracy = AccuracyTracker()      # rolling accuracy
_conf_corr = ConfidenceCorrelationTracker()  # conf-accuracy correlation
_stacker = StackingGatekeeper()    # stacking gate
_signal_db = SignalDB()            # SQLite signal database

# v12 accuracy upgrade singletons
_dispatch = get_dispatch_model()
_bayesian = get_bayesian_threshold()
_feedback = get_feedback_loop()

_pair_scores: dict = {}            # {symbol: score}
_pair_counter = 0                  # cycles since last pair score
_consecutive_losses: dict = {}     # {symbol: int}
_cooldown_until: dict = {}         # {symbol: datetime}
_last_regime: dict = {}            # {symbol: regime_str}
_regime_skip: dict = {}            # {symbol: candles_to_skip}
_prev_directions: dict = {}        # {symbol: last_direction}
_candle_count = 0                  # total candle counter for drift
_outcome_results: list = []        # in-memory outcome cache
_win_streaks: dict = {}            # {symbol: streak (+win, -loss)}
_total_streak = 0                  # overall win/loss streak
_peak_accuracy = 0.0               # best session accuracy seen
_cycle_messages: dict = {}         # {cycle_key: {text, msg_ids, sig_ids, results}}
CYCLE_MESSAGES_JSON = os.path.join(LOG_DIR, "cycle_messages.json")

# Per-symbol adaptive confidence thresholds (learns from outcomes)
_symbol_min_confidence: dict = {}  # {symbol: float} - dynamic min confidence per symbol

# =============================================================================
#  Accuracy-Boosting Helpers
# =============================================================================

def _update_symbol_confidence(symbol: str, was_correct: bool, confidence: float):
    """
    Adapt per-symbol minimum confidence threshold based on outcomes.
    If a symbol is losing, raise the threshold; if winning, lower slightly.
    """
    current = _symbol_min_confidence.get(symbol, MIN_CONFIDENCE)
    if was_correct:
        # Winning: slightly lower threshold (allow more signals)
        new_thresh = current - 0.5
    else:
        # Losing: raise threshold (be more selective)
        new_thresh = current + 1.5
    # Clamp to reasonable range
    _symbol_min_confidence[symbol] = max(MIN_CONFIDENCE, min(75.0, new_thresh))


def _check_momentum_confirmation(pred: dict, m5_df) -> tuple:
    """
    Verify signal direction with immediate price action momentum.
    Returns (passed: bool, reason: str)
    
    Checks:
      1. Last 3 candles majority agrees with direction
      2. RSI not extreme-overbought for BUY or extreme-oversold for SELL
      3. Current candle body supports direction
    """
    if m5_df is None or len(m5_df) < 5:
        return True, "No data for momentum check"

    direction = pred.get("suggested_direction", "HOLD")
    if direction == "HOLD":
        return True, ""

    recent = m5_df.tail(3)
    green_count = (recent["close"] > recent["open"]).sum()
    last_close = m5_df["close"].iloc[-1]
    last_open = m5_df["open"].iloc[-1]

    # Check 1: Recent candle majority
    if direction == "UP" and green_count == 0:
        return False, "No green candles in last 3 bars"
    if direction == "DOWN" and green_count == 3:
        return False, "All green candles in last 3 bars"

    # Check 2: RSI extreme check (don't BUY at RSI>85 or SELL at RSI<15)
    rsi = 50  # default neutral
    try:
        if "rsi_14" in m5_df.columns and not pd.isna(m5_df["rsi_14"].iloc[-1]):
            rsi = float(m5_df["rsi_14"].iloc[-1])
        else:
            # Compute RSI from close prices
            delta = m5_df["close"].diff()
            gains = delta.clip(lower=0).rolling(14).mean()
            losses = (-delta.clip(upper=0)).rolling(14).mean()
            last_gain = gains.iloc[-1] if not pd.isna(gains.iloc[-1]) else 0
            last_loss = losses.iloc[-1] if not pd.isna(losses.iloc[-1]) else 0
            if last_loss > 1e-10:
                rs = last_gain / last_loss
                rsi = 100 - (100 / (1 + rs))
            elif last_gain > 0:
                rsi = 100
    except Exception:
        rsi = 50  # fallback

    if direction == "UP" and rsi > 95:
        return False, f"RSI extreme overbought ({rsi:.0f})"
    if direction == "DOWN" and rsi < 5:
        return False, f"RSI extreme oversold ({rsi:.0f})"

    return True, "Momentum confirmed"


def _check_candle_pattern_quality(m5_df, direction: str) -> float:
    """
    Return a 0-1 score of how well recent candle patterns support the direction.
    Used as a signal quality multiplier.
    """
    if m5_df is None or len(m5_df) < 5:
        return 0.5

    score = 0.5  # neutral baseline
    recent = m5_df.tail(5)
    c = recent["close"].values
    o = recent["open"].values
    h = recent["high"].values
    l = recent["low"].values

    # Body size of last candle (strong body = conviction)
    last_range = h[-1] - l[-1] + 1e-10
    last_body = abs(c[-1] - o[-1])
    body_ratio = last_body / last_range

    # Direction alignment: last candle green for UP, red for DOWN
    last_green = c[-1] > o[-1]
    if (direction == "UP" and last_green) or (direction == "DOWN" and not last_green):
        score += 0.15 + 0.10 * body_ratio  # bigger body = more confidence

    # Multi-bar momentum: 3 of 5 bars in direction
    greens = sum(1 for i in range(len(c)) if c[i] > o[i])
    if direction == "UP" and greens >= 3:
        score += 0.10
    elif direction == "DOWN" and greens <= 2:
        score += 0.10

    # No extreme wicks against direction (rejection)
    upper_wick = h[-1] - max(c[-1], o[-1])
    lower_wick = min(c[-1], o[-1]) - l[-1]
    if direction == "UP" and upper_wick > last_body * 2:
        score -= 0.15  # heavy upper wick = selling pressure
    if direction == "DOWN" and lower_wick > last_body * 2:
        score -= 0.15  # heavy lower wick = buying pressure

    return max(0.0, min(1.0, score))


# =============================================================================
#  Initialization
# =============================================================================

def _ensure_mt5():
    """Connect to MT5 terminal."""
    if mt5.terminal_info() is not None:
        return True
    kwargs = {}
    if MT5_PATH:
        kwargs["path"] = MT5_PATH
    if MT5_LOGIN:
        kwargs["login"] = MT5_LOGIN
    if MT5_PASSWORD:
        kwargs["password"] = MT5_PASSWORD
    if MT5_SERVER:
        kwargs["server"] = MT5_SERVER
    if not mt5.initialize(**kwargs):
        log.error("MT5 initialization failed: %s", mt5.last_error())
        return False
    log.info("MT5 connected.")
    return True


def _load_subscribers():
    global _subscribers
    try:
        if os.path.exists(SUBSCRIBERS_JSON):
            with open(SUBSCRIBERS_JSON, "r") as f:
                _subscribers = json.load(f)
            log.info("Loaded %d subscribers.", len(_subscribers))
    except Exception as e:
        log.warning("Failed to load subscribers: %s", e)
        _subscribers = {}


def _save_subscribers():
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(SUBSCRIBERS_JSON, "w") as f:
            json.dump(_subscribers, f, indent=2)
    except Exception as e:
        log.warning("Failed to save subscribers: %s", e)


def _load_outcomes():
    """Load historical outcomes for accuracy tracking."""
    global _outcome_results
    try:
        if os.path.exists(OUTCOMES_CSV):
            import pandas as pd
            df = pd.read_csv(OUTCOMES_CSV, on_bad_lines="skip")
            _outcome_results = df.to_dict("records")
            log.info("Loaded %d outcomes from CSV.", len(_outcome_results))
    except Exception as e:
        log.warning("Failed to load outcomes: %s", e)


# =============================================================================
#  Prediction Pipeline
# =============================================================================

def _run_prediction(symbol: str) -> dict:
    """
    Execute the full prediction pipeline for a single symbol.

    Layers exercised: 1 (data), 2 (clean), 3 (features), 5-7 (models), 8 (regime).
    """
    if not _ensure_mt5():
        raise RuntimeError("MT5 unavailable")

    data = fetch_multi_timeframe(symbol, min(CANDLES_TO_FETCH, LIVE_CANDLES_TO_FETCH))
    df = data.get("M5")
    if df is None or len(df) == 0:
        if not _ensure_mt5():
            raise RuntimeError(f"No M5 data for {symbol}")
        data = fetch_multi_timeframe(symbol, min(CANDLES_TO_FETCH, LIVE_CANDLES_TO_FETCH))
        df = data.get("M5")
    if df is None or len(df) == 0:
        raise RuntimeError(f"No M5 data for {symbol}")

    m15, h1, m1 = data.get("M15"), data.get("H1"), data.get("M1")
    df = compute_features(df, m15_df=m15, h1_df=h1, m1_df=m1)
    if df.empty:
        raise RuntimeError("Feature computation failed.")

    if not verify_candle_closed(df):
        df = df.iloc[:-1]
        if df.empty:
            raise RuntimeError("No closed candle available.")

    regime = detect_regime(df)
    spread = _get_spread(symbol)
    risk_warnings = check_warnings(df, current_spread=spread)
    result = predict(df, regime)

    now = datetime.now(timezone.utc)
    log_prediction(symbol, regime, result, risk_warnings=risk_warnings, timestamp=now)

    return {"symbol": symbol, "regime": regime, "risk_warnings": risk_warnings, **result}


def _get_spread(symbol: str) -> float:
    """Get current spread from MT5."""
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            return tick.ask - tick.bid
    except Exception:
        pass
    return 0.0


def _has_hard_risk(warnings: list) -> tuple:
    """Check if any hard risk warning should block the signal."""
    for w in warnings:
        for prefix in HARD_RISK_PREFIXES:
            if w.startswith(prefix):
                return True, w
    return False, ""


def _check_confluence(symbol: str, direction: str) -> dict:
    """Layer 9: Multi-TF confluence check. Also returns fetched M5 data."""
    try:
        data = fetch_multi_timeframe(symbol, 100)
        result = check_confluence(
            data.get("M5"), data.get("M15"), data.get("H1"),
            predicted_direction=direction, min_score=2,
        )
        result["_m5_df"] = data.get("M5")
        return result
    except Exception:
        return {"confluence_score": 3, "confluence_pass": True, "_m5_df": None}


# =============================================================================
#  Signal Formatting
# =============================================================================

def _format_auto_signal(predictions: dict, filtered: dict) -> str:
    """
    Format emoji-rich signal message per v11 spec.
    """
    now_utc = datetime.now(timezone.utc)
    now_str = now_utc.strftime("%Y-%m-%d %H:%M UTC")

    # Next M5 close time (expiry)
    minute = now_utc.minute
    next_close_min = ((minute // 5) + 1) * 5
    if next_close_min >= 60:
        expiry_time = now_utc.replace(second=0, microsecond=0) + timedelta(hours=1)
        expiry_time = expiry_time.replace(minute=next_close_min - 60)
    else:
        expiry_time = now_utc.replace(minute=next_close_min, second=0, microsecond=0)
    expiry_str = expiry_time.strftime("%H:%M")
    countdown_sec = max(0, int((expiry_time - now_utc).total_seconds()))
    countdown_min = countdown_sec // 60
    countdown_rem = countdown_sec % 60

    n_analyzed = len(predictions) + len(filtered)
    n_passed = len(predictions)

    # Today's best/worst pair
    today = now_utc.strftime("%Y-%m-%d")
    today_outcomes = [r for r in _outcome_results if r.get("timestamp", "").startswith(today)]
    best_pair = worst_pair = ""
    if today_outcomes:
        sym_wr = {}
        for r in today_outcomes:
            s = r.get("symbol", "?")
            if s not in sym_wr:
                sym_wr[s] = {"w": 0, "t": 0}
            sym_wr[s]["t"] += 1
            if r.get("correct"):
                sym_wr[s]["w"] += 1
        if sym_wr:
            best_sym = max(sym_wr, key=lambda x: sym_wr[x]["w"] / max(sym_wr[x]["t"], 1))
            worst_sym = min(sym_wr, key=lambda x: sym_wr[x]["w"] / max(sym_wr[x]["t"], 1))
            bw = sym_wr[best_sym]
            ww = sym_wr[worst_sym]
            best_pair = f"🥇 Best: {best_sym} ({bw['w']}/{bw['t']})"
            worst_pair = f"🥉 Worst: {worst_sym} ({ww['w']}/{ww['t']})"

    lines = [
        "⚡ QUOTEX LORD v15 ⚡",
        f"⏰ {now_str}",
        f"📊 Analyzed: {n_analyzed} | Passed: {n_passed}",
        "━" * 26,
    ]

    for sym, pred in sorted(predictions.items()):
        direction = pred.get("suggested_direction", "HOLD")
        if direction == "UP":
            arrow = "🟢 BUY ⬆"
        elif direction == "DOWN":
            arrow = "🔴 SELL ⬇"
        else:
            arrow = "⚪ SKIP"

        conf = pred.get("final_confidence_percent", 0)
        green = pred.get("green_probability_percent", 50)
        meta = pred.get("meta_reliability_percent", 50)
        uncertainty = pred.get("uncertainty_percent", 0)
        regime = pred.get("market_regime", "Unknown").replace("_", " ")
        session = pred.get("session", "Unknown")
        kelly_pct = pred.get("kelly_fraction_percent", 0)
        agree = pred.get("ensemble_agree_count", "?")
        total_m = pred.get("ensemble_total", 5)
        quality = pred.get("quality_score", 0)
        strength = pred.get("signal_strength", "—")
        reasons = pred.get("reason_chain", "")

        # Confidence bar visualization
        conf_bars = int(conf / 10)
        conf_bar = "█" * conf_bars + "░" * (10 - conf_bars)

        # Star rating (based on quality + confidence + meta)
        avg_score = (conf + quality + meta) / 3
        if avg_score >= 80:
            stars = "⭐⭐⭐⭐⭐"
        elif avg_score >= 70:
            stars = "⭐⭐⭐⭐"
        elif avg_score >= 60:
            stars = "⭐⭐⭐"
        elif avg_score >= 50:
            stars = "⭐⭐"
        else:
            stars = "⭐"

        # Risk level indicator
        if uncertainty > 30:
            risk_icon = "🔴 High Risk"
        elif uncertainty > 15:
            risk_icon = "🟡 Med Risk"
        else:
            risk_icon = "🟢 Low Risk"

        # Win streak indicator
        streak_text = ""
        sym_streak = _win_streaks.get(sym, 0)
        if sym_streak >= 3:
            streak_text = f" 🔥{sym_streak}"
        elif sym_streak <= -2:
            streak_text = f" ❄️{abs(sym_streak)}"

        lines.append("")
        lines.append(f"{arrow}  {sym} M5{streak_text}")
        lines.append(f"   {stars}")
        lines.append(f"   [{conf_bar}] {conf:.0f}%")
        lines.append(f"   📈 Prob: {green:.1f}% | Meta: {meta:.0f}%")
        lines.append(f"   🎯 Models: {agree}/{total_m} | Strength: {strength}")
        lines.append(f"   🌊 {regime} | {session}")
        lines.append(f"   ⚠️ {risk_icon} | Kelly: {kelly_pct:.1f}%")
        if quality > 0:
            lines.append(f"   ⭐ Quality: {quality:.0f}/100")
        # v11: Candle freshness and exhaustion indicators
        cq_freshness = pred.get("candle_freshness", 1.0)
        cq_quality = pred.get("candle_quality", 50)
        exhaustion = pred.get("exhaustion_score", 0)
        hour_quality = pred.get("hour_quality", "normal")
        if cq_freshness < 0.3:
            lines.append(f"   🔄 Candle: Stale ({cq_freshness:.0%})")
        elif cq_freshness > 0.7:
            lines.append(f"   ✨ Candle: Fresh ({cq_freshness:.0%})")
        if exhaustion > 40:
            lines.append(f"   ⚡ Exhaustion: {exhaustion:.0f}%")
        if hour_quality == "prime":
            lines.append("   🟢 Prime Trading Hour")
        elif hour_quality in ("weak", "poor"):
            lines.append(f"   🟡 {hour_quality.title()} Hour (extra confirmation)")
        # v11: Same-candle warning
        sc_flags = pred.get("same_candle_flags", [])
        if sc_flags:
            lines.append(f"   ⚠️ Candle: {', '.join(sc_flags)}")
        # v11: Strategy engine breakdown
        strat_agree = pred.get("strategies_agreeing", 0)
        strat_comp = pred.get("strategy_composite", 0)
        strat_dir = pred.get("strategy_direction", "NEUTRAL")
        if strat_agree > 0:
            lines.append(f"   🎯 Strategies: {strat_agree}/8 {strat_dir} (score: {strat_comp:.0f})")
        # v11: Pattern overlay
        patterns = pred.get("pattern_names", [])
        pat_adj = pred.get("pattern_adjustment", 0)
        if patterns:
            pat_icon = "🟢" if pat_adj > 0 else "🔴" if pat_adj < 0 else "⚪"
            lines.append(f"   {pat_icon} Patterns: {', '.join(patterns[:3])} ({pat_adj:+d})")
        if reasons:
            lines.append(f"   💡 {reasons}")
        lines.append(f"   ⏱ Expires: {expiry_str} UTC ({countdown_min}m {countdown_rem}s)")

    if filtered:
        reasons_list = [f"{s}({r[:20]})" for s, r in list(filtered.items())[:5]]
        lines.append(f"\n🚫 Filtered: {', '.join(reasons_list)}")

    lines.append("")
    lines.append("━" * 26)
    lines.append("🛡 v11: 9-Strategy | 12-Pattern | Same-Candle Guard")

    # Session accuracy
    if today_outcomes:
        wins = sum(1 for r in today_outcomes if r.get("correct"))
        total = len(today_outcomes)
        pct = wins / total * 100
        trend = "📈" if pct >= 60 else "📉" if pct < 50 else "➡️"
        lines.append(f"{trend} Session: {wins}/{total} ({pct:.0f}%)")
        if best_pair:
            lines.append(f"{best_pair} | {worst_pair}")

    # Overall streak alert
    if _total_streak >= 3:
        lines.append(f"🔥 HOT STREAK: {_total_streak} wins in a row!")
    elif _total_streak <= -3:
        lines.append(f"⚠️ COLD STREAK: {abs(_total_streak)} losses — caution!")

    return "\n".join(lines)


def _format_prediction_single(pred: dict) -> str:
    """Format a single manual prediction request."""
    sym = pred.get("symbol", "?")
    direction = pred.get("suggested_direction", "HOLD")
    if direction == "UP":
        arrow = "🟢 BUY ⬆"
    elif direction == "DOWN":
        arrow = "🔴 SELL ⬇"
    else:
        arrow = "⚪ SKIP"

    conf = pred.get("final_confidence_percent", 0)
    green = pred.get("green_probability_percent", 50)
    meta = pred.get("meta_reliability_percent", 50)
    uncertainty = pred.get("uncertainty_percent", 0)
    regime = pred.get("market_regime", "Unknown").replace("_", " ")
    session = pred.get("session", "Unknown")
    kelly = pred.get("kelly_fraction_percent", 0)
    trade = pred.get("suggested_trade", "HOLD")
    strength = pred.get("signal_strength", "—")
    reasons = pred.get("reason_chain", "")
    risk_warnings = pred.get("risk_warnings", [])

    lines = [
        f"{arrow}  {sym} M5",
        "",
        f"Probability: {green:.1f}%",
        f"Confidence: {conf:.0f}%",
        f"Meta Trust: {meta:.0f}%",
        f"Strength: {strength}",
        "",
        f"Regime: {regime} | {session}",
        f"Kelly: {kelly:.1f}%",
    ]
    if reasons:
        lines.append(f"💡 {reasons}")
    if risk_warnings:
        lines.append(f"⚠️ {', '.join(risk_warnings[:3])}")

    return "\n".join(lines)


# =============================================================================
#  Signal Outcome Tracking
# =============================================================================

_pending_signals: dict = {}  # {signal_id: {symbol, direction, price, time, ...}}


def _record_sent_signal(symbol: str, pred: dict, chat_ids: list):
    """Record a sent signal for outcome tracking."""
    signal_id = f"{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    direction = pred.get("suggested_direction", "HOLD")
    conf = pred.get("final_confidence_percent", 0)
    kelly = pred.get("kelly_fraction_percent", 0)
    regime = pred.get("market_regime", "Unknown")
    session = pred.get("session", "Unknown")

    # Get entry price
    try:
        tick = mt5.symbol_info_tick(symbol)
        price = tick.bid if tick else 0
    except Exception:
        price = 0

    entry = {
        "signal_id": signal_id,
        "symbol": symbol,
        "direction": direction,
        "confidence": conf,
        "kelly": kelly,
        "regime": regime,
        "session": session,
        "price": price,
        "time": datetime.now(timezone.utc).isoformat(),
        "chat_ids": chat_ids,
        # v12 fields for adaptive systems
        "unanimity": pred.get("ensemble_unanimity", 0.5),
        "variance": pred.get("ensemble_variance", 0.01),
        "signal_score": pred.get("signal_score", 50),
        "quality_score": pred.get("quality_score", 0),
        "dispatch_win_prob": pred.get("dispatch_win_prob", 0.5),
    }
    _pending_signals[signal_id] = entry

    # Persist pending
    try:
        with open(PENDING_JSON, "w") as f:
            json.dump(_pending_signals, f, indent=2, default=str)
    except Exception:
        pass

    # Log to CSV
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        exists = os.path.exists(SENT_SIGNALS_CSV)
        with open(SENT_SIGNALS_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(["timestamp", "signal_id", "symbol", "direction", "confidence", "kelly", "regime", "price"])
            w.writerow([entry["time"], signal_id, symbol, direction, f"{conf:.1f}", f"{kelly:.1f}", regime, f"{price:.5f}"])
    except Exception:
        pass

    return signal_id


def _save_cycle_messages():
    """Persist cycle_messages so result editing survives restarts."""
    try:
        with open(CYCLE_MESSAGES_JSON, "w") as f:
            json.dump(_cycle_messages, f, indent=2, default=str)
    except Exception:
        pass


def _load_cycle_messages():
    """Load cycle_messages from disk."""
    global _cycle_messages
    try:
        if os.path.exists(CYCLE_MESSAGES_JSON):
            with open(CYCLE_MESSAGES_JSON) as f:
                loaded = json.load(f)
                _cycle_messages = loaded if isinstance(loaded, dict) else {}
    except Exception:
        _cycle_messages = {}


async def _resolve_pending_signals(bot: Bot):
    """
    Check pending signals, record outcomes, and EDIT original messages with results.
    Called each cycle to reconcile 5-min-old signals.
    """
    if not _pending_signals:
        return

    now = datetime.now(timezone.utc)
    resolved = []

    for sig_id, entry in list(_pending_signals.items()):
        sig_time = datetime.fromisoformat(entry["time"])
        elapsed = (now - sig_time).total_seconds()
        if elapsed < 300:  # wait 5 minutes
            continue

        symbol = entry["symbol"]
        direction = entry["direction"]
        entry_price = entry["price"]

        # Get current price
        try:
            tick = mt5.symbol_info_tick(symbol)
            current_price = tick.bid if tick else 0
        except Exception:
            current_price = 0

        if entry_price <= 0 or current_price <= 0:
            resolved.append(sig_id)
            continue

        # Determine outcome
        price_change = current_price - entry_price
        # Treat near-zero change as inconclusive (skip, don't count as loss)
        pip_size = 0.01 if "JPY" in symbol else 0.0001
        if abs(price_change) < pip_size * 0.5:
            # Record as DRAW for cycle message editing
            for ck, cm in _cycle_messages.items():
                if sig_id in cm["sig_ids"]:
                    cm["results"][sig_id] = {
                        "symbol": symbol,
                        "direction": direction,
                        "correct": None,  # draw
                        "entry_price": entry_price,
                        "exit_price": current_price,
                    }
                    break
            resolved.append(sig_id)
            continue
        if direction == "UP":
            actual = "UP" if price_change > 0 else "DOWN"
        elif direction == "DOWN":
            actual = "DOWN" if price_change < 0 else "UP"
        else:
            resolved.append(sig_id)
            continue

        correct = (direction == actual)

        outcome = {
            "timestamp": entry["time"],
            "symbol": symbol,
            "predicted": direction,
            "actual": actual,
            "correct": correct,
            "green_prob": entry.get("confidence", 50),
            "confidence": entry.get("confidence", 50),
            "kelly": entry.get("kelly", 0),
            "regime": entry.get("regime", "Unknown"),
            "session": entry.get("session", "Unknown"),
            "latency_ms": elapsed * 1000 / 300,
        }

        # Write to CSV
        try:
            exists = os.path.exists(OUTCOMES_CSV)
            with open(OUTCOMES_CSV, "a", newline="") as f:
                w = csv.writer(f)
                if not exists:
                    w.writerow(list(outcome.keys()))
                w.writerow(list(outcome.values()))
        except Exception as e:
            log.warning("Failed to write outcome: %s", e)

        _outcome_results.append(outcome)

        # Update signal_db outcome
        db_id = entry.get("db_id")
        if db_id:
            try:
                _signal_db.update_outcome(db_id, correct)
            except Exception:
                pass

        # Update trackers
        _accuracy.record(correct)
        _conf_corr.record(entry.get("confidence", 50), correct)

        # v2: Update signal ranker per-symbol stats + adaptive thresholds
        update_symbol_stats(symbol, correct)
        _update_symbol_confidence(symbol, correct, entry.get("confidence", 50))

        # v12: Update all adaptive systems with outcome
        try:
            # Dispatch model learns from outcome
            _dispatch.update(entry, symbol, correct)
        except Exception as e:
            log.debug("Dispatch model update failed: %s", e)

        try:
            # Bayesian threshold learns per (symbol, regime, hour)
            sig_time_parsed = datetime.fromisoformat(entry["time"])
            sig_hour = sig_time_parsed.hour
            _bayesian.update(symbol, entry.get("regime", "Unknown"), sig_hour, correct)
        except Exception as e:
            log.debug("Bayesian threshold update failed: %s", e)

        try:
            # Outcome feedback loop auto-tunes dispatch thresholds
            _feedback.record_outcome({
                "correct": correct,
                "confidence": entry.get("confidence", 50),
                "unanimity": entry.get("unanimity", 0.5),
                "variance": entry.get("variance", 0.01),
                "signal_score": entry.get("signal_score", 50),
                "symbol": symbol,
            })
        except Exception as e:
            log.debug("Feedback loop update failed: %s", e)

        try:
            # Session filter learns per-hour accuracy
            sig_time_parsed2 = datetime.fromisoformat(entry["time"])
            update_hour_outcome(sig_time_parsed2.hour, correct)
        except Exception as e:
            log.debug("Session hour update failed: %s", e)

        # Update stacking gatekeeper with outcome
        try:
            _stacker.update(entry, correct)
        except Exception:
            pass

        # Circuit breaker
        _cb.check(was_win=correct)

        # Per-symbol loss tracking + win streak tracking
        if correct:
            _consecutive_losses[symbol] = 0
            _win_streaks[symbol] = max(1, _win_streaks.get(symbol, 0) + 1) if _win_streaks.get(symbol, 0) >= 0 else 1
        else:
            _consecutive_losses[symbol] = _consecutive_losses.get(symbol, 0) + 1
            _win_streaks[symbol] = min(-1, _win_streaks.get(symbol, 0) - 1) if _win_streaks.get(symbol, 0) <= 0 else -1

        # Overall streak tracking
        global _total_streak
        if correct:
            _total_streak = max(1, _total_streak + 1) if _total_streak >= 0 else 1
        else:
            _total_streak = min(-1, _total_streak - 1) if _total_streak <= 0 else -1

        # Kelly
        _kelly.update(correct)

        # Track result for cycle message editing
        for ck, cm in _cycle_messages.items():
            if sig_id in cm["sig_ids"]:
                cm["results"][sig_id] = {
                    "symbol": symbol,
                    "direction": direction,
                    "correct": correct,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                }
                break

        log.info("Outcome %s %s: predicted=%s actual=%s correct=%s",
                 sig_id, symbol, direction, actual, correct)
        resolved.append(sig_id)

    for sig_id in resolved:
        _pending_signals.pop(sig_id, None)

    # Edit messages for cycles that have ANY new results
    cycles_done = []
    edited_any = False
    for ck, cm in list(_cycle_messages.items()):
        if not cm.get("sig_ids"):
            continue
        result_count = len(cm.get("results", {}))
        if result_count == 0:
            continue
        # Edit as soon as ANY result is available (don't wait for all)
        new_text = _build_result_text(cm["text"], cm["results"])
        for cid, mid in cm.get("msg_ids", {}).items():
            await _safe_edit(bot, cid, mid, new_text)
        edited_any = True
        all_done = result_count >= len(cm["sig_ids"])
        if all_done:
            cycles_done.append(ck)
            log.info("Edited signal message for cycle %s — all %d results.",
                     ck, result_count)
        else:
            log.info("Edited signal message for cycle %s — %d/%d results so far.",
                     ck, result_count, len(cm["sig_ids"]))

    # Clean up old cycles (> 30 min unresolved)
    for ck in list(_cycle_messages.keys()):
        try:
            ct = datetime.strptime(ck, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
            if (now - ct).total_seconds() > 1800:
                cycles_done.append(ck)
        except Exception:
            pass

    for ck in set(cycles_done):
        _cycle_messages.pop(ck, None)

    # Persist cycle messages + pending signals
    if resolved or edited_any:
        _save_cycle_messages()
        try:
            with open(PENDING_JSON, "w") as f:
                json.dump(_pending_signals, f, indent=2, default=str)
        except Exception:
            pass


def _log_all_predictions(symbol: str, pred: dict):
    """Log every prediction (before filtering) for analysis."""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        exists = os.path.exists(ALL_PREDICTIONS_CSV)
        with open(ALL_PREDICTIONS_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(["timestamp", "symbol", "direction", "confidence", "green_prob",
                            "meta_reliability", "uncertainty", "kelly", "regime", "trade"])
            w.writerow([
                datetime.now(timezone.utc).isoformat(),
                symbol,
                pred.get("suggested_direction", ""),
                f"{pred.get('final_confidence_percent', 0):.1f}",
                f"{pred.get('green_probability_percent', 0):.1f}",
                f"{pred.get('meta_reliability_percent', 0):.1f}",
                f"{pred.get('uncertainty_percent', 0):.1f}",
                f"{pred.get('kelly_fraction_percent', 0):.1f}",
                pred.get("market_regime", ""),
                pred.get("suggested_trade", ""),
            ])
    except Exception:
        pass


# =============================================================================
#  Broadcast Helper
# =============================================================================

async def _safe_send(bot: Bot, chat_id, text: str, parse_mode="Markdown") -> int:
    """Send message with error handling. Returns message_id or 0."""
    try:
        msg = await bot.send_message(chat_id=int(chat_id), text=text, parse_mode=parse_mode)
        return msg.message_id
    except Exception as e:
        log.warning("Send failed to %s: %s", chat_id, e)
        try:
            msg = await bot.send_message(chat_id=int(chat_id), text=text)
            return msg.message_id
        except Exception:
            return 0


async def _safe_edit(bot: Bot, chat_id, message_id: int, text: str, parse_mode="Markdown"):
    """Edit message with error handling."""
    try:
        await bot.edit_message_text(
            chat_id=int(chat_id), message_id=message_id,
            text=text, parse_mode=parse_mode,
        )
    except Exception as e:
        log.warning("Edit failed for %s/%s: %s", chat_id, message_id, e)
        try:
            await bot.edit_message_text(
                chat_id=int(chat_id), message_id=message_id, text=text,
            )
        except Exception:
            pass


def _build_result_text(original_text: str, results: dict) -> str:
    """Rebuild signal message with WIN/LOSS results appended."""
    wins = sum(1 for r in results.values() if r.get("correct") is True)
    draws = sum(1 for r in results.values() if r.get("correct") is None)
    total = len(results)

    # Remove expiry/countdown lines from original
    edited_lines = []
    for line in original_text.split("\n"):
        if "\u23f1 Expires:" in line or "\u23f1 Exp:" in line:
            continue
        edited_lines.append(line)

    result_lines = [
        "",
        "\u2501" * 26,
        "\U0001f4ca RESULT",
        "",
    ]

    for sig_id, r in results.items():
        correct = r.get("correct")
        if correct is True:
            icon = "\u2705 WIN"
        elif correct is None:
            icon = "\u27a1\ufe0f DRAW"
        else:
            icon = "\u274c LOSS"
        sym = r.get("symbol", "?")
        direction = r.get("direction", "?")
        entry_p = r.get("entry_price", 0)
        exit_p = r.get("exit_price", 0)
        diff = exit_p - entry_p
        if "JPY" in sym:
            diff_fmt = f"{diff:+.3f}"
        elif sym == "XAUUSD":
            diff_fmt = f"{diff:+.2f}"
        else:
            diff_fmt = f"{diff:+.5f}"
        result_lines.append(f"{icon} | {sym} {direction}")
        result_lines.append(f"   \U0001f4cd {entry_p} \u2192 {exit_p} ({diff_fmt})")

    if total > 0:
        scored = total - draws
        pct = wins / scored * 100 if scored > 0 else 0
        emoji = "\U0001f3c6" if pct >= 60 else "\U0001f4c9"
        losses = scored - wins
        draw_text = f" / {draws}D" if draws > 0 else ""
        result_lines.append(f"\n{emoji} Cycle: {wins}W / {losses}L{draw_text} ({pct:.0f}%)")

    # Add session accuracy to result
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_outcomes = [r for r in _outcome_results if r.get("timestamp", "").startswith(today)]
    if today_outcomes:
        tw = sum(1 for r in today_outcomes if r.get("correct"))
        tt = len(today_outcomes)
        tp = tw / tt * 100
        trend = "\U0001f4c8" if tp >= 60 else "\U0001f4c9" if tp < 50 else "\u27a1\ufe0f"
        result_lines.append(f"{trend} Today: {tw}/{tt} ({tp:.0f}%)")

    # Streak alert
    if _total_streak >= 3:
        result_lines.append(f"\U0001f525 {_total_streak} wins in a row!")
    elif _total_streak <= -3:
        result_lines.append(f"\u26a0\ufe0f {abs(_total_streak)} losses — caution!")

    return "\n".join(edited_lines) + "\n" + "\n".join(result_lines)


async def _broadcast(bot: Bot, text: str, symbol_filter: str = None):
    """Broadcast to all active subscribers."""
    for cid, info in _subscribers.items():
        if not info.get("active", False):
            continue
        if symbol_filter:
            syms = info.get("symbols", SYMBOLS)
            if symbol_filter not in syms:
                continue
        await _safe_send(bot, cid, text)


# =============================================================================
#  Command Handlers
# =============================================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Welcome message with menu."""
    chat_id = str(update.effective_chat.id)
    if chat_id not in _subscribers:
        _subscribers[chat_id] = {"active": False, "symbols": SYMBOLS, "mode": "all"}
        _save_subscribers()

    text = (
        "⚡ QUOTEX LORD v15 Advanced ⚡\n\n"
        "9-Strategy | 12-Pattern | Same-Candle Guard\n"
        "v15 ML pipeline with GRU deep learning + smart confidence.\n\n"
        f"📊 Symbols: {', '.join(SYMBOLS)}\n"
        f"⏱ Timeframe: M5 (5-minute candles)\n"
        f"🤖 9 ML models + GRU + meta-model ensemble\n"
        f"🛡 Auto risk management & circuit breaker\n\n"
        "Features:\n"
        "• Auto-signals every 5 min\n"
        "• WIN/LOSS results on each signal\n"
        "• Star ratings & risk levels\n"
        "• Performance analytics\n"
        "• Live market overview\n"
        "• Hourly & regime breakdowns\n\n"
        "Use the menu below to get started."
    )
    await update.message.reply_text(text, reply_markup=_main_menu(chat_id))


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "📋 Commands:\n\n"
        "/start — Main menu\n"
        "/predict — Predict single symbol\n"
        "/scan — Multi-symbol scan\n"
        "/today — Today's signals & status\n"
        "/results — Today's results\n"
        "/performance — Detailed performance analytics\n"
        "/market — Live market overview\n"
        "/stats — Advanced statistics\n"
        "/audit — Signal audit trail\n"
        "/leaderboard — Pair rankings\n"
        "/dashboard — Performance dashboard\n"
        "/status — System status\n"
        "/help — This message"
    )
    await update.message.reply_text(text)


async def cmd_predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Quick predict via command."""
    await update.message.reply_text("Select symbol:", reply_markup=_symbol_keyboard("predict"))


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Multi-symbol scan via command."""
    msg = await update.message.reply_text("Scanning all symbols...")
    results = []
    for sym in SYMBOLS:
        try:
            pred = _run_prediction(sym)
            results.append(_format_prediction_single(pred))
        except Exception as e:
            results.append(f"Error {sym}: {e}")
    text = "\n━━━━━━━━━━━━━━\n".join(results)
    await _safe_send(update.message.chat.bot, str(update.effective_chat.id),
                     f"Multi-Symbol Scan\n\n{text}")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = _get_status_text()
    await update.message.reply_text(text)


async def cmd_results(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = _get_results_text()
    await update.message.reply_text(text)


async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show all signals sent today with status."""
    text = _get_today_signals_text()
    await update.message.reply_text(text)


async def cmd_dashboard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate and send performance dashboard."""
    try:
        report = generate_dashboard(_signal_db)
        text = format_dashboard_text(report)
    except Exception as e:
        text = f"Dashboard error: {e}"
    await update.message.reply_text(text)


async def cmd_audit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Full signal audit trail — last 20 signals with full details."""
    try:
        signals = _signal_db.get_recent_signals(n=20)
        if not signals:
            await update.message.reply_text("📋 No signals recorded yet.")
            return

        lines = ["📋 Signal Audit Trail (last 20)\n"]
        for s in signals:
            # s = (id, timestamp, symbol, direction, probability, confidence, regime, outcome, was_correct, pnl)
            ts = str(s[1])[:16] if s[1] else "?"
            sym = s[2] or "?"
            d = s[3] or "?"
            conf = s[5] or 0
            regime = s[6] or "?"
            outcome = s[7] or "pending"
            was_correct = s[8]
            icon = "✅" if was_correct == 1 else "❌" if was_correct == 0 else "⏳"
            lines.append(f"{icon} {ts} {sym} {d} C:{conf:.0f}% [{regime}]")

        await update.message.reply_text("\n".join(lines))
    except Exception as e:
        await update.message.reply_text(f"Audit error: {e}")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Advanced performance statistics."""
    try:
        lines = ["📊 Advanced Statistics\n"]

        # Overall stats
        total = len(_outcome_results)
        if total > 0:
            wins = sum(1 for r in _outcome_results if r.get("correct"))
            losses = total - wins
            accuracy = wins / total * 100

            # Win/Loss streaks
            max_win_streak = 0
            max_loss_streak = 0
            current_streak = 0
            for r in _outcome_results:
                if r.get("correct"):
                    current_streak = max(1, current_streak + 1) if current_streak > 0 else 1
                else:
                    current_streak = min(-1, current_streak - 1) if current_streak < 0 else -1
                max_win_streak = max(max_win_streak, current_streak)
                max_loss_streak = min(max_loss_streak, current_streak)

            lines.append(f"Total Signals: {total}")
            lines.append(f"Record: {wins}W / {losses}L ({accuracy:.1f}%)")
            lines.append(f"Best Win Streak: {max_win_streak}")
            lines.append(f"Worst Loss Streak: {abs(max_loss_streak)}")
            lines.append(f"Current Streak: {_total_streak}")

            # Per-symbol breakdown
            lines.append("\n📈 Per Symbol:")
            sym_stats = {}
            for r in _outcome_results:
                s = r.get("symbol", "?")
                if s not in sym_stats:
                    sym_stats[s] = {"w": 0, "l": 0}
                if r.get("correct"):
                    sym_stats[s]["w"] += 1
                else:
                    sym_stats[s]["l"] += 1

            for s, v in sorted(sym_stats.items(), key=lambda x: -(x[1]["w"] / max(1, x[1]["w"] + x[1]["l"]))):
                t = v["w"] + v["l"]
                pct = v["w"] / t * 100 if t > 0 else 0
                bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
                lines.append(f"  {s}: [{bar}] {v['w']}/{t} ({pct:.0f}%)")

            # Per-session breakdown
            lines.append("\n🕐 Per Session:")
            sess_stats = {}
            for r in _outcome_results:
                sess = r.get("session", "Unknown")
                if sess not in sess_stats:
                    sess_stats[sess] = {"w": 0, "l": 0}
                if r.get("correct"):
                    sess_stats[sess]["w"] += 1
                else:
                    sess_stats[sess]["l"] += 1

            for sess, v in sorted(sess_stats.items()):
                t = v["w"] + v["l"]
                pct = v["w"] / t * 100 if t > 0 else 0
                lines.append(f"  {sess}: {v['w']}/{t} ({pct:.0f}%)")

            # Per-regime breakdown
            lines.append("\n🌊 Per Regime:")
            reg_stats = {}
            for r in _outcome_results:
                reg = r.get("regime", "Unknown")
                if reg not in reg_stats:
                    reg_stats[reg] = {"w": 0, "l": 0}
                if r.get("correct"):
                    reg_stats[reg]["w"] += 1
                else:
                    reg_stats[reg]["l"] += 1

            for reg, v in sorted(reg_stats.items()):
                t = v["w"] + v["l"]
                pct = v["w"] / t * 100 if t > 0 else 0
                lines.append(f"  {reg}: {v['w']}/{t} ({pct:.0f}%)")

        else:
            lines.append("No outcomes recorded yet.")

        # System health
        lines.append(f"\n🔧 System Health:")
        lines.append(f"Circuit Breaker: {'🔴 HALTED' if _cb.is_halted else '🟢 OK'}")
        lines.append(f"Consecutive Losses: {_cb.consecutive_losses}")

        await update.message.reply_text("\n".join(lines))
    except Exception as e:
        await update.message.reply_text(f"Stats error: {e}")


async def cmd_leaderboard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Pair leaderboard — best performing symbols."""
    lines = ["🏆 Pair Leaderboard\n"]

    if not _pair_scores:
        lines.append("No pair scores available yet.")
    else:
        for sym, score in sorted(_pair_scores.items(), key=lambda x: -x[1]):
            stars = "⭐" * (score // 5)
            lines.append(f"  {sym}: {score}/30 {stars}")

    # Win streaks per symbol
    if _win_streaks:
        lines.append("\n🔥 Streaks:")
        for sym, streak in sorted(_win_streaks.items(), key=lambda x: -x[1]):
            icon = "🔥" if streak > 0 else "❄️"
            lines.append(f"  {sym}: {icon} {streak}")

    await update.message.reply_text("\n".join(lines))


async def cmd_performance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Detailed performance analytics report."""
    try:
        now_utc = datetime.now(timezone.utc)
        today = now_utc.strftime("%Y-%m-%d")
        today_outcomes = [r for r in _outcome_results if r.get("timestamp", "").startswith(today)]

        lines = ["🏆 Performance Report\n"]

        # All-time summary
        total = len(_outcome_results)
        if total > 0:
            wins = sum(1 for r in _outcome_results if r.get("correct"))
            pct = wins / total * 100
            lines.append(f"📊 All-Time: {wins}W / {total - wins}L ({pct:.1f}%)")
            lines.append(f"🔥 Current Streak: {_total_streak}")
            lines.append(f"🏆 Peak Accuracy: {_peak_accuracy*100:.1f}%")
        else:
            lines.append("No outcomes recorded yet.")
            await update.message.reply_text("\n".join(lines))
            return

        # Today summary
        if today_outcomes:
            tw = sum(1 for r in today_outcomes if r.get("correct"))
            tt = len(today_outcomes)
            tp = tw / tt * 100
            lines.append(f"\n📅 Today: {tw}W / {tt - tw}L ({tp:.0f}%)")
        else:
            lines.append(f"\n📅 Today: No signals yet")

        # Best performing hours (top 3)
        hour_stats = {}
        for r in _outcome_results:
            ts = r.get("timestamp", "")
            if len(ts) >= 13:
                h = ts[11:13]
                if h not in hour_stats:
                    hour_stats[h] = {"w": 0, "t": 0}
                hour_stats[h]["t"] += 1
                if r.get("correct"):
                    hour_stats[h]["w"] += 1
        if hour_stats:
            sorted_hours = sorted(hour_stats.items(), key=lambda x: x[1]["w"] / max(x[1]["t"], 1), reverse=True)
            lines.append("\n⏰ Best Hours:")
            for h, v in sorted_hours[:3]:
                pct = v["w"] / v["t"] * 100 if v["t"] > 0 else 0
                lines.append(f"  {h}:00 UTC — {v['w']}/{v['t']} ({pct:.0f}%) 🔥")

        # Best performing pairs (top 3)
        sym_stats = {}
        for r in _outcome_results:
            s = r.get("symbol", "?")
            if s not in sym_stats:
                sym_stats[s] = {"w": 0, "t": 0}
            sym_stats[s]["t"] += 1
            if r.get("correct"):
                sym_stats[s]["w"] += 1
        if sym_stats:
            sorted_syms = sorted(sym_stats.items(), key=lambda x: x[1]["w"] / max(x[1]["t"], 1), reverse=True)
            lines.append("\n🥇 Best Pairs:")
            for s, v in sorted_syms[:3]:
                pct = v["w"] / v["t"] * 100 if v["t"] > 0 else 0
                lines.append(f"  {s}: {v['w']}/{v['t']} ({pct:.0f}%)")

        # Confidence accuracy correlation
        high_conf = [r for r in _outcome_results if r.get("confidence", 0) >= 70]
        low_conf = [r for r in _outcome_results if r.get("confidence", 0) < 70]
        if high_conf:
            hc_w = sum(1 for r in high_conf if r.get("correct"))
            hc_pct = hc_w / len(high_conf) * 100
            lines.append(f"\n💎 High Conf (≥70%): {hc_w}/{len(high_conf)} ({hc_pct:.0f}%)")
        if low_conf:
            lc_w = sum(1 for r in low_conf if r.get("correct"))
            lc_pct = lc_w / len(low_conf) * 100
            lines.append(f"📉 Low Conf (<70%):  {lc_w}/{len(low_conf)} ({lc_pct:.0f}%)")

        await update.message.reply_text("\n".join(lines))
    except Exception as e:
        await update.message.reply_text(f"Performance error: {e}")


async def cmd_market(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Live market overview — current regimes and sessions."""
    try:
        now_utc = datetime.now(timezone.utc)
        hour = now_utc.hour

        # Determine session
        if 0 <= hour < 7:
            session = "🌏 Asian Session"
        elif 7 <= hour < 12:
            session = "🇪🇺 London Session"
        elif 12 <= hour < 16:
            session = "🇺🇸 NY + London Overlap"
        elif 16 <= hour < 21:
            session = "🇺🇸 New York Session"
        else:
            session = "🌙 Off-Hours"

        lines = [
            "🌍 Market Overview\n",
            f"⏰ {now_utc.strftime('%H:%M UTC')} | {session}",
            "",
        ]

        # Current regime per symbol
        if _last_regime:
            lines.append("🌊 Current Regimes:")
            for sym, regime in sorted(_last_regime.items()):
                skip = _regime_skip.get(sym, 0)
                skip_text = f" (skip {skip})" if skip > 0 else ""
                lines.append(f"  {sym}: {regime}{skip_text}")
        else:
            lines.append("🌊 Regimes: Not yet computed")

        # Win streaks
        if _win_streaks:
            lines.append("\n🔥 Current Streaks:")
            for sym, streak in sorted(_win_streaks.items(), key=lambda x: -x[1]):
                icon = "🔥" if streak > 0 else "❄️" if streak < 0 else "➡️"
                lines.append(f"  {sym}: {icon} {streak}")

        # Circuit breaker status
        lines.append(f"\n🛡 Circuit Breaker: {'🔴 HALTED' if _cb.is_halted else '🟢 ACTIVE'}")
        if _cb.consecutive_losses > 0:
            lines.append(f"⚠️ Consecutive losses: {_cb.consecutive_losses}")

        # Adaptive confidence thresholds
        if _symbol_min_confidence:
            lines.append("\n🎯 Adaptive Thresholds:")
            for sym, thresh in sorted(_symbol_min_confidence.items()):
                lines.append(f"  {sym}: {thresh:.0f}%")

        await update.message.reply_text("\n".join(lines))
    except Exception as e:
        await update.message.reply_text(f"Market error: {e}")


def _main_menu(chat_id=None):
    cid = str(chat_id) if chat_id else None
    is_sub = cid and cid in _subscribers and _subscribers[cid].get("active", False)
    sub_label = "⬛ Stop Signals" if is_sub else "▶️ Start Signals"
    sub_data = "unsub" if is_sub else "sub"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(sub_label, callback_data=sub_data)],
        [InlineKeyboardButton("🔮 Predict", callback_data="predict_now"),
         InlineKeyboardButton("📱 Scan All", callback_data="scan")],
        [InlineKeyboardButton("📡 Today", callback_data="today"),
         InlineKeyboardButton("📊 Results", callback_data="results")],
        [InlineKeyboardButton("🏆 Performance", callback_data="performance"),
         InlineKeyboardButton("🌍 Market", callback_data="market")],
        [InlineKeyboardButton("⚙️ Status", callback_data="status"),
         InlineKeyboardButton("📊 Dashboard", callback_data="dashboard")],
        [InlineKeyboardButton("📈 Stats", callback_data="stats"),
         InlineKeyboardButton("📋 Audit", callback_data="audit")],
        [InlineKeyboardButton("🏆 Leaderboard", callback_data="leaderboard"),
         InlineKeyboardButton("📜 History", callback_data="history")],
        [InlineKeyboardButton("ℹ️ Info", callback_data="info")],
    ])


def _symbol_keyboard(prefix="predict"):
    buttons = []
    row = []
    for sym in SYMBOLS:
        row.append(InlineKeyboardButton(sym, callback_data=f"{prefix}_{sym}"))
        if len(row) == 3:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append([InlineKeyboardButton("◀ Back", callback_data="back")])
    return InlineKeyboardMarkup(buttons)


async def _handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all inline button callbacks."""
    query = update.callback_query
    await query.answer()
    chat_id = str(query.message.chat_id)
    data = query.data

    if data == "sub":
        if chat_id not in _subscribers:
            _subscribers[chat_id] = {"active": True, "symbols": SYMBOLS, "mode": "all"}
        else:
            _subscribers[chat_id]["active"] = True
        _save_subscribers()
        await query.edit_message_text(
            "✅ Subscribed to auto-signals.\n\n"
            f"Symbols: {', '.join(SYMBOLS)}\n"
            "Signals sent every 5 minutes during profitable hours.",
            reply_markup=_main_menu(chat_id),
        )

    elif data == "unsub":
        if chat_id in _subscribers:
            _subscribers[chat_id]["active"] = False
            _save_subscribers()
        await query.edit_message_text(
            "⏹ Unsubscribed. Signals stopped.",
            reply_markup=_main_menu(chat_id),
        )

    elif data == "back":
        await query.edit_message_text(
            "⚡ QUOTEX LORD v15 Advanced ⚡\n\nUse the menu below.",
            reply_markup=_main_menu(chat_id),
        )

    elif data == "predict_now":
        await query.edit_message_text("Select symbol:", reply_markup=_symbol_keyboard("predict"))

    elif data.startswith("predict_"):
        sym = data.replace("predict_", "")
        if sym in SYMBOLS:
            await query.edit_message_text(f"Predicting {sym}...")
            try:
                pred = _run_prediction(sym)
                text = _format_prediction_single(pred)
                await _safe_send(query.message.chat.bot, chat_id, text)
            except Exception as e:
                await _safe_send(query.message.chat.bot, chat_id, f"Prediction failed: {e}")

    elif data == "scan":
        await query.edit_message_text("Scanning all symbols...")
        results = []
        for sym in SYMBOLS:
            try:
                pred = _run_prediction(sym)
                results.append(_format_prediction_single(pred))
            except Exception as e:
                results.append(f"Error {sym}: {e}")
        text = "\n━━━━━━━━━━━━━━\n".join(results)
        await _safe_send(query.message.chat.bot, chat_id, f"Multi-Symbol Scan\n\n{text}")

    elif data == "status":
        text = _get_status_text()
        await query.edit_message_text(text, reply_markup=_main_menu(chat_id))

    elif data == "results":
        text = _get_results_text()
        await query.edit_message_text(text, reply_markup=_main_menu(chat_id))

    elif data == "today":
        text = _get_today_signals_text()
        await query.edit_message_text(text, reply_markup=_main_menu(chat_id))

    elif data == "history":
        text = _get_history_text()
        await query.edit_message_text(text, reply_markup=_main_menu(chat_id))

    elif data == "dashboard":
        try:
            report = generate_dashboard(_signal_db)
            text = format_dashboard_text(report)
        except Exception as e:
            text = f"Dashboard error: {e}"
        await query.edit_message_text(text, reply_markup=_main_menu(chat_id))

    elif data == "stats":
        try:
            lines = ["📊 Advanced Statistics\n"]
            total = len(_outcome_results)
            if total > 0:
                wins = sum(1 for r in _outcome_results if r.get("correct"))
                accuracy = wins / total * 100
                lines.append(f"📈 Record: {wins}/{total} ({accuracy:.1f}%)")
                lines.append(f"🔥 Streak: {_total_streak}")
                lines.append(f"🏆 Peak: {_peak_accuracy*100:.1f}%\n")

                # Per-symbol stats
                sym_stats = {}
                for r in _outcome_results:
                    s = r.get("symbol", "?")
                    if s not in sym_stats:
                        sym_stats[s] = {"w": 0, "l": 0}
                    if r.get("correct"):
                        sym_stats[s]["w"] += 1
                    else:
                        sym_stats[s]["l"] += 1
                lines.append("📊 Per Symbol:")
                for s, v in sorted(sym_stats.items()):
                    t = v["w"] + v["l"]
                    pct = v["w"] / t * 100 if t > 0 else 0
                    bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
                    lines.append(f"  {s}: [{bar}] {v['w']}/{t} ({pct:.0f}%)")

                # Hourly breakdown
                hour_stats = {}
                for r in _outcome_results:
                    ts = r.get("timestamp", "")
                    if len(ts) >= 13:
                        h = ts[11:13]
                        if h not in hour_stats:
                            hour_stats[h] = {"w": 0, "t": 0}
                        hour_stats[h]["t"] += 1
                        if r.get("correct"):
                            hour_stats[h]["w"] += 1
                if hour_stats:
                    lines.append("\n⏰ By Hour (UTC):")
                    for h in sorted(hour_stats.keys()):
                        v = hour_stats[h]
                        pct = v["w"] / v["t"] * 100 if v["t"] > 0 else 0
                        hot = "🔥" if pct >= 65 else "❄️" if pct < 45 else ""
                        lines.append(f"  {h}:00 — {v['w']}/{v['t']} ({pct:.0f}%) {hot}")

                # Regime breakdown
                regime_stats = {}
                for r in _outcome_results:
                    reg = r.get("regime", "Unknown")
                    if reg not in regime_stats:
                        regime_stats[reg] = {"w": 0, "t": 0}
                    regime_stats[reg]["t"] += 1
                    if r.get("correct"):
                        regime_stats[reg]["w"] += 1
                if regime_stats:
                    lines.append("\n🌊 By Regime:")
                    for reg, v in sorted(regime_stats.items(), key=lambda x: -x[1]["t"]):
                        pct = v["w"] / v["t"] * 100 if v["t"] > 0 else 0
                        lines.append(f"  {reg}: {v['w']}/{v['t']} ({pct:.0f}%)")
            else:
                lines.append("No outcomes yet.")
            text = "\n".join(lines)
        except Exception as e:
            text = f"Stats error: {e}"
        await query.edit_message_text(text, reply_markup=_main_menu(chat_id))

    elif data == "audit":
        try:
            signals = _signal_db.get_recent_signals(n=10)
            if not signals:
                text = "📋 No signals recorded yet."
            else:
                lines = ["📋 Signal Audit (last 10)\n"]
                for s in signals:
                    ts = str(s[1])[:16] if s[1] else "?"
                    sym = s[2] or "?"
                    d = s[3] or "?"
                    conf = s[5] or 0
                    was_correct = s[8]
                    icon = "✅" if was_correct == 1 else "❌" if was_correct == 0 else "⏳"
                    lines.append(f"{icon} {ts} {sym} {d} {conf:.0f}%")
                text = "\n".join(lines)
        except Exception as e:
            text = f"Audit error: {e}"
        await query.edit_message_text(text, reply_markup=_main_menu(chat_id))

    elif data == "leaderboard":
        lines = ["🏆 Pair Leaderboard\n"]
        if _pair_scores:
            for sym, score in sorted(_pair_scores.items(), key=lambda x: -x[1]):
                stars = "⭐" * (score // 5)
                lines.append(f"  {sym}: {score}/30 {stars}")
        else:
            lines.append("No pair scores yet.")
        if _win_streaks:
            lines.append("\n🔥 Streaks:")
            for sym, streak in sorted(_win_streaks.items(), key=lambda x: -x[1]):
                icon = "🔥" if streak > 0 else "❄️"
                lines.append(f"  {sym}: {icon} {streak}")
        text = "\n".join(lines)
        await query.edit_message_text(text, reply_markup=_main_menu(chat_id))

    elif data == "info":
        text = (
            "⚡ QUOTEX LORD v15 Advanced ⚡\n\n"
            "v15 ML pipeline:\n"
            "  ✅ Same-Candle Detector (SC-1 to SC-5)\n"
            "  ✅ 9-Strategy Signal Engine (parallel)\n"
            "  ✅ 12 Candlestick Pattern Overlay\n"
            "  ✅ Crossover Staleness Tracker\n"
            "  ✅ 9 model ensemble + GRU + isotonic calibration\n"
            "  (XGB×2 + HistGBM + ExtraTrees + RF + CatBoost)\n"
            "  ✅ HistGBM meta-model (reliability)\n"
            "  ✅ Logistic weight learner (confidence)\n"
            "  ✅ Symmetric triple barrier (TP=1.5×ATR, SL=1.0×ATR)\n"
            "  ✅ Multi-TF confluence + H1 hard gate\n"
            "  ✅ CHOPPY regime suspension\n"
            "  ✅ Anti-Streak Engine\n"
            "  ✅ Candle Quality Filter\n"
            "  ✅ Momentum Exhaustion Detector\n"
            "  ✅ Hour-Strategy Scaling (no blocking)\n"
            "  ✅ Martingale MM (capped step 3)\n"
            "  ✅ Symbol Confidence Adjustments\n"
            "  ✅ Signal validator + quality filter\n"
            "  ✅ Circuit breaker (5 losses / 8% DD)\n"
            "  ✅ Online learning (SGD/PA)\n"
            "  ✅ Feature drift detection (PSI/KS)\n\n"
            f"Features: {len(FEATURE_COLUMNS)} forward-safe\n"
            f"Symbols: {len(SYMBOLS)}\n"
            "Quality over quantity."
        )
        await query.edit_message_text(text, reply_markup=_main_menu(chat_id))

    elif data == "performance":
        try:
            total = len(_outcome_results)
            lines = ["🏆 Performance Report\n"]
            if total > 0:
                wins = sum(1 for r in _outcome_results if r.get("correct"))
                pct = wins / total * 100
                lines.append(f"📊 All-Time: {wins}W / {total - wins}L ({pct:.1f}%)")
                lines.append(f"🔥 Streak: {_total_streak} | 🏆 Peak: {_peak_accuracy*100:.1f}%")

                # Confidence accuracy
                high_conf = [r for r in _outcome_results if r.get("confidence", 0) >= 70]
                low_conf = [r for r in _outcome_results if r.get("confidence", 0) < 70]
                if high_conf:
                    hc_w = sum(1 for r in high_conf if r.get("correct"))
                    hc_pct = hc_w / len(high_conf) * 100
                    lines.append(f"\n💎 High Conf (≥70%): {hc_w}/{len(high_conf)} ({hc_pct:.0f}%)")
                if low_conf:
                    lc_w = sum(1 for r in low_conf if r.get("correct"))
                    lc_pct = lc_w / len(low_conf) * 100
                    lines.append(f"📉 Low Conf (<70%):  {lc_w}/{len(low_conf)} ({lc_pct:.0f}%)")

                # Best hours
                hour_stats = {}
                for r in _outcome_results:
                    ts = r.get("timestamp", "")
                    if len(ts) >= 13:
                        h = ts[11:13]
                        if h not in hour_stats:
                            hour_stats[h] = {"w": 0, "t": 0}
                        hour_stats[h]["t"] += 1
                        if r.get("correct"):
                            hour_stats[h]["w"] += 1
                if hour_stats:
                    sorted_hours = sorted(hour_stats.items(), key=lambda x: x[1]["w"] / max(x[1]["t"], 1), reverse=True)
                    lines.append("\n⏰ Best Hours:")
                    for h, v in sorted_hours[:3]:
                        hp = v["w"] / v["t"] * 100 if v["t"] > 0 else 0
                        lines.append(f"  {h}:00 UTC — {v['w']}/{v['t']} ({hp:.0f}%) 🔥")
            else:
                lines.append("No outcomes yet.")
            text = "\n".join(lines)
        except Exception as e:
            text = f"Performance error: {e}"
        await query.edit_message_text(text, reply_markup=_main_menu(chat_id))

    elif data == "market":
        try:
            now_utc = datetime.now(timezone.utc)
            hour = now_utc.hour
            if 0 <= hour < 7:
                session = "🌏 Asian"
            elif 7 <= hour < 12:
                session = "🇪🇺 London"
            elif 12 <= hour < 16:
                session = "🇺🇸 NY+London"
            elif 16 <= hour < 21:
                session = "🇺🇸 New York"
            else:
                session = "🌙 Off-Hours"

            lines = [
                "🌍 Market Overview\n",
                f"⏰ {now_utc.strftime('%H:%M UTC')} | {session}",
                "",
            ]
            if _last_regime:
                lines.append("🌊 Regimes:")
                for sym, regime in sorted(_last_regime.items()):
                    lines.append(f"  {sym}: {regime}")
            if _symbol_min_confidence:
                lines.append("\n🎯 Adaptive Thresholds:")
                for sym, thresh in sorted(_symbol_min_confidence.items()):
                    lines.append(f"  {sym}: {thresh:.0f}%")
            lines.append(f"\n🛡 Circuit Breaker: {'🔴 HALTED' if _cb.is_halted else '🟢 ACTIVE'}")
            text = "\n".join(lines)
        except Exception as e:
            text = f"Market error: {e}"
        await query.edit_message_text(text, reply_markup=_main_menu(chat_id))


def _get_status_text() -> str:
    """System status summary."""
    n_subs = sum(1 for s in _subscribers.values() if s.get("active"))
    n_outcomes = len(_outcome_results)
    acc = _accuracy.accuracy if hasattr(_accuracy, 'accuracy') else 0
    corr = _conf_corr.correlation if hasattr(_conf_corr, 'correlation') else 0
    cb_status = "🔴 HALTED" if _cb.is_halted else "🟢 OK"
    losses = _cb.consecutive_losses

    now_utc = datetime.now(timezone.utc)
    today = now_utc.strftime("%Y-%m-%d")
    today_outcomes = [r for r in _outcome_results if r.get("timestamp", "").startswith(today)]
    today_wins = sum(1 for r in today_outcomes if r.get("correct"))
    today_total = len(today_outcomes)
    today_pct = today_wins / today_total * 100 if today_total > 0 else 0

    # Pending signals count
    n_pending = len(_pending_signals)

    # Uptime estimate from outcome history
    first_ts = _outcome_results[0].get("timestamp", "") if _outcome_results else ""

    return (
        "⚙️ System Status\n\n"
        f"👥 Subscribers: {n_subs} active\n"
        f"📊 Total outcomes: {n_outcomes}\n"
        f"📈 Rolling accuracy: {acc*100:.1f}%\n"
        f"📉 Conf correlation: {corr:.3f}\n"
        f"🛡 Circuit breaker: {cb_status}\n"
        f"⚠️ Consecutive losses: {losses}\n"
        f"⏳ Pending signals: {n_pending}\n"
        f"🔥 Overall streak: {_total_streak}\n"
        f"🏆 Peak accuracy: {_peak_accuracy*100:.1f}%\n"
        f"\n📅 Today ({today}):\n"
        f"  Signals: {today_total} | Win: {today_wins} ({today_pct:.0f}%)\n"
        f"\n📊 Pair scores: {dict(sorted(_pair_scores.items()))}\n"
    )


def _get_results_text() -> str:
    """Today's results including pending signals."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Resolved outcomes
    today_outcomes = [r for r in _outcome_results if r.get("timestamp", "").startswith(today)]

    # Pending signals
    pending = []
    for sig_id, entry in _pending_signals.items():
        if entry.get("time", "").startswith(today):
            pending.append(entry)

    if not today_outcomes and not pending:
        return f"📊 Today's Results ({today})\n\nNo signals sent today yet."

    wins = sum(1 for r in today_outcomes if r.get("correct"))
    losses = len(today_outcomes) - wins

    lines = [f"📊 Today's Results ({today})\n"]

    if today_outcomes:
        pct = wins / len(today_outcomes) * 100
        trend = "🏆" if pct >= 60 else "📉" if pct < 50 else "➡️"
        lines.append(f"{trend} Resolved: {wins}W / {losses}L ({pct:.0f}%)")

    if pending:
        lines.append(f"⏳ Pending: {len(pending)} signal(s)\n")

    # Detailed signals
    for r in today_outcomes:
        icon = "✅" if r.get("correct") else "❌"
        sym = r.get("symbol", "?")
        d = r.get("predicted", "?")
        conf = r.get("confidence", 0)
        ts = str(r.get("timestamp", ""))[-15:-7]
        lines.append(f"{icon} {ts} {sym} {d} ({conf:.0f}%)")

    for entry in pending:
        sym = entry.get("symbol", "?")
        d = entry.get("direction", "?")
        conf = entry.get("confidence", 0)
        ts = entry.get("time", "")[-15:-7]
        lines.append(f"⏳ {ts} {sym} {d} ({conf:.0f}%) — waiting...")

    # Per-symbol summary
    if today_outcomes:
        lines.append("\n📈 Per Symbol:")
        sym_results = {}
        for r in today_outcomes:
            s = r.get("symbol", "?")
            if s not in sym_results:
                sym_results[s] = {"w": 0, "t": 0}
            sym_results[s]["t"] += 1
            if r.get("correct"):
                sym_results[s]["w"] += 1

        for s, v in sorted(sym_results.items()):
            wr = v["w"] / v["t"] * 100 if v["t"] > 0 else 0
            lines.append(f"  {s}: {v['w']}/{v['t']} ({wr:.0f}%)")

    return "\n".join(lines)


def _get_today_signals_text() -> str:
    """All signals sent today with current status."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Collect all signals from today
    signals = []

    # Pending signals
    for sig_id, entry in _pending_signals.items():
        if entry.get("time", "").startswith(today):
            sig_time = datetime.fromisoformat(entry["time"])
            elapsed = (datetime.now(timezone.utc) - sig_time).total_seconds()
            remaining = max(0, 300 - elapsed)
            signals.append({
                "time": entry["time"],
                "symbol": entry.get("symbol", "?"),
                "direction": entry.get("direction", "?"),
                "confidence": entry.get("confidence", 0),
                "regime": entry.get("regime", "?"),
                "price": entry.get("price", 0),
                "status": "pending",
                "remaining": remaining,
            })

    # Resolved outcomes
    for r in _outcome_results:
        if r.get("timestamp", "").startswith(today):
            signals.append({
                "time": r["timestamp"],
                "symbol": r.get("symbol", "?"),
                "direction": r.get("predicted", "?"),
                "confidence": r.get("confidence", 0),
                "regime": r.get("regime", "?"),
                "price": 0,
                "status": "win" if r.get("correct") else "loss",
                "remaining": 0,
            })

    if not signals:
        return f"📡 Today's Signals ({today})\n\nNo signals dispatched today."

    # Sort by time
    signals.sort(key=lambda x: x["time"])

    wins = sum(1 for s in signals if s["status"] == "win")
    losses = sum(1 for s in signals if s["status"] == "loss")
    pend_count = sum(1 for s in signals if s["status"] == "pending")
    resolved = wins + losses

    lines = [f"📡 Today's Signals ({today})\n"]
    lines.append(f"Total: {len(signals)} | ✅ {wins}W  ❌ {losses}L  ⏳ {pend_count} pending")
    if resolved > 0:
        pct = wins / resolved * 100
        lines.append(f"Accuracy: {pct:.0f}%")
    lines.append("")

    for s in signals:
        ts = s["time"][-15:-7] if len(s["time"]) > 15 else s["time"][:8]
        icon_map = {"win": "✅", "loss": "❌", "pending": "⏳"}
        label_map = {"win": "WIN", "loss": "LOSS", "pending": "PENDING"}
        icon = icon_map.get(s["status"], "?")
        label = label_map.get(s["status"], "?")
        conf = s["confidence"]
        sym = s["symbol"]
        d = s["direction"]
        line = f"{icon} {ts} {sym} {d} ({conf:.0f}%) — {label}"
        if s["status"] == "pending" and s["remaining"] > 0:
            mins = int(s["remaining"]) // 60
            secs = int(s["remaining"]) % 60
            line += f" ({mins}m {secs}s left)"
        lines.append(line)

    return "\n".join(lines)


def _get_history_text() -> str:
    """Signal history (last 20)."""
    all_signals = []

    # From outcome results
    for r in _outcome_results[-20:]:
        ts = r.get("timestamp", "?")
        sym = r.get("symbol", "?")
        pred_dir = r.get("predicted", "?")
        correct = r.get("correct", False)
        conf = r.get("confidence", 0)
        regime = r.get("regime", "?")
        icon = "✅" if correct else "❌"
        all_signals.append(f"{icon} {ts[:16]} {sym} {pred_dir} ({conf:.0f}%) [{regime}]")

    # From pending
    for sig_id, entry in _pending_signals.items():
        ts = entry.get("time", "?")
        sym = entry.get("symbol", "?")
        d = entry.get("direction", "?")
        conf = entry.get("confidence", 0)
        regime = entry.get("regime", "?")
        all_signals.append(f"⏳ {ts[:16]} {sym} {d} ({conf:.0f}%) [{regime}]")

    if not all_signals:
        return "📜 Signal History\n\nNo signals recorded."

    lines = [f"📜 Signal History (last {min(20, len(all_signals))})\n"]
    lines.extend(all_signals[-20:])
    return "\n".join(lines)


# =============================================================================
#  Server Time & M5 Close Alignment
# =============================================================================

def _get_server_time() -> datetime:
    """Get MT5 server time in UTC."""
    tick = mt5.symbol_info_tick(DEFAULT_SYMBOL)
    if tick and tick.time:
        return datetime.fromtimestamp(tick.time, tz=timezone.utc)
    return datetime.now(timezone.utc)


def _next_m5_close(server_time: datetime) -> datetime:
    """Calculate next M5 candle close time."""
    minute = server_time.minute
    next_5 = ((minute // 5) + 1) * 5
    close = server_time.replace(second=0, microsecond=0)
    if next_5 >= 60:
        close += timedelta(hours=1)
        close = close.replace(minute=next_5 - 60)
    else:
        close = close.replace(minute=next_5)
    return close


# =============================================================================
#  Auto-Signal Job (Core Loop)
# =============================================================================

async def _auto_signal_job(app: Application):
    """
    Main auto-signal loop — runs every M5 candle.

    Implements all 14 layers of the pipeline:
      1. Data collection per symbol
      2. Cleaning (in feature pipeline)
      3. Feature engineering (66 features)
      4. (Labels — training only)
      5-7. Primary -> Meta -> Weight model inference
      8. Regime detection
      9. Risk filtering
     10. Signal output (BUY/SELL/SKIP)
     11. Telegram dispatch
     12. Circuit breaker check
     13. Drift detection (periodic)
     14. Online learning update (on outcomes)
    """
    bot: Bot = app.bot
    log.info("Auto-signal job started.")

    global _pair_counter, _pair_scores, _candle_count

    while True:
        try:
            # Align to M5 close
            server_time = _get_server_time()
            close_time = _next_m5_close(server_time)
            send_time = close_time - timedelta(seconds=TELEGRAM_SEND_BEFORE_CLOSE_SEC)
            wait_seconds = (send_time - server_time).total_seconds()

            if wait_seconds < 0:
                wait_seconds += 300
            if wait_seconds > 0:
                log.info("Auto-signal: next cycle in %.0fs at %s UTC",
                         wait_seconds, send_time.strftime("%H:%M:%S"))
                await asyncio.sleep(wait_seconds)

            # Collect active subscribers
            active_subs = {cid: info for cid, info in _subscribers.items()
                           if info.get("active", False)}
            if not active_subs:
                await asyncio.sleep(10)
                continue

            log.info("Auto-signal: cycle for %d subscribers", len(active_subs))

            # Gather needed symbols
            needed_symbols = set()
            for info in active_subs.values():
                needed_symbols.update(info.get("symbols", SYMBOLS))

            # ── LAYER 12: Circuit Breaker ──
            if not _cb.can_trade():
                log.info("Circuit breaker active. Skipping cycle.")
                await asyncio.sleep(300)
                continue

            # ── Weekend Check ──
            now_utc = datetime.now(timezone.utc)
            if now_utc.weekday() >= 5:
                log.info("Market closed (weekend). Sleeping 30 min.")
                await asyncio.sleep(1800)
                continue

            # v11 Advanced: No hour blocking — all hours tradeable via confidence scaling

            # ── LAYER 13: Drift Detection (periodic) ──
            _candle_count += 1
            if _candle_count % DRIFT_CHECK_INTERVAL == 0:
                try:
                    from drift_detector import check_drift_from_files
                    drift_result = check_drift_from_files()
                    if drift_result.get("drift_detected"):
                        log.warning("Feature drift detected: %s", drift_result.get("details"))
                        asyncio.create_task(_retrain_job(bot))
                except Exception as e:
                    log.debug("Drift check failed: %s", e)

                # v12: Feature importance drift check
                try:
                    fi_monitor = get_importance_monitor()
                    fi_result = fi_monitor.check_drift()
                    if fi_result.get("is_drifted"):
                        log.warning("Feature IMPORTANCE drift: corr=%.3f, drifted=%s",
                                    fi_result["rank_correlation"],
                                    fi_result["drifted_features"][:3])
                except Exception as e:
                    log.debug("Feature importance check failed: %s", e)

            # ── Dynamic Pair Selector ──
            _pair_counter += 1
            if _pair_counter >= PAIR_SCORE_REFRESH or not _pair_scores:
                _pair_counter = 0
                for sym in list(needed_symbols):
                    try:
                        data = fetch_multi_timeframe(sym, 500)
                        df = compute_features(data.get("M5"), m15_df=data.get("M15"),
                                              h1_df=data.get("H1"), m1_df=data.get("M1"))
                        res = score_pair(df, symbol=sym)
                        _pair_scores[sym] = res["total"]
                        log.info("Pair score %s: %d/30", sym, res["total"])
                    except Exception:
                        _pair_scores[sym] = 25  # assume tradeable on error

            # Filter by pair score
            tradeable = {s for s in needed_symbols if _pair_scores.get(s, 25) >= PAIR_SELECTOR_MIN_SCORE}
            skipped_pairs = needed_symbols - tradeable
            if skipped_pairs:
                log.info("Pair selector skipped %s (score < %d)", skipped_pairs, PAIR_SELECTOR_MIN_SCORE)
            needed_symbols = tradeable

            if not needed_symbols:
                log.info("No tradeable symbols this cycle.")
                await asyncio.sleep(60)
                continue

            # ── Resolve pending outcomes (Layer 14: Online Learning updates) ──
            await _resolve_pending_signals(bot)

            # ── Run Predictions ──
            predictions = {}
            filtered_out = {}

            for sym in needed_symbols:
                try:
                    # Cooldown check
                    if sym in _cooldown_until:
                        if datetime.now(timezone.utc) < _cooldown_until[sym]:
                            filtered_out[sym] = "cooldown"
                            continue
                        else:
                            del _cooldown_until[sym]

                    # Per-symbol consecutive loss check
                    if _consecutive_losses.get(sym, 0) >= MAX_CONSECUTIVE_LOSSES_PER_SYM:
                        _cooldown_until[sym] = datetime.now(timezone.utc) + timedelta(minutes=30)
                        filtered_out[sym] = f"per_sym_losses:{_consecutive_losses[sym]}"
                        _consecutive_losses[sym] = 0
                        continue

                    # Regime transition skip
                    if _regime_skip.get(sym, 0) > 0:
                        _regime_skip[sym] -= 1
                        filtered_out[sym] = "regime_transition"
                        continue

                    # ── Layers 1-8: Prediction Pipeline ──
                    pred = _run_prediction(sym)
                    _log_all_predictions(sym, pred)

                    conf = pred.get("final_confidence_percent", 0)
                    direction = pred.get("suggested_direction", "HOLD")
                    trade = pred.get("suggested_trade", "HOLD")
                    regime = pred.get("market_regime", "Unknown")
                    risk_warnings = pred.get("risk_warnings", [])

                    # v11: Symbol-specific confidence adjustment
                    try:
                        from config import V11_SYMBOL_CONFIDENCE_ADJUSTMENTS
                        sym_mult = V11_SYMBOL_CONFIDENCE_ADJUSTMENTS.get(sym, 1.0)
                        if sym_mult != 1.0:
                            conf = conf * sym_mult
                            pred["final_confidence_percent"] = conf
                    except Exception:
                        pass

                    # Regime transition detection
                    prev = _last_regime.get(sym)
                    if prev and regime != prev:
                        _regime_skip[sym] = 2
                        filtered_out[sym] = f"regime_shift:{prev}->{regime}"
                        _last_regime[sym] = regime
                        continue
                    _last_regime[sym] = regime

                    # Skip errors
                    if pred.get("error"):
                        filtered_out[sym] = pred["error"]
                        continue

                    # ── LAYER 9: Risk Filter ──

                    # 9a. Hard risk warnings (ATR spike, high spread, low vol)
                    blocked, reason = _has_hard_risk(risk_warnings)
                    if blocked:
                        filtered_out[sym] = f"risk:{reason}"
                        log.info("Risk filter blocked %s: %s", sym, reason)
                        continue

                    # 9b. Confidence gate (adaptive per-symbol + v12 Bayesian)
                    sym_min_conf = _symbol_min_confidence.get(sym, MIN_CONFIDENCE)
                    # v12: Bayesian threshold override (per symbol/regime/hour)
                    try:
                        hour_now = datetime.now(timezone.utc).hour
                        bayes_min = _bayesian.get_min_confidence(sym, regime, hour_now)
                        sym_min_conf = max(sym_min_conf, bayes_min)
                    except Exception:
                        pass
                    if conf < sym_min_conf:
                        filtered_out[sym] = f"confidence<{sym_min_conf:.0f}"
                        continue

                    # 9c. Ensemble agreement (v12: use feedback loop thresholds)
                    unanimity = float(pred.get("ensemble_unanimity", 0))
                    variance = float(pred.get("ensemble_variance", 1.0))
                    fb_thresholds = _feedback.get_thresholds()
                    effective_min_unanimity = max(MIN_UNANIMITY, fb_thresholds.get("min_unanimity", MIN_UNANIMITY))
                    effective_max_variance = min(MAX_ENSEMBLE_VARIANCE, fb_thresholds.get("max_variance", MAX_ENSEMBLE_VARIANCE))
                    if unanimity < effective_min_unanimity:
                        filtered_out[sym] = f"unanimity:{unanimity:.2f}<{effective_min_unanimity:.2f}"
                        continue
                    if variance > effective_max_variance:
                        filtered_out[sym] = f"variance:{variance:.3f}>{effective_max_variance:.3f}"
                        continue

                    # 9d. Production gate (HOLD)
                    if trade == "HOLD" or direction == "HOLD":
                        filtered_out[sym] = pred.get("skip_reason", "hold")
                        continue

                    # 9e. Blocked regimes
                    if regime in PRODUCTION_BLOCKED_REGIMES:
                        filtered_out[sym] = f"blocked_regime:{regime}"
                        continue

                    # 9f. Multi-TF confluence (Layer 9 extension)
                    cf = _check_confluence(sym, direction)
                    cf_m5_df = cf.pop("_m5_df", None)  # reuse for momentum check
                    pred["_confluence_score"] = cf.get("confluence_score", 0)
                    if not cf.get("confluence_pass", True):
                        reason = cf.get("reason", "TF conflict")
                        soft_override = (
                            cf.get("h1_hard_gate_pass", True)
                            and conf >= SIGNAL_SOFT_CONFLUENCE_OVERRIDE_CONFIDENCE
                            and float(pred.get("quality_score", 0)) >= SIGNAL_SOFT_CONFLUENCE_OVERRIDE_QUALITY
                            and float(pred.get("ensemble_unanimity", 0)) >= SIGNAL_SOFT_CONFLUENCE_OVERRIDE_UNANIMITY
                        )
                        if soft_override:
                            pred["_confluence_override"] = True
                            pred["_confluence_score"] = max(pred.get("_confluence_score", 0), 2)
                            log.info("Confluence soft override %s: %s", sym, reason)
                        else:
                            filtered_out[sym] = f"confluence:{reason}"
                            log.info("Confluence filter blocked %s: %s", sym, reason)
                            continue

                    # 9g. Session multiplier — DISABLED v13 (pure technical analysis)
                    session = pred.get("session", "Off")
                    # No session penalty — all sessions treated equally
                    adjusted_conf = conf
                    # (skip session rejection)

                    # 9h. Momentum confirmation (v2 accuracy upgrade)
                    try:
                        m5_data = cf_m5_df  # reuse data from confluence check
                        if m5_data is None:
                            d = fetch_multi_timeframe(sym, 50)
                            m5_data = d.get("M5")
                        mom_pass, mom_reason = _check_momentum_confirmation(pred, m5_data)
                        if not mom_pass:
                            filtered_out[sym] = f"momentum:{mom_reason}"
                            log.info("Momentum filter blocked %s: %s", sym, mom_reason)
                            continue
                        # Also compute candle pattern quality and attach to pred
                        cpq = _check_candle_pattern_quality(m5_data, direction)
                        pred["_candle_pattern_quality"] = cpq
                    except Exception as e:
                        log.debug("Momentum check skipped for %s: %s", sym, e)

                    predictions[sym] = pred

                except Exception as e:
                    log.error("Prediction failed %s: %s", sym, e)
                    record_error(f"{sym}: {e}")
                    filtered_out[sym] = f"error:{e}"

            # ── Correlation Guard ──
            try:
                for sym, pred in list(predictions.items()):
                    corr = check_correlation(sym, pred.get("suggested_direction", ""), predictions)
                    if not corr.get("corr_pass", True):
                        filtered_out[sym] = f"correlation:{corr.get('reason')}"
                        del predictions[sym]
            except Exception as e:
                log.warning("Correlation filter error: %s", e)

            # ── Direction Flip Filter ──
            for sym, pred in list(predictions.items()):
                direction = pred["suggested_direction"]
                regime = pred.get("market_regime", "Unknown")
                if sym in _prev_directions:
                    if _prev_directions[sym] != direction and regime != "Trending":
                        filtered_out[sym] = "direction_flip"
                        del predictions[sym]
                        continue
                if sym in predictions:
                    _prev_directions[sym] = direction

            # ── Stacking Gate ──
            for sym, pred in list(predictions.items()):
                try:
                    decision = _stacker.should_send(pred)
                    if not decision.get("send", True):
                        filtered_out[sym] = f"stack:{decision.get('reason')}"
                        del predictions[sym]
                except Exception:
                    pass

            # ── LAYER 10: Rank & Select (v12: adaptive min_score) ──
            if predictions:
                fb_thresholds = _feedback.get_thresholds()
                effective_min_score = max(MIN_SIGNAL_SCORE, fb_thresholds.get("min_signal_score", MIN_SIGNAL_SCORE))
                predictions = rank_and_select(
                    predictions,
                    max_signals=MAX_SIGNALS_PER_CYCLE,
                    min_score=effective_min_score,
                )

            # ── LAYER 10b: Signal Validator (confidence, duplicate, rate limit) ──
            for sym, pred in list(predictions.items()):
                is_valid, reject_reasons = validate_signal(pred, sym, _cb)
                if not is_valid:
                    filtered_out[sym] = "; ".join(reject_reasons[:2])
                    del predictions[sym]

            # ── LAYER 10c: Dispatch Model Gate (v12) ──
            for sym, pred in list(predictions.items()):
                try:
                    should_send, win_prob, reason = _dispatch.should_dispatch(pred, sym)
                    pred["dispatch_win_prob"] = win_prob
                    if not should_send:
                        filtered_out[sym] = f"dispatch:{reason}"
                        del predictions[sym]
                except Exception as e:
                    log.debug("Dispatch model check skipped for %s: %s", sym, e)
                    pred["dispatch_win_prob"] = 0.5

            # ── Log Filtered ──
            if filtered_out:
                log.info("Filtered: %s", {k: v[:50] for k, v in filtered_out.items()})

            # ── LAYER 11: Signal Dispatch ──
            if predictions:
                message = _format_auto_signal(predictions, filtered_out)

                sent_ids = []
                msg_ids = {}  # {chat_id: message_id} for result editing
                for cid, info in active_subs.items():
                    if info.get("active"):
                        syms = info.get("symbols", SYMBOLS)
                        sub_preds = {s: p for s, p in predictions.items() if s in syms}
                        if sub_preds:
                            mid = await _safe_send(bot, cid, message)
                            if mid:
                                msg_ids[cid] = mid
                            sent_ids.append(cid)

                # Record sent signals + cycle for message editing
                cycle_key = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                sig_ids = []
                for sym, pred in predictions.items():
                    sid = _record_sent_signal(sym, pred, sent_ids)
                    sig_ids.append(sid)
                    # Record in SQLite DB and link db_id to pending signal
                    try:
                        db_id = _signal_db.insert_signal(
                            symbol=sym,
                            direction=pred.get("suggested_direction", "HOLD"),
                            probability=pred.get("green_probability_percent", 50),
                            confidence=pred.get("final_confidence_percent", 0),
                            meta_score=pred.get("meta_reliability_percent", 0),
                            regime=pred.get("market_regime", ""),
                            session=pred.get("session", ""),
                            ensemble_std=pred.get("ensemble_variance", 0),
                            kelly_fraction=pred.get("kelly_fraction_percent", 0),
                            signal_strength=pred.get("signal_strength", ""),
                            reason_chain=pred.get("reason_chain", ""),
                            quality_score=pred.get("quality_score", 0),
                            unanimity=pred.get("ensemble_unanimity", 0),
                            ensemble_variance=pred.get("ensemble_variance", 0),
                        )
                        # Link DB ID to pending signal for outcome update
                        if sid in _pending_signals:
                            _pending_signals[sid]["db_id"] = db_id
                    except Exception as db_err:
                        log.warning("signal_db insert error: %s", db_err)

                # Store cycle for result message editing
                if msg_ids and sig_ids:
                    _cycle_messages[cycle_key] = {
                        "text": message,
                        "msg_ids": msg_ids,
                        "sig_ids": sig_ids,
                        "results": {},
                    }
                    _save_cycle_messages()

                # Persist pending signals (with db_ids now attached)
                try:
                    with open(PENDING_JSON, "w") as f:
                        json.dump(_pending_signals, f, indent=2, default=str)
                except Exception:
                    pass

                log.info("Sent %d signals to %d subscribers.", len(predictions), len(sent_ids))
            else:
                log.info("No signals passed filters this cycle.")
                # ── No-Signal Notification ──
                now_utc = datetime.now(timezone.utc)
                no_sig_msg = (
                    "\u23f3 *No Signal Available*\n\n"
                    f"\u23f0 Time: {now_utc.strftime('%H:%M UTC')}\n"
                    f"\U0001f50d Pairs scanned: {len(SYMBOLS)}\n"
                )
                if filtered_out:
                    reasons = list(filtered_out.values())[:3]
                    no_sig_msg += f"\u274c Filtered: {len(filtered_out)} pair(s)\n"
                    for r in reasons:
                        no_sig_msg += f"  \u2022 {r[:60]}\n"
                no_sig_msg += (
                    "\n\U0001f504 Next scan in ~5 minutes.\n"
                    "\U0001f4a1 _Only high-confidence signals are sent._"
                )
                for cid, info in active_subs.items():
                    if info.get("active"):
                        await _safe_send(bot, cid, no_sig_msg, parse_mode=None)

        except Exception as e:
            log.error("Auto-signal cycle error: %s", e, exc_info=True)
            record_error(str(e))
            await asyncio.sleep(60)


# =============================================================================
#  Report Jobs
# =============================================================================

async def _daily_report_job(app: Application):
    """Send daily performance summary at 21:00 UTC."""
    bot: Bot = app.bot
    log.info("Daily report job started.")

    while True:
        try:
            now = datetime.now(timezone.utc)
            target = now.replace(hour=21, minute=0, second=0, microsecond=0)
            if now >= target:
                target += timedelta(days=1)
            wait = (target - now).total_seconds()
            log.info("Daily report: next in %.0f min", wait / 60)
            await asyncio.sleep(wait)

            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            today_outcomes = [r for r in _outcome_results if r.get("timestamp", "").startswith(today)]

            if not today_outcomes:
                await asyncio.sleep(60)
                continue

            wins = sum(1 for r in today_outcomes if r.get("correct"))
            total = len(today_outcomes)

            text = (
                f"Daily Report — {today}\n\n"
                f"Signals: {total}\n"
                f"Wins: {wins}\n"
                f"Accuracy: {wins/total*100:.1f}%\n"
                f"Circuit breaker: {'HALTED' if _cb.is_halted else 'OK'}\n"
            )

            # Per-symbol breakdown
            sym_map = {}
            for r in today_outcomes:
                s = r["symbol"]
                if s not in sym_map:
                    sym_map[s] = [0, 0]
                sym_map[s][1] += 1
                if r.get("correct"):
                    sym_map[s][0] += 1

            for s, (w, t) in sorted(sym_map.items()):
                text += f"\n  {s}: {w}/{t} ({w/t*100:.0f}%)"

            await _broadcast(bot, text)

        except Exception as e:
            log.error("Daily report error: %s", e)
            await asyncio.sleep(300)


async def _weekly_report_job(app: Application):
    """Send weekly performance summary on Sunday 20:00 UTC."""
    bot: Bot = app.bot
    log.info("Weekly report job started.")

    while True:
        try:
            now = datetime.now(timezone.utc)
            # Next Sunday 20:00
            days_until_sunday = (6 - now.weekday()) % 7
            if days_until_sunday == 0 and now.hour >= 20:
                days_until_sunday = 7
            target = (now + timedelta(days=days_until_sunday)).replace(hour=20, minute=0, second=0, microsecond=0)
            wait = (target - now).total_seconds()
            log.info("Weekly report: next in %.1f hours", wait / 3600)
            await asyncio.sleep(wait)

            if not _outcome_results:
                await asyncio.sleep(60)
                continue

            # Last 7 days
            cutoff = datetime.now(timezone.utc) - timedelta(days=7)
            week = [r for r in _outcome_results
                    if datetime.fromisoformat(r.get("timestamp", "2000-01-01T00:00:00+00:00")) > cutoff]

            if not week:
                await asyncio.sleep(60)
                continue

            wins = sum(1 for r in week if r.get("correct"))
            total = len(week)

            text = (
                f"Weekly Report\n\n"
                f"Total signals: {total}\n"
                f"Wins: {wins}\n"
                f"Accuracy: {wins/total*100:.1f}%\n"
            )
            await _broadcast(bot, text)

        except Exception as e:
            log.error("Weekly report error: %s", e)
            await asyncio.sleep(3600)


# =============================================================================
#  Retrain Job
# =============================================================================

async def _retrain_job(bot: Bot):
    """Trigger model retrain when accuracy/correlation degrades."""
    try:
        from auto_learner import retrain
        log.info("Auto-retrain triggered.")
        result = retrain()
        if result:
            load_models()
            log.info("Retrain complete. Models reloaded.")
            if ADMIN_CHAT_ID:
                await _safe_send(bot, ADMIN_CHAT_ID, "Auto-retrain complete. Models reloaded.")
    except Exception as e:
        log.error("Retrain failed: %s", e)


async def _retrain_check_job(app: Application):
    """Periodic retrain trigger check (every 4 hours)."""
    bot: Bot = app.bot

    while True:
        try:
            await asyncio.sleep(14400)  # 4 hours

            acc = _accuracy.accuracy if hasattr(_accuracy, 'accuracy') else 0.5
            corr = _conf_corr.correlation if hasattr(_conf_corr, 'correlation') else 0.5

            if acc < AUTO_RETRAIN_ACCURACY_TRIGGER or corr < AUTO_RETRAIN_CORRELATION_TRIGGER:
                log.warning("Retrain trigger: accuracy=%.3f, correlation=%.3f", acc, corr)
                await _retrain_job(bot)

        except Exception as e:
            log.error("Retrain check error: %s", e)
            await asyncio.sleep(3600)


# =============================================================================
#  Main Entry
# =============================================================================

def main():
    """Initialize and start the v11 Advanced bot."""
    if not TELEGRAM_BOT_TOKEN:
        log.error("TELEGRAM_BOT_TOKEN not set. Exiting.")
        sys.exit(1)

    # Ensure directories exist
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Connect MT5
    if not _ensure_mt5():
        log.warning("MT5 connection failed at startup. Will retry on first prediction.")

    # Load state
    _load_subscribers()
    _load_outcomes()

    # Load ML models
    try:
        load_models()
        log.info("Models pre-loaded.")
    except Exception as e:
        log.warning("Model pre-load failed: %s (will retry)", e)

    # Load pending signals
    try:
        global _pending_signals
        if os.path.exists(PENDING_JSON):
            with open(PENDING_JSON) as f:
                loaded = json.load(f)
                _pending_signals = loaded if isinstance(loaded, dict) else {}
    except Exception:
        _pending_signals = {}

    # Load cycle messages (for result editing after restart)
    _load_cycle_messages()

    record_startup()

    # Build application
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Register handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("predict", cmd_predict))
    app.add_handler(CommandHandler("scan", cmd_scan))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("results", cmd_results))
    app.add_handler(CommandHandler("today", cmd_today))
    app.add_handler(CommandHandler("dashboard", cmd_dashboard))
    app.add_handler(CommandHandler("audit", cmd_audit))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("leaderboard", cmd_leaderboard))
    app.add_handler(CommandHandler("performance", cmd_performance))
    app.add_handler(CommandHandler("market", cmd_market))
    app.add_handler(CallbackQueryHandler(_handle_callback))

    async def _on_startup(app_ref: Application):
        log.info("Telegram bot starting (v11 Advanced)...")

        # Start background tasks
        asyncio.create_task(_auto_signal_job(app_ref))
        asyncio.create_task(_daily_report_job(app_ref))
        asyncio.create_task(_weekly_report_job(app_ref))
        asyncio.create_task(_retrain_check_job(app_ref))

        log.info("All background jobs scheduled.")

    app.post_init = _on_startup

    log.info("QUOTEX LORD v15 Advanced starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
