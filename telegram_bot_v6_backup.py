"""
Telegram Bot v3 — Auto-signal mode + interactive menu.

Auto-Signal: Automatically sends predictions to all subscribers
5 seconds before every M5 candle close. No request needed.

Features:
    - Auto-signal every 5 minutes (subscribe to start)
    - Interactive button menu
    - Multi-symbol support
    - Win/loss tracking with actual outcomes
    - Daily performance summary
    - Position sizing via Kelly criterion

Usage:
    set TELEGRAM_BOT_TOKEN=<your_token>
    python telegram_bot.py
"""

import asyncio
import json
import os
import logging
import math
from datetime import datetime, timezone, timedelta

import MetaTrader5 as mt5
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Bot
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, ContextTypes,
)

from config import (
    DEFAULT_SYMBOL, CANDLES_TO_FETCH,
    MT5_PATH, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER,
    TELEGRAM_BOT_TOKEN, TELEGRAM_SEND_BEFORE_CLOSE_SEC,
    LOG_LEVEL, LOG_FORMAT, LOG_DIR, ADMIN_CHAT_ID,
    PRODUCTION_MIN_CONFIDENCE,
)
from data_collector import fetch_multi_timeframe
from feature_engineering import compute_features, FEATURE_COLUMNS
from regime_detection import detect_regime
from predict_engine import (
    predict, load_models, log_prediction,
    verify_candle_closed, update_prediction_history, update_online_learner,
    set_retrain_flag, reload_models
)
from stacking_model import StackingGatekeeper
from adaptive_threshold import AdaptiveThreshold
from gemini_filter import verify_signal
from news_sentiment import fetch_live_news
from risk_filter import check_warnings
from circuit_breaker import CircuitBreaker
from pair_selector import score_pair, MIN_SCORE as PAIR_SELECTOR_MIN_SCORE
from confluence_filter import check_confluence
from signal_ranker import rank_and_select, compute_signal_score
from correlation_filter import check_correlation
from candle_patterns import pattern_confirms
from news_filter import check_news_filter
import threading
from stability import AccuracyTracker, ConfidenceCorrelationTracker
from production_state import (
    record_startup, record_prediction, record_error,
    get_state_summary,
)

log = logging.getLogger("telegram_bot")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

# Initialize adaptive threshold and stacking gatekeeper
_adaptive_thresh = AdaptiveThreshold()
_stacker = StackingGatekeeper()

_tracker = AccuracyTracker()
_conf_tracker = ConfidenceCorrelationTracker()

# Supported symbols (6 pairs)
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "GBPJPY", "AUDUSD"]
LIVE_CANDLES_TO_FETCH = 1000  # enough history for EMA200/warmup while keeping live latency low

# Subscriber storage
SUBS_FILE = os.path.join(LOG_DIR, "subscribers.json")
SIGNAL_CSV = os.path.join(LOG_DIR, "signal_outcomes.csv")
PREDICTION_CSV = os.path.join(LOG_DIR, "all_predictions.csv")
SENT_SIGNALS_CSV = os.path.join(LOG_DIR, "sent_signals.csv")
PENDING_SIGNALS_FILE = os.path.join(LOG_DIR, "pending_signals.json")
GEMINI_LOG_CSV = os.path.join(LOG_DIR, "gemini_decisions.csv")
_subscribers = {}  # {chat_id: {"symbols": [...], "active": True, "mode": "all"|"high"}}
_last_predictions = {}  # {symbol: {pred_dict, timestamp}}
_signal_history = []  # [{symbol, direction, confidence, timestamp, result}]
_outcome_results = []  # [{symbol, predicted, actual, correct, confidence, time}]
_consecutive_losses = {}  # {symbol: int}
_consecutive_wins = {}    # {symbol: int} — for win streak tracking
_cooldown_until = {}  # {symbol: datetime}
_last_regime = {}  # {symbol: regime_name} — for transition detection
_regime_transition_skip = {}  # {symbol: int} — candles to skip

# Signal quality thresholds (DATA-DRIVEN: tuned from 130 signal history)
# Confidence <42% has 45.8% win rate (losing!) vs >=42% has 70.7%
MIN_CONFIDENCE_FILTER = 42.0   # raised from 35 — kills the losing 38-42% tier
HIGH_CONF_THRESHOLD = 45.0     # for "high" mode subscribers
COOLDOWN_CANDLES = 3
MAX_CONSECUTIVE_LOSSES = 2     # tightened from 3 — faster cooldown per symbol
REGIME_SKIP_CANDLES = 2  # skip after regime transition

# Regimes to block entirely (win rate < 50% in historical data)
BLOCKED_REGIMES = {"High_Volatility"}  # 47.8% win rate = losing

# Symbols to block (consistently underperforming)
BLOCKED_SYMBOLS = {"AUDUSD"}  # 43.8% win rate over 16 signals

# Ensemble disagreement threshold — skip when models disagree too much
MAX_ENSEMBLE_VARIANCE = 0.035  # skip if variance > this (models arguing)
MIN_UNANIMITY = 0.857          # Strategy 2: require 6/7 models to agree (85.7%)

# Strategy 3: Trading session windows (UTC hours that are profitable)
# V6 data-driven (403 outcomes): 05=64.7%, 13=54.8%, 14=64.2%, 16=63.0%
# Removed: 07(50%), 08(50%), 11(52.4%), 12(47.1%), 18(50%) — coinflip or worse
# Also removed: 09(37.5%), 10(23.5%), 15(44.7%) — clear losers
PROFITABLE_HOURS_UTC = {5, 13, 14, 16, 17}  # only proven profitable hours
BLOCKED_HOURS_UTC = {7, 8, 9, 10, 11, 12, 15, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 6}  # blocked

# Strategy 4: Max signals per cycle
MAX_SIGNALS_PER_CYCLE = 2     # only send top 2 ranked signals
MIN_SIGNAL_SCORE = 50.0       # minimum composite score to qualify

# Strategy 5: Consecutive candle confirmation
_prev_directions = {}  # {symbol: last_predicted_direction}

# Data-driven live symbol filter from stored outcomes
SYMBOL_ACCURACY_MIN_TOTAL_SAMPLES = 20
SYMBOL_ACCURACY_MIN_OVERALL_WR = 48.0
SYMBOL_ACCURACY_RECENT_WINDOW = 15
SYMBOL_ACCURACY_MIN_RECENT_SAMPLES = 10
SYMBOL_ACCURACY_MIN_RECENT_WR = 50.0
PREFERRED_SYMBOL_MIN_TOTAL_SAMPLES = 30
PREFERRED_SYMBOL_MIN_OVERALL_WR = 53.0
PREFERRED_SYMBOL_MAX_COUNT = 1
PREFERRED_SYMBOL_MIN_RECENT_TOTAL = SYMBOL_ACCURACY_RECENT_WINDOW
PREFERRED_SYMBOL_EXTRA_MIN_TOTAL_SAMPLES = 50
PREFERRED_SYMBOL_EXTRA_MIN_OVERALL_WR = 58.0
ADAPTIVE_MIN_CONFIDENCE_RECENT_WR_CUTOFF = 55.0
ADAPTIVE_MIN_CONFIDENCE_MIN_RECENT_SAMPLES = SYMBOL_ACCURACY_MIN_RECENT_SAMPLES
ADAPTIVE_MIN_CONFIDENCE_WEAK_SYMBOL = 68.0
HARD_RISK_WARNING_PREFIXES = ("High spread", "Low volatility")

# Kelly minimum filter — signals with kelly < 1% show 48.8% WR (losing)
MIN_KELLY_THRESHOLD = 1.0

# DOWN direction premium — DOWN has 48.7% WR vs UP 59.4%
# Require this extra % confidence for DOWN signals
DOWN_DIRECTION_CONFIDENCE_PREMIUM = 3.0

# Symbol × Direction cross-filter: block directions with stored WR < this threshold
# E.g. GBPUSD DOWN=45.8%, GBPJPY DOWN=48.2% — these get blocked
SYMBOL_DIRECTION_MIN_SAMPLES = 15
SYMBOL_DIRECTION_MIN_WR = 48.0

# Symbol × Regime cross-filter: block combos with stored WR < threshold
# E.g. GBPJPY/Ranging=33.3%, AUDUSD/Trending=29.2%
SYMBOL_REGIME_MIN_SAMPLES = 8
SYMBOL_REGIME_MIN_WR = 45.0

# Streak-based adaptive thresholds (pro binary bot feature)
STREAK_LOSS_CONFIDENCE_BUMP = 5.0    # after 2 consecutive losses, raise min confidence by this
STREAK_LOSS_TRIGGER = 2              # number of consecutive losses to trigger bump
STREAK_LOSS_MAX_BUMP = 10.0          # maximum cumulative confidence bump from streaks
STREAK_WIN_THRESHOLD_REDUCTION = 2.0 # after 3 wins, allow small confidence reduction
STREAK_WIN_TRIGGER = 3               # number of consecutive wins to trigger reduction

# Support/Resistance proximity filter
SR_PROXIMITY_ATR_THRESHOLD = 0.3     # block signals within 0.3 ATR of S/R level (likely reversal zone)

# Round-number / psychological level filter
ROUND_LEVEL_PROXIMITY_PIPS = 10      # pips from a round number to consider "near"

# Preferred symbol pair-score bonus — don't let pair selector block top performers
PREFERRED_SYMBOL_PAIR_SCORE_BONUS = 2  # add this to pair score for preferred symbols

# Circuit breaker instance (tightened: 3 losses instead of 5)
_cb = CircuitBreaker(max_losses=3, max_drawdown_pct=6.0, cooldown_min=30)
_cb_just_reset = False   # one-time 'trading resumed' message flag

# Pair selector and drift monitor state
_candle_count = 0
DRIFT_CHECK_INTERVAL = 100

_pair_scores = {}
_pair_score_counter = 0
PAIR_SCORE_REFRESH = 500


# =============================================================================
#  Subscriber Persistence
# =============================================================================

def _load_subscribers():
    global _subscribers
    if os.path.exists(SUBS_FILE):
        try:
            with open(SUBS_FILE, "r") as f:
                _subscribers = json.load(f)
            log.info("Loaded %d subscribers.", len(_subscribers))
        except Exception:
            _subscribers = {}


def _save_subscribers():
    os.makedirs(os.path.dirname(SUBS_FILE), exist_ok=True)
    with open(SUBS_FILE, "w") as f:
        json.dump(_subscribers, f)


def _load_today_from_csv():
    """Load today's signals and outcomes from CSV on startup."""
    global _signal_history, _outcome_results
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    import csv

    # Load outcomes
    if os.path.exists(SIGNAL_CSV):
        try:
            with open(SIGNAL_CSV, "r") as f:
                for row in csv.DictReader(f):
                    if row.get("timestamp", "").startswith(today):
                        _outcome_results.append({
                            "symbol": row.get("symbol", ""),
                            "predicted": row.get("predicted", ""),
                            "actual": row.get("actual", ""),
                            "correct": row.get("correct", "False") == "True",
                            "confidence": float(row.get("confidence", 0)),
                            "time": row.get("timestamp", ""),
                        })
            log.info("Loaded %d outcomes from CSV.", len(_outcome_results))
        except Exception as e:
            log.warning("CSV outcomes load failed: %s", e)

    # Load predictions/signals
    if os.path.exists(PREDICTION_CSV):
        try:
            with open(PREDICTION_CSV, "r") as f:
                for row in csv.DictReader(f):
                    if row.get("timestamp", "").startswith(today):
                        _signal_history.append({
                            "symbol": row.get("symbol", ""),
                            "direction": row.get("direction", ""),
                            "confidence": float(row.get("confidence", 0)),
                            "kelly": float(row.get("kelly", 0)),
                            "trade": row.get("trade", ""),
                            "regime": row.get("regime", ""),
                            "time": row.get("timestamp", ""),
                        })
            log.info("Loaded %d signals from CSV.", len(_signal_history))
        except Exception as e:
            log.warning("CSV signals load failed: %s", e)


def _load_pending_signals() -> list:
    if not os.path.exists(PENDING_SIGNALS_FILE):
        return []
    try:
        with open(PENDING_SIGNALS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        log.warning("Pending signals load failed: %s", e)
        return []


def _save_pending_signals(records: list):
    os.makedirs(os.path.dirname(PENDING_SIGNALS_FILE), exist_ok=True)
    try:
        with open(PENDING_SIGNALS_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
    except Exception as e:
        log.warning("Pending signals save failed: %s", e)


def _remove_pending_signal_ids(signal_ids: set[str]):
    if not signal_ids:
        return

    pending_records = _load_pending_signals()
    if not pending_records:
        return

    remaining = [record for record in pending_records if record.get("signal_id") not in signal_ids]
    if len(remaining) != len(pending_records):
        _save_pending_signals(remaining)


def _record_sent_signals(predictions: dict, sent_symbols_per_sub: dict):
    if not predictions or not sent_symbols_per_sub:
        return {}

    now = datetime.now(timezone.utc)
    evaluate_after = now + timedelta(seconds=TELEGRAM_SEND_BEFORE_CLOSE_SEC + 15 + 300 + 15)
    pending_records = _load_pending_signals()
    header_needed = not os.path.exists(SENT_SIGNALS_CSV)
    os.makedirs(os.path.dirname(SENT_SIGNALS_CSV), exist_ok=True)

    sent_map = {}
    signal_ids = {}
    for chat_id, symbols in sent_symbols_per_sub.items():
        for symbol in symbols:
            sent_map.setdefault(symbol, []).append(str(chat_id))

    try:
        with open(SENT_SIGNALS_CSV, "a", encoding="utf-8") as f:
            if header_needed:
                f.write("timestamp,symbol,predicted,confidence,regime,chat_ids,evaluate_after\n")

            for symbol, pred in predictions.items():
                delivered_chat_ids = sorted(sent_map.get(symbol, []))
                if not delivered_chat_ids:
                    continue

                signal_id = f"{now.isoformat()}::{symbol}"
                signal_ids[symbol] = signal_id
                pred["_signal_id"] = signal_id
                meta_row = _last_predictions.get(symbol, {}).get("meta_row")
                pending_records.append({
                    "signal_id": signal_id,
                    "symbol": symbol,
                    "pred": pred,
                    "meta_row": meta_row,
                    "sent_to": delivered_chat_ids,
                    "sent_at": now.isoformat(),
                    "evaluate_after": evaluate_after.isoformat(),
                })
                f.write(
                    f"{now.isoformat()},"
                    f"{symbol},"
                    f"{pred.get('suggested_direction', '?')},"
                    f"{pred.get('final_confidence_percent', 0):.2f},"
                    f"{pred.get('market_regime', 'Unknown')},"
                    f"{'|'.join(delivered_chat_ids)},"
                    f"{evaluate_after.isoformat()}\n"
                )
    except Exception as e:
        log.error("Sent signals store failed: %s", e)

    _save_pending_signals(pending_records)
    return signal_ids


def _resolve_signal_outcome(symbol: str, pred: dict, meta_row=None) -> dict | None:
    try:
        data = fetch_multi_timeframe(symbol, 10)
        df = data.get("M5")
        if df is None or len(df) < 2:
            return None

        last_candle = df.iloc[-2]
        actual_green = last_candle["close"] > last_candle["open"]
        predicted_up = pred["suggested_direction"] == "UP"
        correct = (predicted_up == actual_green)

        base_ml_up = pred.get("primary_direction") == "GREEN"
        base_ml_correct = (base_ml_up == actual_green)
        if meta_row is not None:
            update_online_learner(meta_row, base_ml_correct)

        _adaptive_thresh.record_outcome(correct)
        _stacker.update(pred, correct)

        global _cb_just_reset
        was_halted = _cb.is_halted
        cb_halted, cb_reason = _cb.check(was_win=correct)
        resumed = was_halted and not _cb.is_halted
        triggered = cb_halted and not was_halted
        if resumed:
            _cb_just_reset = True

        conf = pred["final_confidence_percent"]
        outcome_time = datetime.now(timezone.utc).isoformat()
        _outcome_results.append({
            "symbol": symbol,
            "predicted": pred["suggested_direction"],
            "actual": "UP" if actual_green else "DOWN",
            "correct": correct,
            "confidence": conf,
            "time": outcome_time,
        })

        update_prediction_history(
            pred["green_probability_percent"] / 100,
            correct,
            confidence=conf,
        )
        record_prediction(correct, conf, pred.get("latency_ms", 0))

        if correct:
            _consecutive_losses[symbol] = 0
            _consecutive_wins[symbol] = _consecutive_wins.get(symbol, 0) + 1
        else:
            _consecutive_losses[symbol] = _consecutive_losses.get(symbol, 0) + 1
            _consecutive_wins[symbol] = 0
            if _consecutive_losses[symbol] >= MAX_CONSECUTIVE_LOSSES:
                cooldown_end = datetime.now(timezone.utc) + timedelta(minutes=5 * COOLDOWN_CANDLES)
                _cooldown_until[symbol] = cooldown_end
                log.warning("COOLDOWN: %s has %d consecutive losses, pausing until %s",
                            symbol, _consecutive_losses[symbol], cooldown_end.strftime("%H:%M"))

        _log_outcome_csv(symbol, pred, actual_green, correct)

        recent = _outcome_results[-50:]
        wins = sum(1 for r in recent if r["correct"])
        win_rate = wins / len(recent) * 100 if recent else 0.0

        return {
            "symbol": symbol,
            "correct": correct,
            "triggered_cb": triggered,
            "cb_reason": cb_reason,
            "line": (
                f"{'✅' if correct else '❌'} *{symbol}*: Predicted {pred['suggested_direction']} "
                f"→ Actual {'UP' if actual_green else 'DOWN'} "
                f"({conf:.0f}% conf) | Win rate: {win_rate:.0f}%"
            ),
        }
    except Exception as e:
        log.error("Outcome check failed for %s: %s", symbol, e)
        return None


async def _reconcile_pending_signals(bot: Bot | None = None):
    pending_records = _load_pending_signals()
    if not pending_records:
        return

    now = datetime.now(timezone.utc)
    remaining = []
    results_by_chat = {}

    for record in pending_records:
        try:
            due_time = datetime.fromisoformat(record["evaluate_after"])
        except Exception:
            due_time = now

        if due_time > now:
            remaining.append(record)
            continue

        outcome = _resolve_signal_outcome(
            record.get("symbol", "?"),
            record.get("pred", {}),
            meta_row=record.get("meta_row"),
        )
        if not outcome:
            remaining.append(record)
            continue

        for chat_id in record.get("sent_to", []):
            results_by_chat.setdefault(str(chat_id), []).append(outcome["line"])

    if len(remaining) != len(pending_records):
        _save_pending_signals(remaining)

    if bot is not None:
        for chat_id, lines in results_by_chat.items():
            msg = "📋 *Signal Results*\n━━━━━━━━━━━━━━━━━━━━━━━\n\n" + "\n".join(lines)
            try:
                await _safe_send_message(bot, int(chat_id), msg, parse_mode="Markdown")
            except Exception as e:
                log.error("Failed to send reconciled result to %s: %s", chat_id, e)


async def _pending_outcome_job(app: Application):
    """Reconcile pending signals every minute so accuracy survives restarts."""
    while True:
        try:
            await _reconcile_pending_signals(app.bot)
        except Exception as e:
            log.warning("Pending outcome reconciliation failed: %s", e)
        await asyncio.sleep(60)


# =============================================================================
#  MT5 Helpers
# =============================================================================

def _connect_mt5():
    kwargs = {}
    if MT5_PATH: kwargs["path"] = MT5_PATH
    if MT5_LOGIN: kwargs["login"] = MT5_LOGIN
    if MT5_PASSWORD: kwargs["password"] = MT5_PASSWORD
    if MT5_SERVER: kwargs["server"] = MT5_SERVER
    if not mt5.initialize(**kwargs):
        log.error("MT5 init failed: %s", mt5.last_error())
        return False
    log.info("MT5 connected.")
    return True


def _get_server_time():
    """Get current time — always use system UTC clock for reliability."""
    return datetime.now(timezone.utc)


def _ensure_mt5_connection() -> bool:
    """Ensure MT5 IPC is available for live fetches; reconnect if needed."""
    try:
        info = mt5.terminal_info()
        tick = mt5.symbol_info_tick(DEFAULT_SYMBOL)
        if info is not None and tick is not None:
            return True
    except Exception:
        pass

    log.warning("MT5 session unavailable; attempting reconnect.")
    try:
        mt5.shutdown()
    except Exception:
        pass
    return _connect_mt5()


def _next_m5_close_time(server_time: datetime) -> datetime:
    minute = server_time.minute
    next_bar_minute = (math.ceil((minute + 1) / 5)) * 5
    if next_bar_minute >= 60:
        close_time = server_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        close_time = server_time.replace(minute=next_bar_minute, second=0, microsecond=0)
    return close_time


def _validate_symbol(symbol: str) -> bool:
    info = mt5.symbol_info(symbol)
    return info is not None


def _get_spread(symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    return float(info.spread) if info else 0.0


def _is_market_open() -> bool:
    """Check if forex market is open (not weekend, MT5 has fresh data)."""
    now = datetime.now(timezone.utc)
    # Forex closes Friday ~22:00 UTC, opens Sunday ~22:00 UTC
    if now.weekday() == 5:  # Saturday — always closed
        return False
    if now.weekday() == 6 and now.hour < 22:  # Sunday before 22:00
        return False
    if now.weekday() == 4 and now.hour >= 22:  # Friday after 22:00
        return False
    # Check if MT5 has recent tick
    tick = mt5.symbol_info_tick(DEFAULT_SYMBOL)
    if tick is None:
        return False
    tick_age = now.timestamp() - tick.time
    return tick_age < 600  # stale if >10 min


async def _safe_send_message(bot: Bot, chat_id: int, text: str,
                             parse_mode: str = "Markdown", **kwargs):
    """Send Telegram messages with a plain-text fallback if markup parsing fails."""
    try:
        return await bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=parse_mode,
            **kwargs,
        )
    except Exception as exc:
        if parse_mode and "Can't parse entities" in str(exc):
            log.warning("Telegram markup rejected for %s; retrying without parse mode.", chat_id)
            return await bot.send_message(
                chat_id=chat_id,
                text=text,
                **kwargs,
            )
        raise


def _market_closed_msg() -> str:
    """Friendly message when market is closed."""
    now = datetime.now(timezone.utc)
    ist = now + timedelta(hours=5, minutes=30)
    # Next Monday 00:05 UTC
    days_to_mon = (7 - now.weekday()) % 7
    if days_to_mon == 0:
        days_to_mon = 7
    next_open = (now + timedelta(days=days_to_mon)).replace(
        hour=0, minute=5, second=0, microsecond=0
    )
    hours_left = (next_open - now).total_seconds() / 3600
    return (
        "🔒 *Market Closed*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Forex market is closed for the weekend.\n\n"
        f"📅 Opens: *Monday ~00:05 UTC*\n"
        f"🇮🇳 IST: *~05:35 Monday*\n"
        f"⏳ In: *~{hours_left:.0f} hours*\n\n"
        "💡 _Auto-signals will resume automatically._"
    )


# =============================================================================
#  Prediction Pipeline
# =============================================================================

def _run_prediction(symbol: str) -> dict:
    if not _ensure_mt5_connection():
        raise RuntimeError("MT5 unavailable")

    data = fetch_multi_timeframe(symbol, min(CANDLES_TO_FETCH, LIVE_CANDLES_TO_FETCH))
    df = data.get("M5")
    if df is None or len(df) == 0:
        log.warning("Initial fetch failed for %s; reconnecting MT5 and retrying once.", symbol)
        if not _ensure_mt5_connection():
            raise RuntimeError(f"No M5 data for {symbol}")
        data = fetch_multi_timeframe(symbol, min(CANDLES_TO_FETCH, LIVE_CANDLES_TO_FETCH))
        df = data.get("M5")

    if df is None or len(df) == 0:
        raise RuntimeError(f"No M5 data for {symbol}")
    m15, h1 = data.get("M15"), data.get("H1")
    m1 = data.get("M1")

    df = compute_features(df, m15_df=m15, h1_df=h1, m1_df=m1)
    if df.empty:
        raise RuntimeError("Not enough data for features.")

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

    return {
        "symbol": symbol,
        "regime": regime,
        "risk_warnings": risk_warnings,
        **result,
    }


# =============================================================================
#  Keyboard Builders
# =============================================================================

def _main_menu_keyboard(chat_id=None):
    is_subscribed = str(chat_id) in _subscribers and _subscribers[str(chat_id)].get("active", False)
    sub_btn = ("🔴 Stop Signals", "unsub") if is_subscribed else ("🟢 Start Auto-Signal", "sub")

    return InlineKeyboardMarkup([
        [InlineKeyboardButton(sub_btn[0], callback_data=sub_btn[1])],
        [InlineKeyboardButton("🔮 Predict Now", callback_data="predict_now"),
         InlineKeyboardButton("⏳ Next Candle", callback_data="predict_next")],
        [InlineKeyboardButton("📊 Status", callback_data="status"),
         InlineKeyboardButton("🌊 Regime", callback_data="regime")],
        [InlineKeyboardButton("📈 Multi-Symbol", callback_data="multi_symbol"),
         InlineKeyboardButton("ℹ️ Model Info", callback_data="model_info")],
        [InlineKeyboardButton("💰 Position Guide", callback_data="position_guide"),
         InlineKeyboardButton("📜 Today's Signals", callback_data="today_signals")],
        [InlineKeyboardButton("🎯 Results", callback_data="today_results"),
         InlineKeyboardButton("🏭 Prod State", callback_data="prod_state")],
    ])


def _symbol_keyboard(action_prefix="predict"):
    buttons = []
    row = []
    for sym in SYMBOLS:
        row.append(InlineKeyboardButton(sym, callback_data=f"{action_prefix}_{sym}"))
        if len(row) == 3:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append([InlineKeyboardButton("🔙 Back to Menu", callback_data="menu")])
    return InlineKeyboardMarkup(buttons)


def _sub_symbol_keyboard():
    """Symbol selection for auto-signal subscription."""
    buttons = []
    row = []
    for sym in SYMBOLS:
        row.append(InlineKeyboardButton(f"✅ {sym}", callback_data=f"sub_{sym}"))
        if len(row) == 3:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append([InlineKeyboardButton("📊 All Symbols", callback_data="sub_ALL")])
    buttons.append([InlineKeyboardButton("🔙 Back", callback_data="menu")])
    return InlineKeyboardMarkup(buttons)


def _alert_mode_keyboard():
    """Choose signal filter mode."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📡 All Signals", callback_data="mode_all")],
        [InlineKeyboardButton("🎯 High-Conf Only (≥70%)", callback_data="mode_high")],
        [InlineKeyboardButton("🔙 Back", callback_data="menu")],
    ])


def _back_keyboard(chat_id=None):
    is_sub = str(chat_id) in _subscribers and _subscribers[str(chat_id)].get("active", False)
    row = [InlineKeyboardButton("🔙 Menu", callback_data="menu")]
    if not is_sub:
        row.append(InlineKeyboardButton("🟢 Start Signals", callback_data="sub"))
    row.append(InlineKeyboardButton("🔮 Predict", callback_data="predict_now"))
    return InlineKeyboardMarkup([row])


# =============================================================================
#  Message Formatters
# =============================================================================

def _format_prediction(pred: dict, is_auto=False) -> str:
    """Format a single prediction — V3 industry-grade format."""
    def clean_text(value) -> str:
        return str(value).replace("_", " ").replace("*", "").replace("`", "'")

    if pred["suggested_direction"] == "UP":
        direction = "🟢 UP"
        arrow = "⬆"
    elif pred["suggested_direction"] == "DOWN":
        direction = "🔴 DOWN"
        arrow = "⬇"
    else:
        direction = "⚪ HOLD"
        arrow = "•"
    conf = pred["final_confidence_percent"]
    green = pred["green_probability_percent"]
    kelly = pred.get("kelly_fraction_percent", 0.0)
    regime_raw = clean_text(pred.get("market_regime", "Unknown"))
    regime_clean = regime_raw.replace("_", " ")
    trade = clean_text(pred["suggested_trade"])
    latency = pred.get("latency_ms", -1)
    uncertainty = pred.get("uncertainty_percent", 0)
    meta_rel = pred.get("meta_reliability_percent", 0)
    session = clean_text(pred.get("session", "Off"))
    stability = pred.get("regime_stability", 0)
    adaptive_conf = pred.get("adaptive_confidence_percent", conf)
    variance = pred.get("ensemble_variance", 0)

    regime_emoji = {"Trending": "📈", "Ranging": "↔️",
                   "High_Volatility": "🔥", "Low_Volatility": "❄️"}.get(regime_raw, "❓")
    session_emoji = {"London": "🇬🇧", "Overlap": "🌐", "Asian": "🌏", "Off": "💤"}.get(session, "")

    # Confidence bar
    filled = int(conf / 10)
    bar = "█" * filled + "░" * (10 - filled)

    # Position size
    if kelly > 3: size = "🟢 STRONG"
    elif kelly > 2: size = "🟡 MEDIUM"
    elif kelly > 1: size = "🟠 LIGHT"
    else: size = "⚪ SKIP"

    now_utc = datetime.now(timezone.utc)
    ist = now_utc + timedelta(hours=5, minutes=30)
    time_str = f"{now_utc.strftime('%H:%M')} UTC │ {ist.strftime('%H:%M')} IST"

    header = "🤖 AUTO-SIGNAL" if is_auto else "🔮 PREDICTION"

    return (
        f"*{header}*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{direction} *{clean_text(pred['symbol'])}* M5 {arrow}\n\n"
        f"📊 Probability: *{green:.1f}%*\n"
        f"🎯 Confidence:  `[{bar}]` *{conf:.0f}%*\n"
        f"🧠 Meta Trust:  *{meta_rel:.0f}%*\n"
        f"📉 Uncertainty: *{uncertainty:.1f}%*\n\n"
        f"⏳ Expiry: *5 MINUTES (1 CANDLE)*\n"
        f"{regime_emoji} Regime: *{regime_clean}* (stab: {stability:.0%})\n"
        f"{session_emoji} Session: *{session}*\n"
        f"💰 Kelly: *{kelly:.1f}%* │ Size: {size}\n"
        f"💡 Action: *{trade}*\n\n"
        f"⏱ {time_str} │ {latency:.0f}ms"
    )


def _format_combined_signal(predictions: dict, filtered: dict) -> str:
    """V3 industry-grade combined signal message with all metrics."""
    def clean_text(value) -> str:
        return str(value).replace("_", " ").replace("*", "").replace("`", "'")

    now_utc = datetime.now(timezone.utc)
    ist = now_utc + timedelta(hours=5, minutes=30)
    time_str = f"{now_utc.strftime('%H:%M')} UTC │ {ist.strftime('%H:%M')} IST"

    # Win rate from stored outcomes (more accurate)
    import csv as _csv
    stored_total, stored_wins = 0, 0
    if os.path.exists(SIGNAL_CSV):
        try:
            with open(SIGNAL_CSV, "r", encoding="utf-8") as _f:
                for _row in _csv.DictReader(_f):
                    stored_total += 1
                    stored_wins += _row.get("correct") == "True"
        except Exception:
            pass
    recent = _outcome_results[-20:] if _outcome_results else []
    total_r = len(recent)
    wins_r = sum(1 for r in recent if r["correct"])

    # Accuracy badge
    if stored_total >= 20:
        stored_wr = stored_wins / stored_total * 100
        if stored_wr >= 60: acc_badge = "🏆 ELITE"
        elif stored_wr >= 55: acc_badge = "🥇 HIGH"
        elif stored_wr >= 50: acc_badge = "🥈 MODERATE"
        else: acc_badge = "🥉 BUILDING"
    else:
        stored_wr = 0
        acc_badge = "📊 CALIBRATING"

    # Confidence grade for this batch
    confs = [p["final_confidence_percent"] for p in predictions.values()]
    avg_conf = sum(confs) / len(confs) if confs else 0
    if avg_conf >= 80: conf_grade = "S"
    elif avg_conf >= 70: conf_grade = "A"
    elif avg_conf >= 65: conf_grade = "B"
    else: conf_grade = "C"

    # Active filters count
    n_filtered = len(filtered)
    n_passed = len(predictions)
    filter_ratio = f"{n_passed}/{n_passed + n_filtered}"

    lines = [
        f"🤖 *QUOTEX LORD V6*",
        f"⏱ {time_str}",
        f"🛡 *{n_passed + n_filtered}* analyzed │ *{n_passed}* passed │ Grade *{conf_grade}*",
        f"━━━━━━━━━━━━━━━━━━━━━━━",
    ]

    for sym, pred in sorted(predictions.items()):
        if pred["suggested_direction"] == "UP":
            d = "🟢"
            arrow = "⬆"
            direction_text = "CALL (UP)"
        elif pred["suggested_direction"] == "DOWN":
            d = "🔴"
            arrow = "⬇"
            direction_text = "PUT (DOWN)"
        else:
            d = "⚪"
            arrow = "•"
            direction_text = "HOLD"
        conf = pred["final_confidence_percent"]
        green = pred["green_probability_percent"]
        kelly = pred.get("kelly_fraction_percent", 0.0)
        trade = clean_text(pred["suggested_trade"])
        regime_raw = clean_text(pred.get("market_regime", ""))
        regime_clean = regime_raw.replace("_", " ")
        meta_rel = pred.get("meta_reliability_percent", 0)
        uncertainty = pred.get("uncertainty_percent", 0)
        session = clean_text(pred.get("session", ""))
        stability = pred.get("regime_stability", 0)
        r_e = {"Trending": "📈", "Ranging": "↔️",
               "High_Volatility": "🔥", "Low_Volatility": "❄️"}.get(regime_raw, "")
        s_e = {"London": "🇬🇧", "Overlap": "🌐", "Asian": "🌏", "Off": "💤"}.get(session, "")

        # Position size from Kelly
        if kelly > 5: size = "🟢 AGGRESSIVE"
        elif kelly > 3: size = "🟢 STRONG"
        elif kelly > 2: size = "🟡 MEDIUM"
        elif kelly > 1: size = "🟠 LIGHT"
        else: size = "⚪ MIN"

        # Confidence bar
        filled = int(conf / 10)
        bar = "█" * filled + "░" * (10 - filled)

        # Signal strength indicator
        if conf >= 80: strength = "🔥 PREMIUM"
        elif conf >= 70: strength = "💎 STRONG"
        elif conf >= 65: strength = "✅ SOLID"
        else: strength = "📊 STANDARD"

        # Dynamic expiry
        expiry_info = pred.get("_expiry_info", {})
        expiry_label = expiry_info.get("expiry_label", "5 Min")

        # Gemini verification badge
        gemini_badge = "🤖✓" if pred.get("gemini_approved") else ""

        # Confluence badge
        confl_score = pred.get("_confluence_score", 0)
        confl_badge = f"TF:{confl_score}/3" if confl_score > 0 else ""

        # Breakout badge
        bb_info = pred.get("_bb_breakout", {})
        breakout_badge = "🚀" if pred.get("_breakout_aligned") else ""

        lines.append(f"\n{d} {arrow} *{clean_text(sym)}* │ {direction_text} {gemini_badge} {breakout_badge}")
        lines.append(f"   {strength}")
        lines.append(f"   ⏳ Expiry: *{expiry_label}* │ {confl_badge}")
        lines.append(f"   📊 `[{bar}]` *{conf:.0f}%*")
        lines.append(f"   🎲 Win Prob: *{green:.1f}%* │ Meta: *{meta_rel:.0f}%*")
        lines.append(f"   {r_e} {regime_clean} │ {s_e} {session}")
        lines.append(f"   💰 Kelly: *{kelly:.1f}%* │ {size}")

        # S/R context
        sr_info = pred.get("_sr_info", {})
        if sr_info.get("support") and sr_info.get("resistance"):
            lines.append(f"   🔰 S: {sr_info['support']:.5f} │ R: {sr_info['resistance']:.5f}")

        # Round number alert
        if pred.get("_near_round"):
            lines.append(f"   ⚠️ {pred['_near_round']}")

        # Breakout detail
        if bb_info.get("is_breakout"):
            lines.append(f"   🚀 {bb_info['detail']}")

        lines.append(f"   💡 *{trade}*")


    # Footer
    lines.append(f"\n━━━━━━━━━━━━━━━━━━━━━━━")
    if stored_total >= 10:
        lines.append(f"{acc_badge} Stored: *{stored_wins}/{stored_total}* ({stored_wr:.1f}%)")
    if total_r > 0:
        wr_pct = wins_r / total_r * 100
        lines.append(f"🎯 Session: *{wins_r}/{total_r}* ({wr_pct:.0f}%)")
    lines.append(f"🛡 Filters: *{filter_ratio}* passed │ 20+ Layers")
    lines.append(f"⚙️ V6 │ Confluence │ BB Breakout │ Cross-Filtered")

    if filtered:
        reasons = [f"{clean_text(s)}({clean_text(r).replace('_', '-')})" for s, r in list(filtered.items())[:4]]
        lines.append(f"⏭ _{', '.join(reasons)}_")

    return "\n".join(lines)


# =============================================================================
#  Auto-Signal Background Job
# =============================================================================

async def _auto_signal_job(app: Application):
    """
    Background loop: sends predictions to all subscribers
    5 seconds before every M5 candle close.
    """
    bot: Bot = app.bot
    log.info("Auto-signal job started.")

    while True:
        try:
            server_time = _get_server_time()
            close_time = _next_m5_close_time(server_time)
            send_time = close_time - timedelta(seconds=TELEGRAM_SEND_BEFORE_CLOSE_SEC)
            wait_seconds = (send_time - server_time).total_seconds()

            if wait_seconds < 0:
                # Already past send time, wait for next candle
                wait_seconds += 300  # 5 minutes

            if wait_seconds > 0:
                log.info("Auto-signal: next send in %.0fs at %s UTC",
                         wait_seconds, send_time.strftime("%H:%M:%S"))
                await asyncio.sleep(wait_seconds)

            # Collect active subscribers
            active_subs = {
                cid: info for cid, info in _subscribers.items()
                if info.get("active", False)
            }

            if not active_subs:
                await asyncio.sleep(10)
                continue

            log.info("Auto-signal: sending to %d subscribers", len(active_subs))

            # Run predictions for all needed symbols
            needed_symbols = set()
            for info in active_subs.values():
                needed_symbols.update(info.get("symbols", [DEFAULT_SYMBOL]))

            # -----------------------------------------------------------------
            # NEW COMPONENT: CIRCUIT BREAKER (GAP 2)
            # -----------------------------------------------------------------
            global _cb_just_reset
            if not _cb.can_trade():
                status = _cb.get_status()
                halt_msg = (
                    f'🛑 *CIRCUIT BREAKER ACTIVE*\n'
                    f'Reason: _{status["reason"]}_\n'
                    f'Consecutive losses: *{status["consecutive_losses"]}*\n'
                    f'Cooldown: 60 min. Signals resume automatically.'
                )
                await _broadcast_to_subscribers(bot, halt_msg)
                await asyncio.sleep(300)
                continue

            # -----------------------------------------------------------------
            # NEW COMPONENT: DRIFT DETECTOR / AUTO-RETRAIN (GAP 4)
            # -----------------------------------------------------------------
            global _candle_count
            _candle_count += 1
            if _candle_count % DRIFT_CHECK_INTERVAL == 0:
                tradeable_symbols = list(active_subs.values())[0].get("symbols", [DEFAULT_SYMBOL])
                asyncio.create_task(_run_drift_check(bot, list(tradeable_symbols)))

            # -----------------------------------------------------------------
            # NEW COMPONENT: DYNAMIC PAIR SELECTOR (GAP 5)
            # -----------------------------------------------------------------
            needed_symbols = set()
            for info in active_subs.values():
                needed_symbols.update(info.get("symbols", [DEFAULT_SYMBOL]))

            # Weekend check (Sat/Sun) — do this BEFORE pair scoring to avoid wasting API calls
            now_utc = datetime.now(timezone.utc)
            if now_utc.weekday() >= 5:  # Saturday=5, Sunday=6
                log.info("Auto-signal: market closed (weekend). Sleeping 30 min.")
                await asyncio.sleep(1800)
                continue

            if now_utc.hour in BLOCKED_HOURS_UTC or now_utc.hour not in PROFITABLE_HOURS_UTC:
                log.info("Auto-signal: blocked hour %02d UTC for accuracy-first mode.", now_utc.hour)
                await asyncio.sleep(300)
                continue

            global _pair_score_counter, _pair_scores
            _pair_score_counter += 1
            if _pair_score_counter >= PAIR_SCORE_REFRESH or not _pair_scores:
                _pair_score_counter = 0
                for sym in list(needed_symbols):
                    try:
                        data = fetch_multi_timeframe(sym, 500)
                        df   = compute_features(data.get('M5'), m15_df=data.get('M15'), h1_df=data.get('H1'), m1_df=data.get('M1'))
                        res  = score_pair(df, symbol=sym)
                        _pair_scores[sym] = res['total']
                        log.info('Pair score %s: %d/30', sym, res['total'])
                    except Exception:
                        _pair_scores[sym] = 25  # assume tradeable if scoring fails

            # Filter: only allow pairs that meet the selector's recommended score.
            # Preferred symbols get a bonus so top-performers aren't blocked by momentary dips
            accuracy_snapshot_early = _get_symbol_accuracy_snapshot()
            preferred_early = _get_preferred_symbols(needed_symbols, accuracy_snapshot_early)
            tradeable = set()
            for s in needed_symbols:
                raw_score = _pair_scores.get(s, 25)
                bonus = PREFERRED_SYMBOL_PAIR_SCORE_BONUS if s in preferred_early else 0
                effective_score = raw_score + bonus
                if effective_score >= PAIR_SELECTOR_MIN_SCORE:
                    tradeable.add(s)
                elif bonus > 0:
                    log.info('Pair selector: preferred %s score %d+%d=%d still below %d — blocked',
                             s, raw_score, bonus, effective_score, PAIR_SELECTOR_MIN_SCORE)
            skipped   = needed_symbols - tradeable
            if skipped:
                log.info('Pair selector skipped %s (score < %d)', skipped, PAIR_SELECTOR_MIN_SCORE)
            needed_symbols = tradeable

            blocked_symbols = needed_symbols & BLOCKED_SYMBOLS
            if blocked_symbols:
                log.info("Accuracy filter blocked symbols: %s", blocked_symbols)
                needed_symbols -= BLOCKED_SYMBOLS

            accuracy_snapshot = _get_symbol_accuracy_snapshot()
            for sym in list(needed_symbols):
                allowed, reason = _is_symbol_accuracy_allowed(sym, accuracy_snapshot)
                if not allowed:
                    log.info("Stored-accuracy filter blocked %s: %s", sym, reason)
                    needed_symbols.remove(sym)

            preferred_symbols = _get_preferred_symbols(needed_symbols, accuracy_snapshot)
            if preferred_symbols:
                skipped_symbols = needed_symbols - preferred_symbols
                if skipped_symbols:
                    log.info("Preferred-symbol filter blocked %s; strongest set is %s", skipped_symbols, preferred_symbols)
                needed_symbols = preferred_symbols

            predictions = {}
            filtered_out = {}
            # Pre-compute cross-filter stats once per cycle
            _sd_stats = _get_symbol_direction_stats()
            _sr_stats = _get_symbol_regime_stats()
            for sym in needed_symbols:
                try:
                    # 1. Cooldown check
                    if sym in _cooldown_until:
                        if datetime.now(timezone.utc) < _cooldown_until[sym]:
                            filtered_out[sym] = "cooldown"
                            continue
                        else:
                            del _cooldown_until[sym]

                    # 2. Regime transition skip
                    if _regime_transition_skip.get(sym, 0) > 0:
                        _regime_transition_skip[sym] -= 1
                        filtered_out[sym] = f"regime shift ({_regime_transition_skip[sym]+1} left)"
                        continue

                    pred = _run_prediction(sym)
                    conf = pred["final_confidence_percent"]
                    trade = pred["suggested_trade"]
                    regime = pred.get("market_regime", "Unknown")
                    risk_warnings = pred.get("risk_warnings", [])

                    # Compute multi-TF confluence score for this signal
                    try:
                        _cf_data = fetch_multi_timeframe(sym, 100)
                        _cf_result = check_confluence(
                            _cf_data.get('M5'), _cf_data.get('M15'), _cf_data.get('H1'),
                            predicted_direction=pred.get("suggested_direction", "UP"),
                            min_score=2,
                        )
                        pred["_confluence_score"] = _cf_result.get("confluence_score", 0)
                        pred["_confluence_pass"] = _cf_result.get("confluence_pass", True)
                        pred["_tf_directions"] = _cf_result.get("tf_directions", {})
                        if not _cf_result.get("confluence_pass", True):
                            filtered_out[sym] = f"confluence_fail:{_cf_result.get('reason', 'TF conflict')}"
                            log.info("Confluence filter blocked %s: %s", sym, _cf_result.get("reason", ""))
                            continue
                    except Exception as e:
                        log.debug("Confluence check failed for %s: %s", sym, e)
                        pred["_confluence_score"] = 3  # benefit of doubt on error

                    # 3. Regime transition detection
                    prev_regime = _last_regime.get(sym)
                    if prev_regime and regime != prev_regime:
                        _regime_transition_skip[sym] = REGIME_SKIP_CANDLES
                        filtered_out[sym] = f"regime changed {prev_regime}→{regime}"
                        log.info("REGIME SHIFT %s: %s -> %s, pausing %d candles",
                                 sym, prev_regime, regime, REGIME_SKIP_CANDLES)
                    _last_regime[sym] = regime

                    # Log EVERY prediction to CSV (before filtering)
                    _log_prediction_csv(sym, pred)

                    # =========================================================
                    # Phase 6 (Always-On High Frequency Override):
                    # We bypass ALL legacy ML confidence thresholds, regime blockers, 
                    # and session filters. Every single 5-minute candle passes through 
                    # to the Gemini Oracle for a forced directional verdict.
                    # =========================================================
                    if pred.get("error"):
                        filtered_out[sym] = pred["error"]
                        continue

                    hard_risk_blocked, hard_risk_reason = _has_hard_risk_warning(risk_warnings)
                    if hard_risk_blocked:
                        filtered_out[sym] = f"risk_veto:{hard_risk_reason}"
                        log.info("Risk veto blocked %s: %s", sym, hard_risk_reason)
                        continue

                    if conf < MIN_CONFIDENCE_FILTER:
                        filtered_out[sym] = f"confidence<{MIN_CONFIDENCE_FILTER:.0f}"
                        continue

                    symbol_min_conf = _get_symbol_min_confidence(sym, accuracy_snapshot)
                    if conf < symbol_min_conf:
                        filtered_out[sym] = f"Low confidence ({conf:.0f}% < {symbol_min_conf:.0f}%)"
                        continue

                    # Kelly minimum filter — kelly<1 has 48.8% WR (losing)
                    kelly_pct = pred.get("kelly_fraction_percent", 0.0)
                    if kelly_pct < MIN_KELLY_THRESHOLD:
                        filtered_out[sym] = f"kelly_too_low:{kelly_pct:.1f}%<{MIN_KELLY_THRESHOLD}%"
                        continue

                    direction = pred.get("suggested_direction", "")

                    # DOWN direction confidence premium
                    if direction == "DOWN":
                        down_min = symbol_min_conf + DOWN_DIRECTION_CONFIDENCE_PREMIUM
                        if conf < down_min:
                            filtered_out[sym] = f"DOWN_premium({conf:.0f}%<{down_min:.0f}%)"
                            continue

                    # Symbol × Direction cross-filter
                    sd_ok, sd_reason = _is_symbol_direction_allowed(sym, direction, _sd_stats)
                    if not sd_ok:
                        filtered_out[sym] = f"sym_dir_block:{sd_reason}"
                        log.info("Symbol×Direction blocked %s %s: %s", sym, direction, sd_reason)
                        continue

                    # Symbol × Regime cross-filter
                    sr_ok, sr_reason = _is_symbol_regime_allowed(sym, regime, _sr_stats)
                    if not sr_ok:
                        filtered_out[sym] = f"sym_regime_block:{sr_reason}"
                        log.info("Symbol×Regime blocked %s/%s: %s", sym, regime, sr_reason)
                        continue

                    # Streak-based adaptive confidence adjustment
                    streak_adj = _get_streak_confidence_adjustment(sym)
                    if streak_adj > 0:
                        adjusted_min = symbol_min_conf + streak_adj
                        if conf < adjusted_min:
                            filtered_out[sym] = f"streak_loss_bump({conf:.0f}%<{adjusted_min:.0f}%)"
                            log.info("Streak filter raised threshold for %s: +%.0f%% (losses=%d)",
                                     sym, streak_adj, _consecutive_losses.get(sym, 0))
                            continue

                    # Support/Resistance proximity filter — block signals in reversal zones
                    try:
                        data_for_sr = fetch_multi_timeframe(sym, 100)
                        sr_df = data_for_sr.get('M5')
                        if sr_df is not None and len(sr_df) > 0:
                            sr_df = compute_features(sr_df, m15_df=data_for_sr.get('M15'),
                                                     h1_df=data_for_sr.get('H1'), m1_df=data_for_sr.get('M1'))
                            sr_info = _detect_support_resistance(sr_df)
                            if sr_info["near_sr"]:
                                # Near S/R: block if signal direction conflicts with level
                                if direction == "UP" and sr_info.get("resistance_dist_atr", 99) < SR_PROXIMITY_ATR_THRESHOLD:
                                    filtered_out[sym] = f"near_resistance:{sr_info['sr_detail']}"
                                    log.info("S/R filter blocked %s UP near resistance", sym)
                                    continue
                                if direction == "DOWN" and sr_info.get("support_dist_atr", 99) < SR_PROXIMITY_ATR_THRESHOLD:
                                    filtered_out[sym] = f"near_support:{sr_info['sr_detail']}"
                                    log.info("S/R filter blocked %s DOWN near support", sym)
                                    continue
                            # Store S/R info for signal message
                            pred["_sr_info"] = sr_info
                    except Exception as e:
                        log.debug("S/R detection failed for %s: %s", sym, e)

                    # Round-number psychological level filter
                    try:
                        current_price = pred.get("_last_close", 0)
                        if current_price <= 0:
                            # Get current price from last row
                            tick = mt5.symbol_info_tick(sym)
                            current_price = tick.bid if tick else 0
                        if current_price > 0:
                            near_round, round_detail = _is_near_round_number(sym, current_price)
                            if near_round:
                                pred["_near_round"] = round_detail
                                log.info("Round number proximity %s: %s (price=%.5f)", sym, round_detail, current_price)
                    except Exception as e:
                        log.debug("Round number check failed for %s: %s", sym, e)

                    # Compute dynamic expiry recommendation
                    try:
                        atr_val = pred.get("_atr", 0)
                        if atr_val <= 0 and sr_df is not None and "atr_14" in sr_df.columns:
                            atr_val = sr_df["atr_14"].values[-1]
                        atr_median = sr_df["atr_14"].median() if (sr_df is not None and "atr_14" in sr_df.columns) else atr_val
                        expiry_info = _compute_optimal_expiry(regime, atr_val, atr_median)
                        pred["_expiry_info"] = expiry_info
                    except Exception as e:
                        pred["_expiry_info"] = {"expiry_minutes": 5, "expiry_label": "5 Min", "reason": "error"}

                    # Bollinger Band breakout detection
                    try:
                        bb_info = _detect_bb_breakout(sr_df if 'sr_df' in dir() and sr_df is not None else None)
                        pred["_bb_breakout"] = bb_info
                        if bb_info["is_breakout"]:
                            if bb_info["breakout_direction"] == direction:
                                # Breakout aligns with signal — high-edge: give a confidence boost in display
                                pred["_breakout_aligned"] = True
                                log.info("BB breakout ALIGNED %s %s: %s", sym, direction, bb_info["detail"])
                            else:
                                # Breakout contradicts signal direction — block
                                filtered_out[sym] = f"bb_breakout_conflict:{bb_info['detail']}"
                                log.info("BB breakout CONFLICT %s: signal=%s but breakout=%s", sym, direction, bb_info["breakout_direction"])
                                continue
                    except Exception as e:
                        log.debug("BB breakout detection failed for %s: %s", sym, e)

                    if regime in BLOCKED_REGIMES:
                        filtered_out[sym] = f"blocked_regime:{regime}"
                        continue

                    unanimity = float(pred.get("ensemble_unanimity", 0.0))
                    variance = float(pred.get("ensemble_variance", 1.0))
                    if unanimity < MIN_UNANIMITY:
                        filtered_out[sym] = f"low_unanimity:{unanimity:.2f}"
                        continue
                    if variance > MAX_ENSEMBLE_VARIANCE:
                        filtered_out[sym] = f"high_variance:{variance:.3f}"
                        continue

                    if trade == "HOLD" or pred.get("suggested_direction") == "HOLD":
                        filtered_out[sym] = pred.get("skip_reason", "hold_gate")
                        continue
                        
                    predictions[sym] = pred
                    _last_predictions[sym] = {
                        "pred": pred,
                        "time": datetime.now(timezone.utc).isoformat(),
                        "meta_row": pred.get("_meta_row"),
                    }
                    _signal_history.append({
                        "symbol": sym,
                        "direction": pred["suggested_direction"],
                        "confidence": conf,
                        "kelly": pred.get("kelly_fraction_percent", 0),
                        "trade": trade,
                        "regime": regime,
                        "time": datetime.now(timezone.utc).isoformat(),
                    })
                except Exception as e:
                    log.error("Prediction failed %s: %s", sym, e)
                    record_error(f"{sym}: {e}")

            # =================================================================
            # Phase 10: Restore Correlation Filter
            # =================================================================
            try:
                for sym, pred in list(predictions.items()):
                    corr = check_correlation(sym, pred.get("suggested_direction", ""), predictions)
                    if not corr.get("corr_pass", True):
                        log.info("Correlation filter blocked %s: %s", sym, corr.get("reason", "conflict"))
                        filtered_out[sym] = corr.get("reason", "correlation_conflict")
                        del predictions[sym]
            except Exception as e:
                log.warning("Correlation EXCEPTION — signal BLOCKED: %s", e)
                predictions.clear()

            # =================================================================
            # Phase 10: Restore Direction Flip Filter (Strategy 5)
            # =================================================================
            for sym, pred in list(predictions.items()):
                direction = pred["suggested_direction"]
                regime = pred.get("market_regime", "Unknown")
                if sym in _prev_directions:
                    if _prev_directions[sym] != direction and regime != "Trending":
                        log.info("Blocked direction flip for %s in %s", sym, regime)
                        filtered_out[sym] = "direction_flip"
                        del predictions[sym]
                if sym in predictions:
                    _prev_directions[sym] = direction

            # =================================================================
            # Phase 10: Restore Stacking Gatekeeper & Rank/Select
            # =================================================================
            if predictions:
                for sym, pred in list(predictions.items()):
                    try:
                        stack_decision = _stacker.should_send(pred)
                        if not stack_decision.get("send", True):
                            stack_reason = stack_decision.get("reason", "stack_reject")
                            log.info("Stack blocker %s: %s", sym, stack_reason)
                            filtered_out[sym] = f"stack_reject: {stack_reason}"
                            del predictions[sym]
                    except Exception as e:
                        log.warning("Stacking EXCEPTION on %s — signal BLOCKED: %s", sym, e)
                        filtered_out[sym] = "stack_error"
                        if sym in predictions: del predictions[sym]
                
                if predictions:
                    ranked = rank_and_select(
                        predictions,
                        max_signals=MAX_SIGNALS_PER_CYCLE,
                        min_score=MIN_SIGNAL_SCORE,
                    )
                    predictions = ranked

            # =================================================================
            # LAYER 13: Gemini AI Master Trader Verification
            # =================================================================
            if predictions:
                gemini_filtered = {}
                for sym, pred in predictions.items():
                    try:
                        data_mtf = fetch_multi_timeframe(sym, 250)
                        m5_df = data_mtf.get('M5')
                        m1_df = data_mtf.get('M1')
                        if m5_df is not None and not m5_df.empty and len(m5_df) > 5:
                            m5_feats = compute_features(m5_df, m15_df=data_mtf.get('M15'), h1_df=data_mtf.get('H1'), m1_df=m1_df)
                            if m5_feats is None or m5_feats.empty:
                                raise ValueError("Computed features returned empty df")
                                
                            last_row = m5_feats.iloc[-1]
                            price = last_row.get("close", 0)
                            e20 = last_row.get("ema_20", 0)
                            e50 = last_row.get("ema_50", 0)
                            e100 = last_row.get("ema_100", 0)
                            e200 = last_row.get("ema_200", 0)
                            atr = last_row.get("atr_14", 0)
                            adx = last_row.get("adx", 0)
                            rsi = last_row.get("rsi_14", 50)
                            vol_imb = last_row.get("volume_imbalance", 0)
                            # Phase 7 Microstructure Extractions
                            dist_to_res = last_row.get("dist_to_resistance", 5.0)
                            dist_to_sup = last_row.get("dist_to_support", 5.0)
                            bull_div = last_row.get("bullish_divergence", 0.0)
                            bear_div = last_row.get("bearish_divergence", 0.0)
                            vol_accel = last_row.get("tick_vol_accel_1", 0.0)
                            
                            # Phase 8 Microstructure Extractions
                            cum_delta = last_row.get("cumulative_delta", 0.0)
                            m1_absorp = last_row.get("m1_absorption", 0.0)
                            trade_int = last_row.get("trade_intensity", 0.0)
                            
                            recent_candles_df = m5_df.tail(5)[['time', 'open', 'high', 'low', 'close', 'tick_volume']].copy()
                            recent_candles_str = recent_candles_df.to_string(index=False)
                            
                            # Phase 8: Live News & Macro Asset Scraping
                            live_news_text = fetch_live_news()
                            
                            def safe_tick_price(s):
                                try:
                                    t = mt5.symbol_info_tick(s)
                                    return t.bid if t else 0.0
                                except: return 0.0

                            dxy_proxy = safe_tick_price("USDX") or safe_tick_price("DXY")
                            gold_price = safe_tick_price("XAUUSD")
                            eurgbp_price = safe_tick_price("EURGBP")
                            
                            v = await verify_signal(
                                symbol=sym,
                                timestamp=datetime.now(timezone.utc).isoformat(),
                                direction=pred["suggested_direction"],
                                green_prob=pred.get("green_probability_percent", 50.0) / 100.0,
                                confidence=pred.get("final_confidence_percent", 0.0),
                                meta_trust=pred.get("meta_reliability_percent", 50.0),
                                uncertainty=pred.get("uncertainty_percent", 0.0),
                                regime=pred.get("market_regime", "Unknown"),
                                session=pred.get("session", "Unknown"),
                                kelly=pred.get("kelly_fraction_percent", 0.0),
                                recent_candles=recent_candles_str,
                                support1="N/A",  # Pending auto-SR implementation
                                resistance1="N/A",
                                ema20=e20,
                                ema50=e50,
                                rsi=rsi,
                                atr=atr,
                                dist_to_res=dist_to_res,
                                dist_to_sup=dist_to_sup,
                                bull_div=bull_div,
                                bear_div=bear_div,
                                vol_accel=vol_accel,
                                cumulative_delta=cum_delta,
                                m1_absorption=m1_absorp,
                                trade_intensity=trade_int,
                                news_sentiment=live_news_text,
                                dxy_price=dxy_proxy,
                                gold_price=gold_price,
                                eurgbp_price=eurgbp_price
                            )
                            # Gemini as VETO-ONLY gate — never overrides ML direction or confidence
                            # LLM hallucinations previously flipped validated ML decisions
                            ml_direction = pred["suggested_direction"]
                            gemini_direction = v["verdict"]
                            gemini_agrees = (gemini_direction == ml_direction) or (
                                (gemini_direction == "UP" and ml_direction == "UP") or
                                (gemini_direction == "DOWN" and ml_direction == "DOWN")
                            )

                            if v["verdict"] == "NO_TRADE" or not v.get("approved", True):
                                filtered_out[sym] = f"Gemini Veto: {v['reason']}"
                                log.info("Gemini veto %s: %s", sym, v["reason"])
                                _log_gemini_decision(sym, ml_direction, gemini_direction, "VETOED")
                            elif not gemini_agrees:
                                # Gemini disagrees with ML direction — block signal (don't flip)
                                filtered_out[sym] = f"Gemini disagrees: ML={ml_direction} vs Gemini={gemini_direction}"
                                log.info("Gemini directional conflict %s: ML=%s Gemini=%s — BLOCKED", sym, ml_direction, gemini_direction)
                                _log_gemini_decision(sym, ml_direction, gemini_direction, "CONFLICT")
                            else:
                                # Gemini agrees — keep ML direction and confidence intact
                                pred["gemini_approved"] = True
                                pred["gemini_confidence"] = v.get("gemini_confidence", 0)
                                gemini_filtered[sym] = pred
                                log.info("Gemini CONFIRMED %s %s (Gemini conf: %s%%): %s", sym, ml_direction, v.get("gemini_confidence", "?"), v["reason"])
                                _log_gemini_decision(sym, ml_direction, gemini_direction, "CONFIRMED")
                    except Exception as e:
                        log.warning("Gemini EXCEPTION on %s — signal BLOCKED: %s", sym, e)
                        filtered_out[sym] = f"Gemini Error: {e}"
                predictions = gemini_filtered

            # Send ONE combined message per subscriber (respecting alert mode)
            # Track which symbols were actually sent per subscriber
            sent_symbols_per_sub = {}  # {chat_id: set(symbols)}
            for chat_id, info in active_subs.items():
                symbols = info.get("symbols", [DEFAULT_SYMBOL])
                mode = info.get("mode", "all")
                # Filter predictions for this subscriber
                sub_preds = {}
                for sym in symbols:
                    if sym in predictions:
                        pred = predictions[sym]
                        conf = pred["final_confidence_percent"]
                        if mode == "high" and conf < HIGH_CONF_THRESHOLD:
                            continue
                        sub_preds[sym] = pred

                if not sub_preds:
                    continue

                sent_symbols_per_sub[chat_id] = set(sub_preds.keys())

                msg = _format_combined_signal(sub_preds, filtered_out)
                try:
                    await _safe_send_message(bot, int(chat_id), msg, parse_mode="Markdown")
                except Exception as e:
                    log.error("Failed to send to %s: %s", chat_id, e)

            if filtered_out:
                log.info("Filtered signals: %s", filtered_out)

            _record_sent_signals(predictions, sent_symbols_per_sub)

            # Schedule outcome check after candle closes (+30s buffer)
            asyncio.create_task(
                _check_outcomes(bot, predictions, active_subs, sent_symbols_per_sub)
            )

            # Small delay to avoid double-sending
            await asyncio.sleep(15)

        except Exception as e:
            log.exception("Auto-signal job error: %s", e)
            await asyncio.sleep(30)


async def _check_outcomes(bot: Bot, predictions: dict, subscribers: dict,
                         sent_symbols_per_sub: dict = None):
    """
    Wait for the current candle to close, then check if predictions were correct.
    Send result notifications ONLY for symbols each subscriber actually received.
    """
    # STAGE 1: Wait for current candle to close (trade opens) + 15s buffer
    # TELEGRAM_SEND_BEFORE_CLOSE_SEC is usually 15s, so this waits ~30s total
    await asyncio.sleep(TELEGRAM_SEND_BEFORE_CLOSE_SEC + 15)

    # Send ENTRY CONFIRMATION to subscribers who received these signals
    for chat_id, info in subscribers.items():
        if not info.get("active", False):
            continue

        if sent_symbols_per_sub and chat_id in sent_symbols_per_sub:
            sub_syms = sent_symbols_per_sub[chat_id]
        else:
            sub_syms = set(predictions.keys())

        if not sub_syms:
            continue

        entry_lines = []
        for sym in sub_syms:
            if sym in predictions:
                pred = predictions[sym]
                if pred["suggested_direction"] == "UP":
                    d = "🟢 UP"
                elif pred["suggested_direction"] == "DOWN":
                    d = "🔴 DOWN"
                else:
                    continue
                entry_lines.append(f"⏱ *{sym}* Trade is LIVE \u2192 *{d}*")

        if entry_lines:
            entry_msg = "⏳ *ENTRY CONFIRMED*\n━━━━━━━━━━━━━━━━━━━━━━━\n" + "\n".join(entry_lines) + "\n\n_Waiting 5 mins for candle close..._"
            try:
                await _safe_send_message(bot, int(chat_id), entry_msg, parse_mode="Markdown")
            except Exception as e:
                log.error("Failed to send entry confirmation to %s: %s", chat_id, e)

    # STAGE 2: Wait for the PREDICTED 5-minute candle to fully form
    # We already waited ~30s, now we wait the remaining 5 minutes + 15s buffer
    await asyncio.sleep(300 + 15)

    # Build all outcome data first
    all_outcomes = {}  # {sym: {emoji, msg_line, correct, conf, pred}}
    resolved_signal_ids = set()
    for sym, pred in predictions.items():
        outcome = _resolve_signal_outcome(
            sym,
            pred,
            meta_row=_last_predictions.get(sym, {}).get("meta_row"),
        )
        if not outcome:
            continue

        if outcome.get("triggered_cb"):
            cb_msg = (
                f'🛑 *CIRCUIT BREAKER TRIGGERED* — {sym}\n'
                f"Reason: _{outcome.get('cb_reason', 'risk guard')}_\n"
                f'All signals halted 60 min. Resumes automatically.'
            )
            await _broadcast_to_subscribers(bot, cb_msg)
            await _send_admin_alert(bot, cb_msg)

        all_outcomes[sym] = {"line": outcome["line"]}
        signal_id = pred.get("_signal_id")
        if signal_id:
            resolved_signal_ids.add(signal_id)

    _remove_pending_signal_ids(resolved_signal_ids)

    if not all_outcomes:
        return

    # Send MATCHED results to each subscriber (only their sent symbols)
    for chat_id, info in subscribers.items():
        if not info.get("active", False):
            continue

        # Determine which symbols this subscriber actually received
        if sent_symbols_per_sub and chat_id in sent_symbols_per_sub:
            sub_syms = sent_symbols_per_sub[chat_id]
        else:
            # Fallback: send all outcomes
            sub_syms = set(all_outcomes.keys())

        # Build result lines only for symbols this subscriber got signals for
        sub_lines = [all_outcomes[s]["line"] for s in sub_syms if s in all_outcomes]
        if not sub_lines:
            continue

        msg = "\U0001f4cb *Signal Results*\n\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n\n" + "\n".join(sub_lines)
        try:
            await _safe_send_message(bot, int(chat_id), msg, parse_mode="Markdown")
        except Exception as e:
            log.error("Failed to send result to %s: %s", chat_id, e)



def _log_prediction_csv(symbol, pred):
    """Log EVERY prediction to CSV (before filtering). For weekly analysis."""
    os.makedirs(os.path.dirname(PREDICTION_CSV), exist_ok=True)
    header_needed = not os.path.exists(PREDICTION_CSV)
    try:
        with open(PREDICTION_CSV, "a") as f:
            if header_needed:
                f.write("timestamp,symbol,direction,green_prob,confidence,kelly,"
                        "regime,trade,uncertainty,latency_ms\n")
            f.write(
                f"{datetime.now(timezone.utc).isoformat()},"
                f"{symbol},"
                f"{pred.get('suggested_direction', '?')},"
                f"{pred.get('green_probability_percent', 50):.2f},"
                f"{pred.get('final_confidence_percent', 0):.2f},"
                f"{pred.get('kelly_fraction_percent', 0):.1f},"
                f"{pred.get('market_regime', 'Unknown')},"
                f"{pred.get('suggested_trade', 'HOLD')},"
                f"{pred.get('uncertainty_percent', 0):.2f},"
                f"{pred.get('latency_ms', -1)}\n"
            )
    except Exception as e:
        log.error("Prediction CSV write failed: %s", e)


def _log_outcome_csv(symbol, pred, actual_green, correct):
    """Append every signal outcome to CSV for offline analysis."""
    os.makedirs(os.path.dirname(SIGNAL_CSV), exist_ok=True)
    header_needed = not os.path.exists(SIGNAL_CSV)
    try:
        with open(SIGNAL_CSV, "a") as f:
            if header_needed:
                f.write("timestamp,symbol,predicted,actual,correct,green_prob,confidence,kelly,regime,latency_ms\n")
            f.write(
                f"{datetime.now(timezone.utc).isoformat()},"
                f"{symbol},"
                f"{pred['suggested_direction']},"
                f"{'UP' if actual_green else 'DOWN'},"
                f"{correct},"
                f"{pred['green_probability_percent']:.2f},"
                f"{pred['final_confidence_percent']:.2f},"
                f"{pred.get('kelly_fraction_percent', 0):.1f},"
                f"{pred.get('market_regime', 'Unknown')},"
                f"{pred.get('latency_ms', -1)}\n"
            )
    except Exception as e:
        log.error("CSV write failed: %s", e)


def _get_accuracy_snapshot(window: int = 50) -> dict:
    import csv

    if not os.path.exists(SIGNAL_CSV):
        return {
            "total": 0,
            "overall_win_rate": 0.0,
            "recent_count": 0,
            "recent_win_rate": 0.0,
        }

    try:
        with open(SIGNAL_CSV, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        log.warning("Accuracy snapshot load failed: %s", e)
        return {
            "total": 0,
            "overall_win_rate": 0.0,
            "recent_count": 0,
            "recent_win_rate": 0.0,
        }

    total = len(rows)
    if total == 0:
        return {
            "total": 0,
            "overall_win_rate": 0.0,
            "recent_count": 0,
            "recent_win_rate": 0.0,
        }

    wins = sum(row.get("correct") == "True" for row in rows)
    recent = rows[-window:]
    recent_wins = sum(row.get("correct") == "True" for row in recent)
    return {
        "total": total,
        "overall_win_rate": wins / total * 100,
        "recent_count": len(recent),
        "recent_win_rate": recent_wins / len(recent) * 100 if recent else 0.0,
    }


def _get_symbol_accuracy_snapshot(recent_window: int = SYMBOL_ACCURACY_RECENT_WINDOW) -> dict:
    import csv
    from collections import defaultdict

    if not os.path.exists(SIGNAL_CSV):
        return {}

    try:
        with open(SIGNAL_CSV, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        log.warning("Symbol accuracy snapshot load failed: %s", e)
        return {}

    totals = defaultdict(lambda: {"wins": 0, "total": 0})
    rows_by_symbol = defaultdict(list)

    for row in rows:
        symbol = row.get("symbol", "")
        if not symbol:
            continue
        totals[symbol]["total"] += 1
        totals[symbol]["wins"] += row.get("correct") == "True"
        rows_by_symbol[symbol].append(row)

    snapshot = {}
    for symbol, symbol_rows in rows_by_symbol.items():
        total = totals[symbol]["total"]
        recent_rows = symbol_rows[-recent_window:] if recent_window > 0 else []
        recent_total = len(recent_rows)
        recent_wins = sum(row.get("correct") == "True" for row in recent_rows)
        snapshot[symbol] = {
            "overall_total": total,
            "overall_win_rate": (totals[symbol]["wins"] / total * 100) if total else 0.0,
            "recent_total": recent_total,
            "recent_win_rate": (recent_wins / recent_total * 100) if recent_total else 0.0,
        }
    return snapshot


def _is_symbol_accuracy_allowed(symbol: str, stats_snapshot: dict | None = None) -> tuple[bool, str]:
    stats = (stats_snapshot or _get_symbol_accuracy_snapshot()).get(symbol)
    if not stats:
        return True, "no_accuracy_history"

    if (
        stats["overall_total"] >= SYMBOL_ACCURACY_MIN_TOTAL_SAMPLES and
        stats["overall_win_rate"] < SYMBOL_ACCURACY_MIN_OVERALL_WR
    ):
        return False, f"overall_wr:{stats['overall_win_rate']:.1f}%"

    if (
        stats["recent_total"] >= SYMBOL_ACCURACY_MIN_RECENT_SAMPLES and
        stats["recent_win_rate"] < SYMBOL_ACCURACY_MIN_RECENT_WR
    ):
        return False, f"recent_wr:{stats['recent_win_rate']:.1f}%"

    return True, "accuracy_ok"


def _get_symbol_min_confidence(symbol: str, stats_snapshot: dict | None = None) -> float:
    stats = (stats_snapshot or _get_symbol_accuracy_snapshot()).get(symbol)
    if not stats:
        return PRODUCTION_MIN_CONFIDENCE

    if (
        stats["recent_total"] >= ADAPTIVE_MIN_CONFIDENCE_MIN_RECENT_SAMPLES and
        stats["recent_win_rate"] < ADAPTIVE_MIN_CONFIDENCE_RECENT_WR_CUTOFF
    ):
        return ADAPTIVE_MIN_CONFIDENCE_WEAK_SYMBOL

    return PRODUCTION_MIN_CONFIDENCE


def _get_symbol_direction_stats() -> dict:
    """Return {(symbol, direction): {'wins': int, 'total': int}} from stored outcomes."""
    import csv
    from collections import defaultdict

    if not os.path.exists(SIGNAL_CSV):
        return {}

    try:
        with open(SIGNAL_CSV, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return {}

    stats = defaultdict(lambda: {"wins": 0, "total": 0})
    for row in rows:
        sym = row.get("symbol", "")
        direction = row.get("predicted", "")
        if sym and direction:
            stats[(sym, direction)]["total"] += 1
            stats[(sym, direction)]["wins"] += row.get("correct") == "True"
    return dict(stats)


def _get_symbol_regime_stats() -> dict:
    """Return {(symbol, regime): {'wins': int, 'total': int}} from stored outcomes."""
    import csv
    from collections import defaultdict

    if not os.path.exists(SIGNAL_CSV):
        return {}

    try:
        with open(SIGNAL_CSV, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return {}

    stats = defaultdict(lambda: {"wins": 0, "total": 0})
    for row in rows:
        sym = row.get("symbol", "")
        regime = row.get("regime", "")
        if sym and regime:
            stats[(sym, regime)]["total"] += 1
            stats[(sym, regime)]["wins"] += row.get("correct") == "True"
    return dict(stats)


def _is_symbol_direction_allowed(symbol: str, direction: str, sd_stats: dict | None = None) -> tuple[bool, str]:
    """Check if this symbol+direction combo has acceptable historical WR."""
    if sd_stats is None:
        sd_stats = _get_symbol_direction_stats()

    key = (symbol, direction)
    stats = sd_stats.get(key)
    if not stats or stats["total"] < SYMBOL_DIRECTION_MIN_SAMPLES:
        return True, "insufficient_data"

    wr = stats["wins"] / stats["total"] * 100
    if wr < SYMBOL_DIRECTION_MIN_WR:
        return False, f"{symbol}_{direction}_wr:{wr:.1f}%<{SYMBOL_DIRECTION_MIN_WR}%"
    return True, f"wr:{wr:.1f}%"


def _is_symbol_regime_allowed(symbol: str, regime: str, sr_stats: dict | None = None) -> tuple[bool, str]:
    """Check if this symbol+regime combo has acceptable historical WR."""
    if sr_stats is None:
        sr_stats = _get_symbol_regime_stats()

    key = (symbol, regime)
    stats = sr_stats.get(key)
    if not stats or stats["total"] < SYMBOL_REGIME_MIN_SAMPLES:
        return True, "insufficient_data"

    wr = stats["wins"] / stats["total"] * 100
    if wr < SYMBOL_REGIME_MIN_WR:
        return False, f"{symbol}/{regime}_wr:{wr:.1f}%<{SYMBOL_REGIME_MIN_WR}%"
    return True, f"wr:{wr:.1f}%"


def _has_hard_risk_warning(risk_warnings: list[str]) -> tuple[bool, str]:
    for warning in risk_warnings or []:
        if any(warning.startswith(prefix) for prefix in HARD_RISK_WARNING_PREFIXES):
            return True, warning
    return False, ""


def _log_gemini_decision(symbol: str, ml_direction: str, gemini_direction: str, outcome: str):
    """Track Gemini decisions for accuracy analysis."""
    os.makedirs(os.path.dirname(GEMINI_LOG_CSV), exist_ok=True)
    header_needed = not os.path.exists(GEMINI_LOG_CSV)
    try:
        with open(GEMINI_LOG_CSV, "a", encoding="utf-8") as f:
            if header_needed:
                f.write("timestamp,symbol,ml_direction,gemini_direction,outcome\n")
            f.write(f"{datetime.now(timezone.utc).isoformat()},{symbol},{ml_direction},{gemini_direction},{outcome}\n")
    except Exception as e:
        log.warning("Gemini log write failed: %s", e)


def _get_streak_confidence_adjustment(symbol: str) -> float:
    """
    Returns confidence adjustment based on win/loss streaks.
    After STREAK_LOSS_TRIGGER consecutive losses: raise min conf.
    After STREAK_WIN_TRIGGER consecutive wins: allow small reduction.
    """
    losses = _consecutive_losses.get(symbol, 0)
    wins = _consecutive_wins.get(symbol, 0)

    if losses >= STREAK_LOSS_TRIGGER:
        bump = min(STREAK_LOSS_CONFIDENCE_BUMP * (losses - STREAK_LOSS_TRIGGER + 1), STREAK_LOSS_MAX_BUMP)
        return bump
    elif wins >= STREAK_WIN_TRIGGER:
        return -STREAK_WIN_THRESHOLD_REDUCTION
    return 0.0


def _detect_bb_breakout(df, squeeze_lookback: int = 20, squeeze_percentile: float = 20.0) -> dict:
    """
    Detect Bollinger Band breakout — price breaking out of a squeeze (low vol → expansion).
    
    A BB squeeze occurs when BB width is at a historical low (compressed volatility).
    When price then breaks above upper BB or below lower BB, it signals strong directional move.
    
    Returns:
        dict with:
            is_breakout: bool — True if a breakout is detected
            breakout_direction: 'UP' | 'DOWN' | None
            squeeze_active: bool — was recently in a squeeze
            bb_width_percentile: float — current BB width relative to recent history
            detail: str — human-readable description
    """
    result = {
        "is_breakout": False,
        "breakout_direction": None,
        "squeeze_active": False,
        "bb_width_percentile": 50.0,
        "detail": "",
    }
    
    if df is None or len(df) < squeeze_lookback + 10:
        return result
    
    try:
        import numpy as np
        close = df["close"].values
        
        # Compute BB if not already in df
        if "bb_upper" in df.columns and "bb_lower" in df.columns:
            bb_upper = df["bb_upper"].values
            bb_lower = df["bb_lower"].values
            bb_mid = df["bb_mid"].values if "bb_mid" in df.columns else (bb_upper + bb_lower) / 2
        else:
            sma_arr = np.convolve(close, np.ones(20)/20, mode='full')[:len(close)]
            sma_arr[:19] = np.nan
            # Rolling std via numpy
            std_arr = np.full(len(close), np.nan)
            for i in range(19, len(close)):
                std_arr[i] = np.std(close[i-19:i+1], ddof=1)
            bb_upper = sma_arr + 2 * std_arr
            bb_lower = sma_arr - 2 * std_arr
            bb_mid = sma_arr
        
        # BB width as percentage of middle band
        bb_width = (bb_upper - bb_lower) / np.where(bb_mid > 0, bb_mid, 1)
        
        # Current and recent BB width
        current_width = bb_width[-1]
        recent_widths = bb_width[-squeeze_lookback:]
        recent_widths = recent_widths[~np.isnan(recent_widths)]
        
        if len(recent_widths) < 10:
            return result
        
        # Percentile of current width in recent history
        width_percentile = (np.sum(recent_widths < current_width) / len(recent_widths)) * 100
        result["bb_width_percentile"] = round(width_percentile, 1)
        
        # Detect squeeze: BB width in bottom percentile
        squeeze_threshold = np.percentile(recent_widths, squeeze_percentile)
        
        # Check if squeeze was active in last 5 candles (just broke out)
        squeeze_recently = any(bb_width[i] <= squeeze_threshold for i in range(-6, -1) if not np.isnan(bb_width[i]))
        result["squeeze_active"] = squeeze_recently
        
        # Current candle breakout detection
        last_close = close[-1]
        last_upper = bb_upper[-1]
        last_lower = bb_lower[-1]
        
        if np.isnan(last_upper) or np.isnan(last_lower):
            return result
        
        if squeeze_recently:
            if last_close > last_upper:
                result["is_breakout"] = True
                result["breakout_direction"] = "UP"
                result["detail"] = f"BB squeeze breakout UP (width P{width_percentile:.0f})"
            elif last_close < last_lower:
                result["is_breakout"] = True
                result["breakout_direction"] = "DOWN"
                result["detail"] = f"BB squeeze breakout DOWN (width P{width_percentile:.0f})"
                
        # Even without squeeze, flag extreme BB penetration  
        if not result["is_breakout"]:
            if last_close > last_upper and width_percentile > 60:
                result["is_breakout"] = True
                result["breakout_direction"] = "UP"
                result["detail"] = f"BB upper breakout (expanding vol, P{width_percentile:.0f})"
            elif last_close < last_lower and width_percentile > 60:
                result["is_breakout"] = True
                result["breakout_direction"] = "DOWN"
                result["detail"] = f"BB lower breakout (expanding vol, P{width_percentile:.0f})"
    except Exception as e:
        log.debug("BB breakout detection failed: %s", e)
    
    return result


def _detect_support_resistance(df, lookback: int = 50, n_levels: int = 3) -> dict:
    """
    Auto-detect support/resistance levels from swing highs/lows.
    Returns dict with closest support, resistance, and proximity info.
    """
    if df is None or len(df) < lookback:
        return {"support": None, "resistance": None, "near_sr": False}

    highs = df["high"].values[-lookback:]
    lows = df["low"].values[-lookback:]
    close = df["close"].values[-1]
    atr = df["atr_14"].values[-1] if "atr_14" in df.columns else 0

    # Find swing highs (local maxima)
    swing_highs = []
    swing_lows = []
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append(lows[i])

    # Find closest support (below price) and resistance (above price)
    supports = sorted([l for l in swing_lows if l < close], reverse=True)
    resistances = sorted([r for r in swing_highs if r > close])

    support = supports[0] if supports else None
    resistance = resistances[0] if resistances else None

    # Check proximity to S/R
    near_sr = False
    sr_detail = ""
    if atr > 0:
        if support and (close - support) / atr < SR_PROXIMITY_ATR_THRESHOLD:
            near_sr = True
            sr_detail = f"near_support({(close - support) / atr:.2f}ATR)"
        if resistance and (resistance - close) / atr < SR_PROXIMITY_ATR_THRESHOLD:
            near_sr = True
            sr_detail = f"near_resistance({(resistance - close) / atr:.2f}ATR)"

    return {
        "support": support,
        "resistance": resistance,
        "near_sr": near_sr,
        "sr_detail": sr_detail,
        "support_dist_atr": (close - support) / atr if support and atr > 0 else 99,
        "resistance_dist_atr": (resistance - close) / atr if resistance and atr > 0 else 99,
    }


def _is_near_round_number(symbol: str, price: float) -> tuple[bool, str]:
    """
    Check if price is near a psychological round number.
    Forex: .000, .500 levels. Gold: $10 levels.
    """
    if "XAU" in symbol:
        # Gold: near $10 round levels (e.g., 2920, 2930)
        remainder = price % 10
        dist = min(remainder, 10 - remainder)
        if dist < 1.0:  # within $1 of round $10 level
            level = round(price / 10) * 10
            return True, f"near_${level}"
    else:
        # Forex pairs: near .000 or .500 levels
        if "JPY" in symbol:
            # JPY pairs: round yen levels (e.g., 148.000, 148.500)
            remainder = price % 0.5
            dist = min(remainder, 0.5 - remainder)
            if dist < 0.05:  # within 5 pips
                level = round(price / 0.5) * 0.5
                return True, f"near_{level:.1f}"
        else:
            # Other forex: .00000 or .00500 levels
            scaled = price * 10000  # convert to pips
            remainder = scaled % 500  # every 500 pips = .00500
            dist = min(remainder, 500 - remainder)
            if dist < ROUND_LEVEL_PROXIMITY_PIPS:
                return True, f"near_round({dist:.0f}pips)"
    return False, ""


def _compute_optimal_expiry(regime: str, atr_val: float, atr_median: float, bb_width: float = 0) -> dict:
    """
    Recommend optimal expiry based on volatility and regime.
    Professional binary bots adjust expiry per signal instead of fixed 5min.
    """
    if atr_val <= 0 or atr_median <= 0:
        return {"expiry_minutes": 5, "expiry_label": "5 Min (Default)", "reason": "insufficient_data"}

    vol_ratio = atr_val / atr_median

    if regime == "Ranging":
        if vol_ratio < 0.7:
            return {"expiry_minutes": 3, "expiry_label": "3 Min", "reason": "low_vol_ranging"}
        elif vol_ratio > 1.5:
            return {"expiry_minutes": 5, "expiry_label": "5 Min", "reason": "high_vol_ranging"}
        else:
            return {"expiry_minutes": 3, "expiry_label": "3 Min", "reason": "normal_ranging"}
    elif regime == "Trending":
        if vol_ratio > 2.0:
            return {"expiry_minutes": 3, "expiry_label": "3 Min", "reason": "strong_trend_high_vol"}
        else:
            return {"expiry_minutes": 5, "expiry_label": "5 Min", "reason": "normal_trend"}
    elif regime == "High_Volatility":
        return {"expiry_minutes": 3, "expiry_label": "3 Min", "reason": "high_volatility"}
    else:
        return {"expiry_minutes": 5, "expiry_label": "5 Min", "reason": "default"}


def _get_preferred_symbols(candidates: set[str], stats_snapshot: dict | None = None) -> set[str]:
    stats_snapshot = stats_snapshot or _get_symbol_accuracy_snapshot()
    ranked = []

    for symbol in candidates:
        stats = stats_snapshot.get(symbol)
        if not stats:
            continue
        if stats["overall_total"] < PREFERRED_SYMBOL_MIN_TOTAL_SAMPLES:
            continue
        if stats["recent_total"] < PREFERRED_SYMBOL_MIN_RECENT_TOTAL:
            continue
        if stats["overall_win_rate"] < PREFERRED_SYMBOL_MIN_OVERALL_WR:
            continue
        ranked.append((
            stats["overall_win_rate"],
            stats["recent_win_rate"],
            stats["overall_total"],
            symbol,
        ))

    ranked.sort(reverse=True)
    if not ranked:
        return set()

    selected = {ranked[0][3]}
    for overall_win_rate, _, overall_total, symbol in ranked[1:]:
        if len(selected) >= PREFERRED_SYMBOL_MAX_COUNT + 1:
            break
        if overall_total < PREFERRED_SYMBOL_EXTRA_MIN_TOTAL_SAMPLES:
            continue
        if overall_win_rate < PREFERRED_SYMBOL_EXTRA_MIN_OVERALL_WR:
            continue
        selected.add(symbol)

    return selected


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
            wait_secs = (target - now).total_seconds()
            log.info("Daily report: next send in %.0f min at 21:00 UTC", wait_secs / 60)
            await asyncio.sleep(wait_secs)

            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            today_res = [r for r in _outcome_results if r["time"].startswith(today)]

            if not today_res:
                await asyncio.sleep(60)
                continue

            total = len(today_res)
            wins = sum(1 for r in today_res if r["correct"])
            losses = total - wins
            win_rate = wins / total * 100

            sym_stats = {}
            for r in today_res:
                s = r["symbol"]
                if s not in sym_stats:
                    sym_stats[s] = {"w": 0, "l": 0}
                if r["correct"]:
                    sym_stats[s]["w"] += 1
                else:
                    sym_stats[s]["l"] += 1

            best_sym = max(sym_stats, key=lambda s: sym_stats[s]["w"] / max(sym_stats[s]["w"] + sym_stats[s]["l"], 1))
            worst_sym = min(sym_stats, key=lambda s: sym_stats[s]["w"] / max(sym_stats[s]["w"] + sym_stats[s]["l"], 1))
            best_rate = sym_stats[best_sym]["w"] / max(sym_stats[best_sym]["w"] + sym_stats[best_sym]["l"], 1) * 100
            worst_rate = sym_stats[worst_sym]["w"] / max(sym_stats[worst_sym]["w"] + sym_stats[worst_sym]["l"], 1) * 100

            ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
            msg = (
                f"📋 *Daily Report — {today}*\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"📊 Signals: *{total}* | ✅ *{wins}* | ❌ *{losses}*\n"
                f"🎯 Win Rate: *{win_rate:.1f}%*\n\n"
                f"🏆 Best: *{best_sym}* ({best_rate:.0f}%)\n"
                f"📉 Worst: *{worst_sym}* ({worst_rate:.0f}%)\n\n"
            )
            for s, st in sorted(sym_stats.items()):
                t = st["w"] + st["l"]
                r = st["w"] / t * 100 if t > 0 else 0
                emoji = "🟢" if r >= 60 else "🔴"
                msg += f"  {emoji} {s}: {st['w']}/{t} ({r:.0f}%)\n"

            await _broadcast_to_subscribers(bot, msg)
            await asyncio.sleep(60)

        except Exception as e:
            log.exception("Daily report error: %s", e)
            await asyncio.sleep(300)


async def _weekly_report_job(app: Application):
    """Send weekly accuracy report every Sunday at 21:05 UTC."""
    bot: Bot = app.bot
    log.info("Weekly report job started.")

    while True:
        try:
            now = datetime.now(timezone.utc)
            # Find next Sunday
            days_until_sunday = (6 - now.weekday()) % 7
            if days_until_sunday == 0 and now.hour >= 21:
                days_until_sunday = 7
            target = (now + timedelta(days=days_until_sunday)).replace(
                hour=21, minute=5, second=0, microsecond=0
            )
            wait_secs = (target - now).total_seconds()
            log.info("Weekly report: next send in %.1f hours", wait_secs / 3600)
            await asyncio.sleep(wait_secs)

            # Build 7-day summary from outcome results
            cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            week_res = [r for r in _outcome_results if r["time"] >= cutoff]

            if len(week_res) < 5:
                await asyncio.sleep(60)
                continue

            total = len(week_res)
            wins = sum(1 for r in week_res if r["correct"])
            win_rate = wins / total * 100

            # Per-symbol
            sym_stats = {}
            for r in week_res:
                s = r["symbol"]
                if s not in sym_stats:
                    sym_stats[s] = {"w": 0, "l": 0}
                if r["correct"]:
                    sym_stats[s]["w"] += 1
                else:
                    sym_stats[s]["l"] += 1

            # Confidence bins
            high_conf = [r for r in week_res if r["confidence"] >= 70]
            low_conf = [r for r in week_res if r["confidence"] < 70]
            hc_rate = sum(1 for r in high_conf if r["correct"]) / max(len(high_conf), 1) * 100
            lc_rate = sum(1 for r in low_conf if r["correct"]) / max(len(low_conf), 1) * 100

            week_str = datetime.now(timezone.utc).strftime("%b %d")
            msg = (
                f"📊 *Weekly Report — Week of {week_str}*\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"📈 Total: *{total}* signals\n"
                f"✅ Wins: *{wins}* | ❌ Losses: *{total - wins}*\n"
                f"🎯 Win Rate: *{win_rate:.1f}%*\n\n"
                f"🔥 High Conf (≥70%): *{hc_rate:.0f}%* ({len(high_conf)} signals)\n"
                f"📉 Low Conf (<70%): *{lc_rate:.0f}%* ({len(low_conf)} signals)\n\n"
            )
            for s, st in sorted(sym_stats.items()):
                t = st["w"] + st["l"]
                r_pct = st["w"] / t * 100 if t > 0 else 0
                emoji = "🟢" if r_pct >= 60 else "🟡" if r_pct >= 50 else "🔴"
                msg += f"  {emoji} *{s}*: {st['w']}/{t} ({r_pct:.0f}%)\n"

            msg += f"\n💡 _Retrain recommended if win rate < 55%_"

            await _broadcast_to_subscribers(bot, msg)
            await asyncio.sleep(60)

        except Exception as e:
            log.exception("Weekly report error: %s", e)
            await asyncio.sleep(300)


async def _broadcast_to_subscribers(bot: Bot, msg: str):
    """Send a message to all active subscribers."""
    for chat_id, info in _subscribers.items():
        if info.get("active", False):
            try:
                await _safe_send_message(bot, int(chat_id), msg, parse_mode="Markdown")
            except Exception as e:
                log.error("Broadcast failed for %s: %s", chat_id, e)


# =============================================================================
#  Bot Handlers
# =============================================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    n_subs = sum(1 for s in _subscribers.values() if s.get("active"))
    welcome = (
        "🤖 *QUOTEX LORD V3 — ML Trading Engine*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "🔔 *Auto-Signal Mode:*\n"
        "  Predictions 5s before every M5 close\n"
        "  6 pairs │ Win/loss tracking │ Daily reports\n\n"
        "🏗 *V3 Architecture (14 Layers):*\n"
        "  • 65 features + multi-timeframe (M5/M15/H1)\n"
        "  • Triple Barrier labeling (1.5x TP / 1.0x SL)\n"
        "  • XGBoost + LightGBM + RandomForest ensemble\n"
        "  • Purged Walk-Forward CV (leak-proof)\n"
        "  • Regime-specific model routing\n"
        "  • 10-gate pre-flight filter system\n"
        "  • Meta-model + Kelly sizing + Circuit breaker\n\n"
        "📊 *Backtest: 56.2% (honest, leak-free)*\n"
        f"🎯 *High-Conf (≥70%): 75%*\n"
        f"👥 Active subscribers: *{n_subs}*\n\n"
        "👇 *Tap below to start!*"
    )
    await update.message.reply_text(
        welcome,
        parse_mode="Markdown",
        reply_markup=_main_menu_keyboard(chat_id),
    )


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    chat_id = query.message.chat_id

    # ----- Menu -----
    if data == "menu":
        await query.edit_message_text(
            "🤖 *QUOTEX LORD — Main Menu*\n\nChoose an option 👇",
            parse_mode="Markdown",
            reply_markup=_main_menu_keyboard(chat_id),
        )

    # ----- Subscribe -----
    elif data == "sub":
        await query.edit_message_text(
            "🟢 *Start Auto-Signal*\n\nWhich symbols do you want signals for?",
            parse_mode="Markdown",
            reply_markup=_sub_symbol_keyboard(),
        )

    elif data.startswith("sub_"):
        symbol = data.replace("sub_", "")
        cid = str(chat_id)
        if symbol == "ALL":
            symbols = SYMBOLS.copy()
        else:
            symbols = [symbol]

        # Store symbols temporarily, ask for alert mode
        _subscribers[cid] = {"symbols": symbols, "active": False, "mode": "all"}
        _save_subscribers()

        sym_list = ", ".join(symbols)
        await query.edit_message_text(
            f"📊 Symbols: *{sym_list}*\n\n"
            f"🔔 *Choose alert mode:*\n\n"
            f"📡 *All Signals* — every M5 candle\n"
            f"🎯 *High-Conf Only* — only when ≥70% confidence",
            parse_mode="Markdown",
            reply_markup=_alert_mode_keyboard(),
        )

    elif data in ("mode_all", "mode_high"):
        cid = str(chat_id)
        mode = "high" if data == "mode_high" else "all"
        if cid in _subscribers:
            _subscribers[cid]["active"] = True
            _subscribers[cid]["mode"] = mode
            _save_subscribers()

        mode_label = "🎯 High-Conf Only (≥70%)" if mode == "high" else "📡 All Signals"
        syms = _subscribers.get(cid, {}).get("symbols", [DEFAULT_SYMBOL])
        await query.edit_message_text(
            f"✅ *Auto-Signal Activated!*\n\n"
            f"📊 Symbols: *{', '.join(syms)}*\n"
            f"🔔 Mode: *{mode_label}*\n"
            f"⏰ Predictions every 5 minutes\n\n"
            f"Signals will start at the next candle.\n"
            f"Send /start anytime to see the menu.",
            parse_mode="Markdown",
            reply_markup=_main_menu_keyboard(chat_id),
        )
        log.info("Subscriber activated: %s mode=%s syms=%s", cid, mode, syms)

    # ----- Unsubscribe -----
    elif data == "unsub":
        cid = str(chat_id)
        if cid in _subscribers:
            _subscribers[cid]["active"] = False
            _save_subscribers()
        await query.edit_message_text(
            "🔴 *Auto-Signal Stopped*\n\n"
            "You won't receive automatic signals anymore.\n"
            "Tap the button below to reactivate.",
            parse_mode="Markdown",
            reply_markup=_main_menu_keyboard(chat_id),
        )

    # ----- Predict Now -----
    elif data == "predict_now":
        await query.edit_message_text(
            "🔮 *Instant Prediction*\n\nSelect symbol:",
            parse_mode="Markdown",
            reply_markup=_symbol_keyboard("now"),
        )

    elif data.startswith("now_"):
        symbol = data.replace("now_", "")
        if not _is_market_open():
            await query.edit_message_text(
                _market_closed_msg(), parse_mode="Markdown",
                reply_markup=_back_keyboard(chat_id))
            return
        await query.edit_message_text(f"⏳ Running *{symbol}*...", parse_mode="Markdown")
        try:
            pred = _run_prediction(symbol)
            msg = _format_prediction(pred)
            await query.edit_message_text(msg, parse_mode="Markdown",
                                          reply_markup=_back_keyboard(chat_id))
        except Exception as e:
            log.exception("Prediction failed for %s", symbol)
            await query.edit_message_text(f"❌ Error: {e}",
                                          reply_markup=_back_keyboard(chat_id))

    # ----- Next Candle -----
    elif data == "predict_next":
        await query.edit_message_text(
            "⏳ *Next Candle Prediction*\n\nSelect symbol:",
            parse_mode="Markdown",
            reply_markup=_symbol_keyboard("next"),
        )

    elif data.startswith("next_"):
        symbol = data.replace("next_", "")
        if not _is_market_open():
            await query.edit_message_text(
                _market_closed_msg(), parse_mode="Markdown",
                reply_markup=_back_keyboard(chat_id))
            return
        if not _validate_symbol(symbol):
            await query.edit_message_text(f"❌ `{symbol}` not found.",
                                          reply_markup=_back_keyboard(chat_id))
            return

        server_time = _get_server_time()
        close_time = _next_m5_close_time(server_time)
        send_time = close_time - timedelta(seconds=TELEGRAM_SEND_BEFORE_CLOSE_SEC)
        wait_seconds = max(0, (send_time - server_time).total_seconds())

        await query.edit_message_text(
            f"⏳ *{symbol}* M5 close at `{close_time.strftime('%H:%M:%S')}` UTC\n"
            f"Prediction in ~{int(wait_seconds)}s...",
            parse_mode="Markdown",
        )

        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds)

        try:
            pred = _run_prediction(symbol)
            msg = _format_prediction(pred)
            await query.edit_message_text(msg, parse_mode="Markdown",
                                          reply_markup=_back_keyboard(chat_id))
        except Exception as e:
            await query.edit_message_text(f"❌ Error: {e}",
                                          reply_markup=_back_keyboard(chat_id))

    # ----- Multi-Symbol -----
    elif data == "multi_symbol":
        if not _is_market_open():
            await query.edit_message_text(
                _market_closed_msg(), parse_mode="Markdown",
                reply_markup=_back_keyboard(chat_id))
            return
        await query.edit_message_text("📈 *Scanning all symbols...*", parse_mode="Markdown")
        results = []
        for sym in SYMBOLS:
            try:
                if not _validate_symbol(sym):
                    results.append(f"  {sym}: ❌ N/A")
                    continue
                pred = _run_prediction(sym)
                d = "🟢" if pred["suggested_direction"] == "UP" else "🔴"
                conf = pred["final_confidence_percent"]
                kelly = pred.get("kelly_fraction_percent", 0)
                trade = pred["suggested_trade"]
                results.append(f"  {d} *{sym}*: {conf:.0f}% | K:{kelly:.1f}% | {trade}")
            except Exception as e:
                results.append(f"  {sym}: ⚠️ {str(e)[:25]}")

        msg = "📈 *Multi-Symbol Scan*\n━━━━━━━━━━━━━━━━━━━━━\n\n" + "\n".join(results)
        await query.edit_message_text(msg, parse_mode="Markdown",
                                      reply_markup=_back_keyboard(chat_id))

    # ----- Status -----
    elif data == "status":
        acc_status = _tracker.status()
        conf_status = _conf_tracker.status()
        stored_acc = _get_accuracy_snapshot()
        n_subs = sum(1 for s in _subscribers.values() if s.get("active"))
        msg = (
            f"📊 *Engine Status*\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🎯 Rolling Accuracy: *{acc_status['rolling_accuracy']:.1%}*\n"
            f"📝 Predictions: *{acc_status['predictions_tracked']}*\n"
            f"💾 Stored WR: *{stored_acc['overall_win_rate']:.1f}%* ({stored_acc['total']} signals)\n"
            f"📉 Recent {stored_acc['recent_count']}: *{stored_acc['recent_win_rate']:.1f}%*\n"
            f"📈 Conf Correlation: *{conf_status['confidence_correlation']:.3f}*\n"
            f"🔔 Alert: *{conf_status['alert']}*\n"
            f"🔄 Retrain: *{acc_status['should_retrain']}*\n\n"
            f"👥 Active Subscribers: *{n_subs}*\n"
            f"📡 Signals Today: *{_count_today_signals()}*"
        )
        await query.edit_message_text(msg, parse_mode="Markdown",
                                      reply_markup=_back_keyboard(chat_id))

    # ----- Regime -----
    elif data == "regime":
        if not _is_market_open():
            await query.edit_message_text(
                _market_closed_msg(), parse_mode="Markdown",
                reply_markup=_back_keyboard(chat_id))
            return
        try:
            data_dict = fetch_multi_timeframe(DEFAULT_SYMBOL, CANDLES_TO_FETCH)
            df = data_dict.get("M5")
            m15, h1 = data_dict.get("M15"), data_dict.get("H1")
            df = compute_features(df, m15_df=m15, h1_df=h1)
            regime = detect_regime(df)

            last = df.iloc[-1]
            adx = last.get("adx", 0)
            atr = last.get("atr_14", 0)
            vol_z = last.get("volatility_zscore", 0)
            comp = last.get("compression_ratio", 0)
            imb = last.get("imbalance_10", 0)

            regime_emoji = {"Trending": "📈", "Ranging": "↔️",
                           "High_Volatility": "🔥", "Low_Volatility": "❄️"}.get(regime, "❓")

            msg = (
                f"🌊 *Market Regime*\n━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"{regime_emoji} *{regime}*\n\n"
                f"ADX: *{adx:.1f}* | ATR: *{atr:.5f}*\n"
                f"Vol Z: *{vol_z:.2f}* | Comp: *{comp:.2f}*\n"
                f"Imbalance: *{imb:.0f}*"
            )
        except Exception as e:
            msg = f"❌ Error: {e}"

        await query.edit_message_text(msg, parse_mode="Markdown",
                                      reply_markup=_back_keyboard(chat_id))

    # ----- Model Info -----
    elif data == "model_info":
        msg = (
            "ℹ️ *V3 Architecture — 14 Layers*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "🏗 *Ensemble:* XGB + LGB + RF (5 seeds)\n"
            "🎯 *Labels:* Triple Barrier (1.5x/1.0x ATR)\n"
            "⚖️ *Training:* Purged WF-CV + time weights\n"
            "🧠 *Meta:* GBM + Sigmoid (14 features)\n"
            "📊 *Features:* 65 (multi-TF: M5/M15/H1)\n"
            "🌊 *Regime:* ADX+ATR+BB classifier → routing\n\n"
            "*V3 Backtest Results:*\n"
            "  📈 Overall: *56.2%* (leak-free)\n"
            "  🇬🇧 London: *58.6%*\n"
            "  ≥70% conf: *75.0%*\n"
            "  BSS: *+0.012* (beats naive)\n\n"
            "*10-Gate Filter:*\n"
            "  G1-Conf │ G2-Meta │ G3-Regime\n"
            "  G4-Stability │ G5-Variance\n"
            "  G6-Session │ G7-Cooldown\n"
            "  G8-Spread │ G9-Kelly │ G10-Circuit"
        )
        await query.edit_message_text(msg, parse_mode="Markdown",
                                      reply_markup=_back_keyboard(chat_id))

    # ----- Production State -----
    elif data == "prod_state":
        msg = get_state_summary()
        await query.edit_message_text(msg, parse_mode="Markdown",
                                      reply_markup=_back_keyboard(chat_id))

    # ----- Position Guide -----
    elif data == "position_guide":
        msg = (
            "💰 *Position Sizing (¼ Kelly)*\n━━━━━━━━━━━━━━━━━━━━━\n\n"
            "🔴 Kelly <1% → *0.5x*\n"
            "🟠 Kelly 1-2% → *0.8x*\n"
            "🟡 Kelly 2-3% → *1.2x*\n"
            "🟢 Kelly >3% → *1.5x*\n\n"
            "⚠️ Never exceed 1.5x\n"
            "⚠️ 3 losses in a row → pause"
        )
        await query.edit_message_text(msg, parse_mode="Markdown",
                                      reply_markup=_back_keyboard(chat_id))

    # ----- Today's Signals -----
    elif data == "today_signals":
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_sigs = [s for s in _signal_history
                      if s["time"].startswith(today)]

        if not today_sigs:
            msg = "📜 *Today's Signals*\n\nNo signals sent yet today."
        else:
            lines = [f"📜 *Today's Signals* ({len(today_sigs)} total)\n━━━━━━━━━━━━━━━━━━━━━\n"]
            for s in today_sigs[-20:]:  # Last 20
                t = s["time"][11:16]
                d = "🟢" if s["direction"] == "UP" else "🔴"
                lines.append(f"  {t} {d} *{s['symbol']}* {s['confidence']:.0f}% | {s['trade']}")
            msg = "\n".join(lines)

        await query.edit_message_text(msg, parse_mode="Markdown",
                                      reply_markup=_back_keyboard(chat_id))

    # ----- Today's Results (Outcome Feedback) -----
    elif data == "today_results":
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_res = [r for r in _outcome_results
                     if r["time"].startswith(today)]

        if not today_res:
            msg = "🎯 *Today's Results*\n\nNo results yet. Outcomes appear ~35s after each candle closes."
        else:
            wins = sum(1 for r in today_res if r["correct"])
            total = len(today_res)
            win_rate = wins / total * 100

            lines = [
                f"🎯 *Today's Results*",
                f"━━━━━━━━━━━━━━━━━━━━━",
                f"✅ Wins: *{wins}* | ❌ Losses: *{total - wins}*",
                f"📊 Win Rate: *{win_rate:.1f}%*",
                f"",
            ]
            for r in today_res[-15:]:
                t = r["time"][11:16]
                emoji = "✅" if r["correct"] else "❌"
                lines.append(
                    f"  {t} {emoji} *{r['symbol']}*: "
                    f"{r['predicted']}→{r['actual']} ({r['confidence']:.0f}%)"
                )
            msg = "\n".join(lines)

        await query.edit_message_text(msg, parse_mode="Markdown",
                                      reply_markup=_back_keyboard(chat_id))

def _count_today_signals():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return sum(1 for s in _signal_history if s["time"].startswith(today))


# Legacy command handlers
async def cmd_next(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    symbol = args[0].upper() if args else DEFAULT_SYMBOL

    if not _validate_symbol(symbol):
        await update.message.reply_text(f"❌ `{symbol}` not found.")
        return

    server_time = _get_server_time()
    close_time = _next_m5_close_time(server_time)
    send_time = close_time - timedelta(seconds=TELEGRAM_SEND_BEFORE_CLOSE_SEC)
    wait_seconds = max(0, (send_time - server_time).total_seconds())

    await update.message.reply_text(
        f"⏳ *{symbol}* M5 close at `{close_time.strftime('%H:%M:%S')}` UTC\n"
        f"Prediction in ~{int(wait_seconds)}s...",
        parse_mode="Markdown",
    )
    if wait_seconds > 0:
        await asyncio.sleep(wait_seconds)

    try:
        pred = _run_prediction(symbol)
        msg = _format_prediction(pred)
        chat_id = update.effective_chat.id
        await update.message.reply_text(msg, parse_mode="Markdown",
                                        reply_markup=_back_keyboard(chat_id))
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    acc_status = _tracker.status()
    conf_status = _conf_tracker.status()
    stored_acc = _get_accuracy_snapshot()
    await update.message.reply_text(
        f"📊 *Engine Status*\n\n"
        f"Accuracy: *{acc_status['rolling_accuracy']:.1%}*\n"
        f"Predictions: *{acc_status['predictions_tracked']}*\n"
        f"Stored WR: *{stored_acc['overall_win_rate']:.1f}%* ({stored_acc['total']} signals)\n"
        f"Recent {stored_acc['recent_count']}: *{stored_acc['recent_win_rate']:.1f}%*\n"
        f"Correlation: *{conf_status['confidence_correlation']:.3f}*\n"
        f"Retrain: *{acc_status['should_retrain']}*",
        parse_mode="Markdown",
    )


async def _send_admin_alert(bot: Bot, msg: str):
    '''Send alert to ADMIN_CHAT_ID only (not all subscribers).'''
    from config import ADMIN_CHAT_ID
    if not ADMIN_CHAT_ID:
        return
    try:
        await _safe_send_message(
            bot,
            int(ADMIN_CHAT_ID),
            f'🔔 *ADMIN ALERT*\n{msg}',
            parse_mode='Markdown')
    except Exception as e:
        log.error('Admin alert failed: %s', e)

async def _broadcast_to_subscribers(bot: Bot, text: str):
    '''Send message to all active subscribers.'''
    for str_cid, info in _subscribers.items():
        if info.get("active", False):
            try:
                await _safe_send_message(
                    bot,
                    int(str_cid),
                    text,
                    parse_mode="Markdown"
                )
            except Exception as e:
                log.error("Broadcast failed to %s: %s", str_cid, e)

async def _run_drift_check(bot: Bot, symbols):
    try:
        sym  = symbols[0] if symbols else DEFAULT_SYMBOL
        data = fetch_multi_timeframe(sym, 1000)
        df   = compute_features(data.get('M5'))
        from auto_learner import check_drift_triggers
        triggers = check_drift_triggers(sym, recent_df=df, feature_names=FEATURE_COLUMNS)
        if triggers:
            msg = f'⚠️ Drift detected: {triggers}\nLite retrain starting...'
            await _send_admin_alert(bot, msg)
            from auto_learner import retrain_lite
            import threading
            threading.Thread(target=retrain_lite, args=(sym,), kwargs={"validate": True}, daemon=True).start()
    except Exception as e:
        log.warning('Drift check coroutine failed: %s', e)

async def _weekly_retrain_job(app: Application):
    '''Lite retrain every Sunday 21:15 UTC. Notifies admin. Never blocks.'''
    bot = app.bot
    while True:
        try:
            now = datetime.now(timezone.utc)
            days_left = (6 - now.weekday()) % 7
            if days_left == 0 and (now.hour > 21 or (now.hour == 21 and now.minute >= 15)):
                days_left = 7
            target = (now + timedelta(days=days_left)).replace(
                hour=21, minute=15, second=0, microsecond=0)
            await asyncio.sleep((target - now).total_seconds())

            await _send_admin_alert(bot, f'🔄 Weekly lite retrain starting for {DEFAULT_SYMBOL}...')

            result = {}
            def _run():
                from auto_learner import retrain_lite
                result['ok'] = retrain_lite(DEFAULT_SYMBOL, validate=True)
            import threading
            t = threading.Thread(target=_run, daemon=True)
            t.start(); t.join(timeout=600)

            ok = result.get('ok', False)
            await _send_admin_alert(bot, f"{'✅' if ok else '❌'} Weekly retrain {'complete' if ok else 'FAILED'}.")
            await asyncio.sleep(60)
        except Exception as e:
            log.exception('Weekly retrain job error: %s', e)
            await asyncio.sleep(3600)


# =============================================================================
#  Main
# =============================================================================

async def post_init(app: Application):
    """Start background jobs after bot initializes."""
    asyncio.create_task(_pending_outcome_job(app))
    asyncio.create_task(_auto_signal_job(app))
    asyncio.create_task(_daily_report_job(app))
    asyncio.create_task(_weekly_report_job(app))
    asyncio.create_task(_weekly_retrain_job(app))
    record_startup()
    log.info("All background jobs scheduled (auto-signal + daily + weekly + retrain).")

def main():
    if not TELEGRAM_BOT_TOKEN:
        print('[X] STARTUP FAILED: TELEGRAM_BOT_TOKEN is not set.')
        print('Copy .env.example to .env and fill your values.')
        return

    # Warn (but don't block) if MT5 credentials are missing
    if not MT5_LOGIN or int(MT5_LOGIN) == 0:
        log.info('MT5_LOGIN is 0 or missing -- MT5 will try default terminal.')
    if not MT5_SERVER:
        log.info('MT5_SERVER is empty -- MT5 will try default terminal.')

    if not _connect_mt5():
        print("WARNING: MT5 not connected.")

    _load_subscribers()
    _load_today_from_csv()

    try:
        load_models()
        log.info("Models pre-loaded.")
    except Exception as e:
        log.warning("Could not pre-load models: %s", e)

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(30)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("next", cmd_next))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CallbackQueryHandler(button_handler))

    log.info("Telegram bot starting (with auto-signal)...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
