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
    LOG_LEVEL, LOG_FORMAT, LOG_DIR,
)
from data_collector import fetch_multi_timeframe
from feature_engineering import compute_features, FEATURE_COLUMNS
from regime_detection import detect_regime
from predict_engine import (
    predict, load_models, log_prediction,
    verify_candle_closed, update_prediction_history,
)
from risk_filter import check_warnings
from stability import AccuracyTracker, ConfidenceCorrelationTracker
from production_state import (
    record_startup, record_prediction, record_error,
    get_state_summary,
)

log = logging.getLogger("telegram_bot")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

_tracker = AccuracyTracker()
_conf_tracker = ConfidenceCorrelationTracker()

# Supported symbols
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "GBPJPY"]

# Subscriber storage
SUBS_FILE = os.path.join(LOG_DIR, "subscribers.json")
SIGNAL_CSV = os.path.join(LOG_DIR, "signal_outcomes.csv")
PREDICTION_CSV = os.path.join(LOG_DIR, "all_predictions.csv")
_subscribers = {}  # {chat_id: {"symbols": [...], "active": True, "mode": "all"|"high"}}
_last_predictions = {}  # {symbol: {pred_dict, timestamp}}
_signal_history = []  # [{symbol, direction, confidence, timestamp, result}]
_outcome_results = []  # [{symbol, predicted, actual, correct, confidence, time}]
_consecutive_losses = {}  # {symbol: int}
_cooldown_until = {}  # {symbol: datetime}
_last_regime = {}  # {symbol: regime_name} — for transition detection
_regime_transition_skip = {}  # {symbol: int} — candles to skip

# Signal quality thresholds
MIN_CONFIDENCE_FILTER = 55.0
HIGH_CONF_THRESHOLD = 70.0
COOLDOWN_CANDLES = 3
MAX_CONSECUTIVE_LOSSES = 3
REGIME_SKIP_CANDLES = 2  # skip after regime transition


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
    data = fetch_multi_timeframe(symbol, CANDLES_TO_FETCH)
    df = data.get("M5")
    if df is None or len(df) == 0:
        raise RuntimeError(f"No M5 data for {symbol}")
    m15, h1 = data.get("M15"), data.get("H1")

    df = compute_features(df, m15_df=m15, h1_df=h1)
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
    """Format a single prediction — compact version."""
    direction = "🟢 UP" if pred["suggested_direction"] == "UP" else "🔴 DOWN"
    conf = pred["final_confidence_percent"]
    green = pred["green_probability_percent"]
    kelly = pred.get("kelly_fraction_percent", 0.0)
    regime = pred.get("market_regime", "Unknown")
    trade = pred["suggested_trade"]
    latency = pred.get("latency_ms", -1)

    regime_emoji = {"↑Trend": "📈", "Trending": "📈", "Ranging": "↔️",
                   "High_Volatility": "🔥", "Low_Volatility": "❄️"}.get(regime, "❓")

    now_utc = datetime.now(timezone.utc)
    ist = now_utc + timedelta(hours=5, minutes=30)
    time_str = f"{now_utc.strftime('%H:%M')} UTC / {ist.strftime('%H:%M')} IST"

    header = "🤖 AUTO" if is_auto else "📊 PRED"

    return (
        f"*{header}* — {time_str}\n"
        f"📊 *{pred['symbol']}* M5 {regime_emoji}\n"
        f"{direction} | Prob *{green:.0f}%* | Conf *{conf:.0f}%*\n"
        f"💰 Kelly *{kelly:.1f}%* | 💡 *{trade}*"
    )


def _format_combined_signal(predictions: dict, filtered: dict) -> str:
    """Industry-grade combined signal message."""
    now_utc = datetime.now(timezone.utc)
    ist = now_utc + timedelta(hours=5, minutes=30)
    time_str = f"{now_utc.strftime('%H:%M')} UTC | {ist.strftime('%H:%M')} IST"

    # Win streak from recent outcomes
    recent = _outcome_results[-20:] if _outcome_results else []
    total_r = len(recent)
    wins_r = sum(1 for r in recent if r["correct"])
    wr = f"{wins_r}/{total_r}" if total_r > 0 else "—"

    lines = [
        f"🤖 *QUOTEX LORD* — {time_str}",
        "━━━━━━━━━━━━━━━━━━━━━━━",
    ]

    for sym, pred in sorted(predictions.items()):
        d = "🟢" if pred["suggested_direction"] == "UP" else "🔴"
        arrow = "⬆" if pred["suggested_direction"] == "UP" else "⬇"
        conf = pred["final_confidence_percent"]
        green = pred["green_probability_percent"]
        kelly = pred.get("kelly_fraction_percent", 0.0)
        trade = pred["suggested_trade"]
        regime = pred.get("market_regime", "")
        r_e = {"↑Trend": "📈", "Trending": "📈", "Ranging": "↔️",
               "High_Volatility": "🔥", "Low_Volatility": "❄️"}.get(regime, "")

        # Position size from Kelly
        if kelly > 3:
            size = "🟢 1.5x"
        elif kelly > 2:
            size = "🟡 1.2x"
        elif kelly > 1:
            size = "🟠 0.8x"
        else:
            size = "⚪ 0.5x"

        # Confidence bar
        filled = int(conf / 10)
        bar = "█" * filled + "░" * (10 - filled)

        lines.append(
            f"\n{d} *{sym}* {r_e} {arrow} *{pred['suggested_direction']}*\n"
            f"  `[{bar}]` *{conf:.0f}%*\n"
            f"  Prob *{green:.0f}%* │ Kelly *{kelly:.1f}%* │ {size}\n"
            f"  💡 *{trade}*"
        )

    # Footer
    lines.append(f"\n━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"🎯 Recent: *{wr}* correct")

    if filtered:
        skipped = ", ".join(f"{s}" for s in filtered)
        lines.append(f"⏭ Skipped: _{skipped}_")

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

            # Weekend check (Sat/Sun)
            now_utc = datetime.now(timezone.utc)
            if now_utc.weekday() >= 5:  # Saturday=5, Sunday=6
                log.info("Auto-signal: market closed (weekend). Sleeping 30 min.")
                await asyncio.sleep(1800)
                continue

            predictions = {}
            filtered_out = {}
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

                    # 4. Signal quality filters
                    if pred.get("error"):
                        filtered_out[sym] = pred["error"]
                        continue
                    if conf < MIN_CONFIDENCE_FILTER:
                        filtered_out[sym] = f"low-conf ({conf:.0f}%)"
                        continue
                    if trade == "HOLD":
                        filtered_out[sym] = "HOLD"
                        continue
                    if sym in filtered_out:  # regime transition
                        continue

                    predictions[sym] = pred
                    _last_predictions[sym] = {
                        "pred": pred,
                        "time": datetime.now(timezone.utc).isoformat(),
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

            # Send ONE combined message per subscriber (respecting alert mode)
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

                msg = _format_combined_signal(sub_preds, filtered_out)
                try:
                    await bot.send_message(
                        chat_id=int(chat_id),
                        text=msg,
                        parse_mode="Markdown",
                    )
                except Exception as e:
                    log.error("Failed to send to %s: %s", chat_id, e)

            if filtered_out:
                log.info("Filtered signals: %s", filtered_out)

            # Schedule outcome check after candle closes (+30s buffer)
            asyncio.create_task(
                _check_outcomes(bot, predictions, active_subs)
            )

            # Small delay to avoid double-sending
            await asyncio.sleep(15)

        except Exception as e:
            log.exception("Auto-signal job error: %s", e)
            await asyncio.sleep(30)


async def _check_outcomes(bot: Bot, predictions: dict, subscribers: dict):
    """
    Wait for the current candle to close, then check if predictions were correct.
    Send result notifications to subscribers.
    """
    # Wait for candle to close + 30s buffer for data availability
    await asyncio.sleep(TELEGRAM_SEND_BEFORE_CLOSE_SEC + 30)

    results_msgs = []
    for sym, pred in predictions.items():
        try:
            data = fetch_multi_timeframe(sym, 10)
            df = data.get("M5")
            if df is None or len(df) < 2:
                continue

            # The last closed candle is the one we predicted
            last_candle = df.iloc[-2]  # second to last (fully closed)
            actual_green = last_candle["close"] > last_candle["open"]
            predicted_up = pred["suggested_direction"] == "UP"
            correct = (predicted_up == actual_green)

            emoji = "✅" if correct else "❌"
            conf = pred["final_confidence_percent"]

            _outcome_results.append({
                "symbol": sym,
                "predicted": pred["suggested_direction"],
                "actual": "UP" if actual_green else "DOWN",
                "correct": correct,
                "confidence": conf,
                "time": datetime.now(timezone.utc).isoformat(),
            })

            # Update prediction history for adaptive confidence
            update_prediction_history(
                pred["green_probability_percent"] / 100,
                correct,
                confidence=conf,
            )
            # Update production state
            record_prediction(correct, conf, pred.get("latency_ms", 0))

            # Consecutive loss tracking + cooldown
            if correct:
                _consecutive_losses[sym] = 0
            else:
                _consecutive_losses[sym] = _consecutive_losses.get(sym, 0) + 1
                if _consecutive_losses[sym] >= MAX_CONSECUTIVE_LOSSES:
                    cooldown_end = datetime.now(timezone.utc) + timedelta(minutes=5 * COOLDOWN_CANDLES)
                    _cooldown_until[sym] = cooldown_end
                    log.warning("COOLDOWN: %s has %d consecutive losses, pausing until %s",
                                sym, _consecutive_losses[sym], cooldown_end.strftime("%H:%M"))

            # CSV logging
            _log_outcome_csv(sym, pred, actual_green, correct)

            # Rolling win rate
            recent = _outcome_results[-50:]
            wins = sum(1 for r in recent if r["correct"])
            win_rate = wins / len(recent) * 100

            results_msgs.append(
                f"{emoji} *{sym}*: Predicted {pred['suggested_direction']} "
                f"→ Actual {'UP' if actual_green else 'DOWN'} "
                f"({conf:.0f}% conf) | Win rate: {win_rate:.0f}%"
            )
        except Exception as e:
            log.error("Outcome check failed for %s: %s", sym, e)

    if results_msgs:
        msg = "📋 *Signal Results*\n━━━━━━━━━━━━━━━━━━━━━\n\n" + "\n".join(results_msgs)

        for chat_id, info in subscribers.items():
            if info.get("active", False):
                try:
                    await bot.send_message(
                        chat_id=int(chat_id),
                        text=msg,
                        parse_mode="Markdown",
                    )
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
                await bot.send_message(
                    chat_id=int(chat_id), text=msg, parse_mode="Markdown",
                )
            except Exception as e:
                log.error("Broadcast failed for %s: %s", chat_id, e)


# =============================================================================
#  Bot Handlers
# =============================================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    welcome = (
        "🤖 *QUOTEX LORD — Auto-Signal Engine*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n\n"
        "🔔 *Auto-Signal Mode:*\n"
        "  Predictions sent automatically\n"
        "  5 seconds before every M5 candle!\n\n"
        "🧠 *v7 Engine:*\n"
        "  • 66 features + signal quality filter\n"
        "  • 5-seed temporal XGBoost ensemble\n"
        "  • Auto win/loss tracking + cooldown\n"
        "  • Daily report at 21:00 UTC\n\n"
        "📊 *Live Expectation: ~63%*\n"
        "🎯 *High-Conf (≥80%): ~85%*\n\n"
        "👇 *Tap Start Signals to begin!*"
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
        n_subs = sum(1 for s in _subscribers.values() if s.get("active"))
        msg = (
            f"📊 *Engine Status*\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🎯 Rolling Accuracy: *{acc_status['rolling_accuracy']:.1%}*\n"
            f"📝 Predictions: *{acc_status['predictions_tracked']}*\n"
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
            "ℹ️ *Model Architecture v7*\n━━━━━━━━━━━━━━━━━━━━━\n\n"
            "🏗️ *Primary:* 5-seed temporal XGB\n"
            "   Windows: 40/40/70/70/100%\n\n"
            "🧠 *Meta:* GBM + Sigmoid (11 feat)\n"
            "⚖️ *Weight:* Logistic regression\n"
            "📊 *Features:* 66 total (v7)\n\n"
            "🎯 Backtest: 72.4% (all bars)\n"
            "   ≥80% conf: 84.1%\n"
            "   Brier Skill: 0.2526\n"
            "   Live exp: ~63%"
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
    await update.message.reply_text(
        f"📊 *Engine Status*\n\n"
        f"Accuracy: *{acc_status['rolling_accuracy']:.1%}*\n"
        f"Predictions: *{acc_status['predictions_tracked']}*\n"
        f"Correlation: *{conf_status['confidence_correlation']:.3f}*\n"
        f"Retrain: *{acc_status['should_retrain']}*",
        parse_mode="Markdown",
    )


# =============================================================================
#  Main
# =============================================================================

async def post_init(app: Application):
    """Start background jobs after bot initializes."""
    asyncio.create_task(_auto_signal_job(app))
    asyncio.create_task(_daily_report_job(app))
    asyncio.create_task(_weekly_report_job(app))
    record_startup()
    log.info("All background jobs scheduled (auto-signal + daily + weekly).")


def main():
    if not TELEGRAM_BOT_TOKEN:
        print("ERROR: Set TELEGRAM_BOT_TOKEN environment variable.")
        return

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
