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

log = logging.getLogger("telegram_bot")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

_tracker = AccuracyTracker()
_conf_tracker = ConfidenceCorrelationTracker()

# Supported symbols
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "GBPJPY"]

# Subscriber storage
SUBS_FILE = os.path.join(LOG_DIR, "subscribers.json")
_subscribers = {}  # {chat_id: {"symbols": [...], "active": True}}
_last_predictions = {}  # {symbol: {pred_dict, timestamp}}
_signal_history = []  # [{symbol, direction, confidence, timestamp, result}]
_outcome_results = []  # [{symbol, predicted, actual, correct, confidence, time}]


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
    tick = mt5.symbol_info_tick(DEFAULT_SYMBOL)
    if tick is None:
        return datetime.now(timezone.utc)
    return datetime.fromtimestamp(tick.time, tz=timezone.utc)


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
        [InlineKeyboardButton("🎯 Results", callback_data="today_results")],
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
    green = pred["green_probability_percent"]
    red = pred["red_probability_percent"]
    direction = "🟢 UP" if pred["suggested_direction"] == "UP" else "🔴 DOWN"
    conf = pred["final_confidence_percent"]
    adaptive = pred.get("adaptive_confidence_percent", conf)
    meta = pred["meta_reliability_percent"]
    unc = pred["uncertainty_percent"]
    kelly = pred.get("kelly_fraction_percent", 0.0)
    regime = pred.get("market_regime", pred.get("regime", "Unknown"))
    trade = pred["suggested_trade"]
    conf_level = pred["confidence_level"]

    if kelly > 3:
        size_emoji, size_label = "🟢", "1.5x"
    elif kelly > 2:
        size_emoji, size_label = "🟡", "1.2x"
    elif kelly > 1:
        size_emoji, size_label = "🟠", "0.8x"
    else:
        size_emoji, size_label = "🔴", "0.5x"

    filled = int(conf / 10)
    conf_bar = "█" * filled + "░" * (10 - filled)

    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    header = "🤖 AUTO-SIGNAL" if is_auto else "📊 PREDICTION"

    regime_emoji = {"Trending": "📈", "Ranging": "↔️",
                   "High_Volatility": "🔥", "Low_Volatility": "❄️"}.get(regime, "❓")

    lines = [
        f"*{header}* — {now}",
        f"━━━━━━━━━━━━━━━━━━━━━",
        f"📊 *{pred['symbol']}* M5 | {regime_emoji} {regime}",
        "",
        f"📈 *{direction}*",
        f"🟢 {green:.1f}%  |  🔴 {red:.1f}%",
        "",
        f"🎯 Confidence: *{conf:.1f}%* ({conf_level})",
        f"`[{conf_bar}]`",
        f"📎 Adaptive: *{adaptive:.1f}%*",
        "",
        f"🔬 Meta: *{meta:.1f}%* | Unc: *{unc:.1f}%*",
        f"💰 Kelly: *{kelly:.1f}%* → {size_emoji} Size *{size_label}*",
        "",
        f"━━━━━━━━━━━━━━━━━━━━━",
        f"💡 *{trade}*",
    ]

    if pred.get("risk_warnings"):
        lines.append("")
        for w in pred["risk_warnings"]:
            lines.append(f"⚠️ {w}")

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

            predictions = {}
            for sym in needed_symbols:
                try:
                    pred = _run_prediction(sym)
                    predictions[sym] = pred
                    _last_predictions[sym] = {
                        "pred": pred,
                        "time": datetime.now(timezone.utc).isoformat(),
                    }
                    # Track for daily summary
                    _signal_history.append({
                        "symbol": sym,
                        "direction": pred["suggested_direction"],
                        "confidence": pred["final_confidence_percent"],
                        "kelly": pred.get("kelly_fraction_percent", 0),
                        "trade": pred["suggested_trade"],
                        "time": datetime.now(timezone.utc).isoformat(),
                    })
                except Exception as e:
                    log.error("Auto-signal prediction failed for %s: %s", sym, e)

            # Send to each subscriber
            for chat_id, info in active_subs.items():
                symbols = info.get("symbols", [DEFAULT_SYMBOL])
                for sym in symbols:
                    if sym in predictions:
                        msg = _format_prediction(predictions[sym], is_auto=True)
                        try:
                            await bot.send_message(
                                chat_id=int(chat_id),
                                text=msg,
                                parse_mode="Markdown",
                            )
                        except Exception as e:
                            log.error("Failed to send to %s: %s", chat_id, e)

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
        "  • 66 features (structural + microstructure)\n"
        "  • Hybrid ensemble (3 XGB + 2 ExtraTrees)\n"
        "  • Kelly position sizing\n"
        "  • Outcome tracking (win/loss results)\n\n"
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

        _subscribers[cid] = {"symbols": symbols, "active": True}
        _save_subscribers()

        sym_list = ", ".join(symbols)
        await query.edit_message_text(
            f"✅ *Auto-Signal Activated!*\n\n"
            f"📊 Symbols: *{sym_list}*\n"
            f"⏰ Predictions every 5 minutes\n"
            f"   (5s before each M5 candle close)\n\n"
            f"Signals will start at the next candle.\n"
            f"Send /start anytime to see the menu.",
            parse_mode="Markdown",
            reply_markup=_main_menu_keyboard(chat_id),
        )
        log.info("Subscriber added: %s -> %s", cid, sym_list)

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
            "ℹ️ *Model Architecture*\n━━━━━━━━━━━━━━━━━━━━━\n\n"
            "🏗️ *Primary:* 5-seed temporal XGB\n"
            "   Windows: 40/40/70/70/100%\n\n"
            "🧠 *Meta:* GBM + Sigmoid (11 feat)\n"
            "⚖️ *Weight:* Logistic regression\n"
            "📊 *Features:* 55 total\n\n"
            "🎯 Backtest: 72.5% (all bars)\n"
            "   ≥80% conf: 85.4%\n"
            "   Live exp: ~63%"
        )
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
    """Start auto-signal background job after bot initializes."""
    asyncio.create_task(_auto_signal_job(app))
    log.info("Auto-signal background job scheduled.")


def main():
    if not TELEGRAM_BOT_TOKEN:
        print("ERROR: Set TELEGRAM_BOT_TOKEN environment variable.")
        return

    if not _connect_mt5():
        print("WARNING: MT5 not connected.")

    _load_subscribers()

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
