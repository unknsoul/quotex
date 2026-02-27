"""
Telegram Bot — Interactive M5 candle prediction delivery.

Features:
    - Interactive button menu (no typing commands)
    - Instant prediction (no wait) or scheduled (before candle close)
    - Multi-symbol support via buttons
    - Model status, regime info, position sizing
    - Candle integrity verification (only closed candles)
    - Race condition safe (retrain never blocks predictions)
    - Full prediction logging

Usage:
    set TELEGRAM_BOT_TOKEN=<your_token>
    python telegram_bot.py
"""

import asyncio
import logging
import math
from datetime import datetime, timezone, timedelta

import MetaTrader5 as mt5
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, ContextTypes,
)

from config import (
    DEFAULT_SYMBOL, CANDLES_TO_FETCH,
    MT5_PATH, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER,
    TELEGRAM_BOT_TOKEN, TELEGRAM_SEND_BEFORE_CLOSE_SEC,
    LOG_LEVEL, LOG_FORMAT,
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
_retrain_lock = asyncio.Lock()

# Supported symbols
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "GBPJPY"]


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
    """Full prediction pipeline using only closed candles."""
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

def _main_menu_keyboard():
    """Main interactive menu with all options."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔮 Predict Now", callback_data="predict_now"),
         InlineKeyboardButton("⏳ Next Candle", callback_data="predict_next")],
        [InlineKeyboardButton("📊 Status", callback_data="status"),
         InlineKeyboardButton("🌊 Regime", callback_data="regime")],
        [InlineKeyboardButton("📈 Multi-Symbol", callback_data="multi_symbol"),
         InlineKeyboardButton("ℹ️ Model Info", callback_data="model_info")],
        [InlineKeyboardButton("💰 Position Guide", callback_data="position_guide")],
    ])


def _symbol_keyboard(action_prefix="predict"):
    """Symbol selection keyboard."""
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


def _back_keyboard():
    """Simple back to menu button."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔙 Menu", callback_data="menu"),
         InlineKeyboardButton("🔮 Predict Again", callback_data="predict_now")]
    ])


# =============================================================================
#  Message Formatters
# =============================================================================

def _format_prediction(pred: dict) -> str:
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
    conf_rel = pred.get("confidence_reliability_score", 0.0)

    # Position sizing
    if kelly > 3:
        size_emoji, size_label = "🟢", "1.5x (Strong)"
    elif kelly > 2:
        size_emoji, size_label = "🟡", "1.2x (Moderate)"
    elif kelly > 1:
        size_emoji, size_label = "🟠", "0.8x (Light)"
    else:
        size_emoji, size_label = "🔴", "0.5x (Minimal)"

    # Confidence bar (visual)
    filled = int(conf / 10)
    conf_bar = "█" * filled + "░" * (10 - filled)

    lines = [
        f"📊 *{pred['symbol']}* — M5 Prediction",
        f"━━━━━━━━━━━━━━━━━━━━━",
        "",
        f"📈 Direction: *{direction}*",
        f"🟢 Green: *{green:.1f}%*  |  🔴 Red: *{red:.1f}%*",
        "",
        f"🎯 Confidence: *{conf:.1f}%*",
        f"   `[{conf_bar}]`",
        f"   Adaptive: *{adaptive:.1f}%* ({conf_level})",
        "",
        f"🔬 Meta Reliability: *{meta:.1f}%*",
        f"📉 Uncertainty: *{unc:.1f}%*",
        f"🌊 Regime: *{regime}*",
        "",
        f"💰 Kelly: *{kelly:.1f}%* → {size_emoji} *{size_label}*",
        f"📊 Reliability Score: *{conf_rel:.2f}*",
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
#  Bot Handlers
# =============================================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Welcome message with interactive menu."""
    welcome = (
        "🤖 *QUOTEX LORD — Adaptive Candle Engine*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n\n"
        "🧠 *v6 Features:*\n"
        "  • 55 structural + microstructure features\n"
        "  • 5-seed temporal XGBoost ensemble\n"
        "  • Sigmoid-calibrated meta model\n"
        "  • Kelly criterion position sizing\n"
        "  • Adaptive confidence (rolling calibration)\n"
        "  • Uncertainty-adjusted predictions\n\n"
        "📊 *Live Expectation: ~63%*\n"
        "🎯 *High-Conf (≥80%): ~85%*\n\n"
        "Choose an option below 👇"
    )
    await update.message.reply_text(
        welcome,
        parse_mode="Markdown",
        reply_markup=_main_menu_keyboard(),
    )


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all inline keyboard button presses."""
    query = update.callback_query
    await query.answer()
    data = query.data

    # ----- Menu -----
    if data == "menu":
        await query.edit_message_text(
            "🤖 *QUOTEX LORD — Main Menu*\n\nChoose an option 👇",
            parse_mode="Markdown",
            reply_markup=_main_menu_keyboard(),
        )

    # ----- Predict Now (instant) -----
    elif data == "predict_now":
        await query.edit_message_text(
            "🔮 *Instant Prediction*\n\nSelect symbol:",
            parse_mode="Markdown",
            reply_markup=_symbol_keyboard("now"),
        )

    elif data.startswith("now_"):
        symbol = data.replace("now_", "")
        await query.edit_message_text(f"⏳ Running prediction for *{symbol}*...", parse_mode="Markdown")
        try:
            pred = _run_prediction(symbol)
            msg = _format_prediction(pred)
            await query.edit_message_text(msg, parse_mode="Markdown", reply_markup=_back_keyboard())
        except Exception as e:
            log.exception("Prediction failed for %s", symbol)
            await query.edit_message_text(
                f"❌ Error: {e}",
                reply_markup=_back_keyboard(),
            )

    # ----- Predict Next Candle -----
    elif data == "predict_next":
        await query.edit_message_text(
            "⏳ *Next Candle Prediction*\n\nSelect symbol:",
            parse_mode="Markdown",
            reply_markup=_symbol_keyboard("next"),
        )

    elif data.startswith("next_"):
        symbol = data.replace("next_", "")
        if not _validate_symbol(symbol):
            await query.edit_message_text(
                f"❌ Symbol `{symbol}` not found on MT5.",
                reply_markup=_back_keyboard(),
            )
            return

        server_time = _get_server_time()
        close_time = _next_m5_close_time(server_time)
        send_time = close_time - timedelta(seconds=TELEGRAM_SEND_BEFORE_CLOSE_SEC)
        wait_seconds = max(0, (send_time - server_time).total_seconds())

        await query.edit_message_text(
            f"⏳ Waiting for *{symbol}* M5 candle close at "
            f"`{close_time.strftime('%H:%M:%S')}` UTC\n"
            f"Prediction in ~{int(wait_seconds)}s...",
            parse_mode="Markdown",
        )

        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds)

        try:
            pred = _run_prediction(symbol)
            msg = _format_prediction(pred)
            await query.edit_message_text(msg, parse_mode="Markdown", reply_markup=_back_keyboard())
        except Exception as e:
            log.exception("Prediction failed for %s", symbol)
            await query.edit_message_text(f"❌ Error: {e}", reply_markup=_back_keyboard())

    # ----- Multi-Symbol Scan -----
    elif data == "multi_symbol":
        await query.edit_message_text("📈 *Scanning all symbols...*", parse_mode="Markdown")
        results = []
        for sym in SYMBOLS:
            try:
                if not _validate_symbol(sym):
                    results.append(f"  {sym}: ❌ Not available")
                    continue
                pred = _run_prediction(sym)
                d = "🟢" if pred["suggested_direction"] == "UP" else "🔴"
                conf = pred["final_confidence_percent"]
                kelly = pred.get("kelly_fraction_percent", 0)
                trade = pred["suggested_trade"]
                results.append(f"  {d} *{sym}*: {conf:.0f}% conf | Kelly {kelly:.1f}% | {trade}")
            except Exception as e:
                results.append(f"  {sym}: ⚠️ {str(e)[:30]}")

        msg = "📈 *Multi-Symbol Scan*\n━━━━━━━━━━━━━━━━━━━━━\n\n" + "\n".join(results)
        await query.edit_message_text(msg, parse_mode="Markdown", reply_markup=_back_keyboard())

    # ----- Status -----
    elif data == "status":
        acc_status = _tracker.status()
        conf_status = _conf_tracker.status()
        msg = (
            f"📊 *Engine Status*\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🎯 Rolling Accuracy: *{acc_status['rolling_accuracy']:.1%}*\n"
            f"📝 Predictions Tracked: *{acc_status['predictions_tracked']}*\n"
            f"📈 Confidence Correlation: *{conf_status['confidence_correlation']:.3f}*\n"
            f"🔔 Alert: *{conf_status['alert']}*\n"
            f"🔄 Should Retrain: *{acc_status['should_retrain']}*"
        )
        await query.edit_message_text(msg, parse_mode="Markdown", reply_markup=_back_keyboard())

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
                f"🌊 *Market Regime Analysis*\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"Regime: {regime_emoji} *{regime}*\n\n"
                f"📊 ADX: *{adx:.1f}* (trend strength)\n"
                f"📉 ATR: *{atr:.5f}* (volatility)\n"
                f"⚡ Vol Z-Score: *{vol_z:.2f}*\n"
                f"🔄 Compression: *{comp:.2f}*\n"
                f"⚖️ Imbalance (10): *{imb:.0f}*\n\n"
            )
            if regime == "Trending":
                msg += "💡 _Strong directional movement. Follow the trend._"
            elif regime == "Ranging":
                msg += "💡 _Sideways market. Mean-reversion plays._"
            elif regime == "High_Volatility":
                msg += "💡 _Volatile. Reduce position size._"
            else:
                msg += "💡 _Low vol. Wait for breakout or reduce expectations._"

        except Exception as e:
            msg = f"❌ Regime analysis failed: {e}"

        await query.edit_message_text(msg, parse_mode="Markdown", reply_markup=_back_keyboard())

    # ----- Model Info -----
    elif data == "model_info":
        msg = (
            "ℹ️ *Model Architecture*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n\n"
            "🏗️ *Primary:* 5-seed XGBoost temporal ensemble\n"
            "   Seeds: 40% / 40% / 70% / 70% / 100% windows\n"
            "   Each calibrated with isotonic regression\n\n"
            "🧠 *Meta:* GradientBoosting + Sigmoid calibration\n"
            "   11 features (OOF, no leakage)\n\n"
            "⚖️ *Weight:* Logistic regression\n"
            "   Combines primary + meta + regime + uncertainty\n\n"
            "📊 *Features:* 55 total\n"
            "   Trend (5) | Momentum (5) | Volatility (4)\n"
            "   Candle (7) | Context (5) | Multi-TF (4)\n"
            "   Regime (3) | Structural (12) | Micro (8)\n"
            "   + HTF alignment\n\n"
            "🎯 *Backtest:* 72.5% (all bars)\n"
            "   ≥80% conf: 85.4% accuracy\n"
            "   Live expectation: ~63%"
        )
        await query.edit_message_text(msg, parse_mode="Markdown", reply_markup=_back_keyboard())

    # ----- Position Guide -----
    elif data == "position_guide":
        msg = (
            "💰 *Position Sizing Guide*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Based on Kelly Criterion (¼ Kelly):\n\n"
            "🔴 Kelly < 1%  → *0.5x* (minimal)\n"
            "🟠 Kelly 1-2%  → *0.8x* (light)\n"
            "🟡 Kelly 2-3%  → *1.2x* (moderate)\n"
            "🟢 Kelly > 3%  → *1.5x* (strong)\n\n"
            "📐 *Kelly Formula:*\n"
            "`f = 0.25 × (2p - 1)`\n"
            "where p = win probability\n\n"
            "⚠️ *Rules:*\n"
            "  • Never exceed 1.5x\n"
            "  • High uncertainty → reduce by 50%\n"
            "  • After 3 consecutive losses → pause\n"
            "  • After sub-50% rolling → reduce all sizes"
        )
        await query.edit_message_text(msg, parse_mode="Markdown", reply_markup=_back_keyboard())


# Legacy command handlers (still work if typed)
async def cmd_next(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    symbol = args[0].upper() if args else DEFAULT_SYMBOL

    if not _validate_symbol(symbol):
        await update.message.reply_text(f"❌ Symbol `{symbol}` not found on MT5.")
        return

    server_time = _get_server_time()
    close_time = _next_m5_close_time(server_time)
    send_time = close_time - timedelta(seconds=TELEGRAM_SEND_BEFORE_CLOSE_SEC)
    wait_seconds = max(0, (send_time - server_time).total_seconds())

    await update.message.reply_text(
        f"⏳ Waiting for *{symbol}* M5 candle close at "
        f"`{close_time.strftime('%H:%M:%S')}` UTC\n"
        f"Prediction in ~{int(wait_seconds)}s...",
        parse_mode="Markdown",
    )

    if wait_seconds > 0:
        await asyncio.sleep(wait_seconds)

    try:
        pred = _run_prediction(symbol)
        msg = _format_prediction(pred)
        await update.message.reply_text(msg, parse_mode="Markdown", reply_markup=_back_keyboard())
    except Exception as e:
        log.exception("Prediction failed for %s", symbol)
        await update.message.reply_text(f"❌ Error: {e}")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    acc_status = _tracker.status()
    conf_status = _conf_tracker.status()
    await update.message.reply_text(
        f"📊 *Engine Status*\n\n"
        f"Rolling Accuracy: *{acc_status['rolling_accuracy']:.1%}*\n"
        f"Predictions Tracked: *{acc_status['predictions_tracked']}*\n"
        f"Confidence Correlation: *{conf_status['confidence_correlation']:.3f}*\n"
        f"Conf Alert: *{conf_status['alert']}*\n"
        f"Should Retrain: *{acc_status['should_retrain']}*",
        parse_mode="Markdown",
    )


# =============================================================================
#  Main
# =============================================================================

def main():
    if not TELEGRAM_BOT_TOKEN:
        print("ERROR: Set TELEGRAM_BOT_TOKEN environment variable.")
        print("  Get one from @BotFather on Telegram.")
        return

    if not _connect_mt5():
        print("WARNING: MT5 not connected. Predictions will fail.")

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
        .build()
    )

    # Command handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("next", cmd_next))
    app.add_handler(CommandHandler("status", cmd_status))

    # Button callback handler
    app.add_handler(CallbackQueryHandler(button_handler))

    log.info("Telegram bot starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
