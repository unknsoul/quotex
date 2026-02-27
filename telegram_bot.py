"""
Telegram Bot — Real-time M5 candle prediction delivery.

Commands:
    /start          — Welcome message
    /next <symbol>  — Wait until 5s before next M5 close, send prediction
    /status         — Show accuracy tracker status

Features:
    - Candle integrity verification (only closed candles)
    - Race condition safe (retrain never blocks predictions)
    - Full prediction logging
    - Confidence reliability score in output

Usage:
    set TELEGRAM_BOT_TOKEN=<your_token>
    python telegram_bot.py
"""

import asyncio
import logging
import math
from datetime import datetime, timezone, timedelta

import MetaTrader5 as mt5
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, ContextTypes,
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
    """Get MT5 server time (UTC)."""
    tick = mt5.symbol_info_tick(DEFAULT_SYMBOL)
    if tick is None:
        return datetime.now(timezone.utc)
    return datetime.fromtimestamp(tick.time, tz=timezone.utc)


def _next_m5_close_time(server_time: datetime) -> datetime:
    """Compute the next M5 candle close time."""
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

    # Candle integrity: verify last candle is closed
    if not verify_candle_closed(df):
        # Use second-to-last candle
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
#  Telegram Message Formatter
# =============================================================================

def _format_prediction(pred: dict) -> str:
    green = pred["green_probability_percent"]
    red = pred["red_probability_percent"]
    direction = "🟢 UP" if pred["suggested_direction"] == "UP" else "🔴 DOWN"
    conf = pred["final_confidence_percent"]
    meta = pred["meta_reliability_percent"]
    unc = pred["uncertainty_percent"]
    regime = pred.get("market_regime", pred.get("regime", "Unknown"))
    trade = pred["suggested_trade"]
    conf_level = pred["confidence_level"]
    conf_rel = pred.get("confidence_reliability_score", 0.0)

    lines = [
        f"📊 *{pred['symbol']}* | M5 Prediction",
        "",
        f"🟢 Green: *{green:.1f}%*  🔴 Red: *{red:.1f}%*",
        f"📈 Direction: *{direction}*",
        f"🎯 Confidence: *{conf:.1f}%* ({conf_level})",
        f"🔬 Meta Reliability: *{meta:.1f}%*",
        f"📉 Uncertainty: *{unc:.1f}%*",
        f"🌊 Regime: *{regime}*",
        f"📊 Confidence Reliability: *{conf_rel:.2f}*",
        f"💡 Suggestion: *{trade}*",
    ]

    if pred.get("risk_warnings"):
        lines.append("")
        lines.append("⚠️ " + " | ".join(pred["risk_warnings"]))

    return "\n".join(lines)


# =============================================================================
#  Bot Commands
# =============================================================================

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *Adaptive Candle Probability Engine*\n\n"
        "Commands:\n"
        "  /next EURUSD — Prediction 5s before next M5 close\n"
        "  /status — Accuracy & confidence tracker\n\n"
        "Features:\n"
        "  • 5-seed XGBoost ensemble\n"
        "  • Uncertainty-adjusted confidence\n"
        "  • Leakage-free OOF meta model\n"
        "  • Only closed candle data used",
        parse_mode="Markdown",
    )


async def cmd_next(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    symbol = args[0].upper() if args else DEFAULT_SYMBOL

    if not _validate_symbol(symbol):
        await update.message.reply_text(f"❌ Symbol `{symbol}` not found on MT5.")
        return

    server_time = _get_server_time()
    close_time = _next_m5_close_time(server_time)
    send_time = close_time - timedelta(seconds=TELEGRAM_SEND_BEFORE_CLOSE_SEC)
    wait_seconds = (send_time - server_time).total_seconds()

    if wait_seconds < 0:
        wait_seconds = 0

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
        await update.message.reply_text(msg, parse_mode="Markdown")
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

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("next", cmd_next))
    app.add_handler(CommandHandler("status", cmd_status))

    log.info("Telegram bot starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
