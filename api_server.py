"""
API Server — FastAPI endpoint for live predictions.

This is the ONLY entry point for users / frontend.
It fetches live candles from MT5, runs the full pipeline, and returns JSON.

Run:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import MetaTrader5 as mt5

from config import (
    DEFAULT_SYMBOL, CANDLES_TO_FETCH,
    MT5_PATH, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER,
    LOG_LEVEL, LOG_FORMAT,
)
from feature_engineering import compute_features, FEATURE_COLUMNS
from regime_detection import detect_regime, get_volatility_status
from predict_engine import predict, log_prediction
from risk_filter import apply_filters

log = logging.getLogger("api_server")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


# ─── Schemas ────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    symbol: str = Field(default=DEFAULT_SYMBOL, description="Trading symbol")


class PredictResponse(BaseModel):
    symbol: str
    timestamp: str
    market_regime: str
    green_probability: float
    red_probability: float
    threshold_used: float
    confidence_level: str
    suggested_trade: str
    volatility_status: str
    reason_summary: str


class SkippedResponse(BaseModel):
    status: str = "skipped"
    symbol: str
    reasons: list[str]
    timestamp: str


# ─── MT5 Helpers (local to API) ────────────────────────────────────────────

def _connect_mt5() -> bool:
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
        log.error("MT5 init failed: %s", mt5.last_error())
        return False
    log.info("MT5 connected.")
    return True


def _fetch_live(symbol: str, count: int = CANDLES_TO_FETCH):
    """Fetch live M5 candles from MT5."""
    import pandas as pd
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, count)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No data for {symbol}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


def _get_spread(symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    return float(info.spread) if info else 0.0


# ─── Reason Summary Builder ────────────────────────────────────────────────

def _reason_summary(row, regime: str) -> str:
    parts = [f"Regime: {regime}"]

    # EMA alignment
    if row["ema_20"] > row["ema_50"] > row["ema_200"]:
        parts.append("EMA bullish (20>50>200)")
    elif row["ema_20"] < row["ema_50"] < row["ema_200"]:
        parts.append("EMA bearish (20<50<200)")
    else:
        parts.append("EMAs mixed")

    # RSI
    rsi = row["rsi_14"]
    if rsi > 70:
        parts.append(f"RSI overbought ({rsi:.0f})")
    elif rsi < 30:
        parts.append(f"RSI oversold ({rsi:.0f})")
    else:
        parts.append(f"RSI {rsi:.0f}")

    # MACD
    parts.append("MACD+" if row["macd"] > row["macd_signal"] else "MACD-")

    # ADX
    adx = row["adx"]
    if adx > 40:
        parts.append(f"Strong trend (ADX {adx:.0f})")
    elif adx > 25:
        parts.append(f"Moderate trend (ADX {adx:.0f})")
    else:
        parts.append(f"Weak trend (ADX {adx:.0f})")

    # Patterns
    if row.get("engulfing") == 1:
        parts.append("Bull engulfing")
    elif row.get("engulfing") == -1:
        parts.append("Bear engulfing")
    if row.get("hammer") == 1:
        parts.append("Hammer")
    if row.get("shooting_star") == 1:
        parts.append("Shooting star")

    return "; ".join(parts)


# ─── App ────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not _connect_mt5():
        log.error("MT5 not available — predictions will fail.")
    yield
    mt5.shutdown()
    log.info("MT5 disconnected.")


app = FastAPI(
    title="Regime-Aware Next Candle Probability Engine",
    version="2.0.0",
    description="Predicts M5 candle direction with regime-aware dynamic thresholds.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    """
    Full prediction pipeline:
    1. Fetch live candles
    2. Compute features
    3. Detect regime
    4. Run risk filters
    5. Generate prediction
    6. Log & return
    """
    symbol = req.symbol
    now = datetime.now(timezone.utc)

    # 1) Fetch
    try:
        df = _fetch_live(symbol)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # 2) Features
    df = compute_features(df)
    if df.empty:
        raise HTTPException(status_code=500, detail="Not enough data after feature computation.")

    # 3) Regime
    regime = detect_regime(df)
    vol_status = get_volatility_status(df)

    # 4) Risk filters
    spread = _get_spread(symbol)
    should_skip, reasons = apply_filters(df, current_spread=spread)
    if should_skip:
        return SkippedResponse(
            symbol=symbol,
            reasons=reasons,
            timestamp=now.isoformat(),
        )

    # 5) Predict
    try:
        result = predict(df, regime)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 6) Build response
    reason = _reason_summary(df.iloc[-1], regime)

    response = PredictResponse(
        symbol=symbol,
        timestamp=now.isoformat(),
        market_regime=regime,
        green_probability=result["green_probability"],
        red_probability=result["red_probability"],
        threshold_used=result["threshold_used"],
        confidence_level=result["confidence_level"],
        suggested_trade=result["suggested_trade"],
        volatility_status=vol_status,
        reason_summary=reason,
    )

    # 7) Log
    log_prediction(symbol, regime, result, timestamp=now)
    log.info("Response: %s", response.model_dump())

    return response


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}
