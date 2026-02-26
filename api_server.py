"""
API Server v2 — Dual-layer prediction endpoint.

Returns primary probability + meta reliability + dual-gate decision.

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
from data_collector import fetch_multi_timeframe
from feature_engineering import compute_features, FEATURE_COLUMNS
from regime_detection import detect_regime, get_volatility_status
from predict_engine import predict, log_prediction
from risk_filter import apply_filters
from stability import ProbabilitySmoother, AccuracyTracker

log = logging.getLogger("api_server")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

# Stability instances
_smoother = ProbabilitySmoother()
_tracker = AccuracyTracker()


# --- Schemas -----------------------------------------------------------------

class PredictRequest(BaseModel):
    symbol: str = Field(default=DEFAULT_SYMBOL, description="Trading symbol")


class PredictResponse(BaseModel):
    symbol: str
    timestamp: str
    market_regime: str
    primary_green_probability: float
    meta_reliability_probability: float
    primary_threshold: float
    meta_threshold: float
    final_decision: str
    confidence_level: str
    volatility_status: str
    reason_summary: str


class SkippedResponse(BaseModel):
    status: str = "skipped"
    symbol: str
    reasons: list[str]
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    accuracy_tracker: dict = {}


# --- MT5 Helpers -------------------------------------------------------------

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


def _get_spread(symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    return float(info.spread) if info else 0.0


# --- Reason Summary ---------------------------------------------------------

def _reason_summary(row, regime, result):
    parts = [f"Regime: {regime}"]

    # EMA alignment
    if row["ema_20"] > row["ema_50"] > row["ema_200"]:
        parts.append("EMA bullish")
    elif row["ema_20"] < row["ema_50"] < row["ema_200"]:
        parts.append("EMA bearish")
    else:
        parts.append("EMAs mixed")

    # RSI
    rsi = row["rsi_14"]
    if rsi > 70:
        parts.append(f"RSI OB({rsi:.0f})")
    elif rsi < 30:
        parts.append(f"RSI OS({rsi:.0f})")

    # MACD
    parts.append("MACD+" if row["macd"] > row["macd_signal"] else "MACD-")

    # ADX
    adx = row["adx"]
    if adx > 40:
        parts.append(f"Strong trend(ADX {adx:.0f})")
    elif adx > 25:
        parts.append(f"Trend(ADX {adx:.0f})")

    # Meta
    meta_r = result.get("meta_reliability", 0)
    parts.append(f"Meta:{meta_r:.0%}")

    # HTF
    if row.get("h1_ema_alignment") == 1:
        parts.append("H1 bullish")
    elif row.get("h1_ema_alignment") == -1:
        parts.append("H1 bearish")

    return "; ".join(parts)


# --- App ---------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not _connect_mt5():
        log.error("MT5 not available.")
    yield
    mt5.shutdown()
    log.info("MT5 disconnected.")


app = FastAPI(
    title="Regime-Aware Dual-Layer Meta Learning Engine",
    version="2.0.0",
    description="Predicts M5 candle direction with primary + meta reliability models.",
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
    """Full dual-layer prediction pipeline."""
    symbol = req.symbol
    now = datetime.now(timezone.utc)

    # 1) Fetch multi-timeframe
    import pandas as pd
    try:
        data = fetch_multi_timeframe(symbol, CANDLES_TO_FETCH)
        df = data.get("M5")
        if df is None or len(df) == 0:
            raise RuntimeError(f"No M5 data for {symbol}")
        m15 = data.get("M15")
        h1 = data.get("H1")
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # 2) Features
    df = compute_features(df, m15_df=m15, h1_df=h1)
    if df.empty:
        raise HTTPException(status_code=500, detail="Not enough data after feature computation.")

    # 3) Regime
    regime = detect_regime(df)
    vol_status = get_volatility_status(df)

    # 4) Risk filters
    spread = _get_spread(symbol)
    should_skip, reasons = apply_filters(df, current_spread=spread)
    if should_skip:
        return SkippedResponse(symbol=symbol, reasons=reasons, timestamp=now.isoformat())

    # 5) Dual-layer predict
    try:
        result = predict(df, regime)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 6) Smooth probability
    smoothed_green = _smoother.smooth(result["green_probability"])

    # 7) Build response
    reason = _reason_summary(df.iloc[-1], regime, result)

    response = PredictResponse(
        symbol=symbol,
        timestamp=now.isoformat(),
        market_regime=regime,
        primary_green_probability=result["green_probability"],
        meta_reliability_probability=result["meta_reliability"],
        primary_threshold=result["primary_threshold"],
        meta_threshold=result["meta_threshold"],
        final_decision=result["suggested_trade"],
        confidence_level=result["confidence_level"],
        volatility_status=vol_status,
        reason_summary=reason,
    )

    # 8) Log
    log_prediction(symbol, regime, result, timestamp=now)
    return response


@app.get("/health")
def health():
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        accuracy_tracker=_tracker.status(),
    )
