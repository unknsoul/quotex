"""
API Server v3 — Continuous Prediction (never skip).

Always returns probability percentages for every candle.
Risk warnings are included but do not suppress prediction.

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
from risk_filter import check_warnings
from stability import ProbabilitySmoother, AccuracyTracker

log = logging.getLogger("api_server")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

_smoother = ProbabilitySmoother()
_tracker = AccuracyTracker()


# --- Schemas (v3: always return full prediction) -----------------------------

class PredictRequest(BaseModel):
    symbol: str = Field(default=DEFAULT_SYMBOL)


class PredictResponse(BaseModel):
    symbol: str
    timestamp: str
    market_regime: str
    green_probability_percent: float
    red_probability_percent: float
    primary_direction: str
    meta_reliability_percent: float
    final_confidence_percent: float
    confidence_level: str
    suggested_trade: str
    volatility_status: str
    risk_warnings: list[str]
    reason_summary: str


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

    if row["ema_20"] > row["ema_50"] > row["ema_200"]:
        parts.append("EMA bullish")
    elif row["ema_20"] < row["ema_50"] < row["ema_200"]:
        parts.append("EMA bearish")
    else:
        parts.append("EMAs mixed")

    rsi = row["rsi_14"]
    if rsi > 70:
        parts.append(f"RSI OB({rsi:.0f})")
    elif rsi < 30:
        parts.append(f"RSI OS({rsi:.0f})")

    parts.append("MACD+" if row["macd"] > row["macd_signal"] else "MACD-")

    adx = row["adx"]
    if adx > 40:
        parts.append(f"Strong trend(ADX {adx:.0f})")
    elif adx > 25:
        parts.append(f"Trend(ADX {adx:.0f})")

    parts.append(f"Conf:{result['final_confidence_percent']:.0f}%")
    parts.append(f"Meta:{result['meta_reliability_percent']:.0f}%")

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
    title="Continuous Candle Probability Forecast Engine",
    version="3.0.0",
    description="Predicts green/red probability for EVERY M5 candle. Never skips.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    """
    Generate prediction for current candle.
    ALWAYS returns probabilities — never skips.
    """
    symbol = req.symbol
    now = datetime.now(timezone.utc)

    # 1) Fetch multi-timeframe
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
        raise HTTPException(status_code=500, detail="Not enough data.")

    # 3) Regime
    regime = detect_regime(df)
    vol_status = get_volatility_status(df)

    # 4) Risk warnings (informational only, never blocks)
    spread = _get_spread(symbol)
    risk_warnings = check_warnings(df, current_spread=spread)

    # 5) Predict (ALWAYS)
    try:
        result = predict(df, regime)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 6) Build response
    reason = _reason_summary(df.iloc[-1], regime, result)

    response = PredictResponse(
        symbol=symbol,
        timestamp=now.isoformat(),
        market_regime=regime,
        green_probability_percent=result["green_probability_percent"],
        red_probability_percent=result["red_probability_percent"],
        primary_direction=result["primary_direction"],
        meta_reliability_percent=result["meta_reliability_percent"],
        final_confidence_percent=result["final_confidence_percent"],
        confidence_level=result["confidence_level"],
        suggested_trade=result["suggested_trade"],
        volatility_status=vol_status,
        risk_warnings=risk_warnings,
        reason_summary=reason,
    )

    # 7) Log
    log_prediction(symbol, regime, result, risk_warnings=risk_warnings, timestamp=now)
    return response


@app.get("/health")
def health():
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        accuracy_tracker=_tracker.status(),
    )
