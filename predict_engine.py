"""
Predict Engine — load trained model, generate probabilities, log predictions.

This module is the bridge between the trained model and the API layer.
It NEVER trains. It ONLY loads and predicts.
"""

import os
import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from config import (
    MODEL_PATH, FEATURE_LIST_PATH,
    LOG_DIR, PREDICTION_LOG_CSV, PREDICTION_LOG_JSON,
    CONFIDENCE_LOW_MAX, CONFIDENCE_MED_MAX,
    LOG_LEVEL, LOG_FORMAT,
)
from regime_detection import get_regime_threshold

log = logging.getLogger("predict_engine")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

# Module-level cache
_model = None
_features = None


def load_model():
    """Load model + feature list from disk. Cached after first call."""
    global _model, _features
    if _model is not None:
        return _model, _features

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No model at {MODEL_PATH}. Run train_model.py first.")
    if not os.path.exists(FEATURE_LIST_PATH):
        raise FileNotFoundError(f"No feature list at {FEATURE_LIST_PATH}. Run train_model.py first.")

    _model = joblib.load(MODEL_PATH)
    _features = joblib.load(FEATURE_LIST_PATH)
    log.info("Model loaded (%d features).", len(_features))
    return _model, _features


def predict(df: pd.DataFrame, regime: str) -> dict:
    """
    Generate prediction for the latest row of a feature-engineered DataFrame.

    Args:
        df: DataFrame with feature columns already computed.
        regime: Detected regime string.

    Returns dict:
        green_probability, red_probability, threshold_used,
        confidence_level, suggested_trade
    """
    model, feature_cols = load_model()
    row = df[feature_cols].iloc[-1].values.reshape(1, -1)
    proba = model.predict_proba(row)[0]

    green_p = round(float(proba[1]), 4)
    red_p = round(float(proba[0]), 4)
    threshold = get_regime_threshold(regime)
    dominant = max(green_p, red_p)

    # Confidence
    if dominant >= CONFIDENCE_MED_MAX:
        confidence = "High"
    elif dominant >= CONFIDENCE_LOW_MAX:
        confidence = "Medium"
    else:
        confidence = "Low"

    # Decision: green > threshold -> BUY, green < (1-threshold) -> SELL, else SKIP
    if green_p >= threshold:
        trade = "BUY"
    elif green_p <= (1 - threshold):
        trade = "SELL"
    else:
        trade = "SKIP"

    result = {
        "green_probability": green_p,
        "red_probability": red_p,
        "threshold_used": threshold,
        "confidence_level": confidence,
        "suggested_trade": trade,
    }

    log.info("Prediction: green=%.4f red=%.4f regime=%s threshold=%.2f -> %s",
             green_p, red_p, regime, threshold, trade)
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Prediction Logging
# ═══════════════════════════════════════════════════════════════════════════

def log_prediction(
    symbol: str,
    regime: str,
    prediction: dict,
    timestamp: datetime | None = None,
) -> None:
    """
    Append prediction to CSV and JSON log files.
    Creates log files if they don't exist.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    record = {
        "timestamp": timestamp.isoformat(),
        "symbol": symbol,
        "regime": regime,
        "green_probability": prediction["green_probability"],
        "red_probability": prediction["red_probability"],
        "threshold_used": prediction["threshold_used"],
        "suggested_trade": prediction["suggested_trade"],
        "confidence_level": prediction["confidence_level"],
    }

    # ── CSV ──────────────────────────────────────────────────────────────
    row_df = pd.DataFrame([record])
    header = not os.path.exists(PREDICTION_LOG_CSV)
    row_df.to_csv(PREDICTION_LOG_CSV, mode="a", header=header, index=False)

    # ── JSON (append line) ───────────────────────────────────────────────
    with open(PREDICTION_LOG_JSON, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    log.info("Prediction logged for %s.", symbol)
