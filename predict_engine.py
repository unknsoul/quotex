"""
Predict Engine v3 — Continuous prediction for EVERY candle.

Never skips. Always returns green/red probability percentages (0-100%).
Confidence = primary_strength x meta_reliability.
Trade suggestion is advisory only.
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
    META_MODEL_PATH, META_FEATURE_LIST_PATH,
    LOG_DIR, PREDICTION_LOG_CSV, PREDICTION_LOG_JSON,
    CONFIDENCE_HIGH_MIN, CONFIDENCE_MEDIUM_MIN,
    META_ROLLING_WINDOW, WIN_STREAK_CAP,
    LOG_LEVEL, LOG_FORMAT,
)
from regime_detection import get_regime_thresholds

log = logging.getLogger("predict_engine")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

# Module-level caches
_primary_model = None
_primary_features = None
_meta_model = None
_meta_features = None
_prediction_history = []

REGIME_ENCODING = {"Trending": 0, "Ranging": 1, "High_Volatility": 2, "Low_Volatility": 3}


def load_models():
    global _primary_model, _primary_features, _meta_model, _meta_features

    if _primary_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No primary model at {MODEL_PATH}")
        _primary_model = joblib.load(MODEL_PATH)
        _primary_features = joblib.load(FEATURE_LIST_PATH)
        log.info("Primary model loaded (%d features).", len(_primary_features))

    if _meta_model is None:
        if not os.path.exists(META_MODEL_PATH):
            raise FileNotFoundError(f"No meta model at {META_MODEL_PATH}")
        _meta_model = joblib.load(META_MODEL_PATH)
        _meta_features = joblib.load(META_FEATURE_LIST_PATH)
        log.info("Meta model loaded (%d features).", len(_meta_features))

    return _primary_model, _primary_features, _meta_model, _meta_features


def _build_meta_row(green_p, regime, df_row, regime_encoded):
    global _prediction_history

    if len(_prediction_history) > 0:
        recent = _prediction_history[-META_ROLLING_WINDOW:]
        recent_acc = sum(r["correct"] for r in recent) / len(recent)
        streak = 0
        for r in reversed(recent):
            if r["correct"]:
                streak += 1
            else:
                break
        streak = min(streak, WIN_STREAK_CAP)
    else:
        recent_acc = 0.5
        streak = 0

    return {
        "primary_green_prob": green_p,
        "prob_distance_from_half": abs(green_p - 0.5),
        "regime_encoded": regime_encoded,
        "atr_value": float(df_row.get("atr_14", 0)),
        "spread_ratio": 0.0,
        "volatility_zscore": float(df_row.get("volatility_zscore", 0)),
        "range_position": float(df_row.get("range_position", 0.5)),
        "recent_model_accuracy": recent_acc,
        "recent_win_streak": streak,
    }


def predict(df: pd.DataFrame, regime: str) -> dict:
    """
    Generate prediction for current candle. ALWAYS returns probabilities.

    Returns dict with percentages (0-100) and trade suggestion.
    """
    primary_model, feature_cols, meta_model, meta_feat_cols = load_models()

    # -- Primary --
    row = df[feature_cols].iloc[-1].values.reshape(1, -1)
    proba = primary_model.predict_proba(row)[0]
    green_p = float(proba[1])
    red_p = float(proba[0])

    # -- Meta --
    regime_enc = REGIME_ENCODING.get(regime, 1)
    meta_row = _build_meta_row(green_p, regime, df.iloc[-1], regime_enc)
    meta_input = pd.DataFrame([meta_row])[meta_feat_cols]
    meta_proba = meta_model.predict_proba(meta_input.values)[0]
    meta_reliability = float(meta_proba[1])

    # -- Confidence: primary_strength x meta_reliability --
    primary_strength = abs(green_p - 0.5) * 2  # 0-1 scale (0.5->0, 1.0->1)
    final_confidence = primary_strength * meta_reliability  # 0-1

    # Convert to percentages
    green_pct = round(green_p * 100, 2)
    red_pct = round(red_p * 100, 2)
    meta_pct = round(meta_reliability * 100, 2)
    confidence_pct = round(final_confidence * 100, 2)

    # Direction
    direction = "GREEN" if green_p >= 0.5 else "RED"

    # Confidence level
    if confidence_pct >= CONFIDENCE_HIGH_MIN:
        confidence_level = "High"
    elif confidence_pct >= CONFIDENCE_MEDIUM_MIN:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"

    # Trade suggestion (advisory, based on regime thresholds)
    thresholds = get_regime_thresholds(regime)
    p_thresh = thresholds["primary"]
    m_thresh = thresholds["meta"]

    if green_p >= p_thresh and meta_reliability >= m_thresh:
        suggested_trade = "BUY"
    elif green_p <= (1 - p_thresh) and meta_reliability >= m_thresh:
        suggested_trade = "SELL"
    else:
        suggested_trade = "HOLD"

    result = {
        "green_probability_percent": green_pct,
        "red_probability_percent": red_pct,
        "primary_direction": direction,
        "meta_reliability_percent": meta_pct,
        "final_confidence_percent": confidence_pct,
        "confidence_level": confidence_level,
        "suggested_trade": suggested_trade,
    }

    log.info("Prediction: green=%.1f%% meta=%.1f%% conf=%.1f%% regime=%s -> %s",
             green_pct, meta_pct, confidence_pct, regime, direction)
    return result


def update_prediction_history(was_correct: bool):
    _prediction_history.append({"correct": was_correct})
    if len(_prediction_history) > META_ROLLING_WINDOW * 2:
        _prediction_history.pop(0)


# =============================================================================
#  Logging
# =============================================================================

def log_prediction(symbol, regime, prediction, risk_warnings=None, timestamp=None):
    os.makedirs(LOG_DIR, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    record = {
        "timestamp": timestamp.isoformat(),
        "symbol": symbol,
        "regime": regime,
        "green_probability_percent": prediction["green_probability_percent"],
        "red_probability_percent": prediction["red_probability_percent"],
        "primary_direction": prediction["primary_direction"],
        "meta_reliability_percent": prediction["meta_reliability_percent"],
        "final_confidence_percent": prediction["final_confidence_percent"],
        "confidence_level": prediction["confidence_level"],
        "suggested_trade": prediction["suggested_trade"],
        "risk_warnings": risk_warnings or [],
    }

    row_df = pd.DataFrame([record])
    header = not os.path.exists(PREDICTION_LOG_CSV)
    row_df.to_csv(PREDICTION_LOG_CSV, mode="a", header=header, index=False)

    with open(PREDICTION_LOG_JSON, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    log.info("Prediction logged for %s.", symbol)
