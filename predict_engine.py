"""
Predict Engine v2 — dual-layer prediction (primary + meta).

Trade only if BOTH primary AND meta probabilities exceed their thresholds.
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
    CONFIDENCE_LOW_MAX, CONFIDENCE_MED_MAX,
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

# Rolling prediction history for meta features
_prediction_history = []


def load_models():
    """Load primary + meta models. Cached after first call."""
    global _primary_model, _primary_features, _meta_model, _meta_features

    if _primary_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No primary model at {MODEL_PATH}. Run train_model.py first.")
        _primary_model = joblib.load(MODEL_PATH)
        _primary_features = joblib.load(FEATURE_LIST_PATH)
        log.info("Primary model loaded (%d features).", len(_primary_features))

    if _meta_model is None:
        if not os.path.exists(META_MODEL_PATH):
            raise FileNotFoundError(f"No meta model at {META_MODEL_PATH}. Run meta_model.py first.")
        _meta_model = joblib.load(META_MODEL_PATH)
        _meta_features = joblib.load(META_FEATURE_LIST_PATH)
        log.info("Meta model loaded (%d features).", len(_meta_features))

    return _primary_model, _primary_features, _meta_model, _meta_features


def _build_meta_row(green_p: float, regime: str, df_row, regime_encoded: int) -> dict:
    """Build meta feature dict for a single prediction."""
    global _prediction_history

    # Rolling accuracy from history
    if len(_prediction_history) > 0:
        recent = _prediction_history[-META_ROLLING_WINDOW:]
        recent_acc = sum(r["correct"] for r in recent) / len(recent)
        # Win streak
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

    # Spread ratio
    spread_ratio = 0.0  # will be filled from live data if available

    return {
        "primary_green_prob": green_p,
        "primary_red_prob": 1.0 - green_p,
        "primary_confidence_strength": max(green_p, 1.0 - green_p),
        "regime_encoded": regime_encoded,
        "atr_value": float(df_row.get("atr_14", 0)),
        "spread_ratio": spread_ratio,
        "volatility_zscore": float(df_row.get("volatility_zscore", 0)),
        "range_position": float(df_row.get("range_position", 0.5)),
        "recent_model_accuracy": recent_acc,
        "recent_win_streak": streak,
    }


REGIME_ENCODING = {"Trending": 0, "Ranging": 1, "High_Volatility": 2, "Low_Volatility": 3}


def predict(df: pd.DataFrame, regime: str) -> dict:
    """
    Dual-layer prediction.

    Returns dict with:
        green_probability, red_probability, meta_reliability,
        primary_threshold, meta_threshold,
        confidence_level, suggested_trade
    """
    primary_model, feature_cols, meta_model, meta_feat_cols = load_models()

    # -- Primary prediction --
    row = df[feature_cols].iloc[-1].values.reshape(1, -1)
    proba = primary_model.predict_proba(row)[0]
    green_p = round(float(proba[1]), 4)
    red_p = round(float(proba[0]), 4)

    # -- Meta prediction --
    regime_enc = REGIME_ENCODING.get(regime, 1)
    meta_row = _build_meta_row(green_p, regime, df.iloc[-1], regime_enc)
    meta_input = pd.DataFrame([meta_row])[meta_feat_cols]
    meta_proba = meta_model.predict_proba(meta_input.values)[0]
    meta_reliability = round(float(meta_proba[1]), 4)

    # -- Thresholds --
    thresholds = get_regime_thresholds(regime)
    p_thresh = thresholds["primary"]
    m_thresh = thresholds["meta"]

    # -- Confidence --
    dominant = max(green_p, red_p)
    combined = min(dominant, meta_reliability)
    if combined >= CONFIDENCE_MED_MAX:
        confidence = "High"
    elif combined >= CONFIDENCE_LOW_MAX:
        confidence = "Medium"
    else:
        confidence = "Low"

    # -- Dual-gate decision --
    if green_p >= p_thresh and meta_reliability >= m_thresh:
        trade = "BUY"
    elif green_p <= (1 - p_thresh) and meta_reliability >= m_thresh:
        trade = "SELL"
    else:
        trade = "SKIP"

    result = {
        "green_probability": green_p,
        "red_probability": red_p,
        "meta_reliability": meta_reliability,
        "primary_threshold": p_thresh,
        "meta_threshold": m_thresh,
        "confidence_level": confidence,
        "suggested_trade": trade,
    }

    log.info("Prediction: green=%.4f meta=%.4f regime=%s -> %s",
             green_p, meta_reliability, regime, trade)
    return result


def update_prediction_history(was_correct: bool):
    """Update rolling history after outcome is known (for live tracking)."""
    _prediction_history.append({"correct": was_correct})
    if len(_prediction_history) > META_ROLLING_WINDOW * 2:
        _prediction_history.pop(0)


# =============================================================================
#  Prediction Logging
# =============================================================================

def log_prediction(symbol, regime, prediction, timestamp=None):
    os.makedirs(LOG_DIR, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    record = {
        "timestamp": timestamp.isoformat(),
        "symbol": symbol,
        "regime": regime,
        "green_probability": prediction["green_probability"],
        "red_probability": prediction["red_probability"],
        "meta_reliability": prediction["meta_reliability"],
        "primary_threshold": prediction["primary_threshold"],
        "meta_threshold": prediction["meta_threshold"],
        "suggested_trade": prediction["suggested_trade"],
        "confidence_level": prediction["confidence_level"],
    }

    row_df = pd.DataFrame([record])
    header = not os.path.exists(PREDICTION_LOG_CSV)
    row_df.to_csv(PREDICTION_LOG_CSV, mode="a", header=header, index=False)

    with open(PREDICTION_LOG_JSON, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    log.info("Prediction logged for %s.", symbol)
