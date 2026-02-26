"""
Predict Engine v4 — 5-seed ensemble + uncertainty + learned weights.

Always returns probabilities (0-100%). Never skips.
"""

import os
import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from config import (
    MODEL_PATH, FEATURE_LIST_PATH, ENSEMBLE_MODEL_PATH,
    META_MODEL_PATH, META_FEATURE_LIST_PATH, WEIGHT_MODEL_PATH,
    LOG_DIR, PREDICTION_LOG_CSV, PREDICTION_LOG_JSON,
    CONFIDENCE_HIGH_MIN, CONFIDENCE_MEDIUM_MIN,
    META_ROLLING_WINDOW, WIN_STREAK_CAP, ATR_PERCENTILE_WINDOW,
    LOG_LEVEL, LOG_FORMAT,
)
from regime_detection import get_regime_thresholds
from calibration import CalibratedModel  # needed for joblib.load

log = logging.getLogger("predict_engine")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

_ensemble = None
_primary_features = None
_meta_model = None
_meta_features = None
_weight_model = None
_prediction_history = []

REGIME_ENCODING = {"Trending": 0, "Ranging": 1, "High_Volatility": 2, "Low_Volatility": 3}


def _binary_entropy(p):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def load_models():
    global _ensemble, _primary_features, _meta_model, _meta_features, _weight_model

    if _ensemble is None:
        if os.path.exists(ENSEMBLE_MODEL_PATH):
            _ensemble = joblib.load(ENSEMBLE_MODEL_PATH)
            log.info("Ensemble loaded (%d models).", len(_ensemble))
        elif os.path.exists(MODEL_PATH):
            _ensemble = [joblib.load(MODEL_PATH)]
            log.info("Single model loaded (fallback).")
        else:
            raise FileNotFoundError("No model found.")
        _primary_features = joblib.load(FEATURE_LIST_PATH)

    if _meta_model is None:
        _meta_model = joblib.load(META_MODEL_PATH)
        _meta_features = joblib.load(META_FEATURE_LIST_PATH)
        log.info("Meta model loaded (%d features).", len(_meta_features))

    if _weight_model is None and os.path.exists(WEIGHT_MODEL_PATH):
        _weight_model = joblib.load(WEIGHT_MODEL_PATH)
        log.info("Weight model loaded.")

    return _ensemble, _primary_features, _meta_model, _meta_features, _weight_model


def _build_meta_row(green_p, df_row, regime_enc):
    global _prediction_history
    if _prediction_history:
        recent = _prediction_history[-META_ROLLING_WINDOW:]
        recent_acc = sum(r["correct"] for r in recent) / len(recent)
        streak = 0
        for r in reversed(recent):
            if r["correct"]:
                streak += 1
            else:
                break
        streak = min(streak, WIN_STREAK_CAP)
        # Direction streak
        dstreak = 1
        for r in reversed(_prediction_history[:-1]):
            if (r["green_p"] >= 0.5) == (green_p >= 0.5):
                dstreak += 1
            else:
                break
    else:
        recent_acc = 0.5
        streak = 0
        dstreak = 1

    return {
        "primary_green_prob": green_p,
        "prob_distance_from_half": abs(green_p - 0.5),
        "primary_entropy": float(_binary_entropy(green_p)),
        "regime_encoded": regime_enc,
        "atr_value": float(df_row.get("atr_14", 0)),
        "spread_ratio": 0.0,
        "volatility_zscore": float(df_row.get("volatility_zscore", 0)),
        "range_position": float(df_row.get("range_position", 0.5)),
        "recent_model_accuracy": recent_acc,
        "recent_win_streak": streak,
        "body_percentile_rank": float(df_row.get("body_size", 0.5)),
        "direction_streak": dstreak,
        "rolling_vol_percentile": float(df_row.get("atr_percentile_rank", 0.5)),
    }


def predict(df, regime):
    ensemble, feat_cols, meta_model, meta_feat_cols, weight_model = load_models()

    row = df[feat_cols].iloc[-1].values.reshape(1, -1)

    # Ensemble predictions
    all_probs = np.array([m.predict_proba(row)[0][1] for m in ensemble])
    green_p = float(all_probs.mean())
    red_p = 1.0 - green_p
    variance = float(all_probs.var())
    max_var = 0.25  # max possible variance for [0,1]
    norm_var = min(variance / max_var, 1.0)
    uncertainty_pct = round(norm_var * 100, 2)

    # Meta
    regime_enc = REGIME_ENCODING.get(regime, 1)
    meta_row = _build_meta_row(green_p, df.iloc[-1], regime_enc)
    meta_input = pd.DataFrame([meta_row])[meta_feat_cols]
    meta_proba = meta_model.predict_proba(meta_input.values)[0]
    meta_rel = float(meta_proba[1])

    # Confidence
    primary_strength = abs(green_p - 0.5) * 2
    regime_strength = float(df.iloc[-1].get("adx_normalized", 0.25))

    if weight_model is not None:
        w_input = pd.DataFrame([{
            "primary_strength": primary_strength,
            "meta_reliability": meta_rel,
            "regime_strength": regime_strength,
            "uncertainty": norm_var,
        }])
        confidence = float(weight_model.predict_proba(w_input)[:, 1][0])
    else:
        confidence = primary_strength * meta_rel * (1 - norm_var)

    # Convert to percentages
    green_pct = round(green_p * 100, 2)
    red_pct = round(red_p * 100, 2)
    meta_pct = round(meta_rel * 100, 2)
    confidence_pct = round(confidence * 100, 2)
    direction = "GREEN" if green_p >= 0.5 else "RED"

    if confidence_pct >= CONFIDENCE_HIGH_MIN:
        conf_level = "High"
    elif confidence_pct >= CONFIDENCE_MEDIUM_MIN:
        conf_level = "Medium"
    else:
        conf_level = "Low"

    # Trade suggestion
    thresholds = get_regime_thresholds(regime)
    if green_p >= thresholds["primary"] and meta_rel >= thresholds["meta"]:
        trade = "BUY"
    elif green_p <= (1 - thresholds["primary"]) and meta_rel >= thresholds["meta"]:
        trade = "SELL"
    else:
        trade = "HOLD"

    result = {
        "green_probability_percent": green_pct,
        "red_probability_percent": red_pct,
        "primary_direction": direction,
        "meta_reliability_percent": meta_pct,
        "uncertainty_percent": uncertainty_pct,
        "final_confidence_percent": confidence_pct,
        "confidence_level": conf_level,
        "suggested_trade": trade,
    }

    log.info("Pred: green=%.1f%% meta=%.1f%% unc=%.1f%% conf=%.1f%% -> %s",
             green_pct, meta_pct, uncertainty_pct, confidence_pct, direction)
    return result


def update_prediction_history(green_p, was_correct):
    _prediction_history.append({"correct": was_correct, "green_p": green_p})
    if len(_prediction_history) > META_ROLLING_WINDOW * 2:
        _prediction_history.pop(0)


def log_prediction(symbol, regime, prediction, risk_warnings=None, timestamp=None):
    os.makedirs(LOG_DIR, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    record = {
        "timestamp": timestamp.isoformat(),
        "symbol": symbol, "regime": regime,
        **prediction,
        "risk_warnings": risk_warnings or [],
    }
    row_df = pd.DataFrame([record])
    header = not os.path.exists(PREDICTION_LOG_CSV)
    row_df.to_csv(PREDICTION_LOG_CSV, mode="a", header=header, index=False)
    with open(PREDICTION_LOG_JSON, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
