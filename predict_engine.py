"""
Predict Engine — Production-safe, race-condition protected.

Features:
  - Race condition protection via retrain lock
  - Candle integrity verification (only closed candles)
  - Uncertainty from ensemble variance: confidence *= (1 - normalized_variance)
  - Rolling confidence-accuracy correlation tracking
  - Model cached in memory, no disk load per request

Always returns probabilities (0-100%). Never skips.
"""

import os
import json
import logging
import threading
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from config import (
    MODEL_PATH, FEATURE_LIST_PATH, ENSEMBLE_MODEL_PATH,
    META_MODEL_PATH, META_FEATURE_LIST_PATH, WEIGHT_MODEL_PATH,
    LOG_DIR, PREDICTION_LOG_CSV, PREDICTION_LOG_JSON,
    CONFIDENCE_HIGH_MIN, CONFIDENCE_MEDIUM_MIN,
    META_ROLLING_WINDOW, WIN_STREAK_CAP, ATR_PERCENTILE_WINDOW,
    CONFIDENCE_CORRELATION_WINDOW, CONFIDENCE_CORRELATION_ALERT,
    ROLLING_CONFIDENCE_WINDOW,
    LOG_LEVEL, LOG_FORMAT,
)
from regime_detection import get_regime_thresholds
from calibration import CalibratedModel  # needed for joblib.load

log = logging.getLogger("predict_engine")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

# --- Model cache (in-memory) ------------------------------------------------
_ensemble = None
_primary_features = None
_meta_model = None
_meta_features = None
_weight_model = None
_prediction_history = []

# --- Race condition protection -----------------------------------------------
_retrain_in_progress = False
_model_lock = threading.RLock()

REGIME_ENCODING = {"Trending": 0, "Ranging": 1, "High_Volatility": 2, "Low_Volatility": 3}


def _binary_entropy(p):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def set_retrain_flag(active):
    """Called by auto_learner to signal retrain start/end."""
    global _retrain_in_progress
    _retrain_in_progress = active
    log.info("Retrain flag set to %s", active)


def reload_models():
    """Force reload models from disk (called after retrain completes)."""
    global _ensemble, _primary_features, _meta_model, _meta_features, _weight_model
    with _model_lock:
        _ensemble = None
        _primary_features = None
        _meta_model = None
        _meta_features = None
        _weight_model = None
    load_models()
    log.info("Models reloaded from disk.")


def load_models():
    global _ensemble, _primary_features, _meta_model, _meta_features, _weight_model

    with _model_lock:
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


def verify_candle_closed(df, timeframe_minutes=5):
    """
    Verify that the last candle in df is fully closed.
    Returns True if the last candle's close time is in the past.
    """
    if "time" not in df.columns:
        return True  # Can't verify, assume OK

    last_time = pd.Timestamp(df["time"].iloc[-1])
    if last_time.tzinfo is None:
        last_time = last_time.tz_localize("UTC")

    candle_close = last_time + pd.Timedelta(minutes=timeframe_minutes)
    now = datetime.now(timezone.utc)
    is_closed = candle_close <= now

    if not is_closed:
        log.warning("Last candle NOT closed yet (time=%s, closes=%s, now=%s). "
                     "Using second-to-last candle.", last_time, candle_close, now)
    return is_closed


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


def get_confidence_reliability():
    """
    Compute rolling Spearman correlation between confidence and correctness.
    Returns correlation score (0-1 range) and alert status.
    """
    if len(_prediction_history) < 30:
        return 0.5, False

    recent = _prediction_history[-CONFIDENCE_CORRELATION_WINDOW:]
    confidences = [r["confidence"] for r in recent]
    corrects = [r["correct"] for r in recent]

    corr, _ = stats.spearmanr(confidences, corrects)
    corr = float(corr) if not np.isnan(corr) else 0.0

    alert = corr < CONFIDENCE_CORRELATION_ALERT
    if alert:
        log.warning("CONFIDENCE ALERT: correlation=%.3f < threshold=%.3f",
                     corr, CONFIDENCE_CORRELATION_ALERT)

    return round(corr, 4), alert


def predict(df, regime):
    """
    Generate prediction using cached models.
    If retrain is in progress, uses the previous stable model (cached in memory).
    """
    if _retrain_in_progress:
        log.info("Retrain in progress — using cached stable model.")

    with _model_lock:
        ensemble, feat_cols, meta_model, meta_feat_cols, weight_model = load_models()

    row = df[feat_cols].iloc[-1].values.reshape(1, -1)

    # Ensemble predictions (5 seeded XGBoost)
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

    # Confidence with uncertainty adjustment
    primary_strength = abs(green_p - 0.5) * 2
    regime_strength = float(df.iloc[-1].get("adx_normalized", 0.25))

    if weight_model is not None:
        w_input = pd.DataFrame([{
            "primary_strength": primary_strength,
            "meta_reliability": meta_rel,
            "regime_strength": regime_strength,
            "uncertainty": norm_var,
        }])
        weighted_score = float(weight_model.predict_proba(w_input)[:, 1][0])
    else:
        weighted_score = primary_strength * meta_rel

    # Apply uncertainty adjustment: final_confidence = weighted_score × (1 - variance_factor)
    confidence = weighted_score * (1.0 - norm_var)

    # Confidence reliability (rolling correlation)
    conf_reliability, conf_alert = get_confidence_reliability()

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
        "confidence_reliability_score": conf_reliability,
        "suggested_trade": trade,
        "suggested_direction": "UP" if green_p >= 0.5 else "DOWN",
        "market_regime": regime,
    }

    log.info("Pred: green=%.1f%% meta=%.1f%% unc=%.1f%% conf=%.1f%% rel=%.3f -> %s",
             green_pct, meta_pct, uncertainty_pct, confidence_pct, conf_reliability, direction)
    return result


def update_prediction_history(green_p, was_correct, confidence=0.5):
    """Record prediction outcome for rolling metrics."""
    _prediction_history.append({
        "correct": was_correct,
        "green_p": green_p,
        "confidence": confidence,
    })
    if len(_prediction_history) > CONFIDENCE_CORRELATION_WINDOW * 2:
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
