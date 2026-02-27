"""
Predict Engine — Production-safe, race-condition protected.

Features:
  - Race condition protection via retrain lock
  - Candle integrity verification (only closed candles)
  - Uncertainty from ensemble variance: confidence *= (1 - normalized_variance)
  - Calibrated meta output (isotonic)
  - Rolling confidence-accuracy correlation tracking
  - Model cached in memory

Live parity: meta features built identically to training
  (ensemble_variance, no spread_ratio, no correctness-derived features)
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
    MODEL_DIR, LOG_DIR, PREDICTION_LOG_CSV, PREDICTION_LOG_JSON,
    CONFIDENCE_HIGH_MIN, CONFIDENCE_MEDIUM_MIN,
    ATR_PERCENTILE_WINDOW,
    CONFIDENCE_CORRELATION_WINDOW, CONFIDENCE_CORRELATION_ALERT,
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
_meta_calibrator = None
_meta_features = None
_weight_model = None
_prediction_history = []
_direction_history = []

# --- Race condition protection -----------------------------------------------
_retrain_in_progress = False
_model_lock = threading.RLock()

META_CALIBRATOR_PATH = os.path.join(MODEL_DIR, "meta_calibrator.pkl")

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
    global _ensemble, _primary_features, _meta_model, _meta_calibrator
    global _meta_features, _weight_model
    with _model_lock:
        _ensemble = None
        _primary_features = None
        _meta_model = None
        _meta_calibrator = None
        _meta_features = None
        _weight_model = None
    load_models()
    log.info("Models reloaded from disk.")


def load_models():
    global _ensemble, _primary_features, _meta_model, _meta_calibrator
    global _meta_features, _weight_model

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
            # Load meta calibrator
            if os.path.exists(META_CALIBRATOR_PATH):
                _meta_calibrator = joblib.load(META_CALIBRATOR_PATH)
                log.info("Meta calibrator loaded.")
            else:
                _meta_calibrator = None
                log.warning("No meta calibrator found — using raw probabilities.")

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
        return True

    last_time = pd.Timestamp(df["time"].iloc[-1])
    if last_time.tzinfo is None:
        last_time = last_time.tz_localize("UTC")

    candle_close = last_time + pd.Timedelta(minutes=timeframe_minutes)
    now = datetime.now(timezone.utc)
    is_closed = candle_close <= now

    if not is_closed:
        log.warning("Last candle NOT closed (time=%s, closes=%s, now=%s). "
                     "Using second-to-last.", last_time, candle_close, now)
    return is_closed


def _build_meta_row(green_p, df_row, regime_enc, ensemble_variance):
    """
    Build meta feature row matching META_FEATURE_COLUMNS exactly.
    
    LIVE PARITY: same features as training (no spread_ratio,
    no correctness-derived features, has ensemble_variance).
    """
    global _direction_history

    cur_dir = 1 if green_p >= 0.5 else 0
    dstreak = 1
    for d in reversed(_direction_history):
        if d == cur_dir:
            dstreak += 1
        else:
            break

    return {
        "primary_green_prob": green_p,
        "prob_distance_from_half": abs(green_p - 0.5),
        "primary_entropy": float(_binary_entropy(green_p)),
        "ensemble_variance": ensemble_variance,
        "regime_encoded": regime_enc,
        "atr_value": float(df_row.get("atr_14", 0)),
        "volatility_zscore": float(df_row.get("volatility_zscore", 0)),
        "range_position": float(df_row.get("range_position", 0.5)),
        "body_percentile_rank": float(df_row.get("body_size", 0.5)),
        "direction_streak": dstreak,
        "rolling_vol_percentile": float(df_row.get("atr_percentile_rank", 0.5)),
    }


def get_confidence_reliability():
    """Compute rolling Spearman correlation between confidence and correctness."""
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
    max_var = 0.25
    norm_var = min(variance / max_var, 1.0)
    uncertainty_pct = round(norm_var * 100, 2)

    # Meta (with calibration)
    regime_enc = REGIME_ENCODING.get(regime, 1)
    meta_row = _build_meta_row(green_p, df.iloc[-1], regime_enc, variance)
    meta_input = pd.DataFrame([meta_row])[meta_feat_cols]
    meta_raw = meta_model.predict_proba(meta_input.values)[0][1]

    # Apply meta calibration if available
    if _meta_calibrator is not None:
        meta_rel = float(np.clip(_meta_calibrator.predict([meta_raw])[0], 0, 1))
    else:
        meta_rel = float(meta_raw)

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

    # Update direction history for live parity
    _direction_history.append(1 if green_p >= 0.5 else 0)
    if len(_direction_history) > 500:
        _direction_history.pop(0)

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
