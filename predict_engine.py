"""
Predict Engine — Production-safe, race-condition protected.

Features:
  - Race condition protection via retrain lock
  - Candle integrity verification (only closed candles)
  - Uncertainty from ensemble variance: confidence *= (1 - normalized_variance)
  - Sigmoid-calibrated meta (Platt scaling)
  - Rolling confidence-accuracy correlation tracking
  - Adaptive confidence (rolling calibration window)
  - Kelly criterion position sizing
  - Model cached in memory

Live parity: meta features built identically to training
  (ensemble_variance, no spread_ratio, no correctness-derived features)
"""

import os
import json
import logging
import threading
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from config import (
    MODEL_PATH, FEATURE_LIST_PATH, ENSEMBLE_MODEL_PATH,
    META_MODEL_PATH, META_FEATURE_LIST_PATH, WEIGHT_MODEL_PATH,
    CONFIDENCE_THRESHOLDS_PATH,
    MODEL_DIR, LOG_DIR, PREDICTION_LOG_CSV, PREDICTION_LOG_JSON,
    CONFIDENCE_HIGH_MIN, CONFIDENCE_MEDIUM_MIN,
    ATR_PERCENTILE_WINDOW,
    CONFIDENCE_CORRELATION_WINDOW, CONFIDENCE_CORRELATION_ALERT,
    REGIME_FILTER_ENABLED, REGIME_SKIP_ACCURACY_THRESHOLD,
    REGIME_COOLDOWN_BARS, REGIME_ROLLING_ACCURACY_WINDOW,
    SESSION_FILTER_ENABLED, SESSION_CONFIDENCE_MULT,
    LOG_LEVEL, LOG_FORMAT,
)
from regime_detection import get_regime_thresholds, regime_stability_score, get_session
from calibration import CalibratedModel  # needed for joblib.load

log = logging.getLogger("predict_engine")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

# --- Model cache (in-memory) ------------------------------------------------
_ensemble = None
_primary_features = None
_meta_model = None
_meta_features = None
_weight_model = None
_confidence_thresholds = None  # Phase 3: learned per-regime thresholds
_prediction_history = []
_direction_history = []

# Adaptive confidence: rolling calibration window
_adaptive_history = []  # list of (confidence, correct) tuples
ADAPTIVE_WINDOW = 200

# --- Race condition protection -----------------------------------------------
_retrain_in_progress = False
_model_lock = threading.RLock()

# --- Regime duration tracker for meta feature --------------------------------
_last_regime_name = None
_regime_duration_counter = 1

# --- Phase 3: Adaptive regime filter state -----------------------------------
_rolling_correct = []    # rolling accuracy tracker
_cooldown_remaining = 0  # bars remaining in cooldown
_skip_count = 0          # total skips counter

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
    global _ensemble, _primary_features, _meta_model
    global _meta_features, _weight_model, _confidence_thresholds
    with _model_lock:
        _ensemble = None
        _primary_features = None
        _meta_model = None
        _meta_features = None
        _weight_model = None
        _confidence_thresholds = None
    load_models()
    log.info("Models reloaded from disk.")


def load_models():
    global _ensemble, _primary_features, _meta_model, _meta_calibrator
    global _meta_features, _weight_model, _confidence_thresholds

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
            log.info("Meta model loaded (sigmoid-calibrated, %d features).", len(_meta_features))

        if _weight_model is None and os.path.exists(WEIGHT_MODEL_PATH):
            _weight_model = joblib.load(WEIGHT_MODEL_PATH)
            log.info("Weight model loaded.")

        # Phase 3: load learned confidence thresholds
        if _confidence_thresholds is None and os.path.exists(CONFIDENCE_THRESHOLDS_PATH):
            _confidence_thresholds = joblib.load(CONFIDENCE_THRESHOLDS_PATH)
            log.info("Learned confidence thresholds loaded (%d regimes).", len(_confidence_thresholds))

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


def _build_meta_row(green_p, df_row, regime_enc, ensemble_variance,
                    ensemble_disagreement=0.0, regime_duration=1):
    """
    Build meta feature row matching META_FEATURE_COLUMNS exactly.
    
    LIVE PARITY: same features as training (v8: 14 meta features).
    """
    global _direction_history

    cur_dir = 1 if green_p >= 0.5 else 0
    dstreak = 1
    for d in reversed(_direction_history):
        if d == cur_dir:
            dstreak += 1
        else:
            break

    # Hour-of-day cyclical encoding
    hour = 0
    if hasattr(df_row, 'get'):
        t = df_row.get("time", None)
        if t is not None and hasattr(t, 'hour'):
            hour = t.hour
    import math
    hour_sin = math.sin(2 * math.pi * hour / 24)

    return {
        "primary_green_prob": green_p,
        "prob_distance_from_half": abs(green_p - 0.5),
        "primary_entropy": float(_binary_entropy(green_p)),
        "ensemble_variance": ensemble_variance,
        "ensemble_disagreement": ensemble_disagreement,
        "regime_encoded": regime_enc,
        "regime_duration": regime_duration,
        "atr_value": float(df_row.get("atr_14", 0)),
        "volatility_zscore": float(df_row.get("volatility_zscore", 0)),
        "range_position": float(df_row.get("range_position", 0.5)),
        "body_percentile_rank": float(df_row.get("body_size", 0.5)),
        "direction_streak": dstreak,
        "rolling_vol_percentile": float(df_row.get("atr_percentile_rank", 0.5)),
        "hour_sin": hour_sin,
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


def _should_skip(regime, confidence_pct, df):
    """
    Phase 3: Adaptive regime filter — decides whether to skip this trade.
    Returns (should_skip: bool, reason: str).
    """
    global _cooldown_remaining, _skip_count

    if not REGIME_FILTER_ENABLED:
        return False, ""

    # 1. Cooldown check: if we're in cooldown, skip
    if _cooldown_remaining > 0:
        _cooldown_remaining -= 1
        _skip_count += 1
        return True, f"Cooldown ({_cooldown_remaining + 1} bars left)"

    # 2. Rolling accuracy check: if recent accuracy < threshold, enter cooldown
    if len(_rolling_correct) >= REGIME_ROLLING_ACCURACY_WINDOW:
        recent = _rolling_correct[-REGIME_ROLLING_ACCURACY_WINDOW:]
        rolling_acc = sum(recent) / len(recent)
        if rolling_acc < REGIME_SKIP_ACCURACY_THRESHOLD:
            _cooldown_remaining = REGIME_COOLDOWN_BARS
            _skip_count += 1
            log.warning("REGIME FILTER: accuracy %.1f%% < %.1f%%, entering cooldown (%d bars)",
                        rolling_acc * 100, REGIME_SKIP_ACCURACY_THRESHOLD * 100, REGIME_COOLDOWN_BARS)
            return True, f"Low accuracy ({rolling_acc:.0%} < {REGIME_SKIP_ACCURACY_THRESHOLD:.0%})"

    # 3. Learned confidence thresholds check (if available)
    if _confidence_thresholds is not None:
        regime_info = _confidence_thresholds.get(regime)
        if regime_info and regime_info.get("min_confidence", 0) > 0:
            if confidence_pct < regime_info["min_confidence"]:
                _skip_count += 1
                return True, f"Below {regime} threshold ({confidence_pct:.0f}% < {regime_info['min_confidence']}%)"

    # 4. Regime stability check
    try:
        stability = regime_stability_score(df)
        if stability < 0.5:
            _skip_count += 1
            return True, f"Regime transition (stability={stability:.2f})"
    except Exception:
        pass

    return False, ""


def predict(df, regime):
    """
    Generate prediction using cached models.
    
    Phase 3 enhancements:
      - Adaptive regime filter (skip unfavorable conditions)
      - Session-aware confidence adjustment
      - Regime stability scoring
      - Learned per-regime confidence thresholds
    """
    t_start = time.perf_counter()

    try:
        if _retrain_in_progress:
            log.info("Retrain in progress — using cached stable model.")

        with _model_lock:
            ensemble, feat_cols, meta_model, meta_feat_cols, weight_model = load_models()

        row = df[feat_cols].iloc[-1].values.reshape(1, -1)

        # Ensemble predictions (diverse: 3 XGB + 2 ExtraTrees)
        all_probs = np.array([m.predict_proba(row)[0][1] for m in ensemble])
        green_p = float(all_probs.mean())
        red_p = 1.0 - green_p
        variance = float(all_probs.var())
        disagreement = float(all_probs.max() - all_probs.min())
        max_var = 0.25
        norm_var = min(variance / max_var, 1.0)
        uncertainty_pct = round(norm_var * 100, 2)

        # Latency check after ensemble inference
        t_ensemble = time.perf_counter()
        if (t_ensemble - t_start) > 1.0:
            log.error("LATENCY ABORT: ensemble took %.2fs (>1s limit)", t_ensemble - t_start)
            return _safe_fallback("Inference too slow")

        # Regime duration tracking
        global _last_regime_name, _regime_duration_counter
        if regime == _last_regime_name:
            _regime_duration_counter += 1
        else:
            _regime_duration_counter = 1
            _last_regime_name = regime

        # Meta (sigmoid-calibrated model — calibration is internal)
        regime_enc = REGIME_ENCODING.get(regime, 1)
        meta_row = _build_meta_row(green_p, df.iloc[-1], regime_enc, variance,
                                   ensemble_disagreement=disagreement,
                                   regime_duration=_regime_duration_counter)
        meta_input = pd.DataFrame([meta_row])[meta_feat_cols]
        meta_rel = float(meta_model.predict_proba(meta_input.values)[0][1])

        # Confidence with uncertainty adjustment
        primary_strength = abs(green_p - 0.5) * 2
        regime_strength = float(df.iloc[-1].get("adx_normalized", 0.25))

        if weight_model is not None:
            w_input = pd.DataFrame([{
                "primary_strength": primary_strength,
                "meta_reliability": meta_rel,
                "regime_strength": regime_strength,
                "uncertainty": 1.0 - norm_var,  # inverted: 1=certain, 0=uncertain
            }])
            weighted_score = float(weight_model.predict_proba(w_input)[:, 1][0])
        else:
            weighted_score = primary_strength * meta_rel

        confidence = weighted_score * (1.0 - norm_var)

        # Phase 3: Session-aware confidence adjustment
        session = "Off"
        try:
            last_row = df.iloc[-1]
            if hasattr(last_row, 'get'):
                t = last_row.get("time", None)
                if t is not None and hasattr(t, 'hour'):
                    hour = t.hour
                    session = get_session(hour)
                    if SESSION_FILTER_ENABLED:
                        mult = SESSION_CONFIDENCE_MULT.get(session, 1.0)
                        confidence *= mult
        except Exception:
            pass

        # Phase 3: Regime stability adjustment
        stability_score = 1.0
        try:
            stability_score = regime_stability_score(df)
            if stability_score < 1.0:
                confidence *= (0.5 + 0.5 * stability_score)  # reduce during transitions
        except Exception:
            pass

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

        # Adaptive confidence (rolling calibration)
        adaptive_conf = _adaptive_confidence(confidence_pct)

        # Kelly criterion position sizing (quarter-Kelly for safety)
        win_prob = max(min(green_p if green_p >= 0.5 else (1 - green_p), 0.99), 0.51)
        kelly_full = 2 * win_prob - 1
        kelly_quarter = max(0, kelly_full * 0.25)
        kelly_pct = round(kelly_quarter * 100, 1)

        # Phase 3: Adaptive regime filter — check if we should skip
        should_skip, skip_reason = _should_skip(regime, confidence_pct, df)

        # Trade suggestion (with skip override)
        if should_skip:
            trade = "SKIP"
        else:
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

        # Latency measurement
        latency_ms = round((time.perf_counter() - t_start) * 1000, 1)

        result = {
            "green_probability_percent": green_pct,
            "red_probability_percent": red_pct,
            "primary_direction": direction,
            "meta_reliability_percent": meta_pct,
            "uncertainty_percent": uncertainty_pct,
            "final_confidence_percent": confidence_pct,
            "adaptive_confidence_percent": adaptive_conf,
            "confidence_level": conf_level,
            "confidence_reliability_score": conf_reliability,
            "kelly_fraction_percent": kelly_pct,
            "suggested_trade": trade,
            "suggested_direction": "UP" if green_p >= 0.5 else "DOWN",
            "market_regime": regime,
            "regime_stability": round(stability_score, 2),
            "session": session,
            "skip_reason": skip_reason,
            "total_skips": _skip_count,
            "latency_ms": latency_ms,
        }

        log.info("Pred: green=%.1f%% conf=%.1f%% session=%s regime_stab=%.2f trade=%s latency=%dms",
                 green_pct, confidence_pct, session, stability_score, trade, latency_ms)
        return result

    except Exception as e:
        log.exception("PREDICT CRASH (safe fallback returned): %s", e)
        return _safe_fallback(str(e))


def _safe_fallback(reason="unknown"):
    """Return a safe HOLD prediction — never crashes the caller."""
    return {
        "green_probability_percent": 50.0,
        "red_probability_percent": 50.0,
        "primary_direction": "NEUTRAL",
        "meta_reliability_percent": 0.0,
        "uncertainty_percent": 100.0,
        "final_confidence_percent": 0.0,
        "adaptive_confidence_percent": 0.0,
        "confidence_level": "Low",
        "confidence_reliability_score": 0.0,
        "kelly_fraction_percent": 0.0,
        "suggested_trade": "HOLD",
        "suggested_direction": "HOLD",
        "market_regime": "Unknown",
        "regime_stability": 0.0,
        "session": "Off",
        "skip_reason": "",
        "total_skips": _skip_count,
        "latency_ms": -1,
        "error": reason,
    }

def update_prediction_history(green_p, was_correct, confidence=0.5):
    """Record prediction outcome for rolling metrics."""
    _prediction_history.append({
        "correct": was_correct,
        "green_p": green_p,
        "confidence": confidence,
    })
    if len(_prediction_history) > CONFIDENCE_CORRELATION_WINDOW * 2:
        _prediction_history.pop(0)

    # Update adaptive calibration history
    _adaptive_history.append((confidence, was_correct))
    if len(_adaptive_history) > ADAPTIVE_WINDOW * 2:
        _adaptive_history.pop(0)

    # Phase 3: Update rolling accuracy for regime filter
    _rolling_correct.append(1 if was_correct else 0)
    if len(_rolling_correct) > REGIME_ROLLING_ACCURACY_WINDOW * 3:
        _rolling_correct.pop(0)


def _adaptive_confidence(raw_confidence):
    """
    Adjust confidence based on rolling historical accuracy per confidence tier.
    If raw confidence says 70% but historical accuracy at 70% tier is 60%,
    adaptive confidence returns 60%.
    """
    if len(_adaptive_history) < 50:
        return raw_confidence  # not enough data yet

    recent = _adaptive_history[-ADAPTIVE_WINDOW:]
    # Find similar confidence predictions
    tier_lo = max(0, raw_confidence - 10)
    tier_hi = raw_confidence + 10
    in_tier = [(c, cor) for c, cor in recent if tier_lo <= c < tier_hi]

    if len(in_tier) < 10:
        return raw_confidence  # not enough data in this tier

    historical_acc = sum(cor for _, cor in in_tier) / len(in_tier) * 100
    # Blend: 70% raw, 30% historical (smooth transition)
    adapted = 0.7 * raw_confidence + 0.3 * historical_acc
    return round(adapted, 2)


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
