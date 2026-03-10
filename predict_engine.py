# -*- coding: utf-8 -*-
"""
Predict Engine — v11 Advanced Production Pipeline.

v11 upgrades (full plan):
  - Same-Candle Detector (SC-1 to SC-5): stale/frozen candle detection
  - 9-Strategy Signal Engine: parallel evaluation with composite scoring
  - 12-Pattern Candlestick Overlay: confirm/contradict scoring
  - Staleness Tracker: crossover signal decay
  - CHOPPY regime signal suspension
  - Hour-strategy confidence scaling (replaces hour blocking)
  - Anti-streak engine: prevents directional collapse
  - Candle quality filter: detects stale/repetitive candles
  - Momentum exhaustion: detects trend exhaustion
  - Symbol confidence adjustments: data-driven per-symbol scaling
  - Weak-hour extra confirmation requirement
  - Race condition protection via retrain lock

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

from feature_fingerprint import check_fingerprint

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
    ENSEMBLE_VAR_SKIP_THRESHOLD, ENSEMBLE_VAR_FILTER_ENABLED,
    PRODUCTION_SIGNAL_GATING_ENABLED, PRODUCTION_MIN_CONFIDENCE,
    PRODUCTION_MIN_META_RELIABILITY, PRODUCTION_MIN_UNANIMITY,
    PRODUCTION_MAX_UNCERTAINTY, PRODUCTION_MIN_QUALITY_SCORE,
    PRODUCTION_CONFIDENCE_ALERT_PENALTY, PRODUCTION_REQUIRE_TREND_ALIGNMENT,
    PRODUCTION_BLOCKED_REGIMES,
    LOG_LEVEL, LOG_FORMAT,
)

# v11 imports
from config import (
    V11_ANTI_STREAK_ENABLED, V11_CANDLE_QUALITY_ENABLED,
    V11_CANDLE_FRESHNESS_MIN, V11_CANDLE_QUALITY_MIN,
    V11_EXHAUSTION_ENABLED, V11_EXHAUSTION_SKIP_THRESHOLD,
    V11_EXHAUSTION_PENALTY_START,
    V11_SYMBOL_CONFIDENCE_ADJUSTMENTS,
    V11_SAME_CANDLE_ENABLED, V11_STRATEGY_ENGINE_ENABLED,
    V11_STALENESS_ENABLED, V11_PATTERN_OVERLAY_ENABLED,
    V11_CHOPPY_REGIME_SUSPEND,
)
from candle_quality import candle_quality_score, candle_freshness_score
from anti_streak import apply_streak_correction, get_streak_engine
from momentum_exhaust import compute_exhaustion_score
from session_filter import get_hour_strategy, get_hour_confidence_multiplier
from same_candle_detector import detect_same_candle
from strategy_scorer import score_strategies, get_regime_weight_overrides
from staleness_tracker import track_crossover_staleness
from candle_patterns import compute_pattern_score_adjustment
from regime_detection import (
    get_regime_thresholds, regime_stability_score, get_session,
    get_trend_alignment,
)
from calibration import CalibratedModel  # needed for joblib.load
from online_learner import OnlineLearner

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
_online_learner = OnlineLearner(n_features=14, learning_rate=0.005)

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


def _direction_probability(green_p):
    return green_p if green_p >= 0.5 else (1.0 - green_p)


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

            # Feature fingerprint parity check
            try:
                fp = check_fingerprint(None, _primary_features)
                if fp.get('status') == 'drift':
                    mismatches = fp.get('mismatches', [])
                    log.warning('FEATURE MISMATCH: %d features drifted — model may be stale. %s',
                                len(mismatches), mismatches[:5])
                elif fp.get('status') == 'skipped':
                    log.info('Feature fingerprint loaded (%d features).', fp.get('n_checked', 0))
                else:
                    log.info('Feature fingerprint OK (%d features).', fp.get('n_checked', 0))
            except Exception as e:
                log.warning('Fingerprint check skipped: %s', e)

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
        log.info("Last candle not closed (time=%s, closes=%s, now=%s). "
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


def _should_skip(regime, confidence_pct, df, ensemble_variance=0.0):
    """
    Phase 3+4: Adaptive regime filter -- decides whether to skip this trade.
    Returns (should_skip: bool, reason: str).
    """
    global _cooldown_remaining, _skip_count

    if not REGIME_FILTER_ENABLED:
        return False, ""

    # 0. Phase 4: Ensemble variance hard filter (highest precision filter)
    if ENSEMBLE_VAR_FILTER_ENABLED and ensemble_variance > ENSEMBLE_VAR_SKIP_THRESHOLD:
        _skip_count += 1
        return True, f"High ensemble disagreement (var={ensemble_variance:.4f} > {ENSEMBLE_VAR_SKIP_THRESHOLD})"

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


def _quality_score(confidence_pct, meta_pct, unanimity, stability_score, uncertainty_pct, green_p):
    edge_pct = abs(green_p - 0.5) * 200.0

    # v2: Multi-factor quality with non-linear scaling
    # Strong edges get bonus; weak edges get penalty
    edge_bonus = 0.0
    if edge_pct > 30:  # green_p > 0.65 or < 0.35
        edge_bonus = (edge_pct - 30) * 0.15
    elif edge_pct < 10:  # green_p between 0.45-0.55 (weak)
        edge_bonus = (edge_pct - 10) * 0.20  # negative penalty

    # Unanimity bonus for 5/6 or 6/6
    unanimity_bonus = 0.0
    if unanimity >= 0.833:  # 5/6+
        unanimity_bonus = 8.0
    elif unanimity >= 1.0:  # 6/6
        unanimity_bonus = 15.0

    return round(
        0.28 * confidence_pct +
        0.15 * meta_pct +
        0.22 * (unanimity * 100.0) +
        0.10 * (stability_score * 100.0) +
        0.18 * edge_pct -
        0.30 * uncertainty_pct +
        edge_bonus +
        unanimity_bonus,
        2,
    )


def _apply_production_gate(df, regime, green_p, meta_rel, confidence_pct,
                           adaptive_conf_pct, unanimity, uncertainty_pct,
                           stability_score, conf_alert, ensemble_variance):
    effective_conf = float(min(confidence_pct, adaptive_conf_pct))
    if conf_alert:
        effective_conf *= PRODUCTION_CONFIDENCE_ALERT_PENALTY

    meta_pct = float(meta_rel * 100.0)
    direction_prob = _direction_probability(green_p)
    thresholds = get_regime_thresholds(regime)
    quality_score = _quality_score(
        effective_conf, meta_pct, unanimity, stability_score, uncertainty_pct, green_p
    )

    reasons = []
    soft_failures = []
    should_skip, skip_reason = _should_skip(
        regime, effective_conf, df, ensemble_variance=ensemble_variance
    )
    if should_skip:
        reasons.append(skip_reason)

    if regime in PRODUCTION_BLOCKED_REGIMES:
        reasons.append(f"Blocked regime ({regime})")
    if effective_conf < PRODUCTION_MIN_CONFIDENCE:
        reasons.append(f"Low confidence ({effective_conf:.0f}% < {PRODUCTION_MIN_CONFIDENCE:.0f}%)")
    if meta_pct < max(PRODUCTION_MIN_META_RELIABILITY, thresholds["meta"] * 100.0):
        soft_failures.append(
            f"Meta weak ({meta_pct:.0f}% < {max(PRODUCTION_MIN_META_RELIABILITY, thresholds['meta'] * 100.0):.0f}%)"
        )
    primary_edge_floor = max(0.52, thresholds["primary"] - 0.04)
    if direction_prob < primary_edge_floor:
        soft_failures.append(
            f"Primary edge weak ({direction_prob * 100:.0f}% < {primary_edge_floor * 100:.0f}%)"
        )
    if unanimity < PRODUCTION_MIN_UNANIMITY:
        soft_failures.append(f"Split vote ({unanimity:.0%} < {PRODUCTION_MIN_UNANIMITY:.0%})")
    if uncertainty_pct > PRODUCTION_MAX_UNCERTAINTY:
        reasons.append(
            f"High uncertainty ({uncertainty_pct:.0f}% > {PRODUCTION_MAX_UNCERTAINTY:.0f}%)"
        )

    if PRODUCTION_REQUIRE_TREND_ALIGNMENT:
        trend_alignment = get_trend_alignment(df.iloc[-1])
        trade_alignment = 1 if green_p >= 0.5 else -1
        if trend_alignment != 0 and trend_alignment != trade_alignment:
            trend_name = "bullish" if trend_alignment > 0 else "bearish"
            soft_failures.append(f"Trend conflict ({trend_name} structure)")

    if quality_score < PRODUCTION_MIN_QUALITY_SCORE:
        reasons.append(
            f"Quality score low ({quality_score:.1f} < {PRODUCTION_MIN_QUALITY_SCORE:.1f})"
        )

    if len(soft_failures) >= 3:
        reasons.extend(soft_failures[:2])

    if PRODUCTION_SIGNAL_GATING_ENABLED and reasons:
        return "HOLD", "; ".join(reasons[:3]), round(effective_conf, 2), quality_score

    trade = "BUY" if green_p >= 0.5 else "SELL"
    return trade, "", round(effective_conf, 2), quality_score


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

        row = df[feat_cols].iloc[[-1]]

        # Ensemble predictions (V3: diverse multi-model ensemble)
        all_probs = np.array([m.predict_proba(row)[0][1] for m in ensemble])
        green_p = float(all_probs.mean())
        red_p = 1.0 - green_p
        variance = float(all_probs.var())
        disagreement = float(all_probs.max() - all_probs.min())
        max_var = 0.25
        norm_var = min(variance / max_var, 1.0)
        uncertainty_pct = round(norm_var * 100, 2)

        # Ensemble unanimity: how many models agree on direction?
        n_models = len(all_probs)
        if green_p >= 0.5:
            agree_count = int((all_probs >= 0.5).sum())
        else:
            agree_count = int((all_probs < 0.5).sum())
        unanimity = agree_count / n_models  # 0.0 to 1.0

        # Latency check after ensemble inference
        t_ensemble = time.perf_counter()
        if (t_ensemble - t_start) > 3.0:
            log.error("LATENCY ABORT: ensemble took %.2fs (>3s limit)", t_ensemble - t_start)
            return _safe_fallback("Inference too slow")

        # Regime duration tracking
        global _last_regime_name, _regime_duration_counter
        if regime == _last_regime_name:
            _regime_duration_counter += 1
        else:
            _regime_duration_counter = 1
            _last_regime_name = regime

        # FIX: Update direction history BEFORE building meta row
        # so direction_streak uses current prediction, not stale one
        _direction_history.append(1 if green_p >= 0.5 else 0)
        if len(_direction_history) > 500:
            _direction_history.pop(0)

        # Meta (sigmoid-calibrated model — calibration is internal)
        regime_enc = REGIME_ENCODING.get(regime, 1)
        meta_row = _build_meta_row(green_p, df.iloc[-1], regime_enc, variance,
                                   ensemble_disagreement=disagreement,
                                   regime_duration=_regime_duration_counter)
        meta_input = pd.DataFrame([meta_row])[meta_feat_cols]
        meta_rel = float(meta_model.predict_proba(meta_input)[0][1])

        # Phase 3: Blend online learner prediction after 200+ updates (ramps to 10%)
        if _online_learner.n_updates >= 200:
            try:
                ol_pred     = _online_learner.predict(meta_row)
                blend       = min(0.10, _online_learner.n_updates / 2000)
                meta_rel    = (1 - blend) * meta_rel + blend * float(ol_pred)
            except Exception as e:
                log.debug('Online blend skipped: %s', e)

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

        # v2: Strong signal confidence boost —
        # When ensemble strongly agrees (5/6+) AND variance is very low,
        # boost confidence slightly to reward high-quality setups
        if unanimity >= 0.833 and variance < 0.008:
            boost = 1.0 + 0.05 * (unanimity - 0.667)  # up to ~1.7% boost
            confidence *= boost
        # Penalty for borderline signals (4/6 with moderate variance)
        elif unanimity <= 0.667 and variance > 0.015:
            confidence *= 0.95  # small penalty

        # Phase 3: Session-aware confidence adjustment
        session = "Off"
        try:
            last_row = df.iloc[-1]
            if hasattr(last_row, 'get'):
                t = last_row.get("time", None)
                if t is not None and hasattr(t, 'hour'):
                    hour = t.hour
                    session = get_session(hour)
                    # v13: Session confidence penalty disabled — pure technical analysis
                    # if SESSION_FILTER_ENABLED:
                    #     mult = SESSION_CONFIDENCE_MULT.get(session, 1.0)
                    #     confidence *= mult
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

        # ═══════════════════════════════════════════════════════════
        # ═══ v11: NEW FILTERS (same-candle, strategy engine, staleness, patterns,
        # ═══     anti-streak, candle quality, exhaustion, hour/symbol gating)
        # ═══════════════════════════════════════════════════════════
        
        v11_skip_reasons = []
        predicted_dir = "UP" if green_p >= 0.5 else "DOWN"
        
        # v11 Layer 0: Same-Candle Detector (SC-1 to SC-5) ─── GATE
        sc_info = {"should_skip": False, "flags": [], "confidence_penalty": 1.0, "details": "clear"}
        if V11_SAME_CANDLE_ENABLED:
            try:
                sc_info = detect_same_candle(df)
                if sc_info["should_skip"]:
                    v11_skip_reasons.append(f"Same-candle: {sc_info['details']}")
                elif sc_info["confidence_penalty"] < 1.0:
                    confidence *= sc_info["confidence_penalty"]
            except Exception as e:
                log.debug("Same-candle detector failed: %s", e)
        
        # v11 Layer 0b: CHOPPY Regime Suspension
        if V11_CHOPPY_REGIME_SUSPEND and regime == "CHOPPY":
            v11_skip_reasons.append("CHOPPY regime — all signals suspended")
        
        # v11.1 Layer 1: Hour-Strategy Confidence Scaling (replaces hour blocking)
        hour_val = None
        hour_strategy = {"multiplier": 1.0, "quality": "normal", "require_extra_confirmation": False}
        try:
            last_row = df.iloc[-1]
            if hasattr(last_row, 'get'):
                t = last_row.get("time", None)
                if t is not None and hasattr(t, 'hour'):
                    hour_val = t.hour
        except Exception:
            pass
        
        if hour_val is not None:
            hour_strategy = get_hour_strategy(hour_val)
            # v13: Hour-strategy scaling disabled — no hour penalty or extra confirmation
            # confidence *= hour_strategy["multiplier"]
            # if hour_strategy["require_extra_confirmation"] and unanimity < 0.80:
            #     v11_skip_reasons.append(...)
        
        # v11 Layer 2: Candle Quality Filter (stale/repetitive candles)
        candle_info = {"quality": 50.0, "freshness": 1.0}
        if V11_CANDLE_QUALITY_ENABLED:
            try:
                candle_info = candle_quality_score(df)
                if candle_info["freshness"] < V11_CANDLE_FRESHNESS_MIN:
                    v11_skip_reasons.append(
                        f"Stale candle (freshness={candle_info['freshness']:.2f})"
                    )
                elif candle_info["quality"] < V11_CANDLE_QUALITY_MIN:
                    v11_skip_reasons.append(
                        f"Low candle quality ({candle_info['quality']:.0f})"
                    )
            except Exception as e:
                log.debug("Candle quality check failed: %s", e)
        
        # v11 Layer 3: Anti-Streak Engine (prevents directional collapse)
        streak_reason = ""
        if V11_ANTI_STREAK_ENABLED:
            try:
                adj_conf, force_hold, streak_reason = apply_streak_correction(
                    predicted_dir, confidence * 100
                )
                if force_hold:
                    v11_skip_reasons.append(streak_reason)
                elif streak_reason:
                    confidence = adj_conf / 100.0
            except Exception as e:
                log.debug("Anti-streak check failed: %s", e)
        
        # v11 Layer 4: Momentum Exhaustion
        exhaustion_info = {"exhaustion_score": 0, "should_skip": False}
        if V11_EXHAUSTION_ENABLED:
            try:
                exhaustion_info = compute_exhaustion_score(df, predicted_dir)
                if exhaustion_info["should_skip"]:
                    v11_skip_reasons.append(exhaustion_info.get("reason", "Trend exhausted"))
                elif exhaustion_info["exhaustion_score"] > V11_EXHAUSTION_PENALTY_START:
                    # Proportional penalty between 50-75%
                    excess = exhaustion_info["exhaustion_score"] - V11_EXHAUSTION_PENALTY_START
                    penalty = min(excess / 50.0 * 0.15, 0.15)
                    confidence *= (1.0 - penalty)
            except Exception as e:
                log.debug("Exhaustion check failed: %s", e)
        
        # v11 Layer 5: Symbol-Specific Confidence Adjustment
        # Applied later in telegram_bot when symbol is known, but if df has symbol info:
        # (This is handled in the result dict for the caller to apply)
        
        # v11 Layer 6: Strategy Engine (9 parallel strategies)
        strategy_result = {"signal_valid": True, "composite_score": 0, "composite_direction": "NEUTRAL",
                          "strategies_agreeing": 0, "details": "", "per_strategy": {}}
        if V11_STRATEGY_ENGINE_ENABLED and not v11_skip_reasons:
            try:
                # Determine H1 direction from confluence
                h1_dir = "NEUTRAL"
                regime_weights = get_regime_weight_overrides(regime.upper().replace(" ", "_"))
                strategy_result = score_strategies(
                    df, h1_direction=h1_dir, regime=regime, regime_weights=regime_weights
                )
                # Strategy validation acts as confirmation — not a hard gate
                # but reduces confidence if strategies disagree with ML
                if strategy_result["composite_direction"] != "NEUTRAL":
                    if strategy_result["composite_direction"] != predicted_dir:
                        # Strategy engine disagrees with ML — reduce confidence
                        confidence *= 0.90
                    elif strategy_result["signal_valid"]:
                        # Strategy engine confirms ML with strong agreement — boost
                        confidence *= 1.05
            except Exception as e:
                log.debug("Strategy engine failed: %s", e)
        
        # v11 Layer 7: Staleness Tracker (crossover signal decay)
        staleness_info = {"overall_freshness": 0.5, "is_fresh": True}
        if V11_STALENESS_ENABLED and not v11_skip_reasons:
            try:
                staleness_info = track_crossover_staleness(df)
                if staleness_info["overall_freshness"] < 0.3 and not staleness_info["is_fresh"]:
                    confidence *= 0.92  # slight penalty for stale crossovers
            except Exception as e:
                log.debug("Staleness tracker failed: %s", e)
        
        # v11 Layer 8: Candlestick Pattern Overlay (12 patterns)
        pattern_info = {"adjustment": 0, "patterns": [], "suppressed": False,
                       "momentum_exhaustion": False, "details": ""}
        if V11_PATTERN_OVERLAY_ENABLED and not v11_skip_reasons:
            try:
                pattern_info = compute_pattern_score_adjustment(df, predicted_dir)
                # Apply as confidence adjustment (scaled from ±20 score to ±2% confidence)
                if pattern_info["adjustment"] != 0:
                    conf_adj = pattern_info["adjustment"] / 1000.0  # ±0.02 max
                    confidence *= (1.0 + conf_adj)
            except Exception as e:
                log.debug("Pattern overlay failed: %s", e)
        
        # ═══ END v11 FILTERS ═══════════════════════════════════════

        # Adaptive confidence (rolling calibration)
        raw_confidence_pct = round(confidence * 100, 2)
        adaptive_conf = _adaptive_confidence(raw_confidence_pct)
        
        # If v11 filters say skip, force HOLD before production gate
        if v11_skip_reasons:
            trade = "HOLD"
            skip_reason = "; ".join(v11_skip_reasons[:3])
            confidence_pct = raw_confidence_pct
            quality_score = 0.0
        else:
            trade, skip_reason, confidence_pct, quality_score = _apply_production_gate(
                df=df,
                regime=regime,
                green_p=green_p,
                meta_rel=meta_rel,
                confidence_pct=raw_confidence_pct,
                adaptive_conf_pct=adaptive_conf,
                unanimity=unanimity,
                uncertainty_pct=uncertainty_pct,
                stability_score=stability_score,
                conf_alert=conf_alert,
                ensemble_variance=variance,
            )

        # Convert to percentages
        green_pct = round(green_p * 100, 2)
        red_pct = round(red_p * 100, 2)
        meta_pct = round(meta_rel * 100, 2)
        direction = "GREEN" if green_p >= 0.5 else "RED"

        if confidence_pct >= CONFIDENCE_HIGH_MIN:
            conf_level = "High"
        elif confidence_pct >= CONFIDENCE_MEDIUM_MIN:
            conf_level = "Medium"
        else:
            conf_level = "Low"

        # Kelly criterion position sizing (quarter-Kelly, binary options payoff)
        win_prob = min(_direction_probability(green_p), 0.99)
        payout_ratio = 0.80  # typical binary options payout: 80% on win
        if trade == "HOLD" or confidence_pct < 50.0:
            kelly_pct = 0.0
        else:
            # Kelly for asymmetric payoff: f = (p*b - q) / b
            # where b = payout ratio, p = win prob, q = 1 - p
            kelly_full = (win_prob * payout_ratio - (1 - win_prob)) / payout_ratio
            kelly_quarter = max(0, kelly_full * 0.25)
            kelly_pct = round(kelly_quarter * 100, 1)

        # Direction history already updated above (before meta row building)

        # v11: Record prediction in anti-streak engine
        if V11_ANTI_STREAK_ENABLED and trade != "HOLD":
            try:
                get_streak_engine().record_prediction(predicted_dir)
            except Exception:
                pass

        # Latency measurement
        latency_ms = round((time.perf_counter() - t_start) * 1000, 1)

        # Build signal strength label and reason chain
        _strength_label = "Weak"
        if confidence_pct >= 70:
            _strength_label = "Strong"
        elif confidence_pct >= 58:
            _strength_label = "Moderate"

        _reasons = []
        if unanimity >= 0.667:
            _reasons.append(f"Models agree ({agree_count}/{n_models})")
        if meta_pct >= 55:
            _reasons.append(f"Meta trust {meta_pct:.0f}%")
        if stability_score >= 0.8:
            _reasons.append(f"Stable {regime.replace('_', ' ')} regime")
        if uncertainty_pct < 3:
            _reasons.append("Low uncertainty")
        if kelly_pct > 0:
            _reasons.append(f"Kelly {kelly_pct:.1f}%")
        reason_chain = " | ".join(_reasons[:4]) if _reasons else "Marginal edge"

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
            "suggested_size_percent": kelly_pct,  # Phase 4: alias for clarity
            "ensemble_variance": round(variance, 6),  # Phase 4: exposed for monitoring
            "ensemble_unanimity": round(unanimity, 3),  # Strategy 2: fraction of models agreeing
            "ensemble_agree_count": agree_count,         # e.g. 6/7
            "ensemble_total": n_models,
            "suggested_trade": trade,
            "suggested_direction": "HOLD" if trade == "HOLD" else ("UP" if green_p >= 0.5 else "DOWN"),
            "error": None,
            "_meta_row": meta_row,
            "market_regime": regime,
            "regime_stability": round(stability_score, 2),
            "session": session,
            "skip_reason": skip_reason,
            "quality_score": quality_score,
            "signal_strength": _strength_label,
            "reason_chain": reason_chain,
            "total_skips": _skip_count,
            "latency_ms": latency_ms,
            # v11 fields
            "candle_freshness": candle_info.get("freshness", 1.0),
            "candle_quality": candle_info.get("quality", 50.0),
            "exhaustion_score": exhaustion_info.get("exhaustion_score", 0),
            "streak_length": get_streak_engine().current_streak_len if V11_ANTI_STREAK_ENABLED else 0,
            "hour_quality": hour_strategy.get("quality", "normal"),
            "hour_multiplier": hour_strategy.get("multiplier", 1.0),
            # v11 advanced fields
            "same_candle_flags": sc_info.get("flags", []),
            "same_candle_details": sc_info.get("details", "clear"),
            "strategy_result": strategy_result,
            "strategy_composite": strategy_result.get("composite_score", 0),
            "strategies_agreeing": strategy_result.get("strategies_agreeing", 0),
            "strategy_direction": strategy_result.get("composite_direction", "NEUTRAL"),
            "staleness_freshness": staleness_info.get("overall_freshness", 0.5),
            "pattern_adjustment": pattern_info.get("adjustment", 0),
            "pattern_names": pattern_info.get("patterns", []),
            "pattern_suppressed": pattern_info.get("suppressed", False),
        }

        log.info(
            "Pred: green=%.1f%% conf=%.1f%% quality=%.1f session=%s regime_stab=%.2f trade=%s latency=%dms",
            green_pct, confidence_pct, quality_score, session, stability_score, trade, latency_ms,
        )
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
        # v11 fields
        "candle_freshness": 1.0,
        "candle_quality": 0.0,
        "exhaustion_score": 0,
        "streak_length": 0,
        "hour_quality": "unknown",
        "hour_multiplier": 1.0,
        # v11 advanced fields
        "same_candle_flags": [],
        "same_candle_details": "fallback",
        "strategy_result": {},
        "strategy_composite": 0,
        "strategies_agreeing": 0,
        "strategy_direction": "NEUTRAL",
        "staleness_freshness": 0.5,
        "pattern_adjustment": 0,
        "pattern_names": [],
        "pattern_suppressed": False,
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
    
    v12 upgrade: Isotonic-style recalibration using finer bins and monotonic
    smoothing. If raw=70% but historical accuracy at that tier is 60%,
    outputs ~60%. Also boosts underconfident tiers (raw=40% but actual=55% → 50%).
    """
    if len(_adaptive_history) < 30:
        return raw_confidence  # not enough data yet

    recent = _adaptive_history[-ADAPTIVE_WINDOW:]

    # Build 5-point calibration curve from outcome data
    bins = [(20, 35), (35, 45), (45, 55), (55, 65), (65, 95)]
    cal_points = []  # (midpoint, actual_accuracy)

    for lo, hi in bins:
        in_bin = [(c, cor) for c, cor in recent if lo <= c < hi]
        if len(in_bin) >= 5:
            mid = (lo + hi) / 2.0
            acc = sum(cor for _, cor in in_bin) / len(in_bin) * 100
            cal_points.append((mid, acc))

    if len(cal_points) < 2:
        # Not enough data for calibration curve — simple blend
        tier_lo = max(0, raw_confidence - 10)
        tier_hi = raw_confidence + 10
        in_tier = [(c, cor) for c, cor in recent if tier_lo <= c < tier_hi]
        if len(in_tier) < 5:
            return raw_confidence
        historical_acc = sum(cor for _, cor in in_tier) / len(in_tier) * 100
        return round(0.6 * raw_confidence + 0.4 * historical_acc, 2)

    # Interpolate between calibration points
    cal_points.sort(key=lambda x: x[0])
    
    # Clamp to range of calibration data
    if raw_confidence <= cal_points[0][0]:
        calibrated = cal_points[0][1]
    elif raw_confidence >= cal_points[-1][0]:
        calibrated = cal_points[-1][1]
    else:
        # Linear interpolation between nearest two points
        for i in range(len(cal_points) - 1):
            x0, y0 = cal_points[i]
            x1, y1 = cal_points[i + 1]
            if x0 <= raw_confidence <= x1:
                t = (raw_confidence - x0) / max(x1 - x0, 1e-6)
                calibrated = y0 + t * (y1 - y0)
                break
        else:
            calibrated = raw_confidence

    # Blend: 50% raw signal, 50% calibrated (stronger correction than v11's 70/30)
    adapted = 0.50 * raw_confidence + 0.50 * calibrated
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

def update_online_learner(meta_row: dict, was_correct: bool):
    '''Called from telegram_bot._check_outcomes() after each outcome.'''
    try:
        label = 1 if was_correct else 0
        _online_learner.update(meta_row, label)
        if _online_learner.n_updates % 20 == 0:
            log.info('Online learner: %d updates (accuracy: %.2f%%)',
                     _online_learner.n_updates, _online_learner.accuracy * 100)
    except Exception as e:
        log.warning('Online learner update failed: %s', e)
