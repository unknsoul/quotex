"""
Signal Ranker — v2 (Accuracy-Optimized).

Ranks signals by composite quality score incorporating:
  - Primary confidence + meta reliability (model trust)
  - Ensemble unanimity + low variance (agreement quality)
  - Multi-TF confluence (directional alignment)
  - Kelly edge (mathematical edge)
  - Regime stability (environment quality)
  - Momentum quality (price action confirmation)
  - Win streak bonus / loss streak penalty per symbol

Returns only the top N signals above the minimum score.
"""

import logging
import math
from datetime import datetime, timezone

log = logging.getLogger("signal_ranker")

# Per-symbol performance tracking for adaptive scoring
_symbol_stats: dict = {}  # {symbol: {"wins": int, "total": int, "streak": int}}


def update_symbol_stats(symbol: str, was_correct: bool):
    """Update per-symbol win/loss stats for ranking adjustments."""
    if symbol not in _symbol_stats:
        _symbol_stats[symbol] = {"wins": 0, "total": 0, "streak": 0}
    s = _symbol_stats[symbol]
    s["total"] += 1
    if was_correct:
        s["wins"] += 1
        s["streak"] = max(1, s["streak"] + 1)
    else:
        s["streak"] = min(-1, s["streak"] - 1)
    # Keep rolling window of last 50
    if s["total"] > 50:
        s["wins"] = round(s["wins"] * 40 / s["total"])
        s["total"] = 40


def _symbol_accuracy_bonus(symbol: str) -> float:
    """Return a -0.10 to +0.10 bonus based on recent symbol accuracy."""
    s = _symbol_stats.get(symbol)
    if not s or s["total"] < 5:
        return 0.0
    acc = s["wins"] / s["total"]
    # Bonus/penalty: 70% accuracy → +0.05, 40% → -0.05
    return round((acc - 0.55) * 0.50, 3)


def _streak_bonus(symbol: str) -> float:
    """Return small bonus for win streaks, penalty for loss streaks."""
    s = _symbol_stats.get(symbol)
    if not s:
        return 0.0
    streak = s["streak"]
    if streak >= 3:
        return 0.03
    elif streak >= 2:
        return 0.015
    elif streak <= -3:
        return -0.05
    elif streak <= -2:
        return -0.025
    return 0.0


def _momentum_quality_score(pred: dict) -> float:
    """Score based on how well price action supports the signal direction."""
    quality = pred.get("quality_score", 50)
    regime_stab = pred.get("regime_stability", 0.5)
    # Combine quality and stability into 0-1 range
    q_norm = min(max(quality, 0) / 100, 1.0)
    return q_norm * (0.6 + 0.4 * regime_stab)


def _session_quality_bonus(pred: dict) -> float:
    """v13: No session-based scoring penalty — all sessions equal."""
    return 0.0


def compute_signal_score(pred: dict, symbol: str = "", confluence_score: int = 3) -> float:
    """
    Compute a composite quality score for a single prediction.

    v3 improvements (v12):
      - Ensemble disagreement penalty (high variance = low quality)
      - Dispatch model win probability boost
      - Improved weighting with disagreement factor
      - All v2 features retained
    """
    conf = pred.get("final_confidence_percent", 0)
    meta = pred.get("meta_reliability_percent", 50)
    uncertainty = pred.get("uncertainty_percent", 50)
    unanimity = pred.get("ensemble_unanimity", 0.5)
    kelly = pred.get("kelly_fraction_percent", 0)
    actual_confluence = pred.get("_confluence_score", confluence_score)
    regime_stab = pred.get("regime_stability", 0.5)
    variance = pred.get("ensemble_variance", 0.01)

    # Normalize components to 0-1 range
    conf_norm = min(conf / 100, 1.0)
    meta_norm = min(meta / 100, 1.0)
    uncert_norm = 1.0 - min(uncertainty / 100, 1.0)
    confl_norm = actual_confluence / 3.0
    kelly_norm = min(kelly / 10.0, 1.0)

    # Non-linear unanimity: reward strong agreement (5/6, 6/6) more than 4/6
    unanimity_scaled = math.pow(max(unanimity, 0.5), 0.8)

    # Momentum quality factor
    mom_quality = _momentum_quality_score(pred)

    # Regime stability factor (0.5 - 1.0)
    stab_norm = 0.5 + 0.5 * min(regime_stab, 1.0)

    # v12: Ensemble disagreement factor — penalizes high variance
    # Low variance (0.001) → 1.0 (full quality), High variance (0.05) → 0.5 (halved)
    disagree_factor = max(0.5, 1.0 - variance * 10.0)

    # v12: Dispatch model win probability (if available)
    dispatch_wp = pred.get("dispatch_win_prob", 0.5)
    dispatch_bonus = (dispatch_wp - 0.5) * 0.10  # ±5% based on dispatch model

    # Weighted composite (v3 retuned with disagreement)
    base_score = (
        conf_norm         * 0.20 +   # 20% primary signal strength
        meta_norm         * 0.12 +   # 12% meta model validation
        unanimity_scaled  * 0.16 +   # 16% ensemble agreement (non-linear)
        confl_norm        * 0.12 +   # 12% multi-TF confluence
        uncert_norm       * 0.08 +   # 8% model certainty
        kelly_norm        * 0.08 +   # 8% Kelly edge
        mom_quality       * 0.08 +   # 8% momentum quality
        stab_norm         * 0.06 +   # 6% regime stability
        disagree_factor   * 0.10     # 10% ensemble agreement quality (NEW v12)
    )

    # Additive bonuses
    bonus = (
        _session_quality_bonus(pred) +
        _symbol_accuracy_bonus(symbol) +
        _streak_bonus(symbol) +
        dispatch_bonus  # v12: dispatch model bonus
    )

    final = max(0, min(100, (base_score + bonus) * 100))
    return round(final, 2)


def rank_and_select(predictions: dict, max_signals: int = 2,
                    min_score: float = 55.0) -> dict:
    """
    Rank all qualifying predictions and return only the best N.
    v2: Uses symbol-aware scoring with performance history.
    """
    scored = []

    for sym, pred in predictions.items():
        score = compute_signal_score(pred, symbol=sym)
        pred["signal_score"] = score
        scored.append((sym, pred, score))

    # Sort by score descending
    scored.sort(key=lambda x: x[2], reverse=True)

    # Select top N with minimum score
    selected = {}
    for sym, pred, score in scored[:max_signals]:
        if score >= min_score:
            selected[sym] = pred
            log.info("RANKED: %s score=%.1f (selected)", sym, score)
        else:
            log.info("RANKED: %s score=%.1f (below min %.1f)", sym, score, min_score)

    # Log rejected
    for sym, pred, score in scored[max_signals:]:
        log.info("RANKED: %s score=%.1f (not top-%d)", sym, score, max_signals)

    return selected
