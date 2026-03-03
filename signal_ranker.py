"""
Signal Ranker — Strategy 4.

Instead of sending ALL qualifying signals, ranks them by a
composite quality score and returns only the top N.

Score = confidence × meta_reliability × unanimity × (1 - uncertainty/100)

This ensures only the absolute best signals reach the user.
"""

import logging

log = logging.getLogger("signal_ranker")


def compute_signal_score(pred: dict, confluence_score: int = 3) -> float:
    """
    Compute a composite quality score for a single prediction.

    Components:
        - confidence (0-100) — how strong the model is
        - meta_reliability (0-100) — how trustworthy the meta model finds it
        - unanimity (0-1) — fraction of ensemble models agreeing
        - confluence (0-3) — multi-TF agreement
        - uncertainty (0-100) — lower is better

    Returns: score in [0, 100]
    """
    conf = pred.get("final_confidence_percent", 0)
    meta = pred.get("meta_reliability_percent", 50)
    uncertainty = pred.get("uncertainty_percent", 50)
    unanimity = pred.get("ensemble_unanimity", 0.5)
    kelly = pred.get("kelly_fraction_percent", 0)

    # Normalize components to 0-1 range
    conf_norm = min(conf / 100, 1.0)
    meta_norm = min(meta / 100, 1.0)
    uncert_norm = 1.0 - min(uncertainty / 100, 1.0)  # inverted: low uncertainty = good
    confl_norm = confluence_score / 3.0
    unanimity_norm = max(unanimity, 0.5)

    # Weighted composite (tuned from data analysis)
    score = (
        conf_norm       * 0.30 +   # 30% weight: primary signal strength
        meta_norm       * 0.20 +   # 20% weight: meta model validation
        unanimity_norm  * 0.25 +   # 25% weight: ensemble agreement
        confl_norm      * 0.15 +   # 15% weight: multi-TF confluence
        uncert_norm     * 0.10     # 10% weight: model certainty
    )

    return round(score * 100, 2)


def rank_and_select(predictions: dict, max_signals: int = 2,
                    min_score: float = 55.0) -> dict:
    """
    Rank all qualifying predictions and return only the best N.

    Args:
        predictions: {symbol: pred_dict} — already pre-filtered
        max_signals: maximum signals to send (default 2)
        min_score: minimum composite score to pass (default 55)

    Returns:
        dict: {symbol: pred_dict} — only the top signals
        Also populates pred["signal_score"] for display
    """
    scored = []

    for sym, pred in predictions.items():
        score = compute_signal_score(pred)
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
