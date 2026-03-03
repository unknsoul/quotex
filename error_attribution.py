"""
Error Attribution — V3 Layer 11: Loss classifier for failure analysis.

Classifies every losing trade into one of 8 failure modes:
  F1: Regime misclassification (wrong market state)
  F2: Spread spike (cost exceeded expected)
  F3: News event (fundamental shock)
  F4: Low confidence override (traded despite low conf)
  F5: Ensemble disagreement (models contradicted)
  F6: Stale features (feature drift since training)
  F7: Session effect (wrong time of day)
  F8: Random noise (no identifiable pattern)

Daily summary helps identify systematic issues to fix.
"""

import numpy as np
import logging
from collections import Counter

log = logging.getLogger("error_attribution")


def classify_error(signal_context):
    """
    Classify a losing trade into failure mode F1-F8.
    
    Args:
        signal_context: dict with keys like regime_confidence, ensemble_std,
                        confidence, session, spread_ratio, etc.
    
    Returns:
        str: failure mode (F1-F8)
    """
    regime_conf = signal_context.get("regime_confidence", 1.0)
    ensemble_std = signal_context.get("ensemble_std", 0.0)
    confidence = signal_context.get("confidence", 50)
    session = signal_context.get("session", "London")
    spread_ratio = signal_context.get("spread_ratio", 1.0)
    
    # F2: Spread spike
    if spread_ratio > 2.5:
        return "F2_spread_spike"
    
    # F1: Regime misclassification
    if regime_conf < 0.50:
        return "F1_regime_misclass"
    
    # F5: Ensemble disagreement
    if ensemble_std > 0.15:
        return "F5_ensemble_disagree"
    
    # F4: Low confidence override
    if confidence < 55:
        return "F4_low_confidence"
    
    # F7: Bad session
    if session in ("Asian", "Off"):
        return "F7_session_effect"
    
    # F8: Random noise (default)
    return "F8_random_noise"


def daily_error_summary(errors):
    """
    Generate daily error summary from list of classified errors.
    
    Returns formatted string for Telegram/logging.
    """
    if not errors:
        return "No errors to report"
    
    counts = Counter(errors)
    total = len(errors)
    
    lines = [f"Error Attribution ({total} losses):"]
    for err, count in counts.most_common():
        pct = count / total * 100
        lines.append(f"  {err}: {count} ({pct:.0f}%)")
    
    # Actionable recommendation
    top = counts.most_common(1)[0][0]
    recommendations = {
        "F1_regime_misclass": "Consider regime filter: skip trades with conf < 0.60",
        "F2_spread_spike": "Increase spread guard threshold or skip high-spread periods",
        "F4_low_confidence": "Raise confidence threshold to 60%+",
        "F5_ensemble_disagree": "Lower ensemble variance threshold",
        "F7_session_effect": "Consider session filter: skip Asian/Off hours",
        "F8_random_noise": "Random losses — system performing as expected",
    }
    lines.append(f"\n  Recommendation: {recommendations.get(top, 'Review model performance')}")
    
    return "\n".join(lines)
