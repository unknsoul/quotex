"""
Meta-Labeler — V3 Layer 7: Secondary classifier that predicts model correctness.

Instead of a simple meta model, this is a dedicated binary classifier that
answers: "Is the primary ensemble likely to be correct on THIS bar?"

Trained on OOF predictions from the primary ensemble.
Output: meta_score ∈ [0, 1], where > 0.55 means "trade this signal".

This acts as Gate 2 in the 10-gate system (confidence gating).
"""

import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

log = logging.getLogger("meta_labeler")

META_FEATURES = [
    "primary_prob",          # Raw ensemble probability
    "prob_distance",         # |prob - 0.5| (strength of signal)
    "ensemble_std",          # Disagreement between ensemble members
    "regime_confidence",     # How confident is regime classification
    "atr_percentile",        # Volatility environment
    "adx_normalized",        # Trend strength
    "return_momentum",       # Recent price momentum
    "session_flag",          # Trading session
    "hour_sin",              # Time of day
    "direction_streak",      # Consecutive same-direction predictions
]


class MetaLabeler:
    """Secondary classifier predicting primary model correctness."""
    
    def __init__(self, n_estimators=200, max_depth=3, learning_rate=0.05):
        self.base_model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            random_state=42,
        )
        self.model = None
        self.feature_cols = META_FEATURES
        self.fitted = False
    
    def fit(self, X_meta, y_correct):
        """
        Train meta-labeler on OOF predictions.
        
        Args:
            X_meta: DataFrame with META_FEATURES columns
            y_correct: binary array (1 = primary was correct, 0 = wrong)
        """
        # Calibrated CV for reliable probabilities
        self.model = CalibratedClassifierCV(self.base_model, method="sigmoid", cv=3)
        
        # Fill missing features
        X = X_meta[self.feature_cols].fillna(0)
        
        self.model.fit(X, y_correct)
        self.fitted = True
        
        train_score = self.model.score(X, y_correct)
        log.info("Meta-labeler trained: train_acc=%.1f%%, n=%d", train_score * 100, len(y_correct))
    
    def predict_meta_score(self, features):
        """
        Predict probability that the primary model is correct.
        
        Args:
            features: dict or DataFrame row with META_FEATURES
        
        Returns:
            float: meta_score ∈ [0, 1]
        """
        if not self.fitted:
            return 0.5  # Uninformed prior
        
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        X = features[self.feature_cols].fillna(0)
        return float(self.model.predict_proba(X)[:, 1][0])
    
    def should_trade(self, meta_score, threshold=0.55):
        """Gate check: should this signal be traded?"""
        return meta_score >= threshold


def build_meta_features(primary_prob, ensemble_probs, regime_conf, df_row,
                         direction_history=None):
    """
    Build meta feature row from prediction context.
    
    Args:
        primary_prob: ensemble mean probability
        ensemble_probs: array of individual model probabilities
        regime_conf: regime classification confidence
        df_row: current bar data
        direction_history: list of recent direction predictions
    """
    direction_history = direction_history or []
    
    # Direction streak
    cur_dir = 1 if primary_prob >= 0.5 else 0
    streak = 1
    for d in reversed(direction_history):
        if d == cur_dir:
            streak += 1
        else:
            break
    
    return {
        "primary_prob": primary_prob,
        "prob_distance": abs(primary_prob - 0.5),
        "ensemble_std": float(np.std(ensemble_probs)) if len(ensemble_probs) > 1 else 0,
        "regime_confidence": regime_conf,
        "atr_percentile": float(df_row.get("atr_percentile_rank", 0.5)),
        "adx_normalized": float(df_row.get("adx_normalized", 0.25)),
        "return_momentum": float(df_row.get("momentum_rolling_5", 0)),
        "session_flag": int(df_row.get("session_flag", 0)),
        "hour_sin": float(df_row.get("hour_sin", 0)),
        "direction_streak": streak,
    }
