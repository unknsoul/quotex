# -*- coding: utf-8 -*-
"""
Feature Importance Monitor — detects when feature importance rankings shift.

Complements drift_detector.py (which checks distribution shifts) by tracking
whether the RANKING of feature importances changes between training and
recent live data. A shift in importance ranking suggests the model's learned
patterns may no longer be the most relevant.

Uses Spearman rank correlation between training-time and recent importances.
"""

import json
import logging
import os

import numpy as np

log = logging.getLogger("feature_importance_monitor")

STATE_PATH = os.path.join(os.path.dirname(__file__), "logs", "feature_importance_state.json")


class FeatureImportanceMonitor:
    """Track and compare feature importance rankings over time."""

    def __init__(self):
        self.training_importances = {}  # {feature_name: importance}
        self.recent_importances = {}
        self.rank_correlation = 1.0
        self.drifted_features = []
        self._load()

    def set_training_importances(self, importances: dict):
        """Set the baseline training-time feature importances."""
        self.training_importances = dict(importances)
        self._save()

    def update_recent_importances(self, model_ensemble, feature_names: list):
        """Extract and update recent feature importances from the ensemble."""
        if not model_ensemble or not feature_names:
            return

        agg_importance = np.zeros(len(feature_names))
        count = 0

        for model in model_ensemble:
            try:
                base = model.base_model if hasattr(model, "base_model") else model
                if hasattr(base, "feature_importances_"):
                    imp = base.feature_importances_
                    if len(imp) == len(feature_names):
                        agg_importance += imp
                        count += 1
            except Exception:
                continue

        if count == 0:
            return

        agg_importance /= count
        self.recent_importances = {
            name: float(imp) for name, imp in zip(feature_names, agg_importance)
        }
        self._compute_drift()

    def _compute_drift(self):
        """Compute Spearman rank correlation between training and recent importances."""
        if not self.training_importances or not self.recent_importances:
            self.rank_correlation = 1.0
            self.drifted_features = []
            return

        # Find common features
        common = set(self.training_importances.keys()) & set(self.recent_importances.keys())
        if len(common) < 10:
            self.rank_correlation = 1.0
            return

        features = sorted(common)
        train_vals = [self.training_importances[f] for f in features]
        recent_vals = [self.recent_importances[f] for f in features]

        # Compute ranks
        train_ranks = np.argsort(np.argsort(train_vals)).astype(float)
        recent_ranks = np.argsort(np.argsort(recent_vals)).astype(float)

        # Spearman correlation
        n = len(features)
        d_sq = np.sum((train_ranks - recent_ranks) ** 2)
        self.rank_correlation = 1.0 - (6.0 * d_sq) / (n * (n ** 2 - 1))

        # Find features that shifted most in rank
        rank_shifts = []
        for i, f in enumerate(features):
            shift = abs(train_ranks[i] - recent_ranks[i])
            rank_shifts.append((f, shift))

        rank_shifts.sort(key=lambda x: x[1], reverse=True)
        # Top features that shifted by more than 20% of total features
        threshold = n * 0.2
        self.drifted_features = [
            f for f, shift in rank_shifts if shift > threshold
        ][:10]

        if self.rank_correlation < 0.7:
            log.warning(
                "FEATURE IMPORTANCE DRIFT: rank_corr=%.3f, drifted=%s",
                self.rank_correlation, self.drifted_features[:5],
            )

    def check_drift(self) -> dict:
        """Return drift analysis results."""
        is_drifted = self.rank_correlation < 0.7
        return {
            "rank_correlation": round(self.rank_correlation, 3),
            "is_drifted": is_drifted,
            "drifted_features": self.drifted_features,
            "recommendation": "RETRAIN" if is_drifted else "OK",
            "n_training_features": len(self.training_importances),
            "n_recent_features": len(self.recent_importances),
        }

    def _save(self):
        try:
            state = {
                "training_importances": self.training_importances,
                "recent_importances": self.recent_importances,
                "rank_correlation": self.rank_correlation,
            }
            os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
            with open(STATE_PATH, "w") as f:
                json.dump(state, f)
        except Exception as e:
            log.debug("Feature importance save failed: %s", e)

    def _load(self):
        try:
            if os.path.exists(STATE_PATH):
                with open(STATE_PATH, "r") as f:
                    state = json.load(f)
                self.training_importances = state.get("training_importances", {})
                self.recent_importances = state.get("recent_importances", {})
                self.rank_correlation = state.get("rank_correlation", 1.0)
                log.info("Feature importance monitor loaded (corr=%.3f).",
                         self.rank_correlation)
        except Exception as e:
            log.debug("Feature importance load failed: %s", e)


# Module-level singleton
_monitor = None


def get_importance_monitor() -> FeatureImportanceMonitor:
    global _monitor
    if _monitor is None:
        _monitor = FeatureImportanceMonitor()
    return _monitor
