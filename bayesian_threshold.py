# -*- coding: utf-8 -*-
"""
Bayesian Threshold — per-symbol, per-regime, per-hour dynamic confidence thresholds.

Instead of a single global minimum confidence, each (symbol, regime, hour_bucket)
combination maintains its own optimal threshold based on observed outcomes.
Uses Beta-Binomial Bayesian updating for smooth adaptation with limited data.

Hour buckets: 0-3 (Asian early), 4-7 (Pre-London), 8-11 (London AM),
              12-15 (Overlap), 16-19 (NY PM), 20-23 (Off hours).
"""

import json
import logging
import os
from datetime import datetime, timezone

log = logging.getLogger("bayesian_threshold")

STATE_PATH = os.path.join(os.path.dirname(__file__), "logs", "bayesian_thresholds.json")

# Prior: assume 50% accuracy (Beta(2, 2) prior — mildly informative)
PRIOR_ALPHA = 2.0
PRIOR_BETA = 2.0

# Base minimum confidence (fallback)
BASE_MIN_CONFIDENCE = 42.0

# Hour buckets for grouping (reduces key space)
HOUR_BUCKETS = {
    0: "asian_early", 1: "asian_early", 2: "asian_early", 3: "asian_early",
    4: "pre_london", 5: "pre_london", 6: "pre_london", 7: "pre_london",
    8: "london_am", 9: "london_am", 10: "london_am", 11: "london_am",
    12: "overlap", 13: "overlap", 14: "overlap", 15: "overlap",
    16: "ny_pm", 17: "ny_pm", 18: "ny_pm", 19: "ny_pm",
    20: "off_hours", 21: "off_hours", 22: "off_hours", 23: "off_hours",
}


class BayesianThreshold:
    """Adaptive per-(symbol, regime, hour) confidence thresholds."""

    def __init__(self):
        self._state = {}  # {key: {"alpha": float, "beta": float, "n": int}}
        self._load()

    def _key(self, symbol: str, regime: str, hour: int) -> str:
        bucket = HOUR_BUCKETS.get(hour % 24, "off_hours")
        return f"{symbol}|{regime}|{bucket}"

    def _get_bucket(self, symbol: str, regime: str, hour: int) -> dict:
        key = self._key(symbol, regime, hour)
        if key not in self._state:
            self._state[key] = {
                "alpha": PRIOR_ALPHA,
                "beta": PRIOR_BETA,
                "n": 0,
            }
        return self._state[key]

    def get_min_confidence(self, symbol: str, regime: str, hour: int) -> float:
        """
        Get the optimal minimum confidence threshold for this context.
        
        Uses the posterior mean win rate to set the threshold:
        - If win rate is high (>60%), lower the threshold to allow more trades
        - If win rate is low (<45%), raise the threshold to be more selective
        """
        bucket = self._get_bucket(symbol, regime, hour)
        n = bucket["n"]

        if n < 5:
            # Not enough data — use base threshold
            return BASE_MIN_CONFIDENCE

        # Posterior mean of Beta distribution
        posterior_mean = bucket["alpha"] / (bucket["alpha"] + bucket["beta"])

        # Map win rate to threshold adjustment
        # 60%+ WR → lower threshold by up to 5 points
        # 45%- WR → raise threshold by up to 8 points
        if posterior_mean >= 0.60:
            adjustment = -(posterior_mean - 0.55) * 25  # e.g., 65% → -2.5
        elif posterior_mean < 0.45:
            adjustment = (0.50 - posterior_mean) * 40   # e.g., 40% → +4.0
        else:
            adjustment = 0.0

        threshold = BASE_MIN_CONFIDENCE + adjustment
        return max(35.0, min(65.0, threshold))  # clamp

    def update(self, symbol: str, regime: str, hour: int, was_correct: bool):
        """Bayesian update after observing an outcome."""
        bucket = self._get_bucket(symbol, regime, hour)
        if was_correct:
            bucket["alpha"] += 1.0
        else:
            bucket["beta"] += 1.0
        bucket["n"] += 1

        # Exponential decay: prevent old data from dominating
        # After 100 observations, start decaying
        if bucket["n"] > 100 and bucket["n"] % 20 == 0:
            decay = 0.95
            bucket["alpha"] = max(PRIOR_ALPHA, bucket["alpha"] * decay)
            bucket["beta"] = max(PRIOR_BETA, bucket["beta"] * decay)

        if bucket["n"] % 10 == 0:
            self._save()

    def get_stats(self, symbol: str, regime: str, hour: int) -> dict:
        """Get posterior statistics for this context."""
        bucket = self._get_bucket(symbol, regime, hour)
        alpha = bucket["alpha"]
        beta = bucket["beta"]
        mean = alpha / (alpha + beta)
        return {
            "posterior_mean": round(mean, 3),
            "n_observations": bucket["n"],
            "min_confidence": round(self.get_min_confidence(symbol, regime, hour), 1),
        }

    def get_all_stats(self) -> dict:
        """Return summary of all tracked contexts."""
        summary = {}
        for key, bucket in self._state.items():
            alpha = bucket["alpha"]
            beta = bucket["beta"]
            mean = alpha / (alpha + beta)
            summary[key] = {
                "win_rate": round(mean, 3),
                "n": bucket["n"],
            }
        return summary

    def _save(self):
        try:
            os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
            with open(STATE_PATH, "w") as f:
                json.dump(self._state, f, indent=2)
        except Exception as e:
            log.debug("Bayesian threshold save failed: %s", e)

    def _load(self):
        try:
            if os.path.exists(STATE_PATH):
                with open(STATE_PATH, "r") as f:
                    self._state = json.load(f)
                log.info("Bayesian thresholds loaded (%d contexts).", len(self._state))
        except Exception as e:
            log.debug("Bayesian threshold load failed: %s", e)
            self._state = {}


# Module-level singleton
_bayesian_threshold = None


def get_bayesian_threshold() -> BayesianThreshold:
    global _bayesian_threshold
    if _bayesian_threshold is None:
        _bayesian_threshold = BayesianThreshold()
    return _bayesian_threshold
