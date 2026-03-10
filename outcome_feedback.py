# -*- coding: utf-8 -*-
"""
Outcome Feedback Loop — online threshold tuning from signal outcomes.

Maintains rolling statistics and automatically adjusts key dispatch
parameters based on recent accuracy trends. Acts as a meta-controller
that tunes MIN_SIGNAL_SCORE, MIN_UNANIMITY, and MAX_ENSEMBLE_VARIANCE
based on what's actually working.
"""

import json
import logging
import os
from collections import deque

log = logging.getLogger("outcome_feedback")

STATE_PATH = os.path.join(os.path.dirname(__file__), "logs", "outcome_feedback_state.json")

# Defaults (from config/telegram_bot constants)
DEFAULT_MIN_SCORE = 35.0
DEFAULT_MIN_UNANIMITY = 0.667
DEFAULT_MAX_VARIANCE = 0.055


class OutcomeFeedbackLoop:
    """Auto-tunes dispatch thresholds from rolling outcome data."""

    def __init__(self, window_size=50):
        self.window_size = window_size
        self._outcomes = deque(maxlen=window_size * 2)
        
        # Current tuned thresholds
        self.min_signal_score = DEFAULT_MIN_SCORE
        self.min_unanimity = DEFAULT_MIN_UNANIMITY
        self.max_variance = DEFAULT_MAX_VARIANCE
        
        # Tracking
        self.n_updates = 0
        self.rolling_accuracy = 0.5
        self._load()

    def record_outcome(self, outcome: dict):
        """
        Record a signal outcome for threshold tuning.
        
        outcome should have:
          - correct: bool
          - confidence: float (0-100)
          - unanimity: float (0-1)
          - variance: float
          - signal_score: float (0-100)
        """
        self._outcomes.append(outcome)
        self.n_updates += 1
        
        if self.n_updates >= 20 and self.n_updates % 5 == 0:
            self._retune()
        
        if self.n_updates % 10 == 0:
            self._save()

    def _retune(self):
        """Retune thresholds based on recent outcomes."""
        if len(self._outcomes) < 20:
            return
        
        recent = list(self._outcomes)[-self.window_size:]
        wins = sum(1 for o in recent if o.get("correct"))
        total = len(recent)
        self.rolling_accuracy = wins / total if total > 0 else 0.5
        
        # Strategy: if accuracy is dropping, tighten thresholds; if high, relax slightly
        if self.rolling_accuracy < 0.45:
            # Losing streak — tighten all thresholds
            self.min_signal_score = min(55.0, self.min_signal_score + 1.5)
            self.min_unanimity = min(0.833, self.min_unanimity + 0.02)
            self.max_variance = max(0.025, self.max_variance - 0.003)
            log.info("Feedback loop TIGHTENING: acc=%.1f%%, score>=%.0f, unan>=%.2f, var<=%.3f",
                     self.rolling_accuracy * 100, self.min_signal_score,
                     self.min_unanimity, self.max_variance)
        elif self.rolling_accuracy > 0.58:
            # Winning period — relax slightly to increase volume
            self.min_signal_score = max(30.0, self.min_signal_score - 0.5)
            self.min_unanimity = max(0.60, self.min_unanimity - 0.01)
            self.max_variance = min(0.065, self.max_variance + 0.001)
            log.info("Feedback loop RELAXING: acc=%.1f%%, score>=%.0f, unan>=%.2f, var<=%.3f",
                     self.rolling_accuracy * 100, self.min_signal_score,
                     self.min_unanimity, self.max_variance)
        else:
            # Normal range — nudge toward defaults
            self.min_signal_score += (DEFAULT_MIN_SCORE - self.min_signal_score) * 0.1
            self.min_unanimity += (DEFAULT_MIN_UNANIMITY - self.min_unanimity) * 0.1
            self.max_variance += (DEFAULT_MAX_VARIANCE - self.max_variance) * 0.1

    def get_thresholds(self) -> dict:
        """Return current tuned thresholds."""
        return {
            "min_signal_score": round(self.min_signal_score, 1),
            "min_unanimity": round(self.min_unanimity, 3),
            "max_variance": round(self.max_variance, 4),
            "rolling_accuracy": round(self.rolling_accuracy, 3),
            "n_updates": self.n_updates,
        }

    def _save(self):
        try:
            state = {
                "outcomes": list(self._outcomes),
                "min_signal_score": self.min_signal_score,
                "min_unanimity": self.min_unanimity,
                "max_variance": self.max_variance,
                "n_updates": self.n_updates,
                "rolling_accuracy": self.rolling_accuracy,
            }
            os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
            with open(STATE_PATH, "w") as f:
                json.dump(state, f, default=str)
        except Exception as e:
            log.debug("Outcome feedback save failed: %s", e)

    def _load(self):
        try:
            if os.path.exists(STATE_PATH):
                with open(STATE_PATH, "r") as f:
                    state = json.load(f)
                outcomes = state.get("outcomes", [])
                self._outcomes = deque(outcomes, maxlen=self.window_size * 2)
                self.min_signal_score = state.get("min_signal_score", DEFAULT_MIN_SCORE)
                self.min_unanimity = state.get("min_unanimity", DEFAULT_MIN_UNANIMITY)
                self.max_variance = state.get("max_variance", DEFAULT_MAX_VARIANCE)
                self.n_updates = state.get("n_updates", 0)
                self.rolling_accuracy = state.get("rolling_accuracy", 0.5)
                log.info("Outcome feedback loaded (%d updates, acc=%.1f%%).",
                         self.n_updates, self.rolling_accuracy * 100)
        except Exception as e:
            log.debug("Outcome feedback load failed: %s", e)


# Module-level singleton
_feedback = None


def get_feedback_loop() -> OutcomeFeedbackLoop:
    global _feedback
    if _feedback is None:
        _feedback = OutcomeFeedbackLoop()
    return _feedback
