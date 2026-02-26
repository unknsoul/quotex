"""
Stability v4 — probability smoothing, accuracy tracking, cosine drift detection.
"""

import logging
import numpy as np
import pandas as pd
from collections import deque

from config import (
    PROBABILITY_SMOOTHING_SPAN, ROLLING_ACCURACY_WINDOW,
    RETRAINING_ACCURACY_TRIGGER, DRIFT_COSINE_THRESHOLD,
    LOG_LEVEL, LOG_FORMAT,
)

log = logging.getLogger("stability")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


class ProbabilitySmoother:
    def __init__(self, span=PROBABILITY_SMOOTHING_SPAN):
        self.alpha = 2.0 / (span + 1)
        self._ema = None

    def smooth(self, raw_prob):
        if self._ema is None:
            self._ema = raw_prob
        else:
            self._ema = self.alpha * raw_prob + (1 - self.alpha) * self._ema
        return round(self._ema, 4)

    def reset(self):
        self._ema = None


class AccuracyTracker:
    def __init__(self, window=ROLLING_ACCURACY_WINDOW, trigger=RETRAINING_ACCURACY_TRIGGER):
        self.window = window
        self.trigger = trigger
        self._history = deque(maxlen=window)

    def record(self, was_correct):
        self._history.append(1 if was_correct else 0)

    @property
    def accuracy(self):
        return sum(self._history) / len(self._history) if self._history else 0.5

    @property
    def should_retrain(self):
        if len(self._history) < self.window // 2:
            return False
        return self.accuracy < self.trigger

    def status(self):
        return {
            "rolling_accuracy": round(self.accuracy, 4),
            "predictions_tracked": len(self._history),
            "should_retrain": self.should_retrain,
        }


class CosineDriftDetector:
    """
    Monitor feature importance drift via cosine similarity.
    Compare importance vectors across training cycles.
    """

    def __init__(self, threshold=DRIFT_COSINE_THRESHOLD):
        self.threshold = threshold
        self._baseline = None

    def set_baseline(self, importances):
        self._baseline = np.array(importances, dtype=float)
        log.info("Drift baseline set (%d features).", len(self._baseline))

    def check_drift(self, importances):
        if self._baseline is None:
            self.set_baseline(importances)
            return 1.0, False, "No baseline yet"

        current = np.array(importances, dtype=float)
        if len(current) != len(self._baseline):
            return 0.0, True, "Feature count changed"

        # Cosine similarity
        dot = np.dot(self._baseline, current)
        norm_a = np.linalg.norm(self._baseline)
        norm_b = np.linalg.norm(current)
        if norm_a == 0 or norm_b == 0:
            return 0.0, True, "Zero importance vector"

        cosine_sim = dot / (norm_a * norm_b)
        drifted = cosine_sim < self.threshold

        if drifted:
            log.warning("DRIFT DETECTED: cosine=%.4f < threshold=%.4f", cosine_sim, self.threshold)
        else:
            log.info("No drift: cosine=%.4f", cosine_sim)

        return round(cosine_sim, 4), drifted, f"cosine={cosine_sim:.4f}"

    def update_baseline(self, importances):
        self._baseline = np.array(importances, dtype=float)
