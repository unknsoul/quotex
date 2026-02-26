"""
Stability Module — probability smoothing, accuracy tracking, drift monitoring.
"""

import logging
import numpy as np
import pandas as pd
from collections import deque

from config import (
    PROBABILITY_SMOOTHING_SPAN,
    ROLLING_ACCURACY_WINDOW,
    RETRAINING_ACCURACY_TRIGGER,
    LOG_LEVEL, LOG_FORMAT,
)

log = logging.getLogger("stability")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


class ProbabilitySmoother:
    """EMA-based smoothing of consecutive predictions."""

    def __init__(self, span: int = PROBABILITY_SMOOTHING_SPAN):
        self.span = span
        self.alpha = 2.0 / (span + 1)
        self._ema = None

    def smooth(self, raw_prob: float) -> float:
        if self._ema is None:
            self._ema = raw_prob
        else:
            self._ema = self.alpha * raw_prob + (1 - self.alpha) * self._ema
        return round(self._ema, 4)

    def reset(self):
        self._ema = None


class AccuracyTracker:
    """Rolling window accuracy tracker with retraining trigger."""

    def __init__(self, window: int = ROLLING_ACCURACY_WINDOW,
                 trigger: float = RETRAINING_ACCURACY_TRIGGER):
        self.window = window
        self.trigger = trigger
        self._history: deque = deque(maxlen=window)

    def record(self, was_correct: bool):
        self._history.append(1 if was_correct else 0)

    @property
    def accuracy(self) -> float:
        if len(self._history) == 0:
            return 0.5
        return sum(self._history) / len(self._history)

    @property
    def should_retrain(self) -> bool:
        if len(self._history) < self.window // 2:
            return False
        return self.accuracy < self.trigger

    @property
    def count(self) -> int:
        return len(self._history)

    def status(self) -> dict:
        return {
            "rolling_accuracy": round(self.accuracy, 4),
            "predictions_tracked": self.count,
            "should_retrain": self.should_retrain,
        }


class DriftMonitor:
    """
    Monitor feature importance changes between training sessions.
    Compare stored importances with new ones to detect drift.
    """

    def __init__(self):
        self._baseline: dict[str, float] = {}

    def set_baseline(self, feature_names: list[str], importances: np.ndarray):
        self._baseline = dict(zip(feature_names, importances.tolist()))
        log.info("Drift baseline set with %d features.", len(self._baseline))

    def check_drift(self, feature_names: list[str],
                    importances: np.ndarray,
                    threshold: float = 0.5) -> list[str]:
        """
        Compare current importances to baseline.
        Returns list of features whose importance changed by > threshold (relative).
        """
        if not self._baseline:
            return []

        drifted = []
        for name, imp in zip(feature_names, importances):
            baseline_imp = self._baseline.get(name, 0)
            if baseline_imp == 0:
                if imp > 0.01:
                    drifted.append(f"{name}: new={imp:.4f} (was 0)")
                continue
            change = abs(imp - baseline_imp) / (baseline_imp + 1e-10)
            if change > threshold:
                drifted.append(f"{name}: {baseline_imp:.4f} -> {imp:.4f} ({change:.0%} change)")

        if drifted:
            log.warning("Feature drift detected: %s", drifted)
        else:
            log.info("No significant feature drift.")

        return drifted
