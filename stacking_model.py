"""
Stacking Ensemble — Upgrade 6.

A second-level meta-model that learns WHEN to trust the primary
ensemble. Instead of always using the ensemble prediction, the stacker
observes meta-features (time, regime, volatility, confidence) and
decides whether to PASS or SKIP the signal.

This creates a "gatekeeper" — a model that predicts whether the
primary model will be correct or not.
"""

import logging
import os
import json
import numpy as np
from datetime import datetime, timezone

log = logging.getLogger("stacking_model")

STATE_FILE = os.path.join(os.path.dirname(__file__), "models", "stacker_state.json")
MIN_SAMPLES = 50  # need at least 50 examples before using the stacker


class StackingGatekeeper:
    """
    Online logistic regression that predicts P(primary_model_correct)
    given meta-features about the prediction.

    Features:
        - confidence (normalized)
        - meta_reliability (normalized)
        - ensemble_variance
        - unanimity
        - hour_of_day (sin/cos encoded)
        - regime_encoded
        - kelly_fraction
    """

    def __init__(self):
        self.n_features = 8
        self.weights = np.zeros(self.n_features + 1)  # +1 for bias
        self.lr = 0.01
        self.n_samples = 0
        self._load()

    def _load(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    state = json.load(f)
                self.weights = np.array(state["weights"])
                self.n_samples = state.get("n_samples", 0)
                self.lr = state.get("lr", 0.01)
                log.info("Stacker loaded: %d samples, weights=%s",
                         self.n_samples, self.weights.round(3).tolist())
            except Exception as e:
                log.warning("Stacker load failed: %s", e)

    def _save(self):
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        try:
            with open(STATE_FILE, "w") as f:
                json.dump({
                    "weights": self.weights.tolist(),
                    "n_samples": self.n_samples,
                    "lr": self.lr,
                    "updated": datetime.now(timezone.utc).isoformat(),
                }, f, indent=2)
        except Exception as e:
            log.warning("Stacker save failed: %s", e)

    def _extract_features(self, pred: dict) -> np.ndarray:
        """Extract meta-features from a prediction dict."""
        conf = pred.get("final_confidence_percent", 50) / 100.0
        meta = pred.get("meta_reliability_percent", 50) / 100.0
        var = pred.get("ensemble_variance", 0.01)
        unanimity = pred.get("ensemble_unanimity", 0.5)
        kelly = pred.get("kelly_fraction_percent", 0) / 10.0
        regime_stab = pred.get("regime_stability", 0.5)

        # Hour encoding (sin/cos)
        now = datetime.now(timezone.utc)
        hour_norm = now.hour / 24.0
        hour_sin = np.sin(2 * np.pi * hour_norm)
        hour_cos = np.cos(2 * np.pi * hour_norm)

        return np.array([conf, meta, var, unanimity, kelly,
                         regime_stab, hour_sin, hour_cos])

    def _sigmoid(self, z):
        z = np.clip(z, -10, 10)
        return 1.0 / (1.0 + np.exp(-z))

    def predict_correctness(self, pred: dict) -> float:
        """
        Predict probability that this prediction will be correct.
        Returns P(correct) in [0, 1].
        """
        if self.n_samples < MIN_SAMPLES:
            return 0.5  # not enough data — neutral

        x = self._extract_features(pred)
        x_bias = np.append(x, 1.0)  # add bias term
        return float(self._sigmoid(np.dot(self.weights, x_bias)))

    def update(self, pred: dict, was_correct: bool):
        """
        Update the stacker with a new outcome.
        Uses online logistic regression (SGD).
        """
        x = self._extract_features(pred)
        x_bias = np.append(x, 1.0)
        y = 1.0 if was_correct else 0.0

        # Forward pass
        p = self._sigmoid(np.dot(self.weights, x_bias))

        # Gradient descent step
        error = p - y
        self.weights -= self.lr * error * x_bias

        self.n_samples += 1

        # Decay learning rate gradually
        if self.n_samples % 100 == 0:
            self.lr = max(0.001, self.lr * 0.95)
            self._save()
            log.info("Stacker: %d samples, lr=%.4f", self.n_samples, self.lr)

    def should_send(self, pred: dict, threshold: float = 0.50) -> dict:  # v17.2: raised — 48% was worse than coinflip
        """
        Should we send this signal? Based on gatekeeper's prediction
        of correctness probability.

        Returns:
            dict with:
                send: bool
                p_correct: float (0-1)
                reason: str
        """
        p = self.predict_correctness(pred)

        if self.n_samples < MIN_SAMPLES:
            return {
                "send": True,
                "p_correct": p,
                "reason": f"Stacker: too few samples ({self.n_samples}/{MIN_SAMPLES})",
            }

        if p >= threshold:
            return {
                "send": True,
                "p_correct": round(p, 3),
                "reason": f"Stacker: P(correct)={p:.1%} >= {threshold:.0%}",
            }
        else:
            return {
                "send": False,
                "p_correct": round(p, 3),
                "reason": f"Stacker: P(correct)={p:.1%} < {threshold:.0%}",
            }
