# -*- coding: utf-8 -*-
"""
Dispatch Model — learns from live outcomes which predictions actually win.

Trains a lightweight logistic regression on signal_outcomes.csv to predict
whether a given signal will be correct BEFORE dispatch. Acts as a final
gatekeeper after all other filters.

Features used (all available at dispatch time):
  - confidence, green_prob, meta_reliability, kelly
  - regime (encoded), session (encoded), unanimity, variance
  - hour_sin, hour_cos, symbol_encoded
"""

import csv
import json
import logging
import math
import os
from datetime import datetime, timezone

import numpy as np

log = logging.getLogger("dispatch_model")

OUTCOMES_CSV = os.path.join(os.path.dirname(__file__), "logs", "signal_outcomes.csv")
DISPATCH_STATE = os.path.join(os.path.dirname(__file__), "logs", "dispatch_model_state.json")

# Minimum outcomes needed before the model activates
MIN_TRAINING_SAMPLES = 30

# Symbol encoding (consistent)
SYMBOL_MAP = {
    "EURUSD": 0, "GBPUSD": 1, "USDJPY": 2, "AUDUSD": 3,
    "USDCAD": 4, "USDCHF": 5, "XAUUSD": 6, "GBPJPY": 7,
}
REGIME_MAP = {
    "Trending": 0, "Ranging": 1, "High_Volatility": 2,
    "Low_Volatility": 3, "CHOPPY": 4, "Unknown": 5,
}
SESSION_MAP = {
    "London": 0, "New_York": 1, "Overlap": 2, "Asian": 3, "Off": 4,
}


class DispatchModel:
    """Online logistic regression for dispatch gating."""

    def __init__(self, n_features=11, learning_rate=0.01, l2_reg=0.001):
        self.n_features = n_features
        self.lr = learning_rate
        self.l2 = l2_reg
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.n_updates = 0
        self._load_state()

    def _sigmoid(self, z):
        z = np.clip(z, -20, 20)
        return 1.0 / (1.0 + np.exp(-z))

    def _extract_features(self, pred: dict, symbol: str = "") -> np.ndarray:
        """Extract dispatch features from a prediction dict."""
        conf = pred.get("final_confidence_percent", 50) / 100.0
        green = pred.get("green_probability_percent", 50) / 100.0
        meta = pred.get("meta_reliability_percent", 50) / 100.0
        kelly = pred.get("kelly_fraction_percent", 0) / 10.0
        unanimity = pred.get("ensemble_unanimity", 0.5)
        variance = pred.get("ensemble_variance", 0.01) * 100  # scale up
        quality = pred.get("quality_score", 50) / 100.0

        regime = pred.get("market_regime", "Unknown")
        regime_enc = REGIME_MAP.get(regime, 5) / 5.0

        session = pred.get("session", "Off")
        session_enc = SESSION_MAP.get(session, 4) / 4.0

        sym_enc = SYMBOL_MAP.get(symbol, 0) / 7.0

        # Edge strength: how far from 50%
        edge = abs(green - 0.5) * 2.0

        return np.array([
            conf, green, meta, kelly, unanimity,
            variance, quality, regime_enc, session_enc, sym_enc, edge,
        ])

    def predict_win_probability(self, pred: dict, symbol: str = "") -> float:
        """Predict probability that this signal will be correct."""
        if self.n_updates < MIN_TRAINING_SAMPLES:
            return 0.5  # not enough data, neutral
        x = self._extract_features(pred, symbol)
        z = np.dot(self.weights, x) + self.bias
        return float(self._sigmoid(z))

    def should_dispatch(self, pred: dict, symbol: str = "",
                        min_win_prob: float = 0.44) -> tuple:  # v16.1: lowered from 0.48
        """Return (should_send, win_probability, reason)."""
        wp = self.predict_win_probability(pred, symbol)
        if self.n_updates < MIN_TRAINING_SAMPLES:
            return True, wp, "dispatch_model_warming_up"
        if wp >= min_win_prob:
            return True, wp, f"dispatch_ok({wp:.0%})"
        return False, wp, f"dispatch_reject({wp:.0%}<{min_win_prob:.0%})"

    def update(self, pred: dict, symbol: str, was_correct: bool):
        """Online SGD update with L2 regularization."""
        x = self._extract_features(pred, symbol)
        y = 1.0 if was_correct else 0.0
        p = self._sigmoid(np.dot(self.weights, x) + self.bias)
        error = y - p
        self.weights += self.lr * error * x - self.l2 * self.weights
        self.bias += self.lr * error
        self.n_updates += 1

        if self.n_updates % 20 == 0:
            self._save_state()
            log.info("Dispatch model: %d updates, weights_norm=%.3f",
                     self.n_updates, np.linalg.norm(self.weights))

    def train_from_outcomes(self):
        """Bulk train from historical outcomes CSV."""
        if not os.path.exists(OUTCOMES_CSV):
            return 0

        rows = []
        try:
            with open(OUTCOMES_CSV, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
        except Exception as e:
            log.warning("Failed to read outcomes: %s", e)
            return 0

        if len(rows) < MIN_TRAINING_SAMPLES:
            return 0

        count = 0
        for row in rows:
            try:
                pred = {
                    "final_confidence_percent": float(row.get("confidence", 50)),
                    "green_probability_percent": float(row.get("green_prob", 50)),
                    "meta_reliability_percent": 50.0,
                    "kelly_fraction_percent": float(row.get("kelly", 0)),
                    "ensemble_unanimity": 0.5,
                    "ensemble_variance": 0.01,
                    "quality_score": 50.0,
                    "market_regime": row.get("regime", "Unknown"),
                    "session": row.get("session", "Off"),
                }
                symbol = row.get("symbol", "")
                correct_val = row.get("correct", "")
                if correct_val in ("True", "true", "1", True):
                    was_correct = True
                elif correct_val in ("False", "false", "0", False):
                    was_correct = False
                else:
                    continue
                self.update(pred, symbol, was_correct)
                count += 1
            except Exception:
                continue

        log.info("Dispatch model bulk trained on %d outcomes.", count)
        self._save_state()
        return count

    def _save_state(self):
        try:
            state = {
                "weights": self.weights.tolist(),
                "bias": float(self.bias),
                "n_updates": self.n_updates,
            }
            os.makedirs(os.path.dirname(DISPATCH_STATE), exist_ok=True)
            with open(DISPATCH_STATE, "w") as f:
                json.dump(state, f)
        except Exception as e:
            log.debug("Dispatch model save failed: %s", e)

    def _load_state(self):
        try:
            if os.path.exists(DISPATCH_STATE):
                with open(DISPATCH_STATE, "r") as f:
                    state = json.load(f)
                w = state.get("weights", [])
                if len(w) == self.n_features:
                    self.weights = np.array(w)
                    self.bias = float(state.get("bias", 0))
                    self.n_updates = int(state.get("n_updates", 0))
                    log.info("Dispatch model loaded (%d updates).", self.n_updates)
        except Exception as e:
            log.debug("Dispatch model load failed: %s", e)


# Module-level singleton
_dispatch_model = None


def get_dispatch_model() -> DispatchModel:
    global _dispatch_model
    if _dispatch_model is None:
        _dispatch_model = DispatchModel()
        _dispatch_model.train_from_outcomes()
    return _dispatch_model
