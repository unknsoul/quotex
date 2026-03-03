"""
Dynamic Ensemble — V3 Layer 7: Regime-aware model weighting.

Instead of equal-weight averaging all models, dynamically weights based on:
  1. Regime: different models perform better in different regimes
  2. Rolling accuracy: models that have been right recently get more weight
  3. Softmax normalization: smooth weight distribution

Manages 4 model types × 4 regimes = 16 specialized models.
Falls back to equal weights if < 10 trades in a regime.
"""

import numpy as np
import logging
from collections import defaultdict

log = logging.getLogger("dynamic_ensemble")

MODEL_TYPES = ["xgboost", "lightgbm", "catboost", "random_forest"]
REGIMES = ["Trending", "Ranging", "High_Volatility", "Transition"]
MIN_TRADES_FOR_WEIGHTING = 10
SOFTMAX_TEMP = 2.0  # Temperature for softmax (higher = more uniform)
ROLLING_WINDOW = 50  # Track last 50 predictions per model per regime


class DynamicEnsemble:
    """Regime-aware dynamic ensemble with rolling accuracy tracking."""
    
    def __init__(self, model_types=None, regimes=None):
        self.model_types = model_types or MODEL_TYPES
        self.regimes = regimes or REGIMES
        
        # Rolling accuracy tracker: regime → model → list of (correct, timestamp)
        self.history = defaultdict(lambda: defaultdict(list))
        
        # Default weights per regime (from V3 spec research)
        self.default_weights = {
            "Trending": {"xgboost": 0.35, "lightgbm": 0.25, "catboost": 0.25, "random_forest": 0.15},
            "Ranging": {"xgboost": 0.25, "lightgbm": 0.25, "catboost": 0.30, "random_forest": 0.20},
            "High_Volatility": {"xgboost": 0.30, "lightgbm": 0.30, "catboost": 0.25, "random_forest": 0.15},
            "Transition": {"xgboost": 0.25, "lightgbm": 0.25, "catboost": 0.25, "random_forest": 0.25},
        }
    
    def predict(self, regime, model_probs):
        """
        Combine model probabilities using regime-aware weights.
        
        Args:
            regime: current market regime string
            model_probs: dict of {model_name: probability}
        
        Returns:
            float: weighted ensemble probability
        """
        weights = self.get_weights(regime)
        
        prob = 0.0
        total_weight = 0.0
        for model_name, model_prob in model_probs.items():
            w = weights.get(model_name, 1.0 / len(model_probs))
            prob += w * model_prob
            total_weight += w
        
        if total_weight > 0:
            prob /= total_weight
        
        return float(np.clip(prob, 0, 1))
    
    def get_weights(self, regime):
        """Get current weights for regime (learned or default)."""
        regime = self._normalize_regime(regime)
        
        # Check if enough history to compute learned weights
        regime_history = self.history.get(regime, {})
        total_trades = sum(len(v) for v in regime_history.values())
        
        if total_trades < MIN_TRADES_FOR_WEIGHTING:
            return self.default_weights.get(regime, 
                     {m: 1.0 / len(self.model_types) for m in self.model_types})
        
        # Compute rolling accuracy per model
        accuracies = {}
        for model_name in self.model_types:
            history = regime_history.get(model_name, [])
            if len(history) >= 5:
                recent = history[-ROLLING_WINDOW:]
                accuracies[model_name] = sum(c for c, _ in recent) / len(recent)
            else:
                accuracies[model_name] = 0.5  # Prior
        
        # Softmax weighting
        return self._softmax_weights(accuracies)
    
    def update(self, regime, model_name, was_correct, timestamp=None):
        """Record prediction outcome for a model in a regime."""
        regime = self._normalize_regime(regime)
        self.history[regime][model_name].append((int(was_correct), timestamp))
        
        # Trim to rolling window
        if len(self.history[regime][model_name]) > ROLLING_WINDOW * 2:
            self.history[regime][model_name] = \
                self.history[regime][model_name][-ROLLING_WINDOW:]
    
    def _softmax_weights(self, accuracies):
        """Convert accuracies to softmax weights."""
        values = np.array([accuracies.get(m, 0.5) for m in self.model_types])
        # Scale by temperature
        exp_vals = np.exp((values - values.max()) / SOFTMAX_TEMP)
        weights = exp_vals / exp_vals.sum()
        return {m: float(w) for m, w in zip(self.model_types, weights)}
    
    def _normalize_regime(self, regime):
        """Normalize regime name to match our internal naming."""
        regime_map = {
            "TREND": "Trending", "Trending": "Trending",
            "RANGE": "Ranging", "Ranging": "Ranging",
            "HIGH_VOL": "High_Volatility", "High_Volatility": "High_Volatility",
            "TRANSITION": "Transition", "Transition": "Transition",
            "Low_Volatility": "Ranging",  # Map to closest
        }
        return regime_map.get(regime, "Transition")
    
    def get_stats(self):
        """Return summary statistics."""
        stats = {}
        for regime in self.regimes:
            regime_stats = {}
            for model in self.model_types:
                history = self.history.get(regime, {}).get(model, [])
                if history:
                    acc = sum(c for c, _ in history) / len(history)
                    regime_stats[model] = {"accuracy": acc, "n_trades": len(history)}
            stats[regime] = regime_stats
        return stats
