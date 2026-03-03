"""
Online Learner — V3 Layer 12: Per-trade model updates using River.

Uses River's PassiveAggressiveClassifier for online learning:
  - Updates after every trade outcome
  - < 5ms per update
  - Adapts to concept drift in real-time
  - Combined with offline ensemble as additional signal
"""

import numpy as np
import logging

log = logging.getLogger("online_learner")


class OnlineLearner:
    """
    Lightweight online learner for real-time adaptation.
    
    Uses simple SGD-based approach (compatible without River dependency).
    Falls back gracefully if River is not installed.
    """
    
    def __init__(self, n_features=65, learning_rate=0.01):
        self.n_features = n_features
        self.lr = learning_rate
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.n_updates = 0
        self._try_river = True
        self._river_model = None
        self._init_river()
    
    def _init_river(self):
        """Try to initialize River PassiveAggressive."""
        try:
            from river import linear_model
            self._river_model = linear_model.PAClassifier(C=0.01)
            log.info("Online learner: using River PAClassifier")
        except ImportError:
            self._try_river = False
            log.info("Online learner: using fallback SGD (River not installed)")
    
    def predict(self, features):
        """
        Predict probability using online model.
        
        Args:
            features: dict or array of feature values
        
        Returns:
            float: probability in [0, 1]
        """
        if self._river_model and self.n_updates >= 10:
            try:
                if isinstance(features, dict):
                    proba = self._river_model.predict_proba_one(features)
                    return proba.get(1, 0.5)
                else:
                    feat_dict = {f"f{i}": float(v) for i, v in enumerate(features)}
                    proba = self._river_model.predict_proba_one(feat_dict)
                    return proba.get(1, 0.5)
            except Exception:
                pass
        
        # Fallback: simple linear model
        if isinstance(features, dict):
            x = np.array(list(features.values()))[:self.n_features]
        else:
            x = np.array(features)[:self.n_features]
        
        if len(x) < self.n_features:
            x = np.pad(x, (0, self.n_features - len(x)))
        
        logit = np.dot(self.weights, x) + self.bias
        prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -10, 10)))
        return float(prob)
    
    def update(self, features, label):
        """
        Update model with trade outcome.
        
        Args:
            features: dict or array of feature values
            label: 1 (correct prediction) or 0 (wrong)
        """
        if self._river_model:
            try:
                if isinstance(features, dict):
                    self._river_model.learn_one(features, int(label))
                else:
                    feat_dict = {f"f{i}": float(v) for i, v in enumerate(features)}
                    self._river_model.learn_one(feat_dict, int(label))
                self.n_updates += 1
                return
            except Exception:
                pass
        
        # Fallback SGD update
        if isinstance(features, dict):
            x = np.array(list(features.values()))[:self.n_features]
        else:
            x = np.array(features)[:self.n_features]
        
        if len(x) < self.n_features:
            x = np.pad(x, (0, self.n_features - len(x)))
        
        pred = self.predict(x)
        error = label - pred
        self.weights += self.lr * error * x
        self.bias += self.lr * error
        self.n_updates += 1
    
    def get_stats(self):
        """Return learner statistics."""
        return {
            "n_updates": self.n_updates,
            "backend": "river" if self._river_model else "sgd_fallback",
            "weight_norm": float(np.linalg.norm(self.weights)),
        }
