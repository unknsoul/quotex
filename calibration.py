"""
Calibration v3 — 5 seeded XGBoost + isotonic calibration.

Ensemble: 5 XGBoost models with different seeds and hyperparams.
Each model is calibrated with isotonic regression on a held-out calibration slice.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb


class CalibratedModel:
    """Base model + IsotonicRegression calibration wrapper (picklable)."""

    def __init__(self, base_model, iso_reg, name="unknown"):
        self.base_model = base_model
        self.iso_reg = iso_reg
        self.name = name

    def predict_proba(self, X):
        raw = self._raw_proba(X)
        calibrated = self.iso_reg.predict(raw)
        calibrated = np.clip(calibrated, 0, 1)
        return np.column_stack([1 - calibrated, calibrated])

    def _raw_proba(self, X):
        if hasattr(self.base_model, "predict_proba"):
            return self.base_model.predict_proba(X)[:, 1]
        else:
            return self.base_model.decision_function(X)


def build_calibrated_model(base_model, X_val, y_val, name="model"):
    """Train isotonic calibration on held-out val set."""
    if hasattr(base_model, "predict_proba"):
        raw_val = base_model.predict_proba(X_val)[:, 1]
    else:
        raw_val = base_model.decision_function(X_val)
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(raw_val, y_val)
    return CalibratedModel(base_model, iso, name=name)


def build_seeded_xgb_ensemble(X_train, y_train, X_cal, y_cal, spw=1.0,
                               xgb_params=None, seeds=None, temporal=True):
    """
    Build a 5-model XGBoost ensemble with different seeds and hyperparams.

    Each model uses a different temporal window of training data:
      - Seed 1: depth=6, LR=0.03, 40% window (recent patterns)
      - Seed 2: depth=4, LR=0.05, 70% window (balanced)
      - Seed 3: depth=5, LR=0.05, 100% window (full history)
      - Seed 4: depth=3, LR=0.08, 50% window (shallow, fast learner)
      - Seed 5: depth=5, LR=0.03, 100% window (conservative)

    Returns list of CalibratedModel.
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024]
    if xgb_params is None:
        xgb_params = {}

    n_est = xgb_params.get("n_estimators", 400)
    ss = xgb_params.get("subsample", 0.8)
    cs = xgb_params.get("colsample_bytree", 0.8)

    # Each model spec: (params, temporal_window_ratio)
    # Heavily regularized to prevent overfitting on noisy M5 data
    model_specs = [
        ({"max_depth": 3, "learning_rate": 0.01, "n_estimators": 200}, 0.4),
        ({"max_depth": 3, "learning_rate": 0.02, "n_estimators": 200}, 0.7),
        ({"max_depth": 4, "learning_rate": 0.02, "n_estimators": 200}, 1.0),
        ({"max_depth": 2, "learning_rate": 0.03, "n_estimators": 150}, 0.5),
        ({"max_depth": 3, "learning_rate": 0.01, "n_estimators": 250}, 1.0),
    ]

    if not temporal or len(X_train) <= 1000:
        model_specs = [(p, 1.0) for p, _ in model_specs]

    ensemble = []
    for idx, (params, ratio) in enumerate(model_specs):
        seed = seeds[idx] if idx < len(seeds) else 42 + idx

        # Temporal windowing
        n_rows = len(X_train)
        start = n_rows - int(n_rows * ratio)
        X_tr = X_train.iloc[start:]
        y_tr = y_train[start:]
        window_pct = int(ratio * 100)

        model = xgb.XGBClassifier(
            n_estimators=params.get("n_estimators", n_est),
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=ss, colsample_bytree=cs,
            scale_pos_weight=spw,
            objective="binary:logistic", eval_metric="logloss",
            use_label_encoder=False, random_state=seed, verbosity=0,
            min_child_weight=10, reg_alpha=1.0, reg_lambda=5.0,
            gamma=1.0,
        )
        model.fit(X_tr, y_tr, verbose=False)
        name = f"XGB_d{params['max_depth']}_lr{params['learning_rate']}_{window_pct}pct"

        cal = build_calibrated_model(model, X_cal, y_cal, name=name)
        ensemble.append(cal)

    return ensemble
