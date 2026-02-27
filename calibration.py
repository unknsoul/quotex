"""
Calibration — CalibratedModel + seeded XGBoost ensemble builder.

Ensemble: 5 XGBoost models with different random seeds.
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
                               xgb_params=None, seeds=None):
    """
    Build a 5-model seeded XGBoost ensemble.

    X_train/y_train: training data (train_main split, NOT calibration).
    X_cal/y_cal: held-out calibration slice (used ONLY for isotonic fitting).
    spw: scale_pos_weight for class imbalance.
    seeds: list of random seeds (default: [42, 123, 456, 789, 1024]).

    Returns list of CalibratedModel.
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024]
    if xgb_params is None:
        xgb_params = {}

    n_est = xgb_params.get("n_estimators", 400)
    max_d = xgb_params.get("max_depth", 5)
    lr = xgb_params.get("learning_rate", 0.05)
    ss = xgb_params.get("subsample", 0.8)
    cs = xgb_params.get("colsample_bytree", 0.8)

    ensemble = []
    for seed in seeds:
        model = xgb.XGBClassifier(
            n_estimators=n_est, max_depth=max_d, learning_rate=lr,
            subsample=ss, colsample_bytree=cs, scale_pos_weight=spw,
            objective="binary:logistic", eval_metric="logloss",
            use_label_encoder=False, random_state=seed, verbosity=0,
        )
        # Train on train_main only — cal slice is NEVER seen during training
        model.fit(X_train, y_train, verbose=False)
        # Calibrate on held-out cal slice
        cal = build_calibrated_model(model, X_cal, y_cal, name=f"XGB_{seed}")
        ensemble.append(cal)

    return ensemble
