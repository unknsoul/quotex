"""
Calibration v5 — shared CalibratedModel + diverse ensemble builder.

Ensemble: XGBoost(42), XGBoost(789), ExtraTrees, HistGradientBoosting, LogisticRegression.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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


def build_diverse_ensemble(X_base, y_base, X_val, y_val, spw=1.0,
                           xgb_params=None):
    """
    Build a 5-model diverse ensemble:
      1. XGBoost (seed 42)
      2. XGBoost (seed 789)
      3. ExtraTreesClassifier
      4. HistGradientBoostingClassifier
      5. LogisticRegression

    Each is calibrated with isotonic on the val split.
    Returns list of CalibratedModel.
    """
    if xgb_params is None:
        xgb_params = {}

    n_est = xgb_params.get("n_estimators", 400)
    max_d = xgb_params.get("max_depth", 5)
    lr = xgb_params.get("learning_rate", 0.05)
    ss = xgb_params.get("subsample", 0.8)
    cs = xgb_params.get("colsample_bytree", 0.8)

    models_spec = [
        ("XGB_42", xgb.XGBClassifier(
            n_estimators=n_est, max_depth=max_d, learning_rate=lr,
            subsample=ss, colsample_bytree=cs, scale_pos_weight=spw,
            objective="binary:logistic", eval_metric="logloss",
            use_label_encoder=False, random_state=42, verbosity=0,
        )),
        ("XGB_789", xgb.XGBClassifier(
            n_estimators=n_est, max_depth=max_d, learning_rate=lr,
            subsample=ss, colsample_bytree=cs, scale_pos_weight=spw,
            objective="binary:logistic", eval_metric="logloss",
            use_label_encoder=False, random_state=789, verbosity=0,
        )),
        ("ExtraTrees", ExtraTreesClassifier(
            n_estimators=300, max_depth=max_d + 2, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
        ("HistGB", HistGradientBoostingClassifier(
            max_iter=300, max_depth=max_d, learning_rate=lr,
            min_samples_leaf=20, random_state=42,
        )),
        ("LogReg", LogisticRegression(
            max_iter=1000, C=0.1, class_weight="balanced",
            random_state=42,
        )),
    ]

    ensemble = []
    for name, model in models_spec:
        if isinstance(model, xgb.XGBClassifier):
            model.fit(X_base, y_base,
                      eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_base, y_base)
        cal = build_calibrated_model(model, X_val, y_val, name=name)
        ensemble.append(cal)

    return ensemble
