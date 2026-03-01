"""
Calibration v3 — Diverse hybrid ensemble + isotonic calibration.

Ensemble: 3 XGBoost + 2 ExtraTrees + 1 LightGBM + 1 CatBoost = 7 models.
Each model is calibrated with isotonic regression on a held-out calibration slice.
4-algorithm diversity creates maximally informative ensemble variance.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


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
    Build a diverse 7-model ensemble (Phase 4):
      - XGB deep (depth=6, LR=0.03, 40% window)
      - XGB medium (depth=4, LR=0.05, 70% window)
      - XGB full (depth=5, LR=0.05, 100% window)
      - ExtraTrees shallow (depth=8, 40% window)
      - ExtraTrees full (depth=12, 100% window)
      - LightGBM (num_leaves=31, depth=6, 70% window)
      - CatBoost (depth=6, lr=0.03, 100% window)

    4-algorithm diversity (XGB/ET/LGBM/CatBoost) creates maximally
    informative ensemble variance for the meta model.

    Returns list of CalibratedModel.
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024, 2048, 4096]
    if xgb_params is None:
        xgb_params = {}

    n_est = xgb_params.get("n_estimators", 400)
    ss = xgb_params.get("subsample", 0.8)
    cs = xgb_params.get("colsample_bytree", 0.8)

    # Each model spec: (algorithm, params, temporal_window_ratio)
    model_specs = [
        # XGB deep -- recent patterns, complex interactions
        ("xgb", {"max_depth": 6, "learning_rate": 0.03, "n_estimators": 500}, 0.4),
        # XGB medium -- balanced
        ("xgb", {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 400}, 0.7),
        # XGB full -- full history, standard
        ("xgb", {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 400}, 1.0),
        # ExtraTrees -- random splits capture different signal
        ("et", {"max_depth": 8, "n_estimators": 500}, 0.4),
        # ExtraTrees full -- maximum diversity on full history
        ("et", {"max_depth": 12, "n_estimators": 400}, 1.0),
        # LightGBM -- leaf-wise growth, different split strategy from XGB
        ("lgbm", {"num_leaves": 31, "max_depth": 6, "learning_rate": 0.05,
                  "n_estimators": 400}, 0.7),
        # CatBoost -- ordered boosting, reduces overfitting
        ("cat", {"depth": 6, "learning_rate": 0.03, "iterations": 400}, 1.0),
    ]

    if not temporal or len(X_train) <= 1000:
        model_specs = [(a, p, 1.0) for a, p, _ in model_specs]

    ensemble = []
    for idx, (algo, params, ratio) in enumerate(model_specs):
        seed = seeds[idx] if idx < len(seeds) else 42 + idx

        # Temporal windowing
        n_rows = len(X_train)
        start = n_rows - int(n_rows * ratio)
        X_tr = X_train.iloc[start:]
        y_tr = y_train[start:]
        window_pct = int(ratio * 100)

        if algo == "xgb":
            model = xgb.XGBClassifier(
                n_estimators=params.get("n_estimators", n_est),
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=ss, colsample_bytree=cs,
                scale_pos_weight=spw,
                objective="binary:logistic", eval_metric="logloss",
                use_label_encoder=False, random_state=seed, verbosity=0,
                min_child_weight=3, reg_alpha=0.1, reg_lambda=1.5,
            )
            model.fit(X_tr, y_tr, verbose=False)
            name = f"XGB_d{params['max_depth']}_lr{params['learning_rate']}_{window_pct}pct"

        elif algo == "et":
            model = ExtraTreesClassifier(
                n_estimators=params.get("n_estimators", 400),
                max_depth=params.get("max_depth", 10),
                min_samples_leaf=5,
                random_state=seed, n_jobs=-1,
            )
            model.fit(X_tr, y_tr)
            name = f"ET_d{params.get('max_depth')}_{window_pct}pct"

        elif algo == "lgbm":
            model = lgb.LGBMClassifier(
                num_leaves=params["num_leaves"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                n_estimators=params.get("n_estimators", 400),
                subsample=ss, colsample_bytree=cs,
                scale_pos_weight=spw,
                random_state=seed, verbose=-1,
                min_child_samples=10, reg_alpha=0.1, reg_lambda=1.5,
            )
            model.fit(X_tr, y_tr)
            name = f"LGBM_d{params['max_depth']}_nl{params['num_leaves']}_{window_pct}pct"

        elif algo == "cat":
            model = CatBoostClassifier(
                depth=params["depth"],
                learning_rate=params["learning_rate"],
                iterations=params.get("iterations", 400),
                auto_class_weights="Balanced",
                random_seed=seed, verbose=0,
                l2_leaf_reg=3.0,
            )
            model.fit(X_tr, y_tr)
            name = f"CAT_d{params['depth']}_lr{params['learning_rate']}_{window_pct}pct"

        cal = build_calibrated_model(model, X_cal, y_cal, name=name)
        ensemble.append(cal)

    return ensemble

