"""
Calibration v7 — v15 heterogeneous calibrated ensemble.

v15 upgrades:
  - 8-9 member ensemble (XGB×3 + HistGB + ExtraTrees + RF + CatBoost + LightGBM)
  - Third XGB: recent-window aggressive feature sampling
  - Deeper trees, stronger regularization for 900K+ rows
  - All models: increased estimators for larger dataset
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
import xgboost as xgb

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

try:
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False


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


def _sample_weight(y, spw):
    y_arr = np.asarray(y)
    return np.where(y_arr == 1, float(spw), 1.0)


def _time_decay_weights(n_samples, half_life_ratio=0.3):
    """Exponential time-decay: recent samples weighted more heavily.
    half_life_ratio=0.3 means the weight halves at 30% from the start."""
    positions = np.linspace(0, 1, n_samples)
    decay = np.exp(-np.log(2) * (1 - positions) / max(half_life_ratio, 0.01))
    return decay / decay.mean()  # normalize so mean weight = 1.0


def get_primary_model_specs():
    """v15 production ensemble specs — 8-9 diverse members for 900K+ rows."""
    specs = [
        {
            "model_type": "xgb",
            "name": "XGB_recent_fast",
            "window_ratio": 0.35,
            "params": {"max_depth": 5, "learning_rate": 0.015, "n_estimators": 500,
                       "min_child_weight": 25, "reg_alpha": 1.0, "reg_lambda": 5.0,
                       "gamma": 3.0, "subsample": 0.68, "colsample_bytree": 0.60},
        },
        {
            "model_type": "xgb",
            "name": "XGB_full_deep",
            "window_ratio": 1.0,
            "params": {"max_depth": 6, "learning_rate": 0.003, "n_estimators": 700,
                       "min_child_weight": 30, "reg_alpha": 2.0, "reg_lambda": 8.0,
                       "gamma": 4.0, "subsample": 0.65, "colsample_bytree": 0.55},
        },
        {
            "model_type": "xgb",
            "name": "XGB_medium_diverse",
            "window_ratio": 0.6,
            "params": {"max_depth": 5, "learning_rate": 0.008, "n_estimators": 550,
                       "min_child_weight": 22, "reg_alpha": 1.5, "reg_lambda": 6.0,
                       "gamma": 3.5, "subsample": 0.72, "colsample_bytree": 0.50},
        },
        {
            "model_type": "hist_gb",
            "name": "HistGB_full",
            "window_ratio": 1.0,
            "params": {"max_depth": 6, "learning_rate": 0.015, "max_iter": 600,
                       "min_samples_leaf": 35, "l2_regularization": 2.0},
        },
        {
            "model_type": "extra_trees",
            "name": "ExtraTrees_structural",
            "window_ratio": 1.0,
            "params": {"n_estimators": 700, "max_depth": 8, "min_samples_leaf": 25},
        },
        {
            "model_type": "random_forest",
            "name": "RF_balanced_recent",
            "window_ratio": 0.7,
            "params": {"n_estimators": 600, "max_depth": 7, "min_samples_leaf": 28},
        },
    ]
    # Add CatBoost if installed
    if _HAS_CATBOOST:
        specs.append({
            "model_type": "catboost",
            "name": "CatBoost_conservative",
            "window_ratio": 1.0,
            "params": {"depth": 6, "learning_rate": 0.015, "iterations": 600,
                       "l2_leaf_reg": 7.0, "min_data_in_leaf": 30},
        })
    # Add LightGBM if installed — v15 larger trees
    if _HAS_LIGHTGBM:
        specs.append({
            "model_type": "lightgbm",
            "name": "LightGBM_diverse",
            "window_ratio": 0.8,
            "params": {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.015,
                       "subsample": 0.70, "colsample_bytree": 0.60,
                       "min_child_samples": 30, "reg_alpha": 1.0, "reg_lambda": 3.0},
        })
    return specs


def fit_model_from_spec(spec, X_train, y_train, spw=1.0, seed=42):
    """Fit a single heterogeneous ensemble member from a spec."""
    params = spec.get("params", {})
    model_type = spec["model_type"]
    class_weights = _sample_weight(y_train, spw)
    # Apply time-decay weighting: recent samples get higher weight
    time_weights = _time_decay_weights(len(y_train), half_life_ratio=0.3)
    weights = class_weights * time_weights

    if model_type == "xgb":
        model = xgb.XGBClassifier(
            n_estimators=params.get("n_estimators", 280),
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.015),
            subsample=params.get("subsample", 0.75),
            colsample_bytree=params.get("colsample_bytree", 0.7),
            scale_pos_weight=1.0,  # handled via sample_weight
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=seed,
            verbosity=0,
            min_child_weight=params.get("min_child_weight", 15),
            reg_alpha=params.get("reg_alpha", 0.5),
            reg_lambda=params.get("reg_lambda", 3.0),
            gamma=params.get("gamma", 2.0),
        )
        model.fit(X_train, y_train, sample_weight=weights, verbose=False)
        return model

    if model_type == "hist_gb":
        model = HistGradientBoostingClassifier(
            max_depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.04),
            max_iter=params.get("max_iter", 220),
            min_samples_leaf=params.get("min_samples_leaf", 20),
            l2_regularization=params.get("l2_regularization", 0.2),
            random_state=seed,
        )
        model.fit(X_train, y_train, sample_weight=weights)
        return model

    if model_type == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=params.get("n_estimators", 320),
            max_depth=params.get("max_depth", 7),
            min_samples_leaf=params.get("min_samples_leaf", 12),
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, sample_weight=weights)
        return model

    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 280),
            max_depth=params.get("max_depth", 6),
            min_samples_leaf=params.get("min_samples_leaf", 14),
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, sample_weight=weights)
        return model

    if model_type == "catboost" and _HAS_CATBOOST:
        model = CatBoostClassifier(
            depth=params.get("depth", 4),
            learning_rate=params.get("learning_rate", 0.03),
            iterations=params.get("iterations", 300),
            l2_leaf_reg=params.get("l2_leaf_reg", 3.0),
            min_data_in_leaf=params.get("min_data_in_leaf", 20),
            random_seed=seed,
            verbose=0,
            allow_writing_files=False,
        )
        model.fit(X_train, y_train, sample_weight=weights)
        return model

    if model_type == "lightgbm" and _HAS_LIGHTGBM:
        model = lgb.LGBMClassifier(
            n_estimators=params.get("n_estimators", 400),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.02),
            subsample=params.get("subsample", 0.75),
            colsample_bytree=params.get("colsample_bytree", 0.70),
            min_child_samples=params.get("min_child_samples", 25),
            reg_alpha=params.get("reg_alpha", 0.5),
            reg_lambda=params.get("reg_lambda", 2.0),
            random_state=seed,
            verbose=-1,
        )
        model.fit(X_train, y_train, sample_weight=weights)
        return model

    raise ValueError(f"Unsupported model type: {model_type}")


def build_seeded_xgb_ensemble(X_train, y_train, X_cal, y_cal, spw=1.0,
                               xgb_params=None, seeds=None, temporal=True):
    """
    Build the production primary ensemble.

    The public function name is preserved for compatibility, but the ensemble
    is now heterogeneous to reduce shared blind spots across model members.
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024]
    model_specs = get_primary_model_specs()
    if not temporal or len(X_train) <= 1000:
        model_specs = [{**spec, "window_ratio": 1.0} for spec in model_specs]

    ensemble = []
    for idx, spec in enumerate(model_specs):
        seed = seeds[idx] if idx < len(seeds) else 42 + idx

        # Temporal windowing
        n_rows = len(X_train)
        ratio = spec.get("window_ratio", 1.0)
        start = n_rows - int(n_rows * ratio)
        X_tr = X_train.iloc[start:]
        y_tr = y_train[start:]
        window_pct = int(ratio * 100)

        model = fit_model_from_spec(spec, X_tr, y_tr, spw=spw, seed=seed)
        name = f"{spec['name']}_{window_pct}pct"

        cal = build_calibrated_model(model, X_cal, y_cal, name=name)
        ensemble.append(cal)

    return ensemble
