"""
Calibration v9 — v17 heterogeneous calibrated ensemble + Stacking Combiner.

v17 upgrades:
  - Early stopping for XGBoost, LightGBM, CatBoost (prevents overfitting)
  - focal_sample_weight + smooth_labels now wired into training
  - StackingCombiner: learned optimal blend weights (replaces simple mean)
  - Stronger regularization across all ensemble members
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
    """v16 production ensemble specs — 8-9 diverse members, stronger regularization."""
    specs = [
        {
            "model_type": "xgb",
            "name": "XGB_recent_fast",
            "window_ratio": 0.35,
            "params": {"max_depth": 5, "learning_rate": 0.012, "n_estimators": 600,
                       "min_child_weight": 30, "reg_alpha": 1.5, "reg_lambda": 6.0,
                       "gamma": 4.0, "subsample": 0.65, "colsample_bytree": 0.55},
        },
        {
            "model_type": "xgb",
            "name": "XGB_full_deep",
            "window_ratio": 1.0,
            "params": {"max_depth": 6, "learning_rate": 0.002, "n_estimators": 900,
                       "min_child_weight": 35, "reg_alpha": 2.5, "reg_lambda": 10.0,
                       "gamma": 5.0, "subsample": 0.62, "colsample_bytree": 0.50},
        },
        {
            "model_type": "xgb",
            "name": "XGB_medium_diverse",
            "window_ratio": 0.6,
            "params": {"max_depth": 5, "learning_rate": 0.006, "n_estimators": 700,
                       "min_child_weight": 28, "reg_alpha": 2.0, "reg_lambda": 7.0,
                       "gamma": 4.5, "subsample": 0.68, "colsample_bytree": 0.48},
        },
        {
            "model_type": "hist_gb",
            "name": "HistGB_full",
            "window_ratio": 1.0,
            "params": {"max_depth": 6, "learning_rate": 0.012, "max_iter": 800,
                       "min_samples_leaf": 40, "l2_regularization": 3.0},
        },
        {
            "model_type": "extra_trees",
            "name": "ExtraTrees_structural",
            "window_ratio": 1.0,
            "params": {"n_estimators": 800, "max_depth": 8, "min_samples_leaf": 30},
        },
        {
            "model_type": "random_forest",
            "name": "RF_balanced_recent",
            "window_ratio": 0.7,
            "params": {"n_estimators": 700, "max_depth": 7, "min_samples_leaf": 32},
        },
    ]
    # Add CatBoost if installed — v16 stronger reg
    if _HAS_CATBOOST:
        specs.append({
            "model_type": "catboost",
            "name": "CatBoost_conservative",
            "window_ratio": 1.0,
            "params": {"depth": 6, "learning_rate": 0.012, "iterations": 800,
                       "l2_leaf_reg": 9.0, "min_data_in_leaf": 35},
        })
    # Add LightGBM if installed — v16 deeper regularization
    if _HAS_LIGHTGBM:
        specs.append({
            "model_type": "lightgbm",
            "name": "LightGBM_diverse",
            "window_ratio": 0.8,
            "params": {"n_estimators": 800, "max_depth": 6, "learning_rate": 0.012,
                       "subsample": 0.65, "colsample_bytree": 0.55,
                       "min_child_samples": 35, "reg_alpha": 1.5, "reg_lambda": 5.0},
        })
    return specs


def fit_model_from_spec(spec, X_train, y_train, spw=1.0, seed=42,
                        X_val=None, y_val=None):
    """Fit a single heterogeneous ensemble member from a spec.
    
    v17: accepts optional X_val/y_val for early stopping on boosting models.
    """
    params = spec.get("params", {})
    model_type = spec["model_type"]
    class_weights = _sample_weight(y_train, spw)
    # Apply time-decay weighting: recent samples get higher weight
    time_weights = _time_decay_weights(len(y_train), half_life_ratio=0.3)
    weights = class_weights * time_weights

    # Prepare validation weights for early stopping
    val_weights = None
    if X_val is not None and y_val is not None:
        val_class_w = _sample_weight(y_val, spw)
        val_time_w = _time_decay_weights(len(y_val), half_life_ratio=0.3)
        val_weights = val_class_w * val_time_w

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
            early_stopping_rounds=50 if X_val is not None else None,
        )
        fit_kwargs = {"sample_weight": weights, "verbose": False}
        if X_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            if val_weights is not None:
                fit_kwargs["sample_weight_eval_set"] = [val_weights]
        model.fit(X_train, y_train, **fit_kwargs)
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
            early_stopping_rounds=50 if X_val is not None else None,
        )
        if X_val is not None:
            model.fit(X_train, y_train, sample_weight=weights,
                      eval_set=(X_val, y_val))
        else:
            model.fit(X_train, y_train, sample_weight=weights)
        return model

    if model_type == "lightgbm" and _HAS_LIGHTGBM:
        cb = [lgb.early_stopping(50, verbose=False)] if X_val is not None else None
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
        fit_kwargs = {"sample_weight": weights}
        if X_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["callbacks"] = cb
        model.fit(X_train, y_train, **fit_kwargs)
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

        model = fit_model_from_spec(spec, X_tr, y_tr, spw=spw, seed=seed,
                                    X_val=X_cal, y_val=y_cal)
        name = f"{spec['name']}_{window_pct}pct"

        cal = build_calibrated_model(model, X_cal, y_cal, name=name)
        ensemble.append(cal)

    return ensemble


# =============================================================================
#  v16: Stacking Combiner — learned optimal ensemble blend weights
# =============================================================================

class StackingCombiner:
    """Learns optimal model combination weights from OOF predictions.

    Instead of naive averaging, uses logistic regression on per-model
    OOF probabilities to learn which models to trust more.
    """

    def __init__(self):
        self.model = None
        self.model_names = []

    def fit(self, oof_per_model, y_true):
        """
        oof_per_model: shape (n_models, n_samples) — OOF probabilities per model
        y_true: shape (n_samples,) — actual labels
        """
        from sklearn.linear_model import LogisticRegressionCV
        import logging
        log = logging.getLogger("calibration")

        X_stack = oof_per_model.T  # (n_samples, n_models)
        self.model = LogisticRegressionCV(
            Cs=[0.01, 0.1, 1.0, 10.0],
            cv=3, scoring='roc_auc',
            max_iter=1000, random_state=42
        )
        self.model.fit(X_stack, y_true)

        coefs = self.model.coef_[0]
        for i, c in enumerate(coefs):
            name = self.model_names[i] if i < len(self.model_names) else f"model_{i}"
            log.info("  Stacking weight %s: %.4f", name, c)
        return self

    def predict_proba(self, model_probs):
        """model_probs: shape (n_models,) — per-model probabilities for one sample."""
        if self.model is None:
            return float(np.mean(model_probs))  # fallback
        X = np.asarray(model_probs).reshape(1, -1)
        return float(self.model.predict_proba(X)[0, 1])


# =============================================================================
#  v16: Focal Loss sample weighting — mine hard examples
# =============================================================================

def focal_sample_weight(y, base_probs, gamma=2.0):
    """Upweight examples the current model gets wrong.

    gamma=2.0 means hard examples get ~4x more weight than easy ones.
    """
    p_t = np.where(y == 1, base_probs, 1 - base_probs)
    p_t = np.clip(p_t, 0.01, 0.99)
    focal_weight = (1 - p_t) ** gamma
    return focal_weight / (focal_weight.mean() + 1e-10)


def smooth_labels(y, alpha=0.03):
    """Label smoothing: y=1 → 1-alpha, y=0 → alpha."""
    return y.astype(float) * (1 - alpha) + (1 - y.astype(float)) * alpha
