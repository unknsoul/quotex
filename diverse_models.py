"""
Diverse Model Ensemble — Upgrade 5.

Adds LightGBM and CatBoost models alongside existing XGBoost ensemble
to increase model diversity. Different algorithms capture different
patterns in the data, reducing blind spots.

Usage:
    python diverse_models.py --symbol EURUSD

This trains 2 additional models (LightGBM + CatBoost) and saves them
alongside the existing ensemble. The predict_engine will automatically
use them if they exist.
"""

import argparse
import os
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score

from config import (
    MODEL_DIR, DEFAULT_SYMBOL, TIMESERIES_SPLITS,
    PURGE_EMBARGO_BARS, LOG_LEVEL, LOG_FORMAT,
)
from data_collector import load_csv, load_multi_tf
from feature_engineering import (
    compute_features, add_target_atr_filtered, add_target,
    FEATURE_COLUMNS,
)

log = logging.getLogger("diverse_models")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

DIVERSE_MODEL_PATH = os.path.join(MODEL_DIR, "diverse_ensemble.pkl")


def _try_import_lgb():
    try:
        import lightgbm as lgb
        return lgb
    except ImportError:
        log.warning("LightGBM not installed. Run: pip install lightgbm")
        return None


def _try_import_catboost():
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier
    except ImportError:
        log.warning("CatBoost not installed. Run: pip install catboost")
        return None


class DiverseModelWrapper:
    """Wrapper to make LightGBM/CatBoost have same API as XGBoost in ensemble."""

    def __init__(self, model, model_type):
        self.model = model
        self.model_type = model_type

    def predict_proba(self, X):
        if self.model_type == "lightgbm":
            proba = self.model.predict(X)
            return np.column_stack([1 - proba, proba])
        elif self.model_type == "catboost":
            return self.model.predict_proba(X)
        else:
            return self.model.predict_proba(X)


def train_diverse(symbol: str = DEFAULT_SYMBOL):
    """Train LightGBM and CatBoost models for diversity."""

    print(f"\n{'='*65}")
    print(f"  Diverse Model Training — {symbol}")
    print(f"{'='*65}\n")

    # Load data
    multi_data = load_multi_tf(symbol, source="csv")
    m5 = multi_data.get("M5")
    if m5 is None:
        raise ValueError("No M5 data found")

    df = compute_features(m5)
    df = add_target(df, horizon=1)
    df = df.dropna(subset=["target"])

    use_features = [f for f in FEATURE_COLUMNS if f in df.columns]
    X = df[use_features]
    y = df["target"].values

    print(f"  Data: {len(X)} rows, {len(use_features)} features")

    # Train/test split (80/20 time-series)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    diverse_models = []

    # --- LightGBM ---
    lgb = _try_import_lgb()
    if lgb is not None:
        print("\n  Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, verbose=-1,
        )
        lgb_model.fit(X_train, y_train)
        lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
        lgb_acc = accuracy_score(y_test, (lgb_proba >= 0.5).astype(int))
        lgb_auc = roc_auc_score(y_test, lgb_proba)
        print(f"  LightGBM: Acc={lgb_acc:.4f} AUC={lgb_auc:.4f}")

        diverse_models.append(DiverseModelWrapper(lgb_model, "lightgbm"))

    # --- CatBoost ---
    CatBoostCls = _try_import_catboost()
    if CatBoostCls is not None:
        print("\n  Training CatBoost...")
        cb_model = CatBoostCls(
            iterations=200, depth=5, learning_rate=0.05,
            l2_leaf_reg=3, subsample=0.8, random_seed=42,
            verbose=0, eval_metric="AUC",
        )
        cb_model.fit(X_train, y_train)
        cb_proba = cb_model.predict_proba(X_test)[:, 1]
        cb_acc = accuracy_score(y_test, (cb_proba >= 0.5).astype(int))
        cb_auc = roc_auc_score(y_test, cb_proba)
        print(f"  CatBoost: Acc={cb_acc:.4f} AUC={cb_auc:.4f}")

        diverse_models.append(DiverseModelWrapper(cb_model, "catboost"))

    if diverse_models:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(diverse_models, DIVERSE_MODEL_PATH)
        print(f"\n>> Saved {len(diverse_models)} diverse models -> {DIVERSE_MODEL_PATH}")
    else:
        print("\n  No diverse models trained (missing libraries).")
        print("  Install: pip install lightgbm catboost")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    args = parser.parse_args()
    train_diverse(args.symbol)


if __name__ == "__main__":
    main()
