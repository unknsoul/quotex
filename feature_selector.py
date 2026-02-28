"""
Feature Selector — identifies and removes noisy features using permutation importance.

Computes permutation importance on the trained ensemble using OOF holdout data.
Features with zero or negative importance are candidates for removal.

Usage:
    python feature_selector.py
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.inspection import permutation_importance

from config import (
    MODEL_DIR, ENSEMBLE_MODEL_PATH, SELECTED_FEATURES_PATH,
    DEFAULT_SYMBOL, LOG_LEVEL, LOG_FORMAT,
)
from data_collector import load_csv, load_multi_tf
from feature_engineering import compute_features, add_target, FEATURE_COLUMNS


log = logging.getLogger("feature_selector")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


def select_features(symbol=DEFAULT_SYMBOL, importance_threshold=0.0):
    """Compute permutation importance and select useful features."""

    # Load data
    mtf = load_multi_tf(symbol)
    df = mtf.get("M5")
    if df is None:
        df = load_csv(symbol, "M5")
    m15, h1 = mtf.get("M15"), mtf.get("H1")
    df = compute_features(df, m15_df=m15, h1_df=h1)
    df = add_target(df).dropna(subset=["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)

    # Use last 30% as test set for importance evaluation
    n = len(df)
    split = int(n * 0.7)
    X_test = df[FEATURE_COLUMNS].iloc[split:]
    y_test = df["target"].iloc[split:].values

    # Load ensemble
    ensemble = joblib.load(ENSEMBLE_MODEL_PATH)

    # Extract base model (CalibratedModel wrapper -> .base_model)
    base_model = ensemble[0]
    if hasattr(base_model, 'base_model'):
        base_model = base_model.base_model
    print(f"\n>> Computing permutation importance on {len(X_test)} test bars...")
    print(f"   Features: {len(FEATURE_COLUMNS)}")
    print(f"   Model type: {type(base_model).__name__}")

    result = permutation_importance(
        base_model, X_test, y_test,
        n_repeats=10, random_state=42, n_jobs=-1,
        scoring="accuracy"
    )

    # Rank features
    importances = result.importances_mean
    stds = result.importances_std

    feature_importance = sorted(
        zip(FEATURE_COLUMNS, importances, stds),
        key=lambda x: x[1], reverse=True
    )

    print(f"\n{'='*65}")
    print(f"  FEATURE IMPORTANCE (Permutation)")
    print(f"{'='*65}")
    print(f"  {'Rank':>4} {'Feature':<35} {'Importance':>12} {'Std':>8} {'Status':>8}")
    print(f"  {'-'*70}")

    selected = []
    removed = []
    for rank, (feat, imp, std) in enumerate(feature_importance, 1):
        status = "[KEEP]" if imp > importance_threshold else "[DROP]"
        if imp > importance_threshold:
            selected.append(feat)
        else:
            removed.append(feat)
        print(f"  {rank:>4} {feat:<35} {imp:>11.6f} {std:>7.6f} {status:>8}")

    print(f"\n  SUMMARY:")
    print(f"    Total features: {len(FEATURE_COLUMNS)}")
    print(f"    Selected (importance > {importance_threshold}): {len(selected)}")
    print(f"    Removed: {len(removed)}")

    if removed:
        print(f"\n  Removed features:")
        for f in removed:
            print(f"    - {f}")

    # Save
    joblib.dump(selected, SELECTED_FEATURES_PATH)
    print(f"\n>> Saved {len(selected)} selected features -> {SELECTED_FEATURES_PATH}")

    return selected, removed


if __name__ == "__main__":
    select_features()
