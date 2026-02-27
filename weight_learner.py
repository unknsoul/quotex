"""
Weight Learner — OOF-only, no in-sample bias.

CRITICAL: Uses ONLY OOF predictions from train_model.py.
Does NOT re-run ensemble on OOF rows (that would be in-sample).
Uses stored per-seed OOF probabilities for variance computation.

Fixes applied:
  - OOF indices sorted ascending before slicing
  - Uses meta calibrator for calibrated meta reliability
  - Matches meta feature list exactly (no spread_ratio, has ensemble_variance)

Input features: primary_strength, meta_reliability, regime_strength, uncertainty
"""

import argparse
import os
import sys
import logging

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score

from config import (
    MODEL_DIR, WEIGHT_MODEL_PATH, OOF_PREDICTIONS_PATH,
    META_MODEL_PATH, META_FEATURE_LIST_PATH,
    DEFAULT_SYMBOL,
    ATR_PERCENTILE_WINDOW,
    LOG_LEVEL, LOG_FORMAT,
)
from data_collector import load_csv, load_multi_tf
from feature_engineering import compute_features, add_target, FEATURE_COLUMNS
from regime_detection import detect_regime_series

log = logging.getLogger("weight_learner")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

REGIME_ENCODING = {"Trending": 0, "Ranging": 1, "High_Volatility": 2, "Low_Volatility": 3}

META_CALIBRATOR_PATH = os.path.join(MODEL_DIR, "meta_calibrator.pkl")

WEIGHT_FEATURES = [
    "primary_strength",
    "meta_reliability",
    "regime_strength",
    "uncertainty",
]


def _binary_entropy(p):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def train_weights(symbol):
    # Load OOF + meta model
    if not os.path.exists(OOF_PREDICTIONS_PATH):
        print("ERROR: No OOF predictions. Run train_model.py first.")
        sys.exit(1)
    if not os.path.exists(META_MODEL_PATH):
        print("ERROR: No meta model. Run meta_model.py first.")
        sys.exit(1)

    oof_data = joblib.load(OOF_PREDICTIONS_PATH)
    meta_model = joblib.load(META_MODEL_PATH)
    meta_feat_cols = joblib.load(META_FEATURE_LIST_PATH)

    # Load meta calibrator if available
    meta_calibrator = None
    if os.path.exists(META_CALIBRATOR_PATH):
        meta_calibrator = joblib.load(META_CALIBRATOR_PATH)
        print("   Meta calibrator loaded.")

    # FIX #1: Sort indices ascending before slicing
    raw_indices = oof_data["indices"]
    sort_order = np.argsort(raw_indices)
    indices = raw_indices[sort_order]
    oof_proba = oof_data["oof_proba"][sort_order]
    oof_all = oof_data["oof_all_proba"][:, sort_order]
    actual = oof_data["actual"][sort_order]

    print(f"\n>> Loaded OOF data: {len(oof_proba)} rows, {oof_all.shape[0]} seeds")
    print(f"   (Indices sorted ascending — chronological order guaranteed)")

    # Load features for meta feature building
    mtf = load_multi_tf(symbol)
    df = mtf.get("M5")
    if df is None:
        df = load_csv(symbol, "M5")
    m15, h1 = mtf.get("M15"), mtf.get("H1")
    df = compute_features(df, m15_df=m15, h1_df=h1)
    df = add_target(df)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)

    sub = df.iloc[indices].copy().reset_index(drop=True)

    # Compute correctness from OOF predictions
    primary_dir = (oof_proba >= 0.5).astype(int)
    correct = (primary_dir == actual).astype(int)

    # Uncertainty from stored per-seed OOF probabilities
    oof_var = oof_all.var(axis=0)
    oof_var_norm = oof_var / (oof_var.max() + 1e-10)

    # Primary strength from OOF
    primary_strength = np.abs(oof_proba - 0.5) * 2

    # Build meta features for OOF rows (matching META_FEATURE_COLUMNS exactly)
    proba_s = pd.Series(oof_proba, index=sub.index)
    regimes = detect_regime_series(sub)
    regime_enc = regimes.map(REGIME_ENCODING).fillna(1).astype(int)

    dir_streaks = []
    ds = 1
    dirs = (oof_proba >= 0.5).astype(int)
    for i in range(len(dirs)):
        if i > 0 and dirs[i] == dirs[i - 1]:
            ds += 1
        else:
            ds = 1
        dir_streaks.append(ds)

    meta_rows = pd.DataFrame({
        "primary_green_prob": oof_proba,
        "prob_distance_from_half": np.abs(oof_proba - 0.5),
        "primary_entropy": _binary_entropy(oof_proba),
        "ensemble_variance": oof_var,
        "regime_encoded": regime_enc.values,
        "atr_value": sub["atr_14"].values if "atr_14" in sub.columns else 0,
        "volatility_zscore": sub["volatility_zscore"].values if "volatility_zscore" in sub.columns else 0,
        "range_position": sub["range_position"].values if "range_position" in sub.columns else 0.5,
        "body_percentile_rank": (
            sub["body_size"].rolling(ATR_PERCENTILE_WINDOW, min_periods=1).rank(pct=True).values
            if "body_size" in sub.columns else 0.5
        ),
        "direction_streak": dir_streaks,
        "rolling_vol_percentile": (
            sub["atr_14"].rolling(ATR_PERCENTILE_WINDOW, min_periods=1).rank(pct=True).values
            if "atr_14" in sub.columns else 0.5
        ),
    })
    meta_input = meta_rows[meta_feat_cols].fillna(0)
    meta_raw_proba = meta_model.predict_proba(meta_input.values)[:, 1]

    # Apply meta calibration if available
    if meta_calibrator is not None:
        meta_proba = meta_calibrator.predict(meta_raw_proba)
        meta_proba = np.clip(meta_proba, 0, 1)
        print("   Meta probabilities calibrated.")
    else:
        meta_proba = meta_raw_proba

    # Regime strength
    regime_strength = sub["adx_normalized"].values if "adx_normalized" in sub.columns else np.zeros(len(sub))

    # Build weight training data
    X_weight = pd.DataFrame({
        "primary_strength": primary_strength,
        "meta_reliability": meta_proba,
        "regime_strength": regime_strength,
        "uncertainty": oof_var_norm,
    })
    y_weight = correct

    print(f"\n>> Building weight training data ({len(indices)} rows)...")
    print(f"   X shape: {X_weight.shape}")
    print(f"   Correct: {int(y_weight.sum())} ({y_weight.mean():.1%})")

    # Train with CV
    tscv = TimeSeriesSplit(n_splits=5)
    fold_metrics = []
    print(f"\n{'='*50}")
    print(f"  Weight Learner — 5 folds")
    print(f"{'='*50}\n")

    for fold, (tr, te) in enumerate(tscv.split(X_weight), 1):
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_weight.iloc[tr], y_weight[tr])
        y_prob = lr.predict_proba(X_weight.iloc[te])[:, 1]
        y_pred = lr.predict(X_weight.iloc[te])
        m = {
            "acc": accuracy_score(y_weight[te], y_pred),
            "auc": roc_auc_score(y_weight[te], y_prob) if len(np.unique(y_weight[te])) > 1 else 0.5,
        }
        fold_metrics.append(m)
        print(f"  Fold {fold}: Acc={m['acc']:.4f} AUC={m['auc']:.4f}")

    avg_acc = np.mean([fm["acc"] for fm in fold_metrics])
    avg_auc = np.mean([fm["auc"] for fm in fold_metrics])
    print(f"\n  Avg: Acc={avg_acc:.4f} AUC={avg_auc:.4f}")

    # Final model
    print("\n>> Training final weight model...")
    final = LogisticRegression(random_state=42, max_iter=1000)
    final.fit(X_weight, y_weight)

    coefs = dict(zip(WEIGHT_FEATURES, final.coef_[0]))
    print(f"  Coefficients: {coefs}")
    print(f"  Intercept: {final.intercept_[0]:.4f}")

    # Verify uncertainty coef is negative (higher uncertainty → lower confidence)
    if coefs["uncertainty"] < 0:
        print("  ✅ Uncertainty coefficient is negative (correct)")
    else:
        print("  ⚠️  Uncertainty coefficient is positive (unexpected)")

    joblib.dump(final, WEIGHT_MODEL_PATH)
    print(f"\n>> Saved weight model -> {WEIGHT_MODEL_PATH}")
    print(f"\n>> Done. Next: python backtest.py --symbol {symbol}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    args = parser.parse_args()
    train_weights(args.symbol)


if __name__ == "__main__":
    main()
