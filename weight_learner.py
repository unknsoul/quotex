"""
Weight Learner v4 — learns optimal confidence weighting from validation data.

Replaces fixed strength × reliability formula with a trained LogisticRegression.

Input features: primary_strength, meta_reliability, regime_strength, uncertainty
Output: optimal weighted confidence (probability of being correct)

Usage:
    python weight_learner.py --symbol EURUSD
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
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

from config import (
    MODEL_DIR, WEIGHT_MODEL_PATH, OOF_PREDICTIONS_PATH,
    ENSEMBLE_MODEL_PATH, META_MODEL_PATH, META_FEATURE_LIST_PATH,
    FEATURE_LIST_PATH, DEFAULT_SYMBOL,
    META_ROLLING_WINDOW, WIN_STREAK_CAP, ATR_PERCENTILE_WINDOW,
    LOG_LEVEL, LOG_FORMAT,
)
from data_collector import load_csv, load_multi_tf
from feature_engineering import compute_features, add_target, FEATURE_COLUMNS
from regime_detection import detect_regime_series
from calibration import CalibratedModel  # needed for joblib.load

log = logging.getLogger("weight_learner")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

REGIME_ENCODING = {"Trending": 0, "Ranging": 1, "High_Volatility": 2, "Low_Volatility": 3}

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
    # Load OOF + models
    if not os.path.exists(OOF_PREDICTIONS_PATH):
        print("ERROR: No OOF predictions. Run train_model.py first.")
        sys.exit(1)
    if not os.path.exists(ENSEMBLE_MODEL_PATH):
        print("ERROR: No ensemble models. Run train_model.py first.")
        sys.exit(1)
    if not os.path.exists(META_MODEL_PATH):
        print("ERROR: No meta model. Run meta_model.py first.")
        sys.exit(1)

    oof_data = joblib.load(OOF_PREDICTIONS_PATH)
    ensemble = joblib.load(ENSEMBLE_MODEL_PATH)
    meta_model = joblib.load(META_MODEL_PATH)
    meta_feat_cols = joblib.load(META_FEATURE_LIST_PATH)

    # Load data
    mtf = load_multi_tf(symbol)
    df = mtf.get("M5")
    if df is None:
        df = load_csv(symbol, "M5")
    m15, h1 = mtf.get("M15"), mtf.get("H1")

    df = compute_features(df, m15_df=m15, h1_df=h1)
    df = add_target(df)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)

    indices = oof_data["indices"]
    actual = oof_data["actual"]
    sub = df.iloc[indices].copy().reset_index(drop=True)

    print(f"\n>> Building weight training data ({len(indices)} rows)...")

    # Get ensemble predictions on OOF rows
    X_oof = sub[FEATURE_COLUMNS]
    all_probs = np.array([m.predict_proba(X_oof)[:, 1] for m in ensemble])
    ens_mean = all_probs.mean(axis=0)
    ens_var = all_probs.var(axis=0)
    ens_var_norm = ens_var / (ens_var.max() + 1e-10)

    # Primary strength
    primary_strength = np.abs(ens_mean - 0.5) * 2

    # Meta reliability (build meta features for each row)
    proba_s = pd.Series(ens_mean, index=sub.index)
    primary_dir = (ens_mean >= 0.5).astype(int)
    correct = (primary_dir == actual).astype(int)

    regimes = detect_regime_series(sub)
    regime_enc = regimes.map(REGIME_ENCODING).fillna(1).astype(int)

    correct_s = pd.Series(correct, index=sub.index)
    rolling_acc = correct_s.rolling(META_ROLLING_WINDOW, min_periods=1).mean()
    win_streaks = []
    streak = 0
    for v in correct:
        streak = streak + 1 if v == 1 else 0
        win_streaks.append(min(streak, WIN_STREAK_CAP))

    dir_streaks = []
    ds = 1
    dirs = (ens_mean >= 0.5).astype(int)
    for i in range(len(dirs)):
        if i > 0 and dirs[i] == dirs[i - 1]:
            ds += 1
        else:
            ds = 1
        dir_streaks.append(ds)

    meta_rows = pd.DataFrame({
        "primary_green_prob": ens_mean,
        "prob_distance_from_half": np.abs(ens_mean - 0.5),
        "primary_entropy": _binary_entropy(ens_mean),
        "regime_encoded": regime_enc.values,
        "atr_value": sub["atr_14"].values if "atr_14" in sub.columns else 0,
        "spread_ratio": 0.0,
        "volatility_zscore": sub["volatility_zscore"].values if "volatility_zscore" in sub.columns else 0,
        "range_position": sub["range_position"].values if "range_position" in sub.columns else 0.5,
        "recent_model_accuracy": rolling_acc.values,
        "recent_win_streak": win_streaks,
        "body_percentile_rank": sub["body_size"].rolling(ATR_PERCENTILE_WINDOW, min_periods=1).rank(pct=True).values if "body_size" in sub.columns else 0.5,
        "direction_streak": dir_streaks,
        "rolling_vol_percentile": sub["atr_14"].rolling(ATR_PERCENTILE_WINDOW, min_periods=1).rank(pct=True).values if "atr_14" in sub.columns else 0.5,
    })
    meta_input = meta_rows[meta_feat_cols].fillna(0)
    meta_proba = meta_model.predict_proba(meta_input.values)[:, 1]

    # Regime strength (continuous: use adx_normalized)
    regime_strength = sub["adx_normalized"].values if "adx_normalized" in sub.columns else np.zeros(len(sub))

    # Build weight training data
    X_weight = pd.DataFrame({
        "primary_strength": primary_strength,
        "meta_reliability": meta_proba,
        "regime_strength": regime_strength,
        "uncertainty": ens_var_norm,
    })
    y_weight = correct

    print(f"   X shape: {X_weight.shape}")
    print(f"   Correct: {int(y_weight.sum())} ({y_weight.mean():.1%})")

    # Train with CV
    tscv = TimeSeriesSplit(n_splits=5)
    fold_metrics = []
    print(f"\n{'='*50}")
    print(f"  Weight Learner -- 5 folds")
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
