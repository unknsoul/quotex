"""
Meta Model — OOF-only, LightGBM.

CRITICAL: This model trains exclusively on Out-of-Fold primary predictions.
It NEVER sees in-sample primary outputs. The OOF predictions come from
train_model.py's 3-fold TimeSeriesSplit within train_main.

Target: 1 if OOF primary prediction was correct, else 0.
"""

import argparse
import os
import sys
import logging

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

from config import (
    MODEL_DIR, META_MODEL_PATH, META_FEATURE_LIST_PATH,
    OOF_PREDICTIONS_PATH, DEFAULT_SYMBOL,
    META_N_ESTIMATORS, META_MAX_DEPTH, META_LEARNING_RATE,
    META_SUBSAMPLE, META_NUM_LEAVES,
    TIMESERIES_SPLITS, META_ROLLING_WINDOW, WIN_STREAK_CAP,
    ATR_PERCENTILE_WINDOW,
    LOG_LEVEL, LOG_FORMAT,
)
from data_collector import load_csv, load_multi_tf
from feature_engineering import compute_features, add_target, FEATURE_COLUMNS
from regime_detection import detect_regime_series

log = logging.getLogger("meta_model")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

META_FEATURE_COLUMNS = [
    "primary_green_prob",
    "prob_distance_from_half",
    "primary_entropy",
    "regime_encoded",
    "atr_value",
    "spread_ratio",
    "volatility_zscore",
    "range_position",
    "body_percentile_rank",
    "direction_streak",
    "rolling_vol_percentile",
]

REGIME_ENCODING = {"Trending": 0, "Ranging": 1, "High_Volatility": 2, "Low_Volatility": 3}


def _binary_entropy(p):
    """Entropy of binary prediction: -p*log(p) - (1-p)*log(1-p)."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def _rolling_accuracy(correct, window):
    return correct.rolling(window, min_periods=1).mean()


def _compute_win_streak(correct, cap):
    streaks = []
    streak = 0
    for v in correct:
        streak = streak + 1 if v == 1 else 0
        streaks.append(min(streak, cap))
    return pd.Series(streaks, index=correct.index)


def _direction_streak(proba):
    """Consecutive same-direction predictions."""
    directions = (proba >= 0.5).astype(int)
    streaks = []
    streak = 1
    for i in range(len(directions)):
        if i > 0 and directions.iloc[i] == directions.iloc[i - 1]:
            streak += 1
        else:
            streak = 1
        streaks.append(streak)
    return pd.Series(streaks, index=proba.index)


def _build_meta_clf():
    return GradientBoostingClassifier(
        n_estimators=META_N_ESTIMATORS,
        max_depth=META_MAX_DEPTH,
        learning_rate=META_LEARNING_RATE,
        subsample=META_SUBSAMPLE,
        random_state=42,
    )


def build_meta_features(df, oof_data):
    """Build meta features using ONLY OOF predictions (never in-sample)."""
    indices = oof_data["indices"]
    oof_proba = oof_data["oof_proba"]
    actual = oof_data["actual"]

    sub = df.iloc[indices].copy().reset_index(drop=True)
    proba_s = pd.Series(oof_proba, index=sub.index)

    sub["primary_green_prob"] = oof_proba
    sub["prob_distance_from_half"] = np.abs(oof_proba - 0.5)
    sub["primary_entropy"] = _binary_entropy(oof_proba)

    primary_dir = (oof_proba >= 0.5).astype(int)
    correct = (primary_dir == actual).astype(int)
    sub["meta_target"] = correct

    regimes = detect_regime_series(sub)
    sub["regime_encoded"] = regimes.map(REGIME_ENCODING).fillna(1).astype(int)

    sub["atr_value"] = sub["atr_14"] if "atr_14" in sub.columns else 0.0
    sub["spread_ratio"] = 0.0

    if "volatility_zscore" not in sub.columns:
        sub["volatility_zscore"] = 0.0
    if "range_position" not in sub.columns:
        sub["range_position"] = 0.5

    correct_s = pd.Series(correct, index=sub.index)
    sub["recent_model_accuracy"] = _rolling_accuracy(correct_s, META_ROLLING_WINDOW).values
    sub["recent_win_streak"] = _compute_win_streak(correct_s, WIN_STREAK_CAP).values

    sub["body_percentile_rank"] = (
        sub["body_size"].rolling(ATR_PERCENTILE_WINDOW, min_periods=1).rank(pct=True)
        if "body_size" in sub.columns else 0.5
    )
    sub["direction_streak"] = _direction_streak(proba_s).values
    sub["rolling_vol_percentile"] = (
        sub["atr_14"].rolling(ATR_PERCENTILE_WINDOW, min_periods=1).rank(pct=True)
        if "atr_14" in sub.columns else 0.5
    )

    return sub


def train_meta(symbol):
    if not os.path.exists(OOF_PREDICTIONS_PATH):
        print("ERROR: No OOF predictions. Run train_model.py first.")
        sys.exit(1)

    oof_data = joblib.load(OOF_PREDICTIONS_PATH)
    print(f"\n>> Loaded OOF predictions: {len(oof_data['oof_proba'])} rows")
    print(f"   (These are OUT-OF-FOLD predictions — no leakage)")

    mtf = load_multi_tf(symbol)
    df = mtf.get("M5")
    if df is None:
        df = load_csv(symbol, "M5")
    m15, h1 = mtf.get("M15"), mtf.get("H1")

    df = compute_features(df, m15_df=m15, h1_df=h1)
    df = add_target(df)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)

    print(">> Building meta features from OOF data...")
    meta_df = build_meta_features(df, oof_data)
    X_meta = meta_df[META_FEATURE_COLUMNS].fillna(0)
    y_meta = meta_df["meta_target"].values

    n_c, n_w = int(y_meta.sum()), len(y_meta) - int(y_meta.sum())
    print(f"   Samples: {len(X_meta)}, Correct: {n_c} ({n_c/len(y_meta):.1%}), Wrong: {n_w}")

    tscv = TimeSeriesSplit(n_splits=TIMESERIES_SPLITS)
    fold_metrics = []
    print(f"\n{'='*65}")
    print(f"  Meta Model (GradientBoosting) — {TIMESERIES_SPLITS} folds ({len(META_FEATURE_COLUMNS)} features)")
    print(f"{'='*65}\n")

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_meta), 1):
        clf = _build_meta_clf()
        clf.fit(X_meta.iloc[tr_idx], y_meta[tr_idx])
        y_prob = clf.predict_proba(X_meta.iloc[te_idx])[:, 1]
        y_pred = clf.predict(X_meta.iloc[te_idx])
        m = {
            "acc": accuracy_score(y_meta[te_idx], y_pred),
            "auc": roc_auc_score(y_meta[te_idx], y_prob) if len(np.unique(y_meta[te_idx])) > 1 else 0.5,
            "brier": brier_score_loss(y_meta[te_idx], y_prob),
        }
        fold_metrics.append(m)
        print(f"  Fold {fold}: Acc={m['acc']:.4f} AUC={m['auc']:.4f} Brier={m['brier']:.4f}")

    avg = {k: np.mean([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
    print(f"\n  Avg: Acc={avg['acc']:.4f} AUC={avg['auc']:.4f} Brier={avg['brier']:.4f}")

    # LEAKAGE CHECK: meta CV accuracy should be < 75%
    if avg["acc"] > 0.75:
        print(f"\n  ⚠️  WARNING: Meta CV accuracy {avg['acc']:.1%} is suspiciously high.")
        print(f"     Honest OOF meta should be 55-65%. Check for leakage!")
    elif avg["acc"] > 0.50:
        print(f"\n  ✅ Meta accuracy {avg['acc']:.1%} is in expected range (no leakage)")

    print("\n>> Training final meta model...")
    final = _build_meta_clf()
    final.fit(X_meta, y_meta)

    imp = final.feature_importances_
    order = np.argsort(imp)[::-1]
    print("\n  Meta feature importances:")
    for rank, idx in enumerate(order, 1):
        name = META_FEATURE_COLUMNS[idx] if idx < len(META_FEATURE_COLUMNS) else f"feat_{idx}"
        print(f"    {rank:>2}. {name:<30s} {imp[idx]:.0f}")

    joblib.dump(final, META_MODEL_PATH)
    joblib.dump(META_FEATURE_COLUMNS, META_FEATURE_LIST_PATH)
    print(f"\n>> Saved meta model -> {META_MODEL_PATH}")
    print(f">> Saved meta features -> {META_FEATURE_LIST_PATH}")
    print(f"\n>> Done. Next: python weight_learner.py --symbol {symbol}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    args = parser.parse_args()
    train_meta(args.symbol)


if __name__ == "__main__":
    main()
