"""
Meta Model — GradientBoosting that predicts reliability of primary model predictions.

Trained AFTER primary model using out-of-fold predictions.
Meta target: 1 if primary prediction was correct, 0 if wrong.

Usage:
    python meta_model.py --symbol EURUSD
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
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, brier_score_loss,
)

from config import (
    MODEL_DIR, META_MODEL_PATH, META_FEATURE_LIST_PATH,
    OOF_PREDICTIONS_PATH, DEFAULT_SYMBOL,
    LGBM_N_ESTIMATORS, LGBM_MAX_DEPTH, LGBM_LEARNING_RATE,
    LGBM_SUBSAMPLE,
    TIMESERIES_SPLITS,
    META_ROLLING_WINDOW, WIN_STREAK_CAP,
    LOG_LEVEL, LOG_FORMAT,
)
from data_collector import load_csv, load_multi_tf
from feature_engineering import compute_features, add_target, FEATURE_COLUMNS
from regime_detection import detect_regime_series

log = logging.getLogger("meta_model")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

# Meta feature column names
META_FEATURE_COLUMNS = [
    "primary_green_prob",
    "primary_red_prob",
    "primary_confidence_strength",
    "regime_encoded",
    "atr_value",
    "spread_ratio",
    "volatility_zscore",
    "range_position",
    "recent_model_accuracy",
    "recent_win_streak",
]

REGIME_ENCODING = {
    "Trending": 0,
    "Ranging": 1,
    "High_Volatility": 2,
    "Low_Volatility": 3,
}


def _compute_rolling_accuracy(correct_series: pd.Series, window: int) -> pd.Series:
    return correct_series.rolling(window, min_periods=1).mean()


def _compute_win_streak(correct_series: pd.Series, cap: int) -> pd.Series:
    streaks = []
    streak = 0
    for val in correct_series:
        if val == 1:
            streak += 1
        else:
            streak = 0
        streaks.append(min(streak, cap))
    return pd.Series(streaks, index=correct_series.index)


def _build_clf():
    return GradientBoostingClassifier(
        n_estimators=LGBM_N_ESTIMATORS,
        max_depth=LGBM_MAX_DEPTH,
        learning_rate=LGBM_LEARNING_RATE,
        subsample=LGBM_SUBSAMPLE,
        random_state=42,
    )


def build_meta_features(df: pd.DataFrame, oof_data: dict) -> pd.DataFrame:
    indices = oof_data["indices"]
    oof_proba = oof_data["oof_proba"]
    actual = oof_data["actual"]

    sub = df.iloc[indices].copy().reset_index(drop=True)

    sub["primary_green_prob"] = oof_proba
    sub["primary_red_prob"] = 1.0 - oof_proba
    sub["primary_confidence_strength"] = np.maximum(oof_proba, 1.0 - oof_proba)

    primary_direction = (oof_proba >= 0.5).astype(int)
    correct = (primary_direction == actual).astype(int)
    sub["meta_target"] = correct

    regimes = detect_regime_series(sub)
    sub["regime_encoded"] = regimes.map(REGIME_ENCODING).fillna(1).astype(int)

    sub["atr_value"] = sub["atr_14"] if "atr_14" in sub.columns else 0.0
    if "spread" in sub.columns:
        avg_spread = sub["spread"].rolling(100, min_periods=1).mean()
        sub["spread_ratio"] = sub["spread"] / (avg_spread + 1e-10)
    else:
        sub["spread_ratio"] = 0.0

    if "volatility_zscore" not in sub.columns:
        sub["volatility_zscore"] = 0.0
    if "range_position" not in sub.columns:
        sub["range_position"] = 0.5

    correct_s = pd.Series(correct)
    sub["recent_model_accuracy"] = _compute_rolling_accuracy(correct_s, META_ROLLING_WINDOW).values
    sub["recent_win_streak"] = _compute_win_streak(correct_s, WIN_STREAK_CAP).values

    return sub


def train_meta(symbol: str) -> None:
    if not os.path.exists(OOF_PREDICTIONS_PATH):
        print("ERROR: No OOF predictions found. Run train_model.py first.")
        sys.exit(1)

    oof_data = joblib.load(OOF_PREDICTIONS_PATH)
    print(f"\n>> Loaded OOF predictions: {len(oof_data['oof_proba'])} rows")

    print(f">> Loading feature data for {symbol}...")
    mtf = load_multi_tf(symbol)
    df = mtf.get("M5")
    if df is None:
        df = load_csv(symbol, "M5")
    m15 = mtf.get("M15")
    h1 = mtf.get("H1")

    df = compute_features(df, m15_df=m15, h1_df=h1)
    df = add_target(df)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)

    print(">> Building meta features...")
    meta_df = build_meta_features(df, oof_data)

    X_meta = meta_df[META_FEATURE_COLUMNS].fillna(0)
    y_meta = meta_df["meta_target"].values

    n_correct = int(y_meta.sum())
    n_wrong = len(y_meta) - n_correct
    print(f"\n>> Meta dataset:")
    print(f"   Samples : {len(X_meta)}")
    print(f"   Correct : {n_correct} ({n_correct/len(y_meta):.1%})")
    print(f"   Wrong   : {n_wrong} ({n_wrong/len(y_meta):.1%})")

    tscv = TimeSeriesSplit(n_splits=TIMESERIES_SPLITS)
    fold_metrics = []

    print(f"\n{'='*65}")
    print(f"  Meta Model (GradientBoosting) -- TimeSeriesSplit {TIMESERIES_SPLITS} folds")
    print(f"{'='*65}\n")

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_meta), 1):
        X_tr, X_te = X_meta.iloc[tr_idx], X_meta.iloc[te_idx]
        y_tr, y_te = y_meta[tr_idx], y_meta[te_idx]

        clf = _build_clf()
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

        m = {
            "acc": accuracy_score(y_te, y_pred),
            "auc": roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else 0.5,
            "brier": brier_score_loss(y_te, y_prob),
        }
        fold_metrics.append(m)
        print(f"  Fold {fold}:  Acc={m['acc']:.4f}  AUC={m['auc']:.4f}  Brier={m['brier']:.4f}")

    avg = {k: np.mean([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
    print(f"\n{'-'*65}")
    print(f"  Avg  :  Acc={avg['acc']:.4f}  AUC={avg['auc']:.4f}  Brier={avg['brier']:.4f}")
    print(f"{'='*65}")

    print("\n>> Training final meta model on all data...")
    final = _build_clf()
    final.fit(X_meta, y_meta)

    y_full = final.predict(X_meta)
    print(f"\n  Full-dataset report:\n")
    print(classification_report(y_meta, y_full, target_names=["Wrong (0)", "Correct (1)"]))

    imp = final.feature_importances_
    order = np.argsort(imp)[::-1]
    print("  Meta feature importances:")
    for rank, idx in enumerate(order, 1):
        name = META_FEATURE_COLUMNS[idx] if idx < len(META_FEATURE_COLUMNS) else f"feat_{idx}"
        print(f"    {rank:>2}. {name:<30s} {imp[idx]:.4f}")

    joblib.dump(final, META_MODEL_PATH)
    joblib.dump(META_FEATURE_COLUMNS, META_FEATURE_LIST_PATH)
    print(f"\n>> Saved meta model    -> {META_MODEL_PATH}")
    print(f">> Saved meta features -> {META_FEATURE_LIST_PATH}")
    print(f"\n>> Done. Next: python backtest.py --symbol {symbol}")


def main():
    parser = argparse.ArgumentParser(description="Train meta reliability model.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    args = parser.parse_args()
    train_meta(args.symbol)


if __name__ == "__main__":
    main()
