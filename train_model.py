"""
Train Model — XGBoost with TimeSeriesSplit.

Reads static CSV from data/, computes features, trains, saves model + feature list.
Never runs inside the prediction pipeline.

Usage:
    python train_model.py --symbol EURUSD
"""

import argparse
import os
import sys
import logging

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix,
)

from config import (
    MODEL_DIR, MODEL_PATH, FEATURE_LIST_PATH,
    DEFAULT_SYMBOL, TIMESERIES_SPLITS,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_SUBSAMPLE, XGB_COLSAMPLE_BYTREE,
    LOG_LEVEL, LOG_FORMAT,
)
from data_collector import load_csv
from feature_engineering import compute_features, add_target, FEATURE_COLUMNS

log = logging.getLogger("train_model")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


def _scale_pos_weight(y: np.ndarray) -> float:
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    return n_neg / n_pos if n_pos > 0 else 1.0


def _build_clf(spw: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    )


def train(symbol: str) -> None:
    """Full training pipeline: load CSV -> features -> CV -> save."""

    # ── Load data ────────────────────────────────────────────────────────
    print(f"\n>> Loading CSV data for {symbol}...")
    df = load_csv(symbol)
    print(f"   Rows loaded: {len(df)}")
    print(f"   Range: {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")

    # ── Feature engineering ──────────────────────────────────────────────
    print(f">> Computing {len(FEATURE_COLUMNS)} features...")
    df = compute_features(df)
    df = add_target(df)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)

    X = df[FEATURE_COLUMNS]
    y = df["target"].values

    n_green = int(y.sum())
    n_red = len(y) - n_green
    print(f"\n>> Dataset:")
    print(f"   Samples : {len(X)}")
    print(f"   Features: {len(FEATURE_COLUMNS)}")
    print(f"   Green   : {n_green} ({n_green/len(y):.1%})")
    print(f"   Red     : {n_red} ({n_red/len(y):.1%})")

    spw = _scale_pos_weight(y)
    print(f"   scale_pos_weight: {spw:.4f}")

    # ── Cross-validation ─────────────────────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=TIMESERIES_SPLITS)
    fold_metrics = []

    print(f"\n{'═'*65}")
    print(f"  TimeSeriesSplit — {TIMESERIES_SPLITS} folds")
    print(f"{'═'*65}\n")

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        clf = _build_clf(spw)
        clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

        m = {
            "acc": accuracy_score(y_te, y_pred),
            "prec": precision_score(y_te, y_pred, zero_division=0),
            "rec": recall_score(y_te, y_pred, zero_division=0),
            "f1": f1_score(y_te, y_pred, zero_division=0),
            "auc": roc_auc_score(y_te, y_prob),
        }
        fold_metrics.append(m)
        print(f"  Fold {fold}:  Acc={m['acc']:.4f}  Prec={m['prec']:.4f}  "
              f"Rec={m['rec']:.4f}  F1={m['f1']:.4f}  AUC={m['auc']:.4f}")

    avg = {k: np.mean([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
    print(f"\n{'─'*65}")
    print(f"  Avg  :  Acc={avg['acc']:.4f}  Prec={avg['prec']:.4f}  "
          f"Rec={avg['rec']:.4f}  F1={avg['f1']:.4f}  AUC={avg['auc']:.4f}")
    print(f"{'═'*65}")

    if avg["acc"] > 0.70:
        print("\n  ⚠️  WARNING: Accuracy > 70% — possible overfitting or data leakage!")

    # ── Train final model on all data ────────────────────────────────────
    print("\n>> Training final model on full dataset...")
    final = _build_clf(spw)
    final.fit(X, y, verbose=False)

    y_full = final.predict(X)
    print(f"\n  Full-dataset report:\n")
    print(classification_report(y, y_full, target_names=["Red (0)", "Green (1)"]))
    print("  Confusion matrix:")
    cm = confusion_matrix(y, y_full)
    print(f"    {cm}")

    # Feature importances
    imp = final.feature_importances_
    order = np.argsort(imp)[::-1]
    print("\n  Top 10 features:")
    for rank, idx in enumerate(order[:10], 1):
        name = FEATURE_COLUMNS[idx] if idx < len(FEATURE_COLUMNS) else f"feat_{idx}"
        print(f"    {rank:>2}. {name:<24s} {imp[idx]:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    joblib.dump(FEATURE_COLUMNS, FEATURE_LIST_PATH)
    print(f"\n>> Saved model   -> {MODEL_PATH}")
    print(f">> Saved features -> {FEATURE_LIST_PATH}")
    print(f"\n>> Done. You can now run: python backtest.py --symbol {symbol}")


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost next-candle model.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Symbol")
    args = parser.parse_args()
    train(args.symbol)


if __name__ == "__main__":
    main()
