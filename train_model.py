"""
Train Model v2 — XGBoost + Isotonic calibration + OOF predictions for meta model.

Saves:
  - primary_model.pkl (calibrated)
  - primary_features.pkl
  - oof_predictions.pkl (out-of-fold predictions for meta model training)

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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix,
    brier_score_loss,
)

from config import (
    MODEL_DIR, MODEL_PATH, FEATURE_LIST_PATH, OOF_PREDICTIONS_PATH,
    DEFAULT_SYMBOL, TIMESERIES_SPLITS,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_SUBSAMPLE, XGB_COLSAMPLE_BYTREE,
    LOG_LEVEL, LOG_FORMAT,
)
from data_collector import load_csv, load_multi_tf
from feature_engineering import compute_features, add_target, FEATURE_COLUMNS

log = logging.getLogger("train_model")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


def _scale_pos_weight(y):
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    return n_neg / n_pos if n_pos > 0 else 1.0


def _build_clf(spw):
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
    """Full training pipeline: load -> features -> CV -> calibrate -> OOF -> save."""

    # -- Load data --
    print(f"\n>> Loading data for {symbol}...")
    mtf = load_multi_tf(symbol)
    df = mtf.get("M5")
    if df is None:
        df = load_csv(symbol, "M5")
    m15 = mtf.get("M15")
    h1 = mtf.get("H1")

    print(f"   M5 rows: {len(df)}")

    # -- Features --
    print(f">> Computing {len(FEATURE_COLUMNS)} features...")
    df = compute_features(df, m15_df=m15, h1_df=h1)
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

    # -- Cross-validation + OOF predictions --
    tscv = TimeSeriesSplit(n_splits=TIMESERIES_SPLITS)
    fold_metrics = []

    # OOF arrays
    oof_proba = np.full(len(y), np.nan)
    oof_pred = np.full(len(y), -1, dtype=int)

    print(f"\n{'='*65}")
    print(f"  TimeSeriesSplit -- {TIMESERIES_SPLITS} folds")
    print(f"{'='*65}\n")

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        clf = _build_clf(spw)
        clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

        # Store OOF predictions
        oof_proba[te_idx] = y_prob
        oof_pred[te_idx] = y_pred

        m = {
            "acc": accuracy_score(y_te, y_pred),
            "prec": precision_score(y_te, y_pred, zero_division=0),
            "rec": recall_score(y_te, y_pred, zero_division=0),
            "f1": f1_score(y_te, y_pred, zero_division=0),
            "auc": roc_auc_score(y_te, y_prob),
            "brier": brier_score_loss(y_te, y_prob),
        }
        fold_metrics.append(m)
        print(f"  Fold {fold}:  Acc={m['acc']:.4f}  AUC={m['auc']:.4f}  Brier={m['brier']:.4f}")

    avg = {k: np.mean([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
    print(f"\n{'-'*65}")
    print(f"  Avg  :  Acc={avg['acc']:.4f}  AUC={avg['auc']:.4f}  Brier={avg['brier']:.4f}")
    print(f"{'='*65}")

    if avg["acc"] > 0.70:
        print("\n  WARNING: Accuracy > 70% -- possible overfitting!")

    # -- Train final model with isotonic calibration --
    print("\n>> Training calibrated final model...")
    base = _build_clf(spw)
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
    calibrated.fit(X, y)

    y_cal_prob = calibrated.predict_proba(X)[:, 1]
    brier_cal = brier_score_loss(y, y_cal_prob)
    print(f"   Calibrated Brier score (in-sample): {brier_cal:.4f}")

    y_full = calibrated.predict(X)
    print(f"\n  Full-dataset report:\n")
    print(classification_report(y, y_full, target_names=["Red (0)", "Green (1)"]))

    # Feature importances from base estimator
    # Get from one of the calibrated estimators
    try:
        base_est = calibrated.calibrated_classifiers_[0].estimator
        imp = base_est.feature_importances_
        order = np.argsort(imp)[::-1]
        print("  Top 10 features:")
        for rank, idx in enumerate(order[:10], 1):
            name = FEATURE_COLUMNS[idx] if idx < len(FEATURE_COLUMNS) else f"feat_{idx}"
            print(f"    {rank:>2}. {name:<24s} {imp[idx]:.4f}")
    except Exception:
        print("  (Could not extract feature importances from calibrated model)")

    # -- Save --
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(calibrated, MODEL_PATH)
    joblib.dump(FEATURE_COLUMNS, FEATURE_LIST_PATH)
    print(f"\n>> Saved primary model  -> {MODEL_PATH}")
    print(f">> Saved feature list   -> {FEATURE_LIST_PATH}")

    # -- Save OOF predictions for meta model --
    # Only keep rows that have OOF predictions (from CV test folds)
    oof_mask = ~np.isnan(oof_proba)
    oof_data = {
        "oof_proba": oof_proba[oof_mask],
        "oof_pred": oof_pred[oof_mask],
        "actual": y[oof_mask],
        "indices": np.where(oof_mask)[0],
        "feature_df_index": df.index[oof_mask].tolist(),
    }
    joblib.dump(oof_data, OOF_PREDICTIONS_PATH)
    print(f">> Saved OOF predictions -> {OOF_PREDICTIONS_PATH}")
    print(f"   OOF rows: {int(oof_mask.sum())}")

    print(f"\n>> Done. Next step: python meta_model.py --symbol {symbol}")


def main():
    parser = argparse.ArgumentParser(description="Train primary XGBoost model.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    args = parser.parse_args()
    train(args.symbol)


if __name__ == "__main__":
    main()
