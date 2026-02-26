"""
Train Model v4 — 5-seed XGBoost ensemble + calibration + OOF predictions.

Each seed produces a calibrated model. Ensemble mean = primary probability.
Ensemble variance = uncertainty estimate.

Usage:
    python train_model.py --symbol EURUSD
"""

import argparse
import os
import logging

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss, classification_report,
)
from sklearn.isotonic import IsotonicRegression

from calibration import CalibratedModel, build_calibrated_model

from config import (
    MODEL_DIR, MODEL_PATH, FEATURE_LIST_PATH,
    ENSEMBLE_MODEL_PATH, OOF_PREDICTIONS_PATH,
    DEFAULT_SYMBOL, TIMESERIES_SPLITS, ENSEMBLE_SEEDS,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_SUBSAMPLE, XGB_COLSAMPLE_BYTREE,
    LOG_LEVEL, LOG_FORMAT,
)
from data_collector import load_csv, load_multi_tf
from feature_engineering import compute_features, add_target, FEATURE_COLUMNS

log = logging.getLogger("train_model")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


def _build_clf(spw, seed):
    return xgb.XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE, subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE, scale_pos_weight=spw,
        objective="binary:logistic", eval_metric="logloss",
        use_label_encoder=False, random_state=seed, verbosity=0,
    )


def train(symbol):
    # Load
    print(f"\n>> Loading data for {symbol}...")
    mtf = load_multi_tf(symbol)
    df = mtf.get("M5")
    if df is None:
        df = load_csv(symbol, "M5")
    m15, h1 = mtf.get("M15"), mtf.get("H1")

    # Features
    print(f">> Computing {len(FEATURE_COLUMNS)} features...")
    df = compute_features(df, m15_df=m15, h1_df=h1)
    df = add_target(df)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)

    X = df[FEATURE_COLUMNS]
    y = df["target"].values
    n_green, n_red = int(y.sum()), len(y) - int(y.sum())
    spw = n_red / max(n_green, 1)

    print(f"   Samples: {len(X)}, Features: {len(FEATURE_COLUMNS)}")
    print(f"   Green: {n_green} ({n_green/len(y):.1%}), Red: {n_red} ({n_red/len(y):.1%})")

    # =========================================================================
    # Cross-validation (single seed for metrics)
    # =========================================================================
    tscv = TimeSeriesSplit(n_splits=TIMESERIES_SPLITS)
    oof_proba = np.full(len(y), np.nan)
    fold_metrics = []

    print(f"\n{'='*65}")
    print(f"  TimeSeriesSplit -- {TIMESERIES_SPLITS} folds")
    print(f"{'='*65}\n")

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
        clf = _build_clf(spw, ENSEMBLE_SEEDS[0])
        clf.fit(X.iloc[tr_idx], y[tr_idx], eval_set=[(X.iloc[te_idx], y[te_idx])], verbose=False)
        y_prob = clf.predict_proba(X.iloc[te_idx])[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        oof_proba[te_idx] = y_prob

        m = {
            "acc": accuracy_score(y[te_idx], y_pred),
            "auc": roc_auc_score(y[te_idx], y_prob),
            "brier": brier_score_loss(y[te_idx], y_prob),
        }
        fold_metrics.append(m)
        print(f"  Fold {fold}: Acc={m['acc']:.4f} AUC={m['auc']:.4f} Brier={m['brier']:.4f}")

    avg = {k: np.mean([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
    print(f"\n  Avg: Acc={avg['acc']:.4f} AUC={avg['auc']:.4f} Brier={avg['brier']:.4f}")

    # =========================================================================
    # Train 5-seed ensemble (manual isotonic calibration on held-out val)
    # =========================================================================
    print(f"\n>> Training {len(ENSEMBLE_SEEDS)}-seed ensemble (manual isotonic)...")
    cal_split = int(len(y) * 0.8)
    X_base, y_base = X.iloc[:cal_split], y[:cal_split]
    X_cal, y_cal = X.iloc[cal_split:], y[cal_split:]
    print(f"   Base train: {cal_split}, Calibration val: {len(y) - cal_split}")

    ensemble = []
    for i, seed in enumerate(ENSEMBLE_SEEDS, 1):
        base = _build_clf(spw, seed)
        base.fit(X_base, y_base,
                 eval_set=[(X_cal, y_cal)], verbose=False)
        cal_model = build_calibrated_model(base, X_cal, y_cal)
        ensemble.append(cal_model)
        p = cal_model.predict_proba(X_cal)[:, 1]
        brier = brier_score_loss(y_cal, p)
        print(f"  Seed {seed}: Brier(val)={brier:.4f}")

    # Ensemble predictions
    all_probs = np.array([m.predict_proba(X)[:, 1] for m in ensemble])
    ens_mean = all_probs.mean(axis=0)
    ens_var = all_probs.var(axis=0)
    ens_pred = (ens_mean >= 0.5).astype(int)

    ens_acc = accuracy_score(y, ens_pred)
    ens_brier = brier_score_loss(y, ens_mean)
    print(f"\n  Ensemble (in-sample): Acc={ens_acc:.4f} Brier={ens_brier:.4f}")
    print(f"  Avg variance: {ens_var.mean():.6f}, Max: {ens_var.max():.6f}")

    # Feature importances (from first model's base)
    try:
        base_est = ensemble[0].base_model
        imp = base_est.feature_importances_
        order = np.argsort(imp)[::-1]
        print("\n  Top 10 features:")
        for rank, idx in enumerate(order[:10], 1):
            name = FEATURE_COLUMNS[idx] if idx < len(FEATURE_COLUMNS) else f"feat_{idx}"
            print(f"    {rank:>2}. {name:<26s} {imp[idx]:.4f}")
    except Exception:
        imp = None
        print("  (Could not extract feature importances)")

    # =========================================================================
    # Save
    # =========================================================================
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(ensemble[0], MODEL_PATH)  # backward compat: single model
    joblib.dump(ensemble, ENSEMBLE_MODEL_PATH)
    joblib.dump(FEATURE_COLUMNS, FEATURE_LIST_PATH)
    print(f"\n>> Saved ensemble ({len(ensemble)} models) -> {ENSEMBLE_MODEL_PATH}")
    print(f">> Saved feature list -> {FEATURE_LIST_PATH}")

    # OOF predictions for meta model
    oof_mask = ~np.isnan(oof_proba)
    oof_data = {
        "oof_proba": oof_proba[oof_mask],
        "oof_pred": (oof_proba[oof_mask] >= 0.5).astype(int),
        "actual": y[oof_mask],
        "indices": np.where(oof_mask)[0],
        "feature_importances": imp.tolist() if imp is not None else [],
        "feature_names": list(FEATURE_COLUMNS),
    }
    joblib.dump(oof_data, OOF_PREDICTIONS_PATH)
    print(f">> Saved OOF predictions -> {OOF_PREDICTIONS_PATH} ({int(oof_mask.sum())} rows)")
    print(f"\n>> Done. Next: python meta_model.py --symbol {symbol}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    args = parser.parse_args()
    train(args.symbol)


if __name__ == "__main__":
    main()
