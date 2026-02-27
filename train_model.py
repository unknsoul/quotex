"""
Train Model — Production leakage-free pipeline.

Architecture:
  1. Split data into train_main (80%) + calibration_slice (20%)
  2. Train 5 seeded XGBoost on train_main only
  3. Calibrate each with isotonic on calibration_slice only
  4. Generate OOF predictions from train_main via 3-fold TimeSeriesSplit
  5. Save OOF data (proba + per-member proba) for meta/weight training

Usage:
    python train_model.py --symbol EURUSD
"""

import argparse
import os
import json
import logging

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

from calibration import CalibratedModel, build_seeded_xgb_ensemble

from config import (
    MODEL_DIR, MODEL_PATH, FEATURE_LIST_PATH,
    ENSEMBLE_MODEL_PATH, OOF_PREDICTIONS_PATH,
    DECISION_THRESHOLDS_PATH,
    DEFAULT_SYMBOL, TIMESERIES_SPLITS,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_SUBSAMPLE, XGB_COLSAMPLE_BYTREE,
    CALIBRATION_SPLIT_RATIO, OOF_INTERNAL_SPLITS,
    ENSEMBLE_SEEDS,
    LOG_LEVEL, LOG_FORMAT,
)
from data_collector import load_csv, load_multi_tf
from regime_detection import detect_regime_series
from feature_engineering import (
    compute_features, add_target_atr_filtered, add_target, FEATURE_COLUMNS,
)

log = logging.getLogger("train_model")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


def _xgb_params():
    return {
        "n_estimators": XGB_N_ESTIMATORS, "max_depth": XGB_MAX_DEPTH,
        "learning_rate": XGB_LEARNING_RATE, "subsample": XGB_SUBSAMPLE,
        "colsample_bytree": XGB_COLSAMPLE_BYTREE,
    }


def _generate_oof_predictions(X, y, spw, seeds, n_splits=OOF_INTERNAL_SPLITS):
    """
    Generate Out-of-Fold predictions using TimeSeriesSplit.

    For each fold, trains 5 seeded XGBoost models on the in-fold data
    and predicts on the out-fold data. Returns:
      - oof_mean: mean probability across seeds for each OOF row
      - oof_all: [n_seeds × n_samples] array of per-seed probabilities
      - oof_mask: boolean mask of rows that have OOF predictions
    """
    n = len(y)
    oof_all = np.full((len(seeds), n), np.nan)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for fold_idx, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        for s_idx, seed in enumerate(seeds):
            clf = xgb.XGBClassifier(
                n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
                learning_rate=XGB_LEARNING_RATE, subsample=XGB_SUBSAMPLE,
                colsample_bytree=XGB_COLSAMPLE_BYTREE, scale_pos_weight=spw,
                objective="binary:logistic", eval_metric="logloss",
                use_label_encoder=False, random_state=seed, verbosity=0,
            )
            clf.fit(X.iloc[tr_idx], y[tr_idx], verbose=False)
            oof_all[s_idx, te_idx] = clf.predict_proba(X.iloc[te_idx])[:, 1]

    oof_mask = ~np.isnan(oof_all[0])
    oof_mean = np.nanmean(oof_all, axis=0)

    return oof_mean, oof_all, oof_mask


def _optimize_decision_threshold(y_true, proba, t_min=0.40, t_max=0.60, step=0.01):
    """Find threshold maximizing directional accuracy on provided rows."""
    if len(y_true) == 0:
        return 0.50, 0.0

    best_t, best_acc = 0.50, -1.0
    thresholds = np.arange(t_min, t_max + 1e-10, step)
    for t in thresholds:
        pred = (proba >= t).astype(int)
        acc = float((pred == y_true).mean())
        # tie-breaker: prefer closer to 0.5 (more stable)
        if (acc > best_acc) or (abs(acc - best_acc) < 1e-10 and abs(t - 0.5) < abs(best_t - 0.5)):
            best_acc = acc
            best_t = float(t)
    return best_t, best_acc


def train(symbol):
    print(f"\n>> Loading data for {symbol}...")
    mtf = load_multi_tf(symbol)
    df = mtf.get("M5")
    if df is None:
        df = load_csv(symbol, "M5")
    m15, h1 = mtf.get("M15"), mtf.get("H1")

    print(f">> Computing {len(FEATURE_COLUMNS)} features...")
    df = compute_features(df, m15_df=m15, h1_df=h1)

    # ATR-filtered target for training
    df_train = add_target_atr_filtered(df)
    df_train = df_train.dropna(subset=["target"]).reset_index(drop=True)
    df_train["target"] = df_train["target"].astype(int)
    df_train["regime"] = detect_regime_series(df_train)

    X = df_train[FEATURE_COLUMNS]
    y = df_train["target"].values
    n_green, n_red = int(y.sum()), len(y) - int(y.sum())
    spw = n_red / max(n_green, 1)

    print(f"   ATR-filtered samples: {len(X)}")
    print(f"   Green: {n_green} ({n_green/len(y):.1%}), Red: {n_red} ({n_red/len(y):.1%})")

    # =========================================================================
    # Split: train_main (80%) + calibration_slice (20%)
    # =========================================================================
    cal_split = int(len(y) * CALIBRATION_SPLIT_RATIO)
    X_main, y_main = X.iloc[:cal_split], y[:cal_split]
    X_cal, y_cal = X.iloc[cal_split:], y[cal_split:]

    print(f"\n>> Split: train_main={cal_split}, calibration_slice={len(y) - cal_split}")

    # =========================================================================
    # Step 1: Generate OOF predictions from train_main
    # =========================================================================
    print(f"\n>> Generating OOF predictions ({OOF_INTERNAL_SPLITS}-fold on train_main)...")
    oof_mean, oof_all, oof_mask = _generate_oof_predictions(
        X_main, y_main, spw, ENSEMBLE_SEEDS
    )

    oof_valid_count = int(oof_mask.sum())
    oof_p = oof_mean[oof_mask]
    oof_actual = y_main[oof_mask]
    oof_dir = (oof_p >= 0.5).astype(int)
    oof_acc = (oof_dir == oof_actual).mean()
    print(f"   OOF predictions: {oof_valid_count} rows, accuracy: {oof_acc:.4f}")

    # Per-seed variance on OOF rows
    oof_all_valid = oof_all[:, oof_mask]  # [n_seeds × valid_count]
    oof_var = oof_all_valid.var(axis=0)
    print(f"   OOF ensemble variance: mean={oof_var.mean():.4f} max={oof_var.max():.4f}")

    # OOF threshold optimization (leakage-safe): global + per-regime
    global_t, global_acc = _optimize_decision_threshold(oof_actual, oof_p)
    print(f"   OOF optimized threshold (global): t={global_t:.2f}, acc={global_acc:.4f}")

    oof_global_indices = X_main.index[oof_mask]
    oof_regimes = df_train.loc[oof_global_indices, "regime"].values
    regime_thresholds = {}
    for rg in sorted(set(oof_regimes)):
        m = (oof_regimes == rg)
        if int(m.sum()) < 100:
            continue
        rt, racc = _optimize_decision_threshold(oof_actual[m], oof_p[m])
        regime_thresholds[rg] = round(rt, 2)
        print(f"   OOF optimized threshold ({rg}): t={rt:.2f}, acc={racc:.4f}, n={int(m.sum())}")

    # =========================================================================
    # Step 2: Train production ensemble on full train_main, calibrate on cal_slice
    # =========================================================================
    print(f"\n>> Training 5-seeded XGBoost ensemble on train_main...")
    ensemble = build_seeded_xgb_ensemble(
        X_main, y_main, X_cal, y_cal, spw, _xgb_params(), ENSEMBLE_SEEDS
    )
    for m in ensemble:
        p = m.predict_proba(X_cal)[:, 1]
        brier = brier_score_loss(y_cal, p)
        print(f"  {m.name}: Brier(cal)={brier:.4f}")

    # Ensemble predictions on cal for variance check
    cal_probs = np.array([m.predict_proba(X_cal)[:, 1] for m in ensemble])
    cal_var = cal_probs.var(axis=0)
    print(f"\n  Cal ensemble variance: mean={cal_var.mean():.4f} max={cal_var.max():.4f}")

    # =========================================================================
    # Step 3: Cross-validation metrics (for reporting only)
    # =========================================================================
    tscv = TimeSeriesSplit(n_splits=TIMESERIES_SPLITS)
    fold_metrics = []
    print(f"\n{'='*65}")
    print(f"  TimeSeriesSplit -- {TIMESERIES_SPLITS} folds (ATR-filtered, metrics only)")
    print(f"{'='*65}\n")

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
        clf = xgb.XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE, subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE_BYTREE, scale_pos_weight=spw,
            objective="binary:logistic", eval_metric="logloss",
            use_label_encoder=False, random_state=42, verbosity=0,
        )
        clf.fit(X.iloc[tr_idx], y[tr_idx], verbose=False)
        y_prob = clf.predict_proba(X.iloc[te_idx])[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        m = {
            "acc": accuracy_score(y[te_idx], y_pred),
            "auc": roc_auc_score(y[te_idx], y_prob),
            "brier": brier_score_loss(y[te_idx], y_prob),
        }
        fold_metrics.append(m)
        print(f"  Fold {fold}: Acc={m['acc']:.4f} AUC={m['auc']:.4f} Brier={m['brier']:.4f}")

    avg = {k: np.mean([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
    print(f"\n  Avg: Acc={avg['acc']:.4f} AUC={avg['auc']:.4f} Brier={avg['brier']:.4f}")

    # Feature importances
    imp = None
    try:
        base_est = ensemble[0].base_model
        imp = base_est.feature_importances_
        order = np.argsort(imp)[::-1]
        print("\n  Top 10 features:")
        for rank, idx in enumerate(order[:10], 1):
            name = FEATURE_COLUMNS[idx] if idx < len(FEATURE_COLUMNS) else f"feat_{idx}"
            print(f"    {rank:>2}. {name:<26s} {imp[idx]:.4f}")
    except Exception:
        pass

    # =========================================================================
    # Save
    # =========================================================================
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(ensemble[0], MODEL_PATH)
    joblib.dump(ensemble, ENSEMBLE_MODEL_PATH)
    joblib.dump(FEATURE_COLUMNS, FEATURE_LIST_PATH)
    print(f"\n>> Saved ensemble ({len(ensemble)} models) -> {ENSEMBLE_MODEL_PATH}")

    # OOF data for meta + weight training
    oof_indices = np.where(oof_mask)[0]
    oof_data = {
        "oof_proba": oof_p,                          # mean OOF proba
        "oof_all_proba": oof_all_valid,               # [n_seeds × valid_count] per-seed
        "oof_pred": (oof_p >= 0.5).astype(int),
        "actual": oof_actual,
        "indices": oof_indices,
        "feature_importances": imp.tolist() if imp is not None else [],
        "feature_names": list(FEATURE_COLUMNS),
        "df_train_len": len(X),                       # for integrity check
    }
    joblib.dump(oof_data, OOF_PREDICTIONS_PATH)
    print(f">> Saved OOF predictions -> {OOF_PREDICTIONS_PATH} ({oof_valid_count} rows)")
    print(f"   OOF data includes per-seed probabilities for variance-based uncertainty")

    thresholds_payload = {
        "global": round(global_t, 2),
        "by_regime": regime_thresholds,
        "source": "OOF_train_main",
        "samples": int(oof_valid_count),
    }
    with open(DECISION_THRESHOLDS_PATH, "w", encoding="utf-8") as f:
        json.dump(thresholds_payload, f, indent=2)
    print(f">> Saved learned decision thresholds -> {DECISION_THRESHOLDS_PATH}")
    print(f"\n>> Done. Next: python meta_model.py --symbol {symbol}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    args = parser.parse_args()
    train(args.symbol)


if __name__ == "__main__":
    main()
