"""
Train Model v5 — Diverse ensemble + ATR-filtered target + regime routing.

Changes from v4.1:
  - Diverse ensemble (XGB + ExtraTrees + HistGB + LogReg)
  - ATR-threshold target (drops noise)
  - Two sub-ensembles: trending + ranging (routed by regime)

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
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

from calibration import CalibratedModel, build_diverse_ensemble

from config import (
    MODEL_DIR, MODEL_PATH, FEATURE_LIST_PATH,
    ENSEMBLE_MODEL_PATH, ENSEMBLE_TRENDING_PATH, ENSEMBLE_RANGING_PATH,
    OOF_PREDICTIONS_PATH, DEFAULT_SYMBOL, TIMESERIES_SPLITS,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_SUBSAMPLE, XGB_COLSAMPLE_BYTREE,
    LOG_LEVEL, LOG_FORMAT,
)
from data_collector import load_csv, load_multi_tf
from feature_engineering import (
    compute_features, add_target_atr_filtered, add_target, FEATURE_COLUMNS,
)
from regime_detection import detect_regime_series

log = logging.getLogger("train_model")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

TRENDING_REGIMES = {"Trending", "High_Volatility"}
RANGING_REGIMES = {"Ranging", "Low_Volatility"}


def _xgb_params():
    return {
        "n_estimators": XGB_N_ESTIMATORS, "max_depth": XGB_MAX_DEPTH,
        "learning_rate": XGB_LEARNING_RATE, "subsample": XGB_SUBSAMPLE,
        "colsample_bytree": XGB_COLSAMPLE_BYTREE,
    }


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

    # Simple target for OOF evaluation (all rows)
    df_all = add_target(df)
    df_all = df_all.dropna(subset=["target"]).reset_index(drop=True)
    df_all["target"] = df_all["target"].astype(int)

    X = df_train[FEATURE_COLUMNS]
    y = df_train["target"].values
    n_green, n_red = int(y.sum()), len(y) - int(y.sum())
    spw = n_red / max(n_green, 1)

    print(f"   ATR-filtered samples: {len(X)} (from {len(df_all)} total)")
    print(f"   Green: {n_green} ({n_green/len(y):.1%}), Red: {n_red} ({n_red/len(y):.1%})")

    # =========================================================================
    # Cross-validation (for metrics only, on filtered data)
    # =========================================================================
    tscv = TimeSeriesSplit(n_splits=TIMESERIES_SPLITS)
    oof_proba = np.full(len(y), np.nan)
    fold_metrics = []

    print(f"\n{'='*65}")
    print(f"  TimeSeriesSplit -- {TIMESERIES_SPLITS} folds (ATR-filtered)")
    print(f"{'='*65}\n")

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
        clf = xgb.XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE, subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE_BYTREE, scale_pos_weight=spw,
            objective="binary:logistic", eval_metric="logloss",
            use_label_encoder=False, random_state=42, verbosity=0,
        )
        clf.fit(X.iloc[tr_idx], y[tr_idx],
                eval_set=[(X.iloc[te_idx], y[te_idx])], verbose=False)
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
    # Detect regimes for routing
    # =========================================================================
    regimes = detect_regime_series(df_train)
    trending_mask = regimes.isin(TRENDING_REGIMES)
    ranging_mask = regimes.isin(RANGING_REGIMES)

    print(f"\n>> Regime split: Trending={trending_mask.sum()}, Ranging={ranging_mask.sum()}")

    # =========================================================================
    # Train diverse ensembles (global + regime-routed)
    # =========================================================================
    cal_split = int(len(y) * 0.8)
    X_base, y_base = X.iloc[:cal_split], y[:cal_split]
    X_cal, y_cal = X.iloc[cal_split:], y[cal_split:]
    print(f"   Base: {cal_split}, Cal val: {len(y) - cal_split}")

    # Global ensemble
    print(f"\n>> Training global diverse ensemble...")
    ensemble = build_diverse_ensemble(X_base, y_base, X_cal, y_cal, spw, _xgb_params())
    for m in ensemble:
        p = m.predict_proba(X_cal)[:, 1]
        brier = brier_score_loss(y_cal, p)
        print(f"  {m.name}: Brier(val)={brier:.4f}")

    # Ensemble predictions for variance check
    all_probs = np.array([m.predict_proba(X)[:, 1] for m in ensemble])
    ens_mean = all_probs.mean(axis=0)
    ens_var = all_probs.var(axis=0)
    print(f"\n  Ensemble variance: mean={ens_var.mean():.4f} max={ens_var.max():.4f}")

    # Regime-routed ensembles
    for label, mask, path in [
        ("Trending", trending_mask, ENSEMBLE_TRENDING_PATH),
        ("Ranging", ranging_mask, ENSEMBLE_RANGING_PATH),
    ]:
        idx = mask[mask].index
        if len(idx) < 200:
            print(f"  {label}: too few samples ({len(idx)}), using global ensemble")
            joblib.dump(ensemble, path)
            continue

        X_r = X.iloc[idx]
        y_r = y[idx]
        split_r = int(len(y_r) * 0.8)
        spw_r = np.sum(y_r == 0) / max(np.sum(y_r == 1), 1)
        print(f"\n>> Training {label} ensemble ({len(idx)} samples)...")
        ens_r = build_diverse_ensemble(
            X_r.iloc[:split_r], y_r[:split_r],
            X_r.iloc[split_r:], y_r[split_r:],
            spw_r, _xgb_params()
        )
        for m in ens_r:
            p = m.predict_proba(X_r.iloc[split_r:])[:, 1]
            brier = brier_score_loss(y_r[split_r:], p)
            print(f"  {m.name}: Brier(val)={brier:.4f}")
        joblib.dump(ens_r, path)
        print(f"  Saved -> {path}")

    # Feature importances
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

    # =========================================================================
    # Save
    # =========================================================================
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(ensemble[0], MODEL_PATH)
    joblib.dump(ensemble, ENSEMBLE_MODEL_PATH)
    joblib.dump(FEATURE_COLUMNS, FEATURE_LIST_PATH)
    print(f"\n>> Saved global ensemble -> {ENSEMBLE_MODEL_PATH}")

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
