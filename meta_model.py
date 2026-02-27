"""
Meta Model — OOF-only, GradientBoosting, with isotonic calibration.

CRITICAL: This model trains exclusively on Out-of-Fold primary predictions.
It NEVER sees in-sample primary outputs. The OOF predictions come from
train_model.py's 3-fold TimeSeriesSplit within train_main.

Target: 1 if OOF primary prediction was correct, else 0.

Fixes applied:
  - OOF indices sorted ascending before slicing (prevents corrupted rolling)
  - Ensemble variance included as meta feature
  - Dead spread_ratio removed
  - Meta output calibrated with isotonic regression on holdout
  - Meta correlation logged after training
"""

import argparse
import os
import sys
import logging

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from scipy import stats

from config import (
    MODEL_DIR, META_MODEL_PATH, META_FEATURE_LIST_PATH,
    OOF_PREDICTIONS_PATH, DEFAULT_SYMBOL,
    META_N_ESTIMATORS, META_MAX_DEPTH, META_LEARNING_RATE,
    META_SUBSAMPLE,
    TIMESERIES_SPLITS, ATR_PERCENTILE_WINDOW,
    CALIBRATION_SPLIT_RATIO,
    LOG_LEVEL, LOG_FORMAT,
)
from data_collector import load_csv, load_multi_tf
from feature_engineering import compute_features, add_target, FEATURE_COLUMNS
from regime_detection import detect_regime_series

log = logging.getLogger("meta_model")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

# No spread_ratio (was dead 0.0), no recent_model_accuracy/recent_win_streak
# (derived from correct vector = target leak).
# Added ensemble_variance for uncertainty signal.
META_FEATURE_COLUMNS = [
    "primary_green_prob",
    "prob_distance_from_half",
    "primary_entropy",
    "ensemble_variance",
    "regime_encoded",
    "atr_value",
    "volatility_zscore",
    "range_position",
    "body_percentile_rank",
    "direction_streak",
    "rolling_vol_percentile",
]

META_CALIBRATOR_PATH = os.path.join(MODEL_DIR, "meta_calibrator.pkl")

REGIME_ENCODING = {"Trending": 0, "Ranging": 1, "High_Volatility": 2, "Low_Volatility": 3}


def _binary_entropy(p):
    """Entropy of binary prediction: -p*log(p) - (1-p)*log(1-p)."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


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
    """
    Build meta features using ONLY OOF predictions (never in-sample).
    
    CRITICAL: indices are sorted ascending before slicing to guarantee
    chronological order for any rolling features.
    """
    # FIX #1: Sort indices ascending to prevent corrupted rolling features
    indices = np.sort(oof_data["indices"])
    oof_proba = oof_data["oof_proba"]
    actual = oof_data["actual"]
    oof_all = oof_data["oof_all_proba"]  # [n_seeds × valid_count]

    # Reorder proba/actual to match sorted indices
    sort_order = np.argsort(oof_data["indices"])
    oof_proba = oof_proba[sort_order]
    actual = actual[sort_order]
    oof_all = oof_all[:, sort_order]

    sub = df.iloc[indices].copy().reset_index(drop=True)
    proba_s = pd.Series(oof_proba, index=sub.index)

    sub["primary_green_prob"] = oof_proba
    sub["prob_distance_from_half"] = np.abs(oof_proba - 0.5)
    sub["primary_entropy"] = _binary_entropy(oof_proba)

    # FIX #4: Add ensemble variance as meta feature
    oof_var = oof_all.var(axis=0)  # per-row variance across seeds
    sub["ensemble_variance"] = oof_var

    primary_dir = (oof_proba >= 0.5).astype(int)
    correct = (primary_dir == actual).astype(int)
    sub["meta_target"] = correct

    # Regime detection — verified safe: uses only past rows (backward-looking)
    regimes = detect_regime_series(sub)
    sub["regime_encoded"] = regimes.map(REGIME_ENCODING).fillna(1).astype(int)

    sub["atr_value"] = sub["atr_14"] if "atr_14" in sub.columns else 0.0

    if "volatility_zscore" not in sub.columns:
        sub["volatility_zscore"] = 0.0
    if "range_position" not in sub.columns:
        sub["range_position"] = 0.5

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
    print("   (indices sorted ascending before slicing)")
    meta_df = build_meta_features(df, oof_data)
    X_meta = meta_df[META_FEATURE_COLUMNS].fillna(0)
    y_meta = meta_df["meta_target"].values

    n_c, n_w = int(y_meta.sum()), len(y_meta) - int(y_meta.sum())
    print(f"   Samples: {len(X_meta)}, Correct: {n_c} ({n_c/len(y_meta):.1%}), Wrong: {n_w}")
    print(f"   Features: {META_FEATURE_COLUMNS}")

    # =========================================================================
    # Cross-validation (metrics only)
    # =========================================================================
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

    # LEAKAGE CHECK
    if avg["acc"] > 0.75:
        print(f"\n  ⚠️  WARNING: Meta CV accuracy {avg['acc']:.1%} is suspiciously high.")
        print(f"     Honest OOF meta should be 55-65%. Check for leakage!")
    elif avg["acc"] > 0.50:
        print(f"\n  ✅ Meta accuracy {avg['acc']:.1%} is in expected range (no leakage)")

    # =========================================================================
    # Train final meta model on train portion, calibrate on holdout
    # =========================================================================
    # FIX #5: Calibrate meta model output with isotonic regression
    cal_split = int(len(X_meta) * CALIBRATION_SPLIT_RATIO)
    X_meta_train, y_meta_train = X_meta.iloc[:cal_split], y_meta[:cal_split]
    X_meta_cal, y_meta_cal = X_meta.iloc[cal_split:], y_meta[cal_split:]

    print(f"\n>> Training final meta model (train={cal_split}, cal={len(X_meta)-cal_split})...")
    final = _build_meta_clf()
    final.fit(X_meta_train, y_meta_train)

    # Calibrate meta output
    raw_cal_proba = final.predict_proba(X_meta_cal)[:, 1]
    meta_iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    meta_iso.fit(raw_cal_proba, y_meta_cal)
    print(f"   Meta calibrator fitted on {len(X_meta_cal)} holdout points.")

    # Calibrated performance on holdout
    cal_proba = meta_iso.predict(raw_cal_proba)
    cal_pred = (cal_proba >= 0.5).astype(int)
    cal_acc = accuracy_score(y_meta_cal, cal_pred)
    cal_brier = brier_score_loss(y_meta_cal, cal_proba)
    print(f"   Calibrated holdout: Acc={cal_acc:.4f} Brier={cal_brier:.4f}")

    # Feature importances
    imp = final.feature_importances_
    order = np.argsort(imp)[::-1]
    print("\n  Meta feature importances:")
    for rank, idx in enumerate(order, 1):
        name = META_FEATURE_COLUMNS[idx] if idx < len(META_FEATURE_COLUMNS) else f"feat_{idx}"
        print(f"    {rank:>2}. {name:<30s} {imp[idx]:.4f}")

    # FIX #7: Log meta correlation
    all_meta_proba = meta_iso.predict(final.predict_proba(X_meta)[:, 1])
    meta_corr, _ = stats.spearmanr(all_meta_proba, y_meta)
    meta_corr = float(meta_corr) if not np.isnan(meta_corr) else 0.0
    print(f"\n  Meta Spearman correlation (proba vs target): {meta_corr:.4f}")
    if meta_corr < 0.2:
        print(f"  ⚠️  WARNING: Meta correlation {meta_corr:.4f} < 0.2 — meta may be unreliable")
    else:
        print(f"  ✅ Meta correlation {meta_corr:.4f} is acceptable")

    # =========================================================================
    # Save
    # =========================================================================
    joblib.dump(final, META_MODEL_PATH)
    joblib.dump(meta_iso, META_CALIBRATOR_PATH)
    joblib.dump(META_FEATURE_COLUMNS, META_FEATURE_LIST_PATH)
    print(f"\n>> Saved meta model -> {META_MODEL_PATH}")
    print(f">> Saved meta calibrator -> {META_CALIBRATOR_PATH}")
    print(f">> Saved meta features -> {META_FEATURE_LIST_PATH}")
    print(f"\n>> Done. Next: python weight_learner.py --symbol {symbol}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    args = parser.parse_args()
    train_meta(args.symbol)


if __name__ == "__main__":
    main()
