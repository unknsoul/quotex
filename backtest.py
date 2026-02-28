"""
Backtest — Honest walk-forward with leakage-free per-cycle training.

Per expanding-window cycle:
  1. Split train slice into train_main (80%) + cal_slice (20%)
  2. Train 5 seeded XGBs on train_main
  3. Calibrate on cal_slice
  4. Generate OOF predictions on train_main via 3-fold TimeSeriesSplit
  5. Train meta model (LightGBM) on OOF predictions only
  6. Train weight learner on OOF data only
  7. Predict on unseen test chunk
  8. No model sees future data at any point

Usage:
    python backtest.py --symbol EURUSD
"""

import argparse
import os
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, log_loss
from scipy import stats

from calibration import CalibratedModel, build_seeded_xgb_ensemble
from config import (
    DEFAULT_SYMBOL, CHART_DIR,
    META_ROLLING_WINDOW, WIN_STREAK_CAP, ATR_PERCENTILE_WINDOW,
    CONFIDENCE_HIGH_MIN, CONFIDENCE_MEDIUM_MIN,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_SUBSAMPLE, XGB_COLSAMPLE_BYTREE, ENSEMBLE_SEEDS,
    META_N_ESTIMATORS, META_MAX_DEPTH, META_LEARNING_RATE,
    META_SUBSAMPLE, META_NUM_LEAVES,
    DRIFT_COSINE_THRESHOLD, TARGET_ATR_THRESHOLD,
    CALIBRATION_SPLIT_RATIO, OOF_INTERNAL_SPLITS,
    CONFIDENCE_CORRELATION_WINDOW, CONFIDENCE_CORRELATION_ALERT,
    LOG_LEVEL, LOG_FORMAT,
)
from data_collector import load_csv, load_multi_tf
from feature_engineering import (
    compute_features, add_target, add_target_atr_filtered, FEATURE_COLUMNS,
)
from regime_detection import detect_regime, detect_regime_series, REGIMES, get_session
from stability import CosineDriftDetector

log = logging.getLogger("backtest")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

REGIME_ENCODING = {"Trending": 0, "Ranging": 1, "High_Volatility": 2, "Low_Volatility": 3}

META_FEATURE_COLUMNS = [
    "primary_green_prob", "prob_distance_from_half", "primary_entropy",
    "ensemble_variance",
    "regime_encoded", "atr_value", "volatility_zscore",
    "range_position",
    "body_percentile_rank", "direction_streak", "rolling_vol_percentile",
]

WEIGHT_FEATURES = ["primary_strength", "meta_reliability", "regime_strength", "uncertainty"]


def _binary_entropy(p):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def _xgb_params():
    return {
        "n_estimators": XGB_N_ESTIMATORS, "max_depth": XGB_MAX_DEPTH,
        "learning_rate": XGB_LEARNING_RATE, "subsample": XGB_SUBSAMPLE,
        "colsample_bytree": XGB_COLSAMPLE_BYTREE,
    }


def _build_meta():
    return GradientBoostingClassifier(
        n_estimators=META_N_ESTIMATORS, max_depth=META_MAX_DEPTH,
        learning_rate=META_LEARNING_RATE, subsample=META_SUBSAMPLE,
        random_state=42,
    )


def _regime_for_row(df, idx, lookback=200):
    start = max(0, idx - lookback + 1)
    w = df.iloc[start:idx + 1]
    return detect_regime(w) if len(w) >= 50 else "Ranging"


def _build_meta_row(green_p, regime, row, dir_history, ensemble_variance=0.0):
    """Build meta row matching META_FEATURE_COLUMNS exactly."""
    regime_enc = REGIME_ENCODING.get(regime, 1)

    cur_dir = 1 if green_p >= 0.5 else 0
    dstreak = 1
    for d in reversed(dir_history):
        if d == cur_dir:
            dstreak += 1
        else:
            break

    return {
        "primary_green_prob": green_p,
        "prob_distance_from_half": abs(green_p - 0.5),
        "primary_entropy": float(_binary_entropy(green_p)),
        "ensemble_variance": ensemble_variance,
        "regime_encoded": regime_enc,
        "atr_value": float(row.get("atr_14", 0)),
        "volatility_zscore": float(row.get("volatility_zscore", 0)),
        "range_position": float(row.get("range_position", 0.5)),
        "body_percentile_rank": float(row.get("body_size", 0.5)),
        "direction_streak": dstreak,
        "rolling_vol_percentile": float(row.get("atr_percentile_rank", 0.5)),
    }


# =============================================================================
#  OOF prediction generator (per-cycle, leakage-free)
# =============================================================================

def _generate_oof_predictions(X_tr, y_tr, spw, seeds, n_splits=OOF_INTERNAL_SPLITS):
    """Generate OOF predictions with 5 seeds × n_splits folds."""
    n = len(y_tr)
    oof_all = np.full((len(seeds), n), np.nan)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for fold_idx, (tr_idx, te_idx) in enumerate(tscv.split(X_tr)):
        for s_idx, seed in enumerate(seeds):
            clf = xgb.XGBClassifier(
                n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
                learning_rate=XGB_LEARNING_RATE, subsample=XGB_SUBSAMPLE,
                colsample_bytree=XGB_COLSAMPLE_BYTREE, scale_pos_weight=spw,
                objective="binary:logistic", eval_metric="logloss",
                use_label_encoder=False, random_state=seed, verbosity=0,
            )
            clf.fit(X_tr.iloc[tr_idx], y_tr[tr_idx], verbose=False)
            oof_all[s_idx, te_idx] = clf.predict_proba(X_tr.iloc[te_idx])[:, 1]

    oof_mask = ~np.isnan(oof_all[0])
    oof_mean = np.nanmean(oof_all, axis=0)
    return oof_mean, oof_all, oof_mask


# =============================================================================
#  Walk-Forward Engine
# =============================================================================

def run_walk_forward(symbol, train_ratio=0.6, chunk_ratio=0.1, rolling_window=0):
    mtf = load_multi_tf(symbol)
    df = mtf.get("M5")
    if df is None:
        df = load_csv(symbol, "M5")
    m15, h1 = mtf.get("M15"), mtf.get("H1")

    df = compute_features(df, m15_df=m15, h1_df=h1)

    # ATR-filtered target for training
    df_filtered = add_target_atr_filtered(df)
    # Simple target for test evaluation (all rows)
    df_eval = add_target(df)
    df_eval = df_eval.dropna(subset=["target"]).reset_index(drop=True)
    df_eval["target"] = df_eval["target"].astype(int)

    X_all = df_eval[FEATURE_COLUMNS]
    y_all = df_eval["target"].values
    n = len(df_eval)
    train_end = int(n * train_ratio)
    chunk = int(n * chunk_ratio)

    df_filtered = df_filtered.dropna(subset=["target"]).reset_index(drop=True)
    df_filtered["target"] = df_filtered["target"].astype(int)

    mode_str = f"Rolling({rolling_window})" if rolling_window > 0 else "Expanding"
    print(f"\n>> Walk-Forward ({mode_str}, Leakage-Free) for {symbol}")
    print(f"   Bars: {n}, Train: {train_end}, Chunk: {chunk}")
    if rolling_window > 0:
        print(f"   Rolling window: {rolling_window} bars (~{rolling_window * 5 / 60 / 24:.0f} days)")
    print(f"   ATR threshold: {TARGET_ATR_THRESHOLD}, Ensemble: {len(ENSEMBLE_SEEDS)} seeded XGB")
    print(f"   Cal split: {CALIBRATION_SPLIT_RATIO:.0%}/{1-CALIBRATION_SPLIT_RATIO:.0%}, OOF splits: {OOF_INTERNAL_SPLITS}")

    all_results = []
    meta_history = []
    dir_history = []
    equity = []
    cum_pnl = 0.0
    cycle = 0
    drift_detector = CosineDriftDetector(DRIFT_COSINE_THRESHOLD)
    drift_reports = []

    all_regimes = detect_regime_series(df_eval)

    while train_end < n:
        cycle += 1
        test_start = train_end
        test_end = min(train_end + chunk, n)

        # Filter train data — rolling or expanding window
        if rolling_window > 0:
            # Rolling: use only last N bars before test start
            roll_start = max(0, train_end - rolling_window)
            train_mask = (df_filtered.index >= roll_start) & (df_filtered.index < train_end)
        else:
            # Expanding: use all bars from 0 to train_end
            train_mask = df_filtered.index < train_end
        X_tr_filtered = df_filtered.loc[train_mask, FEATURE_COLUMNS]
        y_tr_filtered = df_filtered.loc[train_mask, "target"].values

        spw = np.sum(y_tr_filtered == 0) / max(np.sum(y_tr_filtered == 1), 1)

        if rolling_window > 0:
            roll_start = max(0, train_end - rolling_window)
            print(f"\n  Cycle {cycle}: train[{roll_start}:{train_end}] test[{test_start}:{test_end}]")
        else:
            print(f"\n  Cycle {cycle}: train[0:{train_end}] test[{test_start}:{test_end}]")
        print(f"    ATR-filtered train: {len(X_tr_filtered)} / {train_end}")

        # =================================================================
        # Step 1: Split into train_main + cal_slice (LEAKAGE-FREE)
        # =================================================================
        cal_split = int(len(X_tr_filtered) * CALIBRATION_SPLIT_RATIO)
        X_main = X_tr_filtered.iloc[:cal_split]
        y_main = y_tr_filtered[:cal_split]
        X_cal = X_tr_filtered.iloc[cal_split:]
        y_cal = y_tr_filtered[cal_split:]

        # =================================================================
        # Step 2: Train ensemble on train_main, calibrate on cal_slice
        # =================================================================
        ensemble = build_seeded_xgb_ensemble(
            X_main, y_main, X_cal, y_cal, spw, _xgb_params(), ENSEMBLE_SEEDS
        )

        # Drift detection
        try:
            base_est = ensemble[0].base_model
            imp = base_est.feature_importances_
            cosine_sim, drifted, msg = drift_detector.check_drift(imp)
            drift_reports.append({"cycle": cycle, "cosine": cosine_sim, "drifted": drifted})
            drift_detector.update_baseline(imp)
            if drifted:
                print(f"    DRIFT: {msg}")
        except Exception:
            pass

        # =================================================================
        # Step 3: Generate OOF predictions on train_main (LEAKAGE-FREE)
        # =================================================================
        oof_mean, oof_all, oof_mask = _generate_oof_predictions(
            X_main, y_main, spw, ENSEMBLE_SEEDS
        )
        oof_indices = np.where(oof_mask)[0]
        oof_p = oof_mean[oof_mask]
        oof_actual = y_main[oof_mask]
        oof_dir = (oof_p >= 0.5).astype(int)
        oof_correct = (oof_dir == oof_actual).astype(int)

        # OOF per-seed for variance
        oof_all_valid = oof_all[:, oof_mask]
        oof_var = oof_all_valid.var(axis=0)
        oof_var_norm = oof_var / (oof_var.max() + 1e-10)

        # =================================================================
        # Step 4: Train meta model on OOF (LEAKAGE-FREE)
        # =================================================================
        sub_main = df_eval.iloc[X_main.index[oof_indices]].copy().reset_index(drop=True) \
            if len(oof_indices) > 0 else pd.DataFrame()

        meta_rows = []
        dh_local = []
        for k in range(len(oof_indices)):
            idx = X_main.index[oof_indices[k]] if oof_indices[k] < len(X_main) else oof_indices[k]
            regime = _regime_for_row(df_eval, idx) if idx > 50 else "Ranging"
            mr = _build_meta_row(oof_p[k], regime, df_eval.iloc[min(idx, len(df_eval)-1)],
                                 dh_local, ensemble_variance=oof_var[k])
            mr["meta_target"] = oof_correct[k]
            meta_rows.append(mr)
            dh_local.append(1 if oof_p[k] >= 0.5 else 0)

        meta_df = pd.DataFrame(meta_rows)
        X_meta_tr = meta_df[META_FEATURE_COLUMNS].fillna(0)
        y_meta_tr = meta_df["meta_target"].values

        # Sigmoid calibration for meta (Platt scaling — stable on noisy data)
        base_meta = _build_meta()
        meta_model = CalibratedClassifierCV(base_meta, method="sigmoid", cv=3)
        meta_model.fit(X_meta_tr, y_meta_tr)

        # =================================================================
        # Step 5: Train weight learner on OOF (LEAKAGE-FREE)
        # =================================================================
        oof_strength = np.abs(oof_p - 0.5) * 2
        meta_rel_oof = meta_model.predict_proba(X_meta_tr.values)[:, 1]
        regime_str_oof = np.zeros(len(oof_indices))
        for k, oi in enumerate(oof_indices):
            idx = X_main.index[oi] if oi < len(X_main) else oi
            if idx < len(df_eval) and "adx_normalized" in df_eval.columns:
                regime_str_oof[k] = df_eval["adx_normalized"].iloc[min(idx, len(df_eval)-1)]

        W_tr = pd.DataFrame({
            "primary_strength": oof_strength,
            "meta_reliability": meta_rel_oof,
            "regime_strength": regime_str_oof,
            "uncertainty": oof_var_norm,
        })
        weight_model = LogisticRegression(random_state=42, max_iter=500)
        weight_model.fit(W_tr, oof_correct)

        # =================================================================
        # Step 6: Test on unseen chunk
        # =================================================================
        cycle_correct = 0
        for i in range(test_start, test_end):
            row_feat = X_all.iloc[[i]]
            regime = all_regimes.iloc[i] if i < len(all_regimes) else "Ranging"

            all_p = np.array([m.predict_proba(row_feat)[0][1] for m in ensemble])
            green_p = float(all_p.mean())
            variance = float(all_p.var())
            norm_var = min(variance / 0.25, 1.0)
            actual = y_all[i]

            meta_row = _build_meta_row(green_p, regime, df_eval.iloc[i], dir_history,
                                       ensemble_variance=variance)
            meta_in = pd.DataFrame([meta_row])[META_FEATURE_COLUMNS]
            meta_rel = float(meta_model.predict_proba(meta_in)[:, 1][0])

            primary_str = abs(green_p - 0.5) * 2
            regime_str = float(df_eval.iloc[i].get("adx_normalized", 0.25))
            w_in = pd.DataFrame([{
                "primary_strength": primary_str,
                "meta_reliability": meta_rel,
                "regime_strength": regime_str,
                "uncertainty": norm_var,
            }])
            weighted_score = float(weight_model.predict_proba(w_in)[:, 1][0])
            confidence = weighted_score * (1.0 - norm_var) * 100

            direction = 1 if green_p >= 0.5 else 0
            correct = 1 if direction == actual else 0
            cycle_correct += correct
            cum_pnl += (1 if correct else -1)
            equity.append(cum_pnl)
            meta_history.append(correct)
            dir_history.append(direction)

            # Detect session
            session = "Off"
            try:
                t = df_eval.iloc[i].get("time", None)
                if t is not None and hasattr(t, 'hour'):
                    session = get_session(t.hour)
            except Exception:
                pass

            all_results.append({
                "bar": i, "green_p": green_p, "meta_rel": meta_rel,
                "confidence": confidence, "uncertainty": norm_var * 100,
                "correct": correct, "regime": regime, "cycle": cycle,
                "session": session,
            })

        test_acc = cycle_correct / max(test_end - test_start, 1)
        print(f"    Test accuracy: {test_acc:.1%} ({cycle_correct}/{test_end - test_start})")
        train_end = test_end

    return {"results": all_results, "equity": equity, "cycles": cycle, "drift": drift_reports}


# =============================================================================
#  Analysis & Reporting
# =============================================================================

def confidence_bin_analysis(results):
    bins = np.arange(0, 101, 10)
    labels = [f"{b}-{b+10}%" for b in bins[:-1]]
    accs, counts = [], []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        in_bin = [r for r in results if lo <= r["confidence"] < hi]
        accs.append(sum(r["correct"] for r in in_bin) / len(in_bin) if in_bin else 0)
        counts.append(len(in_bin))
    return labels, accs, counts


def compute_confidence_correlation(results, window=200):
    """Compute Spearman correlation between confidence and correctness."""
    if len(results) < 30:
        return 0.0
    recent = results[-window:]
    confs = [r["confidence"] for r in recent]
    corrects = [r["correct"] for r in recent]
    corr, _ = stats.spearmanr(confs, corrects)
    return round(float(corr) if not np.isnan(corr) else 0.0, 4)


def stability_warnings(results, window=100):
    warns = []
    corrects = [r["correct"] for r in results]
    if len(corrects) >= window:
        rolling = pd.Series(corrects).rolling(window).mean()
        if rolling.dropna().min() < 0.45:
            warns.append(f"ACCURACY COLLAPSE: min rolling {window}-bar = {rolling.dropna().min():.1%}")

    high = [r for r in results if r["confidence"] >= CONFIDENCE_HIGH_MIN]
    if len(high) > 50:
        h_acc = sum(r["correct"] for r in high) / len(high)
        o_acc = sum(corrects) / len(corrects)
        if h_acc < o_acc:
            warns.append(f"OVERCONFIDENCE: high-conf {h_acc:.1%} < overall {o_acc:.1%}")

    conf_corr = compute_confidence_correlation(results)
    if conf_corr < CONFIDENCE_CORRELATION_ALERT and len(results) > 50:
        warns.append(f"CONFIDENCE UNCORRELATED: Spearman={conf_corr:.3f} < {CONFIDENCE_CORRELATION_ALERT}")

    return warns


def save_charts(results, equity):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib unavailable)")
        return

    os.makedirs(CHART_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(equity, linewidth=0.8, color="#2196F3")
    ax.set_title("Walk-Forward Equity (Leakage-Free)", fontsize=13)
    ax.set_xlabel("Prediction #"); ax.set_ylabel("Cumulative PnL")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "equity_curve.png"), dpi=120)
    plt.close(fig)

    corrects = [r["correct"] for r in results]
    rolling = pd.Series(corrects).rolling(100).mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(rolling, linewidth=0.8, color="#4CAF50")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
    ax.set_title("Rolling Accuracy (100-bar)", fontsize=13)
    ax.set_xlabel("Prediction #"); ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "rolling_accuracy.png"), dpi=120)
    plt.close(fig)

    labels, accs, counts = confidence_bin_analysis(results)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax1.bar(x, [a * 100 for a in accs], color="#FF9800", alpha=0.7, label="Accuracy %")
    ax1.axhline(y=50, color="red", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Confidence Bin"); ax1.set_ylabel("Accuracy %")
    ax1.set_title("Confidence vs Accuracy (Leakage-Free)", fontsize=13)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=45)
    ax2 = ax1.twinx()
    ax2.plot(x, counts, "o-", color="#9C27B0", label="Count")
    ax2.set_ylabel("Count")
    fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95))
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "confidence_vs_accuracy.png"), dpi=120)
    plt.close(fig)

    print(f"\n  Charts saved to {CHART_DIR}/")


def print_report(result):
    results = result["results"]
    equity = result["equity"]
    total = len(results)
    correct = sum(r["correct"] for r in results)
    overall_acc = correct / total * 100

    print(f"\n{'='*75}")
    print(f"  WALK-FORWARD BACKTEST (Leakage-Free)")
    print(f"  Cycles: {result['cycles']} | Predictions: {total}")
    print(f"{'='*75}")

    cycles = {}
    for r in results:
        c = r["cycle"]
        if c not in cycles:
            cycles[c] = {"total": 0, "correct": 0}
        cycles[c]["total"] += 1
        cycles[c]["correct"] += r["correct"]

    print(f"\n  {'Cycle':>6} {'Bars':>7} {'Correct':>8} {'Accuracy':>9}")
    print(f"  {'-'*35}")
    for c in sorted(cycles):
        s = cycles[c]
        print(f"  {c:>6} {s['total']:>7} {s['correct']:>8} {s['correct']/s['total']*100:>8.1f}%")
    print(f"  {'-'*35}")
    print(f"  {'ALL':>6} {total:>7} {correct:>8} {overall_acc:>8.1f}%")

    regime_stats = {r: {"t": 0, "c": 0} for r in REGIMES}
    for r in results:
        regime_stats[r["regime"]]["t"] += 1
        regime_stats[r["regime"]]["c"] += r["correct"]
    print(f"\n  {'Regime':<18} {'Total':>7} {'Acc':>8}")
    print(f"  {'-'*35}")
    for reg in REGIMES:
        s = regime_stats[reg]
        print(f"  {reg:<18} {s['t']:>7} {s['c']/s['t']*100 if s['t'] else 0:>7.1f}%")

    # Session breakdown
    sessions = ["Asian", "London", "Overlap", "New_York", "Off"]
    session_stats = {s: {"t": 0, "c": 0} for s in sessions}
    for r in results:
        s = r.get("session", "Off")
        if s in session_stats:
            session_stats[s]["t"] += 1
            session_stats[s]["c"] += r["correct"]
    print(f"\n  {'Session':<18} {'Total':>7} {'Acc':>8}")
    print(f"  {'-'*35}")
    for sess in sessions:
        s = session_stats[sess]
        if s["t"] > 0:
            print(f"  {sess:<18} {s['t']:>7} {s['c']/s['t']*100:>7.1f}%")

    # ================================================================
    # PROBABILITY RELIABILITY METRICS
    # ================================================================
    green_ps = np.array([r["green_p"] for r in results])
    actuals = np.array([r["correct"] for r in results])
    # For Brier/log loss, use green_p as forecasted P(green)
    # actual direction: correct means predicted=actual, but we need P(actual=green)
    # green_p is P(green), actual target is 1=green, 0=red
    # So: direction = green_p >= 0.5, correct = direction==actual
    # We need the raw target for Brier, reconstruct from green_p and correct:
    directions = (green_ps >= 0.5).astype(int)
    # correct==1 means direction==target, so target = direction if correct else (1-direction)
    targets = np.where(actuals == 1, directions, 1 - directions)

    brier = brier_score_loss(targets, green_ps)
    brier_naive = brier_score_loss(targets, np.full_like(green_ps, 0.5))
    brier_skill = 1 - (brier / brier_naive) if brier_naive > 0 else 0

    try:
        ll = log_loss(targets, np.clip(green_ps, 1e-10, 1 - 1e-10))
        ll_naive = log_loss(targets, np.full_like(green_ps, 0.5))
    except Exception:
        ll, ll_naive = 0.693, 0.693

    print(f"\n  PROBABILITY RELIABILITY:")
    print(f"    Brier Score:       {brier:.4f}  (naive 0.50 baseline: {brier_naive:.4f})")
    print(f"    Brier Skill Score: {brier_skill:.4f}  (>0 = better than naive, 1 = perfect)")
    print(f"    Log Loss:          {ll:.4f}  (naive baseline: {ll_naive:.4f})")

    # Overall Spearman
    confs = [r["confidence"] for r in results]
    corr_all, _ = stats.spearmanr(confs, actuals)
    corr_all = float(corr_all) if not np.isnan(corr_all) else 0.0
    status = "[OK]" if corr_all >= 0.2 else ("[WARN]" if corr_all >= 0.1 else "[BAD]")
    print(f"    Spearman(conf, correct): {corr_all:.4f} {status}")

    # ================================================================
    # CALIBRATION RELIABILITY TABLE (5% bins on green_p)
    # ================================================================
    print(f"\n  CALIBRATION RELIABILITY (5% bins on primary probability):")
    print(f"  {'Bin':<12} {'Count':>6} {'Expected':>9} {'Actual':>8} {'Gap':>6}")
    print(f"  {'-'*44}")
    cal_bins = np.arange(0.50, 1.01, 0.05)
    for i in range(len(cal_bins) - 1):
        lo, hi = cal_bins[i], cal_bins[i + 1]
        # Include both green (p >= 0.5) and red (p < 0.5) sides
        # For green side: green_p in [lo, hi)
        green_mask = (green_ps >= lo) & (green_ps < hi)
        # For red side (symmetry): green_p in (1-hi, 1-lo]
        red_mask = (green_ps > (1 - hi)) & (green_ps <= (1 - lo))
        mask = green_mask | red_mask
        if mask.sum() == 0:
            continue
        expected = (lo + hi) / 2  # average expected probability
        actual_acc = actuals[mask].mean()
        gap = actual_acc - expected
        label = f"{lo:.2f}-{hi:.2f}"
        print(f"  {label:<12} {mask.sum():>6} {expected:>8.1%} {actual_acc:>7.1%} {gap:>+5.1%}")

    # ================================================================
    # ROLLING 100-BAR STABILITY
    # ================================================================
    correct_series = pd.Series([r["correct"] for r in results])
    rolling_acc = correct_series.rolling(100, min_periods=100).mean()
    rolling_clean = rolling_acc.dropna()
    if len(rolling_clean) > 0:
        r_min = rolling_clean.min() * 100
        r_max = rolling_clean.max() * 100
        r_mean = rolling_clean.mean() * 100
        r_std = rolling_clean.std() * 100
        print(f"\n  ROLLING 100-BAR STABILITY:")
        print(f"    Range: {r_min:.1f}% — {r_max:.1f}%  (mean={r_mean:.1f}%, std={r_std:.1f}%)")
        stable = "Stable" if r_std < 5 else ("Moderate" if r_std < 8 else "Unstable")
        print(f"    Verdict: {stable}")

    # ================================================================
    # TRADE FREQUENCY ANALYSIS (Confidence-Gated)
    # ================================================================
    thresholds = [0, 50, 60, 70, 80]
    print(f"\n  TRADE FREQUENCY ANALYSIS:")
    print(f"  {'Threshold':<12} {'Trades':>7} {'Freq':>7} {'Correct':>8} {'Accuracy':>9} {'Lift':>6}")
    print(f"  {'-'*52}")
    for thr in thresholds:
        gated = [r for r in results if r["confidence"] >= thr]
        if gated:
            g_correct = sum(r["correct"] for r in gated)
            g_acc = g_correct / len(gated) * 100
            freq = len(gated) / total * 100
            lift = g_acc - overall_acc
            label = f">={thr}%" if thr > 0 else "All bars"
            print(f"  {label:<12} {len(gated):>7} {freq:>6.1f}% {g_correct:>8} {g_acc:>8.1f}% {lift:>+5.1f}")

    # ================================================================
    # REGIME × CONFIDENCE GATED
    # ================================================================
    print(f"\n  REGIME × CONFIDENCE BREAKDOWN:")
    print(f"  {'Regime':<18} {'All':>10} {'>50%':>10} {'>60%':>10} {'>70%':>10} {'>80%':>10}")
    print(f"  {'-'*70}")
    for reg in REGIMES:
        reg_results = [r for r in results if r["regime"] == reg]
        if not reg_results:
            continue
        parts = []
        for thr in [0, 50, 60, 70, 80]:
            gated = [r for r in reg_results if r["confidence"] >= thr]
            if gated:
                acc = sum(r["correct"] for r in gated) / len(gated) * 100
                parts.append(f"{acc:.0f}%({len(gated)})")
            else:
                parts.append("—")
        print(f"  {reg:<18} {'  '.join(f'{p:>10}' for p in parts)}")

    # ================================================================
    # PER-CYCLE STABILITY
    # ================================================================
    print(f"\n  PER-CYCLE STABILITY:")
    print(f"  {'Cycle':>6} {'Bars':>7} {'Accuracy':>9} {'AvgConf':>8} {'AvgUnc':>7} {'Spearman':>9}")
    print(f"  {'-'*50}")
    for c in sorted(cycles):
        c_results = [r for r in results if r["cycle"] == c]
        c_acc = sum(r["correct"] for r in c_results) / len(c_results) * 100
        c_conf = np.mean([r["confidence"] for r in c_results])
        c_unc = np.mean([r["uncertainty"] for r in c_results])
        c_corr = compute_confidence_correlation(c_results, window=len(c_results))
        print(f"  {c:>6} {len(c_results):>7} {c_acc:>8.1f}% {c_conf:>7.1f}% {c_unc:>6.1f}% {c_corr:>8.4f}")

    # ================================================================
    # FINAL LIVE EXPECTATION
    # ================================================================
    late_cycles = [r for r in results if r["cycle"] >= result["cycles"] - 1]
    if late_cycles:
        late_acc = sum(r["correct"] for r in late_cycles) / len(late_cycles) * 100
        print(f"\n  LIVE EXPECTATION (last cycle): {late_acc:.1f}%")
        print(f"  (This is the most realistic estimate for forward performance)")

    save_charts(results, equity)
    print(f"{'='*75}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--chunk-ratio", type=float, default=0.1)
    parser.add_argument("--rolling-window", type=int, default=0,
                        help="Rolling window size in bars. 0=expanding (default). "
                             "3500≈12 months of M5 data.")
    args = parser.parse_args()
    result = run_walk_forward(args.symbol, args.train_ratio, args.chunk_ratio,
                              args.rolling_window)
    print_report(result)


if __name__ == "__main__":
    main()
