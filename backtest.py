"""
Backtest v4 — Walk-forward with ensemble retraining, uncertainty tracking,
cosine drift detection, and confidence bin analysis.

Usage:
    python backtest.py --symbol EURUSD
"""

import argparse
import os
import logging

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from config import (
    DEFAULT_SYMBOL, CHART_DIR,
    META_ROLLING_WINDOW, WIN_STREAK_CAP, ATR_PERCENTILE_WINDOW,
    CONFIDENCE_HIGH_MIN, CONFIDENCE_MEDIUM_MIN,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_SUBSAMPLE, XGB_COLSAMPLE_BYTREE, ENSEMBLE_SEEDS,
    META_N_ESTIMATORS, META_MAX_DEPTH, META_LEARNING_RATE, META_SUBSAMPLE,
    DRIFT_COSINE_THRESHOLD,
    LOG_LEVEL, LOG_FORMAT,
)
from data_collector import load_csv, load_multi_tf
from feature_engineering import compute_features, add_target, FEATURE_COLUMNS
from regime_detection import detect_regime, get_regime_thresholds, REGIMES
from stability import CosineDriftDetector

log = logging.getLogger("backtest")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

REGIME_ENCODING = {"Trending": 0, "Ranging": 1, "High_Volatility": 2, "Low_Volatility": 3}

META_FEATURE_COLUMNS = [
    "primary_green_prob", "prob_distance_from_half", "primary_entropy",
    "regime_encoded", "atr_value", "spread_ratio", "volatility_zscore",
    "range_position", "recent_model_accuracy", "recent_win_streak",
    "body_percentile_rank", "direction_streak", "rolling_vol_percentile",
]

WEIGHT_FEATURES = ["primary_strength", "meta_reliability", "regime_strength", "uncertainty"]


def _binary_entropy(p):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def _build_primary(spw, seed):
    return xgb.XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE, subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE, scale_pos_weight=spw,
        objective="binary:logistic", eval_metric="logloss",
        use_label_encoder=False, random_state=seed, verbosity=0,
    )


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


def _build_meta_row(green_p, regime, row, history, dir_history):
    regime_enc = REGIME_ENCODING.get(regime, 1)
    if history:
        recent = history[-META_ROLLING_WINDOW:]
        recent_acc = sum(recent) / len(recent)
        streak = 0
        for v in reversed(recent):
            if v == 1:
                streak += 1
            else:
                break
        streak = min(streak, WIN_STREAK_CAP)
    else:
        recent_acc, streak = 0.5, 0

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
        "regime_encoded": regime_enc,
        "atr_value": float(row.get("atr_14", 0)),
        "spread_ratio": 0.0,
        "volatility_zscore": float(row.get("volatility_zscore", 0)),
        "range_position": float(row.get("range_position", 0.5)),
        "recent_model_accuracy": recent_acc,
        "recent_win_streak": streak,
        "body_percentile_rank": float(row.get("body_size", 0.5)),
        "direction_streak": dstreak,
        "rolling_vol_percentile": float(row.get("atr_percentile_rank", 0.5)),
    }


def run_walk_forward(symbol, train_ratio=0.6, chunk_ratio=0.1):
    mtf = load_multi_tf(symbol)
    df = mtf.get("M5")
    if df is None:
        df = load_csv(symbol, "M5")
    m15, h1 = mtf.get("M15"), mtf.get("H1")

    df = compute_features(df, m15_df=m15, h1_df=h1)
    df = add_target(df)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)

    X, y = df[FEATURE_COLUMNS], df["target"].values
    n = len(df)
    train_end = int(n * train_ratio)
    chunk = int(n * chunk_ratio)

    print(f"\n>> Walk-Forward v4 for {symbol}")
    print(f"   Bars: {n}, Train: {train_end}, Chunk: {chunk}, Ensemble: {len(ENSEMBLE_SEEDS)} seeds")

    all_results = []
    meta_history = []
    dir_history = []
    equity = []
    cum_pnl = 0.0
    cycle = 0
    drift_detector = CosineDriftDetector(DRIFT_COSINE_THRESHOLD)
    drift_reports = []

    while train_end < n:
        cycle += 1
        test_start = train_end
        test_end = min(train_end + chunk, n)
        X_tr, y_tr = X.iloc[:train_end], y[:train_end]
        spw = np.sum(y_tr == 0) / max(np.sum(y_tr == 1), 1)

        print(f"\n  Cycle {cycle}: train[0:{train_end}] test[{test_start}:{test_end}]")

        # Train ensemble
        ensemble = []
        for seed in ENSEMBLE_SEEDS:
            base = _build_primary(spw, seed)
            cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
            cal.fit(X_tr, y_tr)
            ensemble.append(cal)

        # Drift detection
        try:
            base_est = ensemble[0].calibrated_classifiers_[0].estimator
            imp = base_est.feature_importances_
            cosine_sim, drifted, msg = drift_detector.check_drift(imp)
            drift_reports.append({"cycle": cycle, "cosine": cosine_sim, "drifted": drifted})
            drift_detector.update_baseline(imp)
            if drifted:
                print(f"    DRIFT: {msg}")
        except Exception:
            pass

        # Train meta on training data
        tr_probs = np.array([m.predict_proba(X_tr)[:, 1] for m in ensemble])
        tr_mean = tr_probs.mean(axis=0)
        tr_dir = (tr_mean >= 0.5).astype(int)
        tr_correct = (tr_dir == y_tr).astype(int)

        meta_rows = []
        mh_local, dh_local = [], []
        for j in range(len(X_tr)):
            regime = _regime_for_row(df, j) if j > 50 else "Ranging"
            mr = _build_meta_row(tr_mean[j], regime, df.iloc[j], mh_local, dh_local)
            mr["meta_target"] = tr_correct[j]
            meta_rows.append(mr)
            mh_local.append(tr_correct[j])
            dh_local.append(1 if tr_mean[j] >= 0.5 else 0)

        meta_df = pd.DataFrame(meta_rows)
        X_meta_tr = meta_df[META_FEATURE_COLUMNS].fillna(0)
        y_meta_tr = meta_df["meta_target"].values
        meta_model = _build_meta()
        meta_model.fit(X_meta_tr, y_meta_tr)

        # Train weight model
        tr_var = tr_probs.var(axis=0)
        tr_var_norm = tr_var / (tr_var.max() + 1e-10)
        tr_strength = np.abs(tr_mean - 0.5) * 2

        meta_input_tr = meta_df[META_FEATURE_COLUMNS].fillna(0)
        meta_rel_tr = meta_model.predict_proba(meta_input_tr.values)[:, 1]
        regime_str_tr = df["adx_normalized"].iloc[:train_end].values if "adx_normalized" in df.columns else np.zeros(train_end)

        W_tr = pd.DataFrame({
            "primary_strength": tr_strength,
            "meta_reliability": meta_rel_tr,
            "regime_strength": regime_str_tr,
            "uncertainty": tr_var_norm,
        })
        weight_model = LogisticRegression(random_state=42, max_iter=500)
        weight_model.fit(W_tr, tr_correct)

        train_acc = accuracy_score(y_tr, tr_dir)
        print(f"    Train acc: {train_acc:.1%}")

        # Test
        cycle_correct = 0
        for i in range(test_start, test_end):
            row_feat = X.iloc[i].values.reshape(1, -1)
            all_p = np.array([m.predict_proba(row_feat)[0][1] for m in ensemble])
            green_p = float(all_p.mean())
            variance = float(all_p.var())
            norm_var = min(variance / 0.25, 1.0)
            actual = y[i]
            regime = _regime_for_row(df, i)

            meta_row = _build_meta_row(green_p, regime, df.iloc[i], meta_history, dir_history)
            meta_in = pd.DataFrame([meta_row])[META_FEATURE_COLUMNS]
            meta_rel = float(meta_model.predict_proba(meta_in.values)[0][1])

            primary_str = abs(green_p - 0.5) * 2
            regime_str = float(df.iloc[i].get("adx_normalized", 0.25))
            w_in = pd.DataFrame([{
                "primary_strength": primary_str,
                "meta_reliability": meta_rel,
                "regime_strength": regime_str,
                "uncertainty": norm_var,
            }])
            confidence = float(weight_model.predict_proba(w_in)[:, 1][0]) * 100

            direction = 1 if green_p >= 0.5 else 0
            correct = 1 if direction == actual else 0
            cycle_correct += correct
            cum_pnl += (1 if correct else -1)
            equity.append(cum_pnl)
            meta_history.append(correct)
            dir_history.append(direction)

            all_results.append({
                "bar": i, "green_p": green_p, "meta_rel": meta_rel,
                "confidence": confidence, "uncertainty": norm_var * 100,
                "correct": correct, "regime": regime, "cycle": cycle,
            })

        test_acc = cycle_correct / max(test_end - test_start, 1)
        print(f"    Test acc:  {test_acc:.1%} ({cycle_correct}/{test_end - test_start})")
        train_end = test_end

    return {"results": all_results, "equity": equity, "cycles": cycle, "drift": drift_reports}


# =============================================================================
#  Analysis
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


def stability_warnings(results, window=100):
    warnings = []
    corrects = [r["correct"] for r in results]
    if len(corrects) >= window:
        rolling = pd.Series(corrects).rolling(window).mean()
        if rolling.dropna().min() < 0.45:
            warnings.append(f"ACCURACY COLLAPSE: min rolling {window}-bar = {rolling.dropna().min():.1%}")

    high = [r for r in results if r["confidence"] >= CONFIDENCE_HIGH_MIN]
    if len(high) > 50:
        h_acc = sum(r["correct"] for r in high) / len(high)
        o_acc = sum(corrects) / len(corrects)
        if h_acc < o_acc:
            warnings.append(f"OVERCONFIDENCE: high-conf {h_acc:.1%} < overall {o_acc:.1%}")

    return warnings


def save_charts(results, equity):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib unavailable)")
        return

    os.makedirs(CHART_DIR, exist_ok=True)

    # Equity
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(equity, linewidth=0.8, color="#2196F3")
    ax.set_title("Walk-Forward Equity Curve (v4 Ensemble)", fontsize=13)
    ax.set_xlabel("Prediction #"); ax.set_ylabel("Cumulative PnL")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "equity_curve.png"), dpi=120)
    plt.close(fig)

    # Rolling accuracy
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

    # Confidence vs Accuracy
    labels, accs, counts = confidence_bin_analysis(results)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax1.bar(x, [a * 100 for a in accs], color="#FF9800", alpha=0.7, label="Accuracy %")
    ax1.axhline(y=50, color="red", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Confidence Bin"); ax1.set_ylabel("Accuracy %")
    ax1.set_title("Confidence vs Actual Accuracy (v4)", fontsize=13)
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
    print(f"  WALK-FORWARD BACKTEST v4 (Ensemble + Uncertainty)")
    print(f"  Cycles: {result['cycles']} | Predictions: {total}")
    print(f"{'='*75}")

    # Per-cycle
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

    # Per-regime
    regime_stats = {r: {"t": 0, "c": 0} for r in REGIMES}
    for r in results:
        regime_stats[r["regime"]]["t"] += 1
        regime_stats[r["regime"]]["c"] += r["correct"]
    print(f"\n  {'Regime':<18} {'Total':>7} {'Acc':>8}")
    print(f"  {'-'*35}")
    for reg in REGIMES:
        s = regime_stats[reg]
        print(f"  {reg:<18} {s['t']:>7} {s['c']/s['t']*100 if s['t'] else 0:>7.1f}%")

    # Confidence bins
    labels, accs, counts = confidence_bin_analysis(results)
    print(f"\n  CONFIDENCE BINS:")
    print(f"  {'Bin':<12} {'Count':>7} {'Accuracy':>9}")
    print(f"  {'-'*30}")
    for l, a, c in zip(labels, accs, counts):
        if c > 0:
            print(f"  {l:<12} {c:>7} {a*100:>8.1f}%")

    high = [r for r in results if r["confidence"] >= CONFIDENCE_HIGH_MIN]
    if high:
        h_acc = sum(r["correct"] for r in high) / len(high) * 100
        print(f"\n  High-conf ({CONFIDENCE_HIGH_MIN}%+): {len(high)} bars, {h_acc:.1f}% (+{h_acc-overall_acc:.1f}pp)")

    # Uncertainty
    unc = [r["uncertainty"] for r in results]
    print(f"\n  Uncertainty: mean={np.mean(unc):.1f}% median={np.median(unc):.1f}% max={max(unc):.1f}%")

    # Drift
    if result["drift"]:
        print(f"\n  DRIFT DETECTION:")
        for d in result["drift"]:
            flag = " !! DRIFT" if d["drifted"] else ""
            print(f"    Cycle {d['cycle']}: cosine={d['cosine']:.4f}{flag}")

    # Equity
    if equity:
        print(f"\n  Equity: final={equity[-1]:.0f} peak={max(equity):.0f}")

    # Warnings
    warnings = stability_warnings(results)
    if warnings:
        print(f"\n  WARNINGS:")
        for w in warnings:
            print(f"    !! {w}")
    else:
        print(f"\n  No stability warnings.")

    save_charts(results, equity)
    print(f"{'='*75}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--chunk-ratio", type=float, default=0.1)
    args = parser.parse_args()
    result = run_walk_forward(args.symbol, args.train_ratio, args.chunk_ratio)
    print_report(result)


if __name__ == "__main__":
    main()
