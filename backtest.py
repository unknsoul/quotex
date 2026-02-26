"""
Backtest v3.1 — Advanced Walk-Forward Stability Backtesting Engine.

Walk-forward with expanding window retraining:
  - Initial train: 60% of data
  - Test chunk: 10% of data
  - Retrain with expanding window each cycle

Predicts EVERY candle. Outputs:
  - Confidence bin accuracy analysis
  - Rolling accuracy chart
  - Equity curve
  - Stability warnings

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
from sklearn.metrics import accuracy_score

from config import (
    DEFAULT_SYMBOL, MODEL_DIR,
    META_ROLLING_WINDOW, WIN_STREAK_CAP,
    CONFIDENCE_HIGH_MIN, CONFIDENCE_MEDIUM_MIN,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_SUBSAMPLE, XGB_COLSAMPLE_BYTREE,
    LGBM_N_ESTIMATORS, LGBM_MAX_DEPTH, LGBM_LEARNING_RATE,
    LGBM_SUBSAMPLE,
    LOG_LEVEL, LOG_FORMAT,
)
from data_collector import load_csv, load_multi_tf
from feature_engineering import compute_features, add_target, FEATURE_COLUMNS
from regime_detection import detect_regime, get_regime_thresholds, REGIMES

log = logging.getLogger("backtest")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

REGIME_ENCODING = {"Trending": 0, "Ranging": 1, "High_Volatility": 2, "Low_Volatility": 3}

META_FEATURE_COLUMNS = [
    "primary_green_prob", "prob_distance_from_half", "regime_encoded",
    "atr_value", "spread_ratio", "volatility_zscore",
    "range_position", "recent_model_accuracy", "recent_win_streak",
]

# --- Chart output dir --------------------------------------------------------
CHART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")


# =============================================================================
#  Model Builders
# =============================================================================

def _build_primary(spw):
    return xgb.XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE, subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE, scale_pos_weight=spw,
        objective="binary:logistic", eval_metric="logloss",
        use_label_encoder=False, random_state=42, verbosity=0,
    )


def _build_meta():
    return GradientBoostingClassifier(
        n_estimators=LGBM_N_ESTIMATORS, max_depth=LGBM_MAX_DEPTH,
        learning_rate=LGBM_LEARNING_RATE, subsample=LGBM_SUBSAMPLE,
        random_state=42,
    )


def _regime_for_row(df, idx, lookback=200):
    start = max(0, idx - lookback + 1)
    window = df.iloc[start:idx + 1]
    if len(window) < 50:
        return "Ranging"
    return detect_regime(window)


def _build_meta_input(green_p, regime, row, history):
    regime_enc = REGIME_ENCODING.get(regime, 1)
    if len(history) > 0:
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
        recent_acc = 0.5
        streak = 0

    return {
        "primary_green_prob": green_p,
        "prob_distance_from_half": abs(green_p - 0.5),
        "regime_encoded": regime_enc,
        "atr_value": float(row.get("atr_14", 0)),
        "spread_ratio": 0.0,
        "volatility_zscore": float(row.get("volatility_zscore", 0)),
        "range_position": float(row.get("range_position", 0.5)),
        "recent_model_accuracy": recent_acc,
        "recent_win_streak": streak,
    }


# =============================================================================
#  Walk-Forward Engine
# =============================================================================

def run_walk_forward(symbol, initial_train_ratio=0.6, test_chunk_ratio=0.1):
    """
    Walk-forward backtest with expanding window retraining.
    """
    # -- Load data --
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

    X = df[FEATURE_COLUMNS]
    y = df["target"].values
    n = len(df)

    initial_train_end = int(n * initial_train_ratio)
    chunk_size = int(n * test_chunk_ratio)

    print(f"\n>> Walk-Forward Backtest for {symbol}")
    print(f"   Total bars: {n}")
    print(f"   Initial train: {initial_train_end} ({initial_train_ratio:.0%})")
    print(f"   Test chunk: {chunk_size} ({test_chunk_ratio:.0%})")

    # -- Accumulators --
    all_results = []         # per-bar: {green_p, meta_rel, confidence, correct, regime}
    meta_history = []        # for rolling accuracy in meta features
    equity = []
    cum_pnl = 0.0

    cycle = 0
    train_end = initial_train_end

    while train_end < n:
        cycle += 1
        test_start = train_end
        test_end = min(train_end + chunk_size, n)

        print(f"\n  Cycle {cycle}: train[0:{train_end}] test[{test_start}:{test_end}] "
              f"({test_end - test_start} bars)")

        # -- Train primary on [0:train_end] --
        X_tr, y_tr = X.iloc[:train_end], y[:train_end]
        spw = np.sum(y_tr == 0) / max(np.sum(y_tr == 1), 1)

        base = _build_primary(spw)
        calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
        calibrated.fit(X_tr, y_tr)

        # -- Generate OOF-like predictions on train set for meta training --
        train_proba = calibrated.predict_proba(X_tr)[:, 1]
        train_direction = (train_proba >= 0.5).astype(int)
        train_correct = (train_direction == y_tr).astype(int)

        # Build meta training data
        meta_rows = []
        meta_hist_local = []
        for j in range(len(X_tr)):
            regime = _regime_for_row(df, j) if j > 50 else "Ranging"
            mr = _build_meta_input(train_proba[j], regime, df.iloc[j], meta_hist_local)
            mr["meta_target"] = train_correct[j]
            meta_rows.append(mr)
            meta_hist_local.append(train_correct[j])

        meta_train_df = pd.DataFrame(meta_rows)
        X_meta_tr = meta_train_df[META_FEATURE_COLUMNS].fillna(0)
        y_meta_tr = meta_train_df["meta_target"].values

        meta_model = _build_meta()
        meta_model.fit(X_meta_tr, y_meta_tr)

        train_acc = accuracy_score(y_tr, train_direction)
        print(f"    Train accuracy: {train_acc:.1%}")

        # -- Predict test chunk (EVERY bar) --
        cycle_correct = 0
        for i in range(test_start, test_end):
            row_feat = X.iloc[i].values.reshape(1, -1)
            proba = calibrated.predict_proba(row_feat)[0]
            green_p = float(proba[1])
            actual = y[i]

            regime = _regime_for_row(df, i)

            # Meta
            meta_input = _build_meta_input(green_p, regime, df.iloc[i], meta_history)
            meta_df = pd.DataFrame([meta_input])[META_FEATURE_COLUMNS]
            meta_proba = meta_model.predict_proba(meta_df.values)[0]
            meta_rel = float(meta_proba[1])

            # Confidence
            primary_strength = abs(green_p - 0.5) * 2
            confidence = primary_strength * meta_rel * 100

            # Correct?
            direction = 1 if green_p >= 0.5 else 0
            correct = 1 if direction == actual else 0
            cycle_correct += correct

            # PnL: +1 correct, -1 wrong
            cum_pnl += (1 if correct else -1)
            equity.append(cum_pnl)

            meta_history.append(correct)

            all_results.append({
                "bar": i,
                "green_p": green_p,
                "meta_rel": meta_rel,
                "confidence": confidence,
                "correct": correct,
                "regime": regime,
                "cycle": cycle,
            })

        test_acc = cycle_correct / (test_end - test_start)
        print(f"    Test accuracy:  {test_acc:.1%} ({cycle_correct}/{test_end - test_start})")

        # Expanding window
        train_end = test_end

    return {
        "results": all_results,
        "equity": equity,
        "total_bars": n,
        "cycles": cycle,
    }


# =============================================================================
#  Analysis & Reporting
# =============================================================================

def confidence_bin_analysis(results):
    """Bin predictions by confidence and compute accuracy per bin."""
    bins = np.arange(0, 101, 10)
    bin_labels = [f"{b}-{b+10}%" for b in bins[:-1]]
    bin_acc = []
    bin_count = []

    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        in_bin = [r for r in results if lo <= r["confidence"] < hi]
        if in_bin:
            acc = sum(r["correct"] for r in in_bin) / len(in_bin)
            bin_acc.append(acc)
            bin_count.append(len(in_bin))
        else:
            bin_acc.append(0)
            bin_count.append(0)

    return bin_labels, bin_acc, bin_count


def stability_warnings(results, window=100):
    """Check for rolling accuracy collapse, confidence inflation."""
    warnings = []

    # Rolling accuracy
    corrects = [r["correct"] for r in results]
    if len(corrects) >= window:
        rolling = pd.Series(corrects).rolling(window).mean()
        min_acc = rolling.dropna().min()
        if min_acc < 0.45:
            warnings.append(f"ACCURACY COLLAPSE: rolling {window}-bar min={min_acc:.1%}")

    # Confidence inflation: check if high confidence has LOW accuracy
    high_conf = [r for r in results if r["confidence"] >= CONFIDENCE_HIGH_MIN]
    if len(high_conf) > 50:
        high_acc = sum(r["correct"] for r in high_conf) / len(high_conf)
        overall_acc = sum(corrects) / len(corrects)
        if high_acc < overall_acc:
            warnings.append(f"OVERCONFIDENCE: high-conf accuracy {high_acc:.1%} < overall {overall_acc:.1%}")

    # Per-cycle accuracy trend
    cycles = {}
    for r in results:
        c = r["cycle"]
        if c not in cycles:
            cycles[c] = []
        cycles[c].append(r["correct"])

    if len(cycles) >= 3:
        cycle_accs = [sum(v) / len(v) for v in cycles.values()]
        if cycle_accs[-1] < cycle_accs[0] - 0.05:
            warnings.append(f"DECLINING TREND: cycle 1 acc={cycle_accs[0]:.1%} -> final={cycle_accs[-1]:.1%}")

    return warnings


def save_charts(results, equity):
    """Save equity curve, rolling accuracy, and confidence-vs-accuracy charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available, skipping charts)")
        return

    os.makedirs(CHART_DIR, exist_ok=True)

    # 1) Equity curve
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(equity, linewidth=0.8, color="#2196F3")
    ax.set_title("Walk-Forward Equity Curve", fontsize=13)
    ax.set_xlabel("Prediction #")
    ax.set_ylabel("Cumulative PnL")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "equity_curve.png"), dpi=120)
    plt.close(fig)

    # 2) Rolling accuracy
    corrects = [r["correct"] for r in results]
    rolling = pd.Series(corrects).rolling(100).mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(rolling, linewidth=0.8, color="#4CAF50")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="50%")
    ax.set_title("Rolling Accuracy (100-bar window)", fontsize=13)
    ax.set_xlabel("Prediction #")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "rolling_accuracy.png"), dpi=120)
    plt.close(fig)

    # 3) Confidence vs Accuracy
    labels, accs, counts = confidence_bin_analysis(results)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax1.bar(x, [a * 100 for a in accs], color="#FF9800", alpha=0.7, label="Accuracy %")
    ax1.axhline(y=50, color="red", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Confidence Bin")
    ax1.set_ylabel("Accuracy %")
    ax1.set_title("Confidence vs Actual Accuracy", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45)

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
    overall_acc = correct / total * 100 if total else 0

    print(f"\n{'='*75}")
    print(f"  WALK-FORWARD BACKTEST REPORT")
    print(f"  Cycles: {result['cycles']}  |  Total predictions: {total}")
    print(f"{'='*75}")

    # Per-cycle summary
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
        acc = s["correct"] / s["total"] * 100
        print(f"  {c:>6} {s['total']:>7} {s['correct']:>8} {acc:>8.1f}%")
    print(f"  {'-'*35}")
    print(f"  {'ALL':>6} {total:>7} {correct:>8} {overall_acc:>8.1f}%")

    # Per-regime
    regime_stats = {r: {"total": 0, "correct": 0} for r in REGIMES}
    for r in results:
        regime_stats[r["regime"]]["total"] += 1
        regime_stats[r["regime"]]["correct"] += r["correct"]

    print(f"\n  {'Regime':<18} {'Total':>7} {'Correct':>8} {'Accuracy':>9}")
    print(f"  {'-'*45}")
    for regime in REGIMES:
        s = regime_stats[regime]
        acc = s["correct"] / s["total"] * 100 if s["total"] else 0
        print(f"  {regime:<18} {s['total']:>7} {s['correct']:>8} {acc:>8.1f}%")

    # Confidence bins
    labels, accs, counts = confidence_bin_analysis(results)
    print(f"\n  CONFIDENCE BIN ANALYSIS:")
    print(f"  {'Bin':<12} {'Count':>7} {'Accuracy':>9}")
    print(f"  {'-'*30}")
    for label, acc, count in zip(labels, accs, counts):
        if count > 0:
            print(f"  {label:<12} {count:>7} {acc*100:>8.1f}%")

    # High-conf vs overall
    high_conf = [r for r in results if r["confidence"] >= CONFIDENCE_HIGH_MIN]
    if high_conf:
        high_acc = sum(r["correct"] for r in high_conf) / len(high_conf) * 100
        print(f"\n  High confidence ({CONFIDENCE_HIGH_MIN}%+): {len(high_conf)} bars, {high_acc:.1f}% accuracy")
        print(f"  Overall:                     {total} bars, {overall_acc:.1f}% accuracy")
        delta = high_acc - overall_acc
        print(f"  Delta: {'+' if delta >= 0 else ''}{delta:.1f}pp")

    # Equity
    if equity:
        peak = max(equity)
        final = equity[-1]
        dd = peak - min(equity[equity.index(peak):]) if peak in equity else 0
        print(f"\n  Equity: final={final:.0f}  peak={peak:.0f}  max_drawdown={dd:.0f}")

    # Stability warnings
    warnings = stability_warnings(results)
    if warnings:
        print(f"\n  STABILITY WARNINGS:")
        for w in warnings:
            print(f"    !! {w}")
    else:
        print(f"\n  No stability warnings.")

    # Save charts
    save_charts(results, equity)

    print(f"\n{'='*75}")


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
