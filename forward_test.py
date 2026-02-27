"""
Forward Test — 500-bar dry run with frozen model.

Simulates live deployment:
  - Loads pre-trained models (from train_model + meta_model + weight_learner)
  - Predicts every new bar as it arrives (no retrain, no tweaks)
  - Tracks accuracy, Brier, Spearman, rolling 100-bar mean, max drawdown streak

Usage:
  python forward_test.py --symbol EURUSD --bars 500

Rules:
  1. No retraining during the test
  2. No parameter changes
  3. No feature additions
  4. Results determine if system is deployable

Pass criteria (from user spec):
  - Forward accuracy > 60%
  - Spearman > 0.15
  - No extended sub-50% streaks (rolling 100-bar never stays below 50% for >50 bars)
"""

import argparse
import sys
import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss

from config import (
    DEFAULT_SYMBOL,
    LOG_LEVEL, LOG_FORMAT,
)
from data_collector import load_csv, load_multi_tf
from feature_engineering import compute_features, add_target, FEATURE_COLUMNS
from regime_detection import detect_regime
from predict_engine import load_models, predict, update_prediction_history

log = logging.getLogger("forward_test")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


def run_forward_test(symbol, n_bars=500, holdout_ratio=0.1):
    """
    Run forward test on the last `n_bars` of available data.
    Uses pre-trained models WITHOUT retraining.
    """

    # Load data
    mtf = load_multi_tf(symbol)
    df = mtf.get("M5")
    if df is None:
        df = load_csv(symbol, "M5")
    m15, h1 = mtf.get("M15"), mtf.get("H1")

    df = compute_features(df, m15_df=m15, h1_df=h1)
    df = add_target(df)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)

    total = len(df)
    test_start = total - n_bars
    if test_start < 200:
        print(f"ERROR: Not enough data. Need at least {n_bars + 200} rows, have {total}.")
        sys.exit(1)

    # Load frozen models
    ensemble, feat_cols, meta_model, meta_feat_cols, weight_model = load_models()
    print(f"\n>> Forward Test for {symbol}")
    print(f"   Total bars: {total}, Test: last {n_bars} bars [{test_start}:{total}]")
    print(f"   Models: FROZEN (no retrain)")
    print(f"   Ensemble: {len(ensemble)} models, Meta features: {len(meta_feat_cols)}")
    print(f"   Pass criteria: Accuracy>60%, Spearman>0.15, no sustained sub-50% streaks\n")

    results = []
    equity = [0]
    cum_pnl = 0

    for i in range(test_start, total):
        # Use all data up to and including current bar (lookback for features)
        window = df.iloc[max(0, i - 200):i + 1]
        if len(window) < 50:
            continue

        regime = detect_regime(window)
        row_df = df.iloc[max(0, i - 200):i + 1].copy()

        pred = predict(row_df, regime)

        actual = df["target"].iloc[i]
        predicted_dir = 1 if pred["primary_direction"] == "GREEN" else 0
        correct = 1 if predicted_dir == actual else 0

        green_p = pred["green_probability_percent"] / 100.0
        confidence = pred["final_confidence_percent"]

        update_prediction_history(green_p, correct, confidence)

        cum_pnl += (1 if correct else -1)
        equity.append(cum_pnl)

        results.append({
            "bar": i,
            "green_p": green_p,
            "actual": actual,
            "predicted": predicted_dir,
            "correct": correct,
            "confidence": confidence,
            "regime": regime,
            "meta_rel": pred["meta_reliability_percent"],
            "uncertainty": pred["uncertainty_percent"],
        })

        # Progress
        if len(results) % 100 == 0:
            running_acc = sum(r["correct"] for r in results) / len(results) * 100
            print(f"  Bar {len(results)}/{n_bars}: running accuracy = {running_acc:.1f}%")

    # =========================================================================
    # Analysis
    # =========================================================================
    total_preds = len(results)
    correct_total = sum(r["correct"] for r in results)
    accuracy = correct_total / total_preds * 100

    green_ps = np.array([r["green_p"] for r in results])
    actuals_correct = np.array([r["correct"] for r in results])
    directions = (green_ps >= 0.5).astype(int)
    targets = np.where(actuals_correct == 1, directions, 1 - directions)

    brier = brier_score_loss(targets, green_ps)
    brier_naive = 0.25
    brier_skill = 1 - (brier / brier_naive)

    try:
        ll = log_loss(targets, np.clip(green_ps, 1e-10, 1 - 1e-10))
    except Exception:
        ll = 0.693

    confs = [r["confidence"] for r in results]
    spearman, _ = stats.spearmanr(confs, actuals_correct)
    spearman = float(spearman) if not np.isnan(spearman) else 0.0

    # Rolling 100-bar
    correct_series = pd.Series([r["correct"] for r in results])
    rolling = correct_series.rolling(100, min_periods=100).mean()
    rolling_clean = rolling.dropna()

    # Max consecutive losses
    max_loss_streak = 0
    current_streak = 0
    for r in results:
        if r["correct"] == 0:
            current_streak += 1
            max_loss_streak = max(max_loss_streak, current_streak)
        else:
            current_streak = 0

    # Sub-50% sustained check
    sub50_sustained = 0
    if len(rolling_clean) > 0:
        below_50 = (rolling_clean < 0.50).astype(int)
        max_below_50 = 0
        current_below = 0
        for v in below_50.values:
            if v:
                current_below += 1
                max_below_50 = max(max_below_50, current_below)
            else:
                current_below = 0
        sub50_sustained = max_below_50

    # Regime breakdown
    regime_stats = {}
    for r in results:
        reg = r["regime"]
        if reg not in regime_stats:
            regime_stats[reg] = {"t": 0, "c": 0}
        regime_stats[reg]["t"] += 1
        regime_stats[reg]["c"] += r["correct"]

    # =========================================================================
    # Report
    # =========================================================================
    print(f"\n{'='*65}")
    print(f"  FORWARD TEST RESULTS (FROZEN MODEL)")
    print(f"  Symbol: {symbol} | Bars: {total_preds}")
    print(f"{'='*65}")

    print(f"\n  CORE METRICS:")
    print(f"    Accuracy:          {accuracy:.1f}%")
    print(f"    Brier Score:       {brier:.4f}  (naive: 0.2500)")
    print(f"    Brier Skill:       {brier_skill:.4f}")
    print(f"    Log Loss:          {ll:.4f}  (naive: 0.6931)")
    sp_status = "✅" if spearman >= 0.15 else "❌"
    print(f"    Spearman:          {spearman:.4f} {sp_status}")

    print(f"\n  STABILITY:")
    if len(rolling_clean) > 0:
        print(f"    Rolling 100-bar:   {rolling_clean.min()*100:.1f}% — {rolling_clean.max()*100:.1f}%  "
              f"(mean={rolling_clean.mean()*100:.1f}%, std={rolling_clean.std()*100:.1f}%)")
    print(f"    Max loss streak:   {max_loss_streak}")
    print(f"    Max sub-50% run:   {sub50_sustained} bars")

    print(f"\n  REGIME:")
    print(f"  {'Regime':<18} {'Total':>7} {'Acc':>8}")
    print(f"  {'-'*35}")
    for reg in sorted(regime_stats.keys()):
        s = regime_stats[reg]
        print(f"  {reg:<18} {s['t']:>7} {s['c']/s['t']*100 if s['t'] else 0:>7.1f}%")

    # Confidence gated
    print(f"\n  CONFIDENCE GATED:")
    for thr in [0, 50, 60, 70, 80]:
        gated = [r for r in results if r["confidence"] >= thr]
        if gated:
            g_acc = sum(r["correct"] for r in gated) / len(gated) * 100
            freq = len(gated) / total_preds * 100
            label = f">={thr}%" if thr > 0 else "All"
            print(f"    {label:<8} {len(gated):>5} bars ({freq:.0f}%) -> {g_acc:.1f}%")

    print(f"\n  EQUITY: final={equity[-1]}, peak={max(equity)}, trough={min(equity)}")

    # =========================================================================
    # Pass/Fail
    # =========================================================================
    print(f"\n  {'='*50}")
    passes = 0
    total_checks = 3

    if accuracy > 60:
        print(f"  ✅ PASS: Accuracy {accuracy:.1f}% > 60%")
        passes += 1
    else:
        print(f"  ❌ FAIL: Accuracy {accuracy:.1f}% <= 60%")

    if spearman > 0.15:
        print(f"  ✅ PASS: Spearman {spearman:.4f} > 0.15")
        passes += 1
    else:
        print(f"  ❌ FAIL: Spearman {spearman:.4f} <= 0.15")

    if sub50_sustained <= 50:
        print(f"  ✅ PASS: No sustained sub-50% streak (max={sub50_sustained})")
        passes += 1
    else:
        print(f"  ❌ FAIL: Sustained sub-50% streak of {sub50_sustained} bars")

    verdict = "DEPLOYABLE" if passes == total_checks else "NOT READY"
    print(f"\n  VERDICT: {verdict} ({passes}/{total_checks} checks passed)")
    print(f"{'='*65}")


def main():
    parser = argparse.ArgumentParser(description="Forward test with frozen model")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--bars", type=int, default=500,
                        help="Number of bars to test (default: 500)")
    args = parser.parse_args()
    run_forward_test(args.symbol, args.bars)


if __name__ == "__main__":
    main()
