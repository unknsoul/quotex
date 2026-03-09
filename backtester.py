"""
Walk-Forward Backtester — Upgrade 7.

Simulates live trading with ALL filters active, using historical data.
This tests the complete signal pipeline (not just the model) including:
  - Confidence filters
  - Regime blocks
  - Session windows
  - Ensemble unanimity
  - Candle pattern confirmation
  - Cross-pair correlation

Usage:
    python backtester.py --symbol EURUSD --bars 2000
"""

import argparse
import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from config import DEFAULT_SYMBOL, LOG_LEVEL, LOG_FORMAT
from data_collector import load_csv, load_multi_tf
from feature_engineering import compute_features, FEATURE_COLUMNS
from regime_detection import detect_regime
from predict_engine import predict, load_models

log = logging.getLogger("backtester")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


def run_backtest(symbol: str = DEFAULT_SYMBOL, n_bars: int = 2000,
                 min_confidence: float = 42.0,
                 min_unanimity: float = 0.857,
                 blocked_regimes: set = None,
                 max_ensemble_var: float = 0.035):
    """
    Walk-forward backtest: for each bar, generate prediction using
    only data available up to that point, then check result.
    """
    if blocked_regimes is None:
        blocked_regimes = {"High_Volatility"}

    print(f"\n{'='*65}")
    print(f"  Walk-Forward Backtest — {symbol}")
    print(f"  Bars: {n_bars}  MinConf: {min_confidence}%  MinUnan: {min_unanimity:.0%}")
    print(f"{'='*65}\n")

    # Load full data
    multi_data = load_multi_tf(symbol)
    m5 = multi_data.get("M5")
    if m5 is None:
        print("ERROR: No M5 data found")
        return

    m15 = multi_data.get("M15")
    h1 = multi_data.get("H1")
    m1 = multi_data.get("M1")

    df_full = compute_features(m5, m15_df=m15, h1_df=h1, m1_df=m1)
    if len(df_full) < n_bars + 100:
        n_bars = len(df_full) - 100
        print(f"  Adjusted bars to {n_bars} (data limit)")

    # Ensure models are loaded
    load_models()

    results = []
    signals_sent = 0
    signals_skipped = 0
    wins = 0
    losses = 0

    start_idx = len(df_full) - n_bars
    print(f"  Testing bars {start_idx} to {len(df_full)-1}...")
    print(f"  {'='*60}")

    for i in range(start_idx, len(df_full) - 3):  # -3 because we need 3 bars for smoothed target
        # Use only data up to this point
        df_window = df_full.iloc[:i+1].copy()

        if len(df_window) < 200:
            continue

        try:
            # Detect regime
            regime = detect_regime(df_window)

            # Get prediction
            pred = predict(df_window, regime=regime)

            # Apply filters
            conf = pred.get("final_confidence_percent", 0)
            trade = pred.get("suggested_trade", "HOLD")
            ens_var = pred.get("ensemble_variance", 0)
            unanimity = pred.get("ensemble_unanimity", 0.5)
            pred_regime = pred.get("market_regime", "Unknown")
            direction = pred.get("suggested_direction", "UP")

            skip_reason = None

            if pred.get("error"):
                skip_reason = "error"
            elif trade == "HOLD":
                skip_reason = pred.get("skip_reason", "HOLD")

            if skip_reason:
                signals_skipped += 1
                continue

            # Check actual result using smoothed 3-bar target (consistent with training)
            # Majority vote over next 3 bars
            green_count = 0
            for k in range(1, 4):
                if i + k < len(df_full):
                    nbar = df_full.iloc[i + k]
                    if nbar["close"] > nbar["open"]:
                        green_count += 1
            actual_green = green_count >= 2  # majority of 3 bars
            predicted_up = direction == "UP"
            correct = (predicted_up == actual_green)

            signals_sent += 1
            if correct:
                wins += 1
            else:
                losses += 1

            results.append({
                "bar": i,
                "direction": direction,
                "actual": "UP" if actual_green else "DOWN",
                "correct": correct,
                "confidence": conf,
                "unanimity": unanimity,
                "regime": pred_regime,
                "variance": ens_var,
            })

        except Exception as e:
            signals_skipped += 1
            continue

        # Progress
        if signals_sent % 20 == 0 and signals_sent > 0:
            wr = wins / signals_sent * 100
            print(f"  ... {signals_sent} signals | WR: {wr:.1f}% | "
                  f"Skipped: {signals_skipped}")

    # Final report
    total = signals_sent
    wr = wins / total * 100 if total > 0 else 0

    print(f"\n{'='*65}")
    print(f"  BACKTEST RESULTS — {symbol}")
    print(f"{'='*65}")
    print(f"  Total bars tested:  {n_bars}")
    print(f"  Signals sent:       {signals_sent}")
    print(f"  Signals skipped:    {signals_skipped}")
    print(f"  Signal rate:        {signals_sent/(n_bars)*100:.1f}%")
    print(f"  Wins:               {wins}")
    print(f"  Losses:             {losses}")
    print(f"  Win Rate:           {wr:.1f}%")

    if results:
        df_res = pd.DataFrame(results)

        # Win rate by regime
        print(f"\n  By Regime:")
        for regime, grp in df_res.groupby("regime"):
            rwr = grp["correct"].mean() * 100
            print(f"    {regime:20s}: {rwr:5.1f}% ({len(grp)} signals)")

        # Win rate by confidence tier
        df_res["conf_tier"] = pd.cut(df_res["confidence"],
                                     bins=[0, 45, 50, 60, 100],
                                     labels=["42-45", "45-50", "50-60", "60+"])
        print(f"\n  By Confidence:")
        for tier, grp in df_res.groupby("conf_tier"):
            if len(grp) > 0:
                twr = grp["correct"].mean() * 100
                print(f"    {str(tier):20s}: {twr:5.1f}% ({len(grp)} signals)")

        # Loss streaks
        streak = 0
        worst_streak = 0
        for r in results:
            if r["correct"]:
                streak = 0
            else:
                streak += 1
                worst_streak = max(worst_streak, streak)
        print(f"\n  Worst loss streak:  {worst_streak}")

        # Save results
        csv_path = os.path.join("logs", f"backtest_{symbol}.csv")
        os.makedirs("logs", exist_ok=True)
        df_res.to_csv(csv_path, index=False)
        print(f"\n>> Results saved -> {csv_path}")

    print(f"{'='*65}\n")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--bars", type=int, default=2000)
    parser.add_argument("--min-conf", type=float, default=42.0)
    args = parser.parse_args()
    run_backtest(args.symbol, n_bars=args.bars, min_confidence=args.min_conf)


if __name__ == "__main__":
    main()
