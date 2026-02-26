"""
Backtesting Engine — simulate predictions across historical data.

Walk-forward: for each bar, detect regime, apply threshold, record outcome.
Reports per-regime win rate, drawdown, profit factor, Sharpe ratio, equity curve.

Usage:
    python backtest.py --symbol EURUSD
    python backtest.py --symbol EURUSD --candles 8000
"""

import argparse
import os
import sys
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib

from config import (
    DEFAULT_SYMBOL, CANDLES_TO_FETCH,
    MODEL_PATH, FEATURE_LIST_PATH,
    LOG_LEVEL, LOG_FORMAT,
)
from data_collector import load_csv
from feature_engineering import compute_features, add_target, FEATURE_COLUMNS
from regime_detection import detect_regime, get_regime_threshold, REGIMES

log = logging.getLogger("backtest")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


def _regime_for_row(df: pd.DataFrame, idx: int, lookback: int = 200) -> str:
    """Detect regime using a trailing window ending at row `idx`."""
    start = max(0, idx - lookback + 1)
    window = df.iloc[start: idx + 1]
    if len(window) < 50:
        return "Ranging"
    return detect_regime(window)


def run_backtest(symbol: str) -> dict:
    """
    Walk-forward backtest.

    Returns:
        {
            "per_regime": {regime: {wins, losses, skips, pnl_list}},
            "equity_curve": [cumulative pnl],
            "total_bars": int,
        }
    """
    # ── Load data ────────────────────────────────────────────────────────
    df = load_csv(symbol)
    df = compute_features(df)
    df = add_target(df)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)

    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_LIST_PATH)

    # ── Accumulators ─────────────────────────────────────────────────────
    stats = {r: {"wins": 0, "losses": 0, "skips": 0, "pnl": []}
             for r in REGIMES}
    equity = []
    cumulative = 0.0

    start = 300  # skip warm-up
    total = len(df) - start
    print(f"\n>> Backtesting {total} bars for {symbol}...")

    for i in range(start, len(df)):
        regime = _regime_for_row(df, i)
        threshold = get_regime_threshold(regime)

        row = df[feature_cols].iloc[i].values.reshape(1, -1)
        proba = model.predict_proba(row)[0]
        green_p = float(proba[1])
        actual = df["target"].iloc[i]

        # Decision
        if green_p >= threshold:
            trade = "BUY"
        elif green_p <= (1 - threshold):
            trade = "SELL"
        else:
            trade = "SKIP"

        if trade == "SKIP":
            stats[regime]["skips"] += 1
            equity.append(cumulative)
            continue

        # Evaluate
        win = (trade == "BUY" and actual == 1) or (trade == "SELL" and actual == 0)
        if win:
            stats[regime]["wins"] += 1
            stats[regime]["pnl"].append(1)
            cumulative += 1
        else:
            stats[regime]["losses"] += 1
            stats[regime]["pnl"].append(-1)
            cumulative -= 1

        equity.append(cumulative)

    return {"per_regime": stats, "equity_curve": equity, "total_bars": total}


def _max_drawdown(pnl_list: list[int]) -> int:
    """Max drawdown in units."""
    if not pnl_list:
        return 0
    cum = np.cumsum(pnl_list)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return int(np.max(dd)) if len(dd) > 0 else 0


def _sharpe(pnl_list: list[int]) -> float:
    """Sharpe ratio (simplified: mean / std of returns)."""
    if len(pnl_list) < 2:
        return 0.0
    arr = np.array(pnl_list, dtype=float)
    std = arr.std()
    if std == 0:
        return 0.0
    return float(arr.mean() / std)


def print_report(result: dict) -> None:
    """Print formatted per-regime report."""
    stats = result["per_regime"]
    equity = result["equity_curve"]

    print(f"\n{'═'*80}")
    print(f"  {'Regime':<18} {'Trades':>7} {'Wins':>6} {'Losses':>7} "
          f"{'WinRate':>8} {'Skips':>7} {'MaxDD':>6} {'PF':>7} {'Sharpe':>7}")
    print(f"{'─'*80}")

    total_t = total_w = total_l = total_s = 0
    all_pnl = []

    for regime in REGIMES:
        s = stats[regime]
        t = s["wins"] + s["losses"]
        w = s["wins"]
        lo = s["losses"]
        sk = s["skips"]
        wr = (w / t * 100) if t > 0 else 0.0
        dd = _max_drawdown(s["pnl"])
        pf = (w / lo) if lo > 0 else float("inf")
        sr = _sharpe(s["pnl"])

        print(f"  {regime:<18} {t:>7} {w:>6} {lo:>7} "
              f"{wr:>7.1f}% {sk:>7} {dd:>6} {pf:>7.2f} {sr:>7.3f}")

        total_t += t
        total_w += w
        total_l += lo
        total_s += sk
        all_pnl.extend(s["pnl"])

    overall_wr = (total_w / total_t * 100) if total_t > 0 else 0
    overall_pf = (total_w / total_l) if total_l > 0 else float("inf")
    overall_dd = _max_drawdown(all_pnl)
    overall_sr = _sharpe(all_pnl)

    print(f"{'─'*80}")
    print(f"  {'TOTAL':<18} {total_t:>7} {total_w:>6} {total_l:>7} "
          f"{overall_wr:>7.1f}% {total_s:>7} {overall_dd:>6} "
          f"{overall_pf:>7.2f} {overall_sr:>7.3f}")
    print(f"{'═'*80}")

    # False positive / negative rates
    if total_t > 0:
        # False positive: predicted BUY but was red, or predicted SELL but was green
        fpr = total_l / total_t * 100
        fnr = 0  # not directly applicable in binary trade context
        print(f"\n  False signal rate: {fpr:.1f}% (losses / total trades)")

    # Equity curve summary
    if equity:
        final_eq = equity[-1]
        peak_eq = max(equity)
        print(f"\n  Equity curve: final={final_eq:.0f}, peak={peak_eq:.0f}")

    # Verdict
    print()
    if overall_wr >= 55:
        print("  ✅ System shows edge (≥55% win rate). Ready for forward testing.")
    elif overall_wr >= 53:
        print("  ⚠️  Marginal edge (53-55%). Consider improving features.")
    else:
        print("  ❌ Below 53% win rate. Do NOT go live. Improve features or model.")
    print()


def main():
    parser = argparse.ArgumentParser(description="Backtest the prediction engine per regime.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Symbol")
    args = parser.parse_args()
    result = run_backtest(args.symbol)
    print_report(result)


if __name__ == "__main__":
    main()
