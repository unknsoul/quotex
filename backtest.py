"""
Backtest v2 — side-by-side comparison: primary-only vs primary+meta.

Walk-forward: for each bar, predict with both layers, record outcomes.

Usage:
    python backtest.py --symbol EURUSD
"""

import argparse
import os
import logging

import numpy as np
import pandas as pd
import joblib

from config import (
    DEFAULT_SYMBOL, MODEL_PATH, FEATURE_LIST_PATH,
    META_MODEL_PATH, META_FEATURE_LIST_PATH,
    META_ROLLING_WINDOW, WIN_STREAK_CAP,
    LOG_LEVEL, LOG_FORMAT,
)
from data_collector import load_csv, load_multi_tf
from feature_engineering import compute_features, add_target, FEATURE_COLUMNS
from regime_detection import detect_regime, get_regime_thresholds, REGIMES

log = logging.getLogger("backtest")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

REGIME_ENCODING = {"Trending": 0, "Ranging": 1, "High_Volatility": 2, "Low_Volatility": 3}


def _regime_for_row(df, idx, lookback=200):
    start = max(0, idx - lookback + 1)
    window = df.iloc[start:idx + 1]
    if len(window) < 50:
        return "Ranging"
    return detect_regime(window)


def _build_meta_input(green_p, regime, row, history, meta_feat_cols):
    """Build meta feature row for backtest."""
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

    meta_row = {
        "primary_green_prob": green_p,
        "primary_red_prob": 1.0 - green_p,
        "primary_confidence_strength": max(green_p, 1.0 - green_p),
        "regime_encoded": regime_enc,
        "atr_value": float(row.get("atr_14", 0)),
        "spread_ratio": 0.0,
        "volatility_zscore": float(row.get("volatility_zscore", 0)),
        "range_position": float(row.get("range_position", 0.5)),
        "recent_model_accuracy": recent_acc,
        "recent_win_streak": streak,
    }
    return pd.DataFrame([meta_row])[meta_feat_cols]


def run_backtest(symbol):
    # -- Load --
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

    primary_model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_LIST_PATH)
    meta_model = joblib.load(META_MODEL_PATH)
    meta_feat_cols = joblib.load(META_FEATURE_LIST_PATH)

    # -- Accumulators --
    primary_stats = {r: {"wins": 0, "losses": 0, "skips": 0, "pnl": []} for r in REGIMES}
    dual_stats = {r: {"wins": 0, "losses": 0, "skips": 0, "pnl": []} for r in REGIMES}
    primary_equity = []
    dual_equity = []
    p_cum = d_cum = 0.0
    meta_history = []

    start = 300
    total = len(df) - start
    print(f"\n>> Backtesting {total} bars for {symbol}...")

    for i in range(start, len(df)):
        regime = _regime_for_row(df, i)
        thresholds = get_regime_thresholds(regime)
        p_thresh = thresholds["primary"]
        m_thresh = thresholds["meta"]

        row_feat = df[feature_cols].iloc[i].values.reshape(1, -1)
        proba = primary_model.predict_proba(row_feat)[0]
        green_p = float(proba[1])
        actual = df["target"].iloc[i]

        # === PRIMARY-ONLY decision ===
        if green_p >= p_thresh:
            p_trade = "BUY"
        elif green_p <= (1 - p_thresh):
            p_trade = "SELL"
        else:
            p_trade = "SKIP"

        if p_trade == "SKIP":
            primary_stats[regime]["skips"] += 1
        else:
            win = (p_trade == "BUY" and actual == 1) or (p_trade == "SELL" and actual == 0)
            if win:
                primary_stats[regime]["wins"] += 1
                primary_stats[regime]["pnl"].append(1)
                p_cum += 1
            else:
                primary_stats[regime]["losses"] += 1
                primary_stats[regime]["pnl"].append(-1)
                p_cum -= 1
        primary_equity.append(p_cum)

        # === DUAL-LAYER decision ===
        meta_input = _build_meta_input(green_p, regime, df.iloc[i], meta_history, meta_feat_cols)
        meta_proba = meta_model.predict_proba(meta_input.values)[0]
        meta_rel = float(meta_proba[1])

        if green_p >= p_thresh and meta_rel >= m_thresh:
            d_trade = "BUY"
        elif green_p <= (1 - p_thresh) and meta_rel >= m_thresh:
            d_trade = "SELL"
        else:
            d_trade = "SKIP"

        if d_trade == "SKIP":
            dual_stats[regime]["skips"] += 1
        else:
            win = (d_trade == "BUY" and actual == 1) or (d_trade == "SELL" and actual == 0)
            if win:
                dual_stats[regime]["wins"] += 1
                dual_stats[regime]["pnl"].append(1)
                d_cum += 1
            else:
                dual_stats[regime]["losses"] += 1
                dual_stats[regime]["pnl"].append(-1)
                d_cum -= 1

        dual_equity.append(d_cum)

        # Update meta history (was primary correct?)
        primary_correct = 1 if ((green_p >= 0.5 and actual == 1) or (green_p < 0.5 and actual == 0)) else 0
        meta_history.append(primary_correct)

    return {
        "primary": primary_stats,
        "dual": dual_stats,
        "primary_equity": primary_equity,
        "dual_equity": dual_equity,
        "total_bars": total,
    }


def _summarize(stats, label):
    """Summarize stats for one mode."""
    total_t = total_w = total_l = total_s = 0
    all_pnl = []

    rows = []
    for regime in REGIMES:
        s = stats[regime]
        t = s["wins"] + s["losses"]
        w, lo, sk = s["wins"], s["losses"], s["skips"]
        wr = (w / t * 100) if t > 0 else 0
        pf = (w / lo) if lo > 0 else float("inf")
        rows.append((regime, t, w, lo, sk, wr, pf))
        total_t += t; total_w += w; total_l += lo; total_s += sk
        all_pnl.extend(s["pnl"])

    overall_wr = (total_w / total_t * 100) if total_t > 0 else 0
    overall_pf = (total_w / total_l) if total_l > 0 else float("inf")

    # Sharpe
    if len(all_pnl) >= 2:
        arr = np.array(all_pnl, dtype=float)
        sharpe = arr.mean() / (arr.std() + 1e-10)
    else:
        sharpe = 0

    # Max drawdown
    if all_pnl:
        cum = np.cumsum(all_pnl)
        peak = np.maximum.accumulate(cum)
        dd = int(np.max(peak - cum))
    else:
        dd = 0

    print(f"\n  [{label}]")
    print(f"  {'Regime':<18} {'Trades':>7} {'Wins':>6} {'WinRate':>8} {'Skips':>7} {'PF':>7}")
    print(f"  {'-'*60}")
    for regime, t, w, lo, sk, wr, pf in rows:
        print(f"  {regime:<18} {t:>7} {w:>6} {wr:>7.1f}% {sk:>7} {pf:>7.2f}")
    print(f"  {'-'*60}")
    print(f"  {'TOTAL':<18} {total_t:>7} {total_w:>6} {overall_wr:>7.1f}% {total_s:>7} {overall_pf:>7.2f}")
    print(f"  Sharpe: {sharpe:.3f}  MaxDD: {dd}  FalseRate: {(total_l/total_t*100) if total_t else 0:.1f}%")

    return {"trades": total_t, "wins": total_w, "wr": overall_wr, "sharpe": sharpe, "dd": dd}


def print_report(result):
    print(f"\n{'='*70}")
    print(f"  BACKTEST COMPARISON: Primary-Only vs Primary+Meta")
    print(f"{'='*70}")

    p = _summarize(result["primary"], "PRIMARY ONLY")
    d = _summarize(result["dual"], "PRIMARY + META")

    print(f"\n{'='*70}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Metric':<20} {'Primary':>12} {'Dual':>12} {'Delta':>12}")
    print(f"  {'-'*56}")
    print(f"  {'Trades':<20} {p['trades']:>12} {d['trades']:>12} {d['trades']-p['trades']:>+12}")
    print(f"  {'Win Rate':<20} {p['wr']:>11.1f}% {d['wr']:>11.1f}% {d['wr']-p['wr']:>+11.1f}%")
    print(f"  {'Sharpe':<20} {p['sharpe']:>12.3f} {d['sharpe']:>12.3f} {d['sharpe']-p['sharpe']:>+12.3f}")
    print(f"  {'Max Drawdown':<20} {p['dd']:>12} {d['dd']:>12} {d['dd']-p['dd']:>+12}")
    print(f"{'='*70}")

    if d['wr'] > p['wr']:
        print(f"\n  Meta layer improved win rate by {d['wr']-p['wr']:.1f}%")
    elif d['trades'] < p['trades']:
        print(f"\n  Meta layer filtered {p['trades']-d['trades']} low-confidence trades")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    args = parser.parse_args()
    result = run_backtest(args.symbol)
    print_report(result)


if __name__ == "__main__":
    main()
