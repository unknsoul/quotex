"""
Confidence Optimizer — learns optimal confidence thresholds per regime from OOF data.

Reads OOF predictions and finds the confidence threshold for each regime that
maximizes accuracy on traded signals. Saves thresholds to models/confidence_thresholds.pkl.

Usage:
    python confidence_optimizer.py
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from collections import defaultdict

from config import (
    MODEL_DIR, OOF_PREDICTIONS_PATH, ENSEMBLE_MODEL_PATH,
    META_MODEL_PATH, META_FEATURE_LIST_PATH, WEIGHT_MODEL_PATH,
    CONFIDENCE_THRESHOLDS_PATH,
    DEFAULT_SYMBOL, LOG_LEVEL, LOG_FORMAT,
)
from data_collector import load_csv, load_multi_tf
from feature_engineering import compute_features, add_target, FEATURE_COLUMNS
from regime_detection import detect_regime, detect_regime_series, REGIMES


log = logging.getLogger("confidence_optimizer")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


REGIME_ENCODING = {"Trending": 0, "Ranging": 1, "High_Volatility": 2, "Low_Volatility": 3}


def _binary_entropy(p):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def optimize_thresholds(symbol=DEFAULT_SYMBOL):
    """Find optimal confidence thresholds per regime using walk-forward OOF."""

    # Load data
    mtf = load_multi_tf(symbol)
    df = mtf.get("M5")
    if df is None:
        df = load_csv(symbol, "M5")
    m15, h1 = mtf.get("M15"), mtf.get("H1")
    df = compute_features(df, m15_df=m15, h1_df=h1)
    df = add_target(df).dropna(subset=["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)

    # Load models
    ensemble = joblib.load(ENSEMBLE_MODEL_PATH)
    meta_model = joblib.load(META_MODEL_PATH)
    meta_features = joblib.load(META_FEATURE_LIST_PATH)
    weight_model = joblib.load(WEIGHT_MODEL_PATH)

    # Detect regimes
    all_regimes = detect_regime_series(df)

    # Generate predictions for last 40% of data (simulated test)
    n = len(df)
    start_idx = int(n * 0.6)

    results = defaultdict(list)  # regime -> list of (confidence, correct)

    print(f"\n>> Optimizing thresholds on {n - start_idx} bars (last 40%)\n")

    for i in range(start_idx, n):
        row_feat = df[FEATURE_COLUMNS].iloc[[i]]
        regime = all_regimes.iloc[i] if i < len(all_regimes) else "Ranging"

        all_p = np.array([m.predict_proba(row_feat)[0][1] for m in ensemble])
        green_p = float(all_p.mean())
        variance = float(all_p.var())
        norm_var = min(variance / 0.25, 1.0)

        direction = 1 if green_p >= 0.5 else 0
        actual = df["target"].iloc[i]
        correct = 1 if direction == actual else 0

        # Simplified confidence (matches predict_engine logic)
        primary_str = abs(green_p - 0.5) * 2
        confidence = primary_str * (1.0 - norm_var) * 100

        results[regime].append((confidence, correct))

    # Find optimal thresholds
    print("=" * 60)
    print("  CONFIDENCE THRESHOLD OPTIMIZATION")
    print("=" * 60)

    optimal_thresholds = {}
    for regime in REGIMES:
        data = results.get(regime, [])
        if len(data) < 30:
            optimal_thresholds[regime] = {"min_confidence": 0, "expected_accuracy": 0.5}
            print(f"\n  {regime}: too few samples ({len(data)}), skipping")
            continue

        confs = np.array([d[0] for d in data])
        corrects = np.array([d[1] for d in data])

        best_threshold = 0
        best_score = 0  # accuracy × sqrt(frequency) — balance accuracy and trade count
        best_acc = 0
        best_count = 0

        for threshold in range(0, 80, 5):
            mask = confs >= threshold
            if mask.sum() < 20:
                continue
            acc = corrects[mask].mean()
            freq = mask.mean()
            score = acc * np.sqrt(freq)  # accuracy weighted by frequency
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_acc = acc
                best_count = mask.sum()

        optimal_thresholds[regime] = {
            "min_confidence": best_threshold,
            "expected_accuracy": round(float(best_acc), 4),
            "sample_count": int(best_count),
            "total_samples": len(data),
            "trade_frequency": round(best_count / len(data), 4) if len(data) > 0 else 0,
        }

        print(f"\n  {regime}:")
        print(f"    Total samples: {len(data)}")
        print(f"    Optimal threshold: >= {best_threshold}%")
        print(f"    Accuracy at threshold: {best_acc:.1%} ({best_count}/{len(data)})")
        print(f"    Trade frequency: {best_count/len(data):.1%}")

    # Save
    joblib.dump(optimal_thresholds, CONFIDENCE_THRESHOLDS_PATH)
    print(f"\n>> Saved thresholds -> {CONFIDENCE_THRESHOLDS_PATH}")

    # Summary
    print(f"\n  SUMMARY:")
    print(f"  {'Regime':<18} {'Threshold':>10} {'Accuracy':>10} {'Frequency':>10}")
    print(f"  {'-'*50}")
    for reg in REGIMES:
        t = optimal_thresholds[reg]
        print(f"  {reg:<18} {'>=' + str(t['min_confidence']) + '%':>10}"
              f" {t['expected_accuracy']:.1%}      {t.get('trade_frequency', 0):.1%}")

    return optimal_thresholds


if __name__ == "__main__":
    optimize_thresholds()
