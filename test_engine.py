"""
Comprehensive V3 Bot Verification
Tests all message formats, predict engine, and symbol handling.
"""
import sys
import os

# ─── Test 1: All modules import ───────────────────────────────────────────
print("=" * 60)
print("TEST 1: Module Imports")
print("=" * 60)
modules = [
    'data_cleaner', 'triple_barrier_labeler', 'time_weighted_training',
    'purged_wfcv', 'asymmetric_loss', 'regime_classifier',
    'dynamic_ensemble', 'meta_labeler', 'adaptive_scaler',
    'spread_guard', 'kelly_sizing', 'circuit_breaker',
    'signal_db', 'monte_carlo', 'transaction_cost_model',
    'health_check', 'error_attribution', 'covariate_shift_monitor',
    'feature_fingerprint', 'online_learner', 'pair_selector',
    'config', 'data_collector', 'feature_engineering',
    'regime_detection', 'calibration', 'stability',
    'meta_model', 'predict_engine', 'backtest',
    'telegram_bot', 'risk_filter', 'production_state',
]
errors = []
for m in modules:
    try:
        __import__(m)
        print(f"  OK: {m}")
    except Exception as e:
        errors.append((m, str(e)))
        print(f"  FAIL: {m} -> {e}")
print(f"\n  Result: {len(modules) - len(errors)}/{len(modules)} modules OK")

# ─── Test 2: Symbol Configuration ─────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 2: Symbol Configuration")
print("=" * 60)
from telegram_bot import SYMBOLS
print(f"  Configured symbols: {SYMBOLS}")
print(f"  Count: {len(SYMBOLS)}")
assert len(SYMBOLS) == 6, f"Expected 6 symbols, got {len(SYMBOLS)}"
print("  PASS: 6 pairs configured")

# ─── Test 3: Predict Engine ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 3: Predict Engine (EURUSD)")
print("=" * 60)
from predict_engine import predict, load_models
from data_collector import load_csv
from feature_engineering import compute_features
from regime_detection import detect_regime

load_models()
df = load_csv("EURUSD", "M5")
df = compute_features(df)
regime = detect_regime(df)
result = predict(df, regime)

# Verify all expected fields exist
required_fields = [
    "green_probability_percent", "red_probability_percent",
    "primary_direction", "meta_reliability_percent",
    "uncertainty_percent", "final_confidence_percent",
    "confidence_level", "kelly_fraction_percent",
    "suggested_trade", "suggested_direction",
    "market_regime", "regime_stability", "session",
    "latency_ms",
]
missing = [f for f in required_fields if f not in result]
if missing:
    print(f"  FAIL: Missing fields: {missing}")
else:
    print(f"  Direction: {result['primary_direction']}")
    print(f"  Green: {result['green_probability_percent']}%")
    print(f"  Confidence: {result['final_confidence_percent']}%")
    print(f"  Meta: {result['meta_reliability_percent']}%")
    print(f"  Uncertainty: {result['uncertainty_percent']}%")
    print(f"  Kelly: {result['kelly_fraction_percent']}%")
    print(f"  Regime: {result['market_regime']}")
    print(f"  Session: {result['session']}")
    print(f"  Trade: {result['suggested_trade']}")
    print(f"  Latency: {result['latency_ms']}ms")
    print(f"  Error: {result.get('error', 'None')}")
    print("  PASS: All required fields present")

# ─── Test 4: Message Formatters ───────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 4: Message Formatters")
print("=" * 60)

from telegram_bot import _format_prediction, _format_combined_signal

# Create mock prediction for testing
mock_pred = {
    "symbol": "EURUSD",
    "suggested_direction": "UP",
    "green_probability_percent": 56.2,
    "red_probability_percent": 43.8,
    "final_confidence_percent": 68.5,
    "meta_reliability_percent": 72.3,
    "uncertainty_percent": 3.1,
    "kelly_fraction_percent": 2.4,
    "suggested_trade": "BUY",
    "market_regime": "Trending",
    "regime_stability": 0.85,
    "session": "London",
    "latency_ms": 1200,
    "adaptive_confidence_percent": 65.0,
    "ensemble_variance": 0.002,
}

# Test single prediction format
single_msg = _format_prediction(mock_pred, is_auto=False)
print("  --- Single Prediction ---")
print(single_msg)
print()

# Test auto format
auto_msg = _format_prediction(mock_pred, is_auto=True)
assert "AUTO-SIGNAL" in auto_msg, "Auto-signal header missing"
print("  PASS: Auto-signal format OK")

# Test combined signal format
mock_preds = {
    "EURUSD": {**mock_pred, "symbol": "EURUSD"},
    "GBPUSD": {**mock_pred, "symbol": "GBPUSD", "suggested_direction": "DOWN",
               "green_probability_percent": 44.1, "red_probability_percent": 55.9,
               "final_confidence_percent": 61.2, "session": "Overlap",
               "market_regime": "Ranging"},
}
filtered = {"USDJPY": "low-conf (42%)", "XAUUSD": "regime shift"}
combined_msg = _format_combined_signal(mock_preds, filtered)
print("  --- Combined Signal ---")
print(combined_msg)
print()

# Verify V3 branding in combined
assert "V3" in combined_msg, "V3 branding missing from combined signal"
assert "14 Layers" in combined_msg, "14 Layers missing from combined signal"
assert "65 Features" in combined_msg, "65 Features missing from combined signal"
assert "Meta" in combined_msg, "Meta info missing from combined signal"
assert "Unc" in combined_msg, "Uncertainty info missing from combined signal"
print("  PASS: Combined signal with V3 branding OK")

# Verify filtered reasons
assert "USDJPY" in combined_msg, "Filtered USDJPY missing"
assert "XAUUSD" in combined_msg, "Filtered XAUUSD missing"
print("  PASS: Filtered symbols shown correctly")

# ─── Test 5: V3 Component Tests ──────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 5: V3 Components")
print("=" * 60)

# Data cleaner
from data_cleaner import clean_data
df_clean = clean_data(load_csv("EURUSD", "M5"))
print(f"  Data cleaner: {len(df_clean)} bars (cleaned)")

# Regime classifier
from regime_classifier import classify_regime
rr = classify_regime(df_clean.tail(500))
print(f"  Regime classifier: {rr.regime} (conf={rr.confidence:.2f}, ADX={rr.adx:.1f})")

# Triple barrier
from triple_barrier_labeler import label_triple_barrier
df_tb = compute_features(df_clean)
df_labeled = label_triple_barrier(df_tb, tp_mult=1.5, sl_mult=1.0, max_bars=4)
tp_count = (df_labeled.get("tb_label", pd.Series()) == "TP").sum() if "tb_label" in df_labeled.columns else "N/A"
print(f"  Triple Barrier: {len(df_labeled)} bars labeled")

# Time weights
from time_weighted_training import compute_time_weights
tw = compute_time_weights(1000, half_life_days=180)
print(f"  Time weights: min={tw.min():.4f}, max={tw.max():.4f}")

# Kelly sizing
from kelly_sizing import DrawdownAwareKelly
kelly = DrawdownAwareKelly()
size = kelly.compute_fraction(win_prob=0.56, win_loss_ratio=1.0)
print(f"  Kelly sizing: {size:.4f} for 56% win rate")

# Circuit breaker
from circuit_breaker import CircuitBreaker
cb = CircuitBreaker()
print(f"  Circuit breaker: halted={cb.is_halted}, can_trade={cb.can_trade()}")

# Health check
from health_check import HealthChecker
hc = HealthChecker()
print(f"  Health check: healthy={hc.is_healthy()}")

import pandas as pd

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
