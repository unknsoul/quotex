"""Test datetime fix with live MT5 data."""
import MetaTrader5 as mt5
mt5.initialize()

from data_collector import fetch_multi_timeframe
data = fetch_multi_timeframe("EURUSD", 1000)
m5 = data.get("M5")
m15 = data.get("M15")
h1 = data.get("H1")

print(f"M5 time dtype: {m5['time'].dtype}")
print(f"M15 time dtype: {m15['time'].dtype}")
print(f"H1 time dtype: {h1['time'].dtype}")

from feature_engineering import compute_features
try:
    df = compute_features(m5, m15_df=m15, h1_df=h1)
    print(f"compute_features OK: {len(df)} rows, {len(df.columns)} cols")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test full prediction pipeline
from regime_detection import detect_regime
from predict_engine import predict, load_models
load_models()
regime = detect_regime(df)
result = predict(df, regime)
print(f"Prediction: {result['primary_direction']} {result['green_probability_percent']}% conf={result['final_confidence_percent']}%")
print(f"Trade: {result['suggested_trade']}, Error: {result.get('error', 'None')}")
print("ALL OK")
mt5.shutdown()
