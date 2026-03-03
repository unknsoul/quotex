import asyncio
from datetime import datetime, timezone
import MetaTrader5 as mt5
mt5.initialize()

from telegram_bot import _run_prediction, _format_combined_signal, MIN_CONFIDENCE_FILTER, SYMBOLS

print(f"Testing Auto-Signal pipeline...")
print(f"MIN_CONFIDENCE_FILTER = {MIN_CONFIDENCE_FILTER}")

predictions = {}
filtered_out = {}

for sym in SYMBOLS:
    try:
        print(f"\n--- {sym} ---")
        pred = _run_prediction(sym)
        conf = pred["final_confidence_percent"]
        trade = pred["suggested_trade"]
        
        print(f"Raw Output: trade={trade}, conf={conf:.1f}%")
        print(f"Green={pred['green_probability_percent']:.1f}%, Meta={pred['meta_reliability_percent']:.1f}%")
        
        if pred.get("error"):
            print(f"Filter: ERROR ({pred['error']})")
            filtered_out[sym] = pred["error"]
            continue
        if conf < MIN_CONFIDENCE_FILTER:
            print(f"Filter: LOW-CONF ({conf:.1f} < {MIN_CONFIDENCE_FILTER})")
            filtered_out[sym] = f"low-conf ({conf:.0f}%)"
            continue
        if trade == "HOLD" and conf < 30:
            print(f"Filter: HOLD & VERY LOW CONF")
            filtered_out[sym] = "very-low-conf HOLD"
            continue
            
        print("✅ PASSED FILTERS")
        predictions[sym] = pred
        
    except Exception as e:
        print(f"CRASH: {e}")

print("\n--- FINAL RESULT ---")
print(f"Passed: {list(predictions.keys())}")
print(f"Filtered: {filtered_out}")

if predictions:
    msg = _format_combined_signal(predictions, filtered_out)
    print("\nMessage that would be sent:")
    print("-" * 40)
    print(msg)
    print("-" * 40)

mt5.shutdown()
