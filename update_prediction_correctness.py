"""Update prediction correctness in logs/predictions.csv after candle close.

Usage:
    python update_prediction_correctness.py --symbol EURUSD
"""

import argparse
from datetime import datetime, timezone, timedelta

import pandas as pd
import MetaTrader5 as mt5

from config import (
    DEFAULT_SYMBOL,
    MT5_PATH, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER,
    PREDICTION_LOG_CSV,
)


def connect_mt5() -> bool:
    kwargs = {}
    if MT5_PATH:
        kwargs["path"] = MT5_PATH
    if MT5_LOGIN:
        kwargs["login"] = MT5_LOGIN
    if MT5_PASSWORD:
        kwargs["password"] = MT5_PASSWORD
    if MT5_SERVER:
        kwargs["server"] = MT5_SERVER
    return mt5.initialize(**kwargs)


def floor_5m(ts: datetime) -> datetime:
    return ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0)


def actual_direction_for_prediction(symbol: str, pred_ts: datetime):
    # Prediction is sent ~5s before a close. Determine close pair [prev_close, this_close].
    close_time = floor_5m(pred_ts) + timedelta(minutes=5)
    start = close_time - timedelta(minutes=10)
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start, close_time + timedelta(minutes=1))
    if rates is None or len(rates) < 2:
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    # Keep candles up to close_time
    df = df[df["time"] <= pd.Timestamp(close_time)]
    if len(df) < 2:
        return None

    prev_close = float(df.iloc[-2]["close"])
    curr_close = float(df.iloc[-1]["close"])
    return "UP" if curr_close > prev_close else "DOWN"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    args = parser.parse_args()

    if not connect_mt5():
        print("ERROR: MT5 connection failed")
        return

    try:
        df = pd.read_csv(PREDICTION_LOG_CSV)
    except FileNotFoundError:
        print("No predictions CSV found.")
        mt5.shutdown()
        return

    if "is_correct" not in df.columns:
        df["is_correct"] = pd.NA
    if "actual_direction" not in df.columns:
        df["actual_direction"] = pd.NA

    now = datetime.now(timezone.utc)
    updated = 0

    for idx, row in df.iterrows():
        if row.get("symbol") != args.symbol:
            continue
        if pd.notna(row.get("is_correct")):
            continue

        ts = pd.to_datetime(row["timestamp"], utc=True).to_pydatetime()
        if ts > now - timedelta(minutes=5):
            continue  # candle not closed yet

        actual = actual_direction_for_prediction(args.symbol, ts)
        if actual is None:
            continue

        predicted = row.get("suggested_direction")
        if isinstance(predicted, str):
            is_correct = 1 if predicted == actual else 0
            df.at[idx, "actual_direction"] = actual
            df.at[idx, "is_correct"] = is_correct
            updated += 1

    df.to_csv(PREDICTION_LOG_CSV, index=False)
    mt5.shutdown()
    print(f"Updated {updated} rows in {PREDICTION_LOG_CSV}")


if __name__ == "__main__":
    main()
