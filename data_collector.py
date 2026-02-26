"""
Data Collector — fetch OHLC data from MT5, validate, save to CSV.

Usage:
    python data_collector.py --symbol EURUSD
    python data_collector.py --symbol EURUSD --candles 15000
    python data_collector.py --symbol EURUSD --update     # incremental
"""

import argparse
import os
import sys
import logging
from datetime import datetime, timezone

import pandas as pd
import MetaTrader5 as mt5

from config import (
    MT5_PATH, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER,
    DATA_DIR, DEFAULT_SYMBOL, CANDLES_TO_FETCH,
    LOG_LEVEL, LOG_FORMAT,
)

log = logging.getLogger("data_collector")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


# ═══════════════════════════════════════════════════════════════════════════
#  MT5 Helpers
# ═══════════════════════════════════════════════════════════════════════════

def connect_mt5() -> bool:
    """Initialize MT5 terminal connection."""
    kwargs = {}
    if MT5_PATH:
        kwargs["path"] = MT5_PATH
    if MT5_LOGIN:
        kwargs["login"] = MT5_LOGIN
    if MT5_PASSWORD:
        kwargs["password"] = MT5_PASSWORD
    if MT5_SERVER:
        kwargs["server"] = MT5_SERVER

    if not mt5.initialize(**kwargs):
        log.error("MT5 init failed: %s", mt5.last_error())
        return False

    info = mt5.terminal_info()
    log.info("MT5 connected — build %s", info.build if info else "?")
    return True


def disconnect_mt5():
    mt5.shutdown()
    log.info("MT5 disconnected.")


# ═══════════════════════════════════════════════════════════════════════════
#  Data Fetching
# ═══════════════════════════════════════════════════════════════════════════

def fetch_candles(symbol: str, count: int = CANDLES_TO_FETCH) -> pd.DataFrame:
    """
    Fetch `count` M5 candles from MT5.
    Returns DataFrame: time, open, high, low, close, tick_volume, spread
    """
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, count)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No data for {symbol}: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    log.info("Fetched %d candles for %s", len(df), symbol)
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════════════════

def validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw OHLC DataFrame:
      1. Chronological sort
      2. Drop duplicate timestamps
      3. Drop rows with any NaN in OHLCV
      4. Reset index
    """
    before = len(df)
    df = df.sort_values("time").reset_index(drop=True)
    df = df.drop_duplicates(subset=["time"], keep="last")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.reset_index(drop=True)
    after = len(df)
    if before != after:
        log.warning("Validation dropped %d rows (%d -> %d)", before - after, before, after)
    else:
        log.info("Validation OK — %d rows clean.", after)
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  Save / Update
# ═══════════════════════════════════════════════════════════════════════════

def _csv_path(symbol: str) -> str:
    return os.path.join(DATA_DIR, f"{symbol}_M5.csv")


def save_csv(df: pd.DataFrame, symbol: str) -> str:
    """Save full dataset to CSV. Returns file path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = _csv_path(symbol)
    df.to_csv(path, index=False)
    log.info("Saved %d rows -> %s", len(df), path)
    return path


def load_csv(symbol: str) -> pd.DataFrame:
    """Load saved CSV for a symbol."""
    path = _csv_path(symbol)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No data file at {path}. Run data_collector.py first.")
    df = pd.read_csv(path, parse_dates=["time"])
    log.info("Loaded %d rows from %s", len(df), path)
    return df


def incremental_update(symbol: str, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge new candles with existing CSV. Keeps only unique timestamps.
    Returns the merged DataFrame.
    """
    path = _csv_path(symbol)
    if os.path.exists(path):
        old = pd.read_csv(path, parse_dates=["time"])
        merged = pd.concat([old, new_df], ignore_index=True)
    else:
        merged = new_df.copy()

    merged = merged.drop_duplicates(subset=["time"], keep="last")
    merged = merged.sort_values("time").reset_index(drop=True)
    log.info("Incremental update: %d rows total for %s", len(merged), symbol)
    return merged


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Fetch M5 OHLC data from MT5 and save to CSV.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Trading symbol")
    parser.add_argument("--candles", type=int, default=CANDLES_TO_FETCH, help="Candles to fetch")
    parser.add_argument("--update", action="store_true", help="Incremental update mode")
    args = parser.parse_args()

    if not connect_mt5():
        print("ERROR: Cannot connect to MT5. Is the terminal running?")
        sys.exit(1)

    try:
        print(f">> Fetching {args.candles} M5 candles for {args.symbol}...")
        df = fetch_candles(args.symbol, args.candles)
        df = validate(df)

        if args.update:
            df = incremental_update(args.symbol, df)

        path = save_csv(df, args.symbol)
        print(f">> Saved {len(df)} rows -> {path}")
        print(f"   Range: {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
    finally:
        disconnect_mt5()


if __name__ == "__main__":
    main()
