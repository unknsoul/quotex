"""
Data Collector v2 — fetch multi-timeframe OHLC data from MT5.

Supports M5, M15, H1 in a single command.

Usage:
    python data_collector.py --symbol EURUSD
    python data_collector.py --symbol EURUSD --candles 15000
    python data_collector.py --symbol EURUSD --update
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

# Timeframe map
TF_MAP = {
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1":  mt5.TIMEFRAME_H1,
}


# =============================================================================
#  MT5 Helpers
# =============================================================================

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

    if not mt5.initialize(**kwargs):
        log.error("MT5 init failed: %s", mt5.last_error())
        return False

    info = mt5.terminal_info()
    log.info("MT5 connected -- build %s", info.build if info else "?")
    return True


def disconnect_mt5():
    mt5.shutdown()
    log.info("MT5 disconnected.")


# =============================================================================
#  Data Fetching
# =============================================================================

def fetch_candles(symbol: str, tf_name: str = "M5",
                  count: int = CANDLES_TO_FETCH) -> pd.DataFrame:
    """Fetch `count` candles for given timeframe from MT5."""
    tf = TF_MAP.get(tf_name)
    if tf is None:
        raise ValueError(f"Unknown timeframe: {tf_name}")

    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No data for {symbol}/{tf_name}: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    log.info("Fetched %d %s candles for %s", len(df), tf_name, symbol)
    return df


def fetch_multi_timeframe(symbol: str,
                          count: int = CANDLES_TO_FETCH) -> dict[str, pd.DataFrame]:
    """Fetch M5, M15, H1 candles. Returns {tf_name: DataFrame}."""
    result = {}
    for tf_name in TF_MAP:
        # Higher timeframes need fewer bars
        tf_count = count if tf_name == "M5" else count // 3
        try:
            result[tf_name] = fetch_candles(symbol, tf_name, tf_count)
        except RuntimeError as e:
            log.warning("Could not fetch %s: %s", tf_name, e)
    return result


# =============================================================================
#  Validation
# =============================================================================

def validate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.sort_values("time").reset_index(drop=True)
    df = df.drop_duplicates(subset=["time"], keep="last")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.reset_index(drop=True)
    after = len(df)
    if before != after:
        log.warning("Validation dropped %d rows (%d -> %d)", before - after, before, after)
    else:
        log.info("Validation OK -- %d rows clean.", after)
    return df


# =============================================================================
#  Save / Load
# =============================================================================

def _csv_path(symbol: str, tf: str = "M5") -> str:
    return os.path.join(DATA_DIR, f"{symbol}_{tf}.csv")


def save_csv(df: pd.DataFrame, symbol: str, tf: str = "M5") -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = _csv_path(symbol, tf)
    df.to_csv(path, index=False)
    log.info("Saved %d rows -> %s", len(df), path)
    return path


def load_csv(symbol: str, tf: str = "M5") -> pd.DataFrame:
    path = _csv_path(symbol, tf)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No data file at {path}. Run data_collector.py first.")
    df = pd.read_csv(path, parse_dates=["time"])
    log.info("Loaded %d rows from %s", len(df), path)
    return df


def load_multi_tf(symbol: str) -> dict[str, pd.DataFrame]:
    """Load all saved timeframe CSVs."""
    result = {}
    for tf in TF_MAP:
        try:
            result[tf] = load_csv(symbol, tf)
        except FileNotFoundError:
            log.warning("No %s data for %s, skipping.", tf, symbol)
    return result


def incremental_update(symbol: str, new_df: pd.DataFrame,
                       tf: str = "M5") -> pd.DataFrame:
    path = _csv_path(symbol, tf)
    if os.path.exists(path):
        old = pd.read_csv(path, parse_dates=["time"])
        merged = pd.concat([old, new_df], ignore_index=True)
    else:
        merged = new_df.copy()

    merged = merged.drop_duplicates(subset=["time"], keep="last")
    merged = merged.sort_values("time").reset_index(drop=True)
    log.info("Incremental update: %d rows total for %s/%s", len(merged), symbol, tf)
    return merged


# =============================================================================
#  CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fetch OHLC data from MT5.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--candles", type=int, default=CANDLES_TO_FETCH)
    parser.add_argument("--update", action="store_true", help="Incremental update")
    args = parser.parse_args()

    if not connect_mt5():
        print("ERROR: Cannot connect to MT5. Is the terminal running?")
        sys.exit(1)

    try:
        # Always fetch all timeframes
        print(f">> Fetching multi-timeframe data for {args.symbol}...")
        data = fetch_multi_timeframe(args.symbol, args.candles)

        for tf_name, df in data.items():
            df = validate(df)
            if args.update:
                df = incremental_update(args.symbol, df, tf_name)
            path = save_csv(df, args.symbol, tf_name)
            print(f"   {tf_name}: {len(df)} rows -> {path}")

        # Print M5 range
        if "M5" in data:
            m5 = data["M5"]
            print(f"   M5 range: {m5['time'].iloc[0]} -> {m5['time'].iloc[-1]}")
    finally:
        disconnect_mt5()


if __name__ == "__main__":
    main()
