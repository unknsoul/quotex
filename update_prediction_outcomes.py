"""Update predictions.csv with realized outcomes after next candle closes."""

import argparse
from datetime import timezone

import numpy as np
import pandas as pd
from scipy import stats

from config import DEFAULT_SYMBOL, PREDICTION_LOG_CSV
from data_collector import load_csv
from production_state import load_state, save_state


def _next_close(ts: pd.Timestamp) -> pd.Timestamp:
    ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    minute = (ts.minute // 5) * 5
    bar_open = ts.replace(minute=minute, second=0, microsecond=0)
    return bar_open + pd.Timedelta(minutes=5)


def _actual_direction(m5: pd.DataFrame, close_ts: pd.Timestamp):
    row = m5[m5["time"] == close_ts]
    if row.empty:
        return None
    r = row.iloc[0]
    return 1 if float(r["close"]) > float(r["open"]) else 0


def update(symbol: str):
    if not pd.io.common.file_exists(PREDICTION_LOG_CSV):
        print(f"No log file found: {PREDICTION_LOG_CSV}")
        return

    log_df = pd.read_csv(PREDICTION_LOG_CSV)
    if log_df.empty:
        print("Prediction log is empty.")
        return

    for col in ["actual_direction", "was_correct", "outcome_timestamp"]:
        if col not in log_df.columns:
            log_df[col] = np.nan

    m5 = load_csv(symbol, "M5")
    m5["time"] = pd.to_datetime(m5["time"], utc=True)
    now = pd.Timestamp.now(tz=timezone.utc)

    updated = 0
    for idx, row in log_df[log_df["symbol"] == symbol].iterrows():
        if pd.notna(row.get("was_correct")):
            continue
        try:
            ts = pd.to_datetime(row["timestamp"], utc=True)
            close_ts = _next_close(ts)
            if close_ts > now:
                continue
            actual = _actual_direction(m5, close_ts)
            if actual is None:
                continue
            pred = 1 if float(row.get("green_probability_percent", 50.0)) >= 50.0 else 0
            log_df.at[idx, "actual_direction"] = int(actual)
            log_df.at[idx, "was_correct"] = int(pred == actual)
            log_df.at[idx, "outcome_timestamp"] = close_ts.isoformat()
            updated += 1
        except Exception:
            continue

    log_df.to_csv(PREDICTION_LOG_CSV, index=False)
    print(f"Updated {updated} prediction outcomes for {symbol}.")

    scored = log_df[(log_df["symbol"] == symbol) & log_df["was_correct"].notna()].copy()
    if len(scored) >= 20:
        accuracy = float(scored["was_correct"].astype(float).tail(200).mean())
        if "final_confidence_percent" in scored.columns:
            conf = scored["final_confidence_percent"].astype(float).tail(200)
            cor = scored["was_correct"].astype(float).tail(200)
            spearman, _ = stats.spearmanr(conf, cor)
            spearman = float(spearman) if not np.isnan(spearman) else 0.0
        else:
            spearman = 0.0

        state = load_state()
        state["rolling_accuracy"] = round(accuracy, 4)
        state["rolling_spearman"] = round(spearman, 4)
        save_state(state)
        print(f"State updated: accuracy={accuracy:.4f}, spearman={spearman:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    args = parser.parse_args()
    update(args.symbol)
