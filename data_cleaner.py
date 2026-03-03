"""
Data Cleaner — V3 Layer 1: Spike removal, weekend gaps, duplicate removal, OHLCV validation.

Cleans raw OHLCV data before feature engineering to remove noise that degrades model accuracy.
"""

import numpy as np
import pandas as pd
import logging

log = logging.getLogger("data_cleaner")


def remove_spikes(df, z_threshold=5.0, col="close"):
    """Remove price spikes > z_threshold standard deviations from rolling mean."""
    df = df.copy()
    rolling_mean = df[col].rolling(20, min_periods=5).mean()
    rolling_std = df[col].rolling(20, min_periods=5).std()
    z_score = (df[col] - rolling_mean) / (rolling_std + 1e-10)
    spike_mask = z_score.abs() > z_threshold
    n_spikes = spike_mask.sum()
    if n_spikes > 0:
        df.loc[spike_mask, col] = rolling_mean[spike_mask]
        log.info("Removed %d spikes (z > %.1f) from %s", n_spikes, z_threshold, col)
    return df


def remove_weekend_gaps(df):
    """Remove bars during weekend gaps (Saturday/Sunday)."""
    if "time" not in df.columns:
        return df
    df = df.copy()
    weekday = df["time"].dt.dayofweek
    weekend_mask = weekday >= 5  # Saturday=5, Sunday=6
    n_weekend = weekend_mask.sum()
    if n_weekend > 0:
        df = df[~weekend_mask].reset_index(drop=True)
        log.info("Removed %d weekend bars", n_weekend)
    return df


def remove_duplicates(df):
    """Remove duplicate timestamps."""
    if "time" not in df.columns:
        return df
    n_before = len(df)
    df = df.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    n_removed = n_before - len(df)
    if n_removed > 0:
        log.info("Removed %d duplicate bars", n_removed)
    return df


def validate_ohlcv(df):
    """Validate OHLCV data integrity: high >= low, high >= open/close, etc."""
    df = df.copy()
    invalid = (
        (df["high"] < df["low"]) |
        (df["high"] < df["open"]) |
        (df["high"] < df["close"]) |
        (df["low"] > df["open"]) |
        (df["low"] > df["close"])
    )
    n_invalid = invalid.sum()
    if n_invalid > 0:
        # Fix by recalculating
        df.loc[invalid, "high"] = df.loc[invalid, ["open", "high", "low", "close"]].max(axis=1)
        df.loc[invalid, "low"] = df.loc[invalid, ["open", "high", "low", "close"]].min(axis=1)
        log.info("Fixed %d invalid OHLCV bars", n_invalid)
    return df


def remove_zero_range(df):
    """Remove bars where high == low (no price movement, likely bad data)."""
    zero_range = (df["high"] - df["low"]).abs() < 1e-10
    n_zero = zero_range.sum()
    if n_zero > 0:
        df = df[~zero_range].reset_index(drop=True)
        log.info("Removed %d zero-range bars", n_zero)
    return df


def clean_data(df):
    """Full cleaning pipeline: validate → spikes → weekends → duplicates → zero-range."""
    n_before = len(df)
    df = validate_ohlcv(df)
    for col in ["open", "high", "low", "close"]:
        df = remove_spikes(df, col=col)
    df = remove_weekend_gaps(df)
    df = remove_duplicates(df)
    df = remove_zero_range(df)
    n_after = len(df)
    log.info("Data cleaning: %d → %d bars (%.1f%% kept)",
             n_before, n_after, n_after / n_before * 100 if n_before else 0)
    return df
