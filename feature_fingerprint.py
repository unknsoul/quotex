"""
Feature Fingerprint — V3 Layer 3: Virtual drift detection.

Saves a fingerprint of feature statistics after training.
At inference startup, compares current features against fingerprint
to detect if feature_engineering.py has been modified.
"""

import json
import os
import numpy as np
import pandas as pd
import logging

log = logging.getLogger("feature_fingerprint")

FINGERPRINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "models", "fingerprint.json")


def save_fingerprint(df, feature_cols, path=FINGERPRINT_PATH):
    """
    Save feature statistics fingerprint after training.
    """
    stats = {}
    for col in feature_cols:
        if col in df.columns:
            values = df[col].dropna().values
            stats[col] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "n": int(len(values)),
            }
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    
    log.info("Feature fingerprint saved: %d features → %s", len(stats), path)
    return stats


def check_fingerprint(df, feature_cols, path=FINGERPRINT_PATH, tolerance=0.3):
    """
    Check current features against saved fingerprint.
    
    Returns:
        dict with comparison results
    """
    if not os.path.exists(path):
        log.warning("No fingerprint found at %s — skipping check", path)
        return {"status": "no_fingerprint", "mismatches": []}
    
    with open(path) as f:
        saved = json.load(f)
    
    mismatches = []
    for col in feature_cols:
        if col not in saved or col not in df.columns:
            mismatches.append(f"{col}: MISSING")
            continue
        
        current_vals = df[col].dropna().values
        if len(current_vals) < 10:
            continue
        
        current_mean = float(np.mean(current_vals))
        saved_mean = saved[col]["mean"]
        saved_std = saved[col]["std"]
        
        if saved_std > 0:
            z = abs(current_mean - saved_mean) / saved_std
            if z > tolerance * 10:
                mismatches.append(f"{col}: drift z={z:.1f}")
    
    if mismatches:
        log.warning("Feature fingerprint mismatches:\n  " + "\n  ".join(mismatches))
    else:
        log.info("Feature fingerprint OK (%d features match)", len(feature_cols))
    
    return {
        "status": "ok" if not mismatches else "drift",
        "mismatches": mismatches,
        "n_checked": len(feature_cols),
    }
