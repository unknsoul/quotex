"""
Covariate Shift Monitor — V3 Layer 11: Weekly PSI feature distribution check.

Monitors Population Stability Index (PSI) for each feature to detect
when feature distributions shift significantly from training baseline.

PSI > 0.10 = minor shift (warning)
PSI > 0.20 = major shift (triggers retrain)
"""

import numpy as np
import logging

log = logging.getLogger("covariate_shift")

PSI_WARNING = 0.10
PSI_RETRAIN = 0.20
N_BINS = 10


def compute_psi(expected, actual, n_bins=N_BINS):
    """
    Compute Population Stability Index between two distributions.
    
    Args:
        expected: array of baseline values (training)
        actual: array of current values (production)
    
    Returns:
        float: PSI value
    """
    # Create bins from expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    # Remove duplicates
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return 0.0
    
    expected_hist = np.histogram(expected, bins=breakpoints)[0].astype(float)
    actual_hist = np.histogram(actual, bins=breakpoints)[0].astype(float)
    
    # Normalize
    expected_pct = (expected_hist + 1) / (len(expected) + len(breakpoints) - 1)
    actual_pct = (actual_hist + 1) / (len(actual) + len(breakpoints) - 1)
    
    # PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    
    return float(psi)


def check_covariate_shift(baseline_df, current_df, feature_cols):
    """
    Check all features for covariate shift.
    
    Returns:
        dict with per-feature PSI and overall verdict
    """
    results = {}
    warnings = []
    retrain_needed = False
    
    for col in feature_cols:
        if col in baseline_df.columns and col in current_df.columns:
            baseline_vals = baseline_df[col].dropna().values
            current_vals = current_df[col].dropna().values
            
            if len(baseline_vals) < 50 or len(current_vals) < 50:
                continue
            
            psi = compute_psi(baseline_vals, current_vals)
            status = "OK"
            
            if psi > PSI_RETRAIN:
                status = "RETRAIN"
                retrain_needed = True
                warnings.append(f"{col}: PSI={psi:.3f} [RETRAIN]")
            elif psi > PSI_WARNING:
                status = "WARNING"
                warnings.append(f"{col}: PSI={psi:.3f} [WARNING]")
            
            results[col] = {"psi": psi, "status": status}
    
    if warnings:
        log.warning("Covariate shift detected:\n  " + "\n  ".join(warnings))
    else:
        log.info("No covariate shift detected (%d features checked)", len(results))
    
    return {
        "features": results,
        "retrain_needed": retrain_needed,
        "n_warnings": len(warnings),
        "warnings": warnings,
    }
