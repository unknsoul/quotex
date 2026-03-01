"""
Drift Detector — Population Stability Index + KS-test for feature drift.

Monitors feature distributions for concept drift between training data
and recent predictions. Triggers alerts when distributions shift significantly.

Phase 4 upgrade: detect when input data shifts before accuracy drops.
"""

import numpy as np
import logging
from scipy import stats

log = logging.getLogger("drift_detector")


def _psi_bucket(expected, actual, bins=10):
    """
    Population Stability Index (PSI) between two distributions.
    PSI < 0.1: no shift, 0.1-0.2: moderate, > 0.2: significant drift.
    """
    # Create bins from expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return 0.0

    exp_counts = np.histogram(expected, bins=breakpoints)[0].astype(float) + 1e-6
    act_counts = np.histogram(actual, bins=breakpoints)[0].astype(float) + 1e-6

    exp_pct = exp_counts / exp_counts.sum()
    act_pct = act_counts / act_counts.sum()

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


def detect_feature_drift(train_features, recent_features, feature_names,
                          psi_threshold=0.2, ks_pvalue=0.01, top_n=10):
    """
    Detect concept drift by comparing training vs recent feature distributions.

    Args:
        train_features: np.ndarray (training data features)
        recent_features: np.ndarray (recent N bars features)
        feature_names: list of feature names
        psi_threshold: PSI threshold for drift alert
        ks_pvalue: KS-test p-value threshold for drift alert
        top_n: number of top features to check

    Returns:
        dict with drift analysis results
    """
    n_features = min(len(feature_names), train_features.shape[1], recent_features.shape[1])
    drift_alerts = []
    feature_results = []

    for i in range(min(n_features, top_n)):
        name = feature_names[i] if i < len(feature_names) else f"feat_{i}"
        train_col = train_features[:, i]
        recent_col = recent_features[:, i]

        # Remove NaN
        train_col = train_col[~np.isnan(train_col)]
        recent_col = recent_col[~np.isnan(recent_col)]

        if len(train_col) < 50 or len(recent_col) < 50:
            continue

        # PSI
        psi = _psi_bucket(train_col, recent_col)

        # KS-test
        ks_stat, ks_p = stats.ks_2samp(train_col, recent_col)

        drifted = psi > psi_threshold or ks_p < ks_pvalue

        result = {
            "feature": name,
            "psi": round(psi, 4),
            "ks_stat": round(ks_stat, 4),
            "ks_pvalue": round(ks_p, 4),
            "drifted": drifted,
        }
        feature_results.append(result)

        if drifted:
            drift_alerts.append(name)
            log.warning("DRIFT DETECTED on '%s': PSI=%.4f, KS p=%.4f", name, psi, ks_p)

    overall_drift = len(drift_alerts) >= 3  # 3+ features drifted = significant

    return {
        "overall_drift": overall_drift,
        "drifted_features": drift_alerts,
        "n_checked": len(feature_results),
        "n_drifted": len(drift_alerts),
        "details": feature_results,
        "recommendation": "RETRAIN" if overall_drift else "OK",
    }


def check_drift_from_files(train_path="models/oof_predictions.pkl",
                            recent_df=None, feature_names=None):
    """
    Convenience function to check drift using saved OOF data.
    Call from auto_learner or predict_engine.
    """
    import joblib
    import os

    if not os.path.exists(train_path):
        return {"overall_drift": False, "recommendation": "NO_BASELINE"}

    oof_data = joblib.load(train_path)
    if "feature_names" not in oof_data:
        return {"overall_drift": False, "recommendation": "NO_FEATURE_DATA"}

    if recent_df is None or feature_names is None:
        return {"overall_drift": False, "recommendation": "NO_RECENT_DATA"}

    feature_cols = [f for f in feature_names if f in recent_df.columns]
    if len(feature_cols) < 10:
        return {"overall_drift": False, "recommendation": "INSUFFICIENT_FEATURES"}

    recent_features = recent_df[feature_cols].values[-500:]  # last 500 bars

    # We need training features - rebuild from saved data
    # For now, use recent data split as proxy
    n_train = max(len(recent_df) - 500, 500)
    train_features = recent_df[feature_cols].values[:n_train]

    return detect_feature_drift(train_features, recent_features, feature_cols)
