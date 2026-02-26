"""
Calibration utilities — shared CalibratedModel wrapper.

This module must be importable by all scripts that load ensemble pkl files.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression


class CalibratedModel:
    """XGBoost + IsotonicRegression calibration wrapper (picklable)."""

    def __init__(self, base_model, iso_reg):
        self.base_model = base_model
        self.iso_reg = iso_reg

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        calibrated = self.iso_reg.predict(raw)
        calibrated = np.clip(calibrated, 0, 1)
        return np.column_stack([1 - calibrated, calibrated])


def build_calibrated_model(base_model, X_val, y_val):
    """Train isotonic calibration on held-out val set."""
    raw_val = base_model.predict_proba(X_val)[:, 1]
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(raw_val, y_val)
    return CalibratedModel(base_model, iso)
