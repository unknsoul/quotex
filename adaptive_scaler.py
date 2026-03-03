"""
Adaptive Scaler — V3 Layer 3: Rolling Z-score normalization.

Replaces frozen StandardScaler with rolling 500-bar Z-score.
This adapts to changing market conditions at inference time,
preventing the "stale scaler" problem where features drift
from training distribution.
"""

import numpy as np
import pandas as pd
import logging

log = logging.getLogger("adaptive_scaler")

ROLLING_WINDOW = 500


class AdaptiveScaler:
    """Rolling Z-score normalizer that adapts to recent data."""
    
    def __init__(self, window=ROLLING_WINDOW):
        self.window = window
        self.history = {}  # feature_name → list of recent values
    
    def transform_row(self, features):
        """
        Transform a single row of features using rolling Z-score.
        
        Args:
            features: dict of {feature_name: value}
        
        Returns:
            dict of normalized features
        """
        normalized = {}
        for name, value in features.items():
            if name not in self.history:
                self.history[name] = []
            
            self.history[name].append(value)
            
            # Trim to window
            if len(self.history[name]) > self.window:
                self.history[name] = self.history[name][-self.window:]
            
            # Z-score normalize
            values = np.array(self.history[name])
            if len(values) >= 10:
                mean = np.mean(values)
                std = np.std(values)
                if std > 1e-10:
                    normalized[name] = (value - mean) / std
                else:
                    normalized[name] = 0.0
            else:
                normalized[name] = value  # Not enough history yet
        
        return normalized
    
    def transform_df(self, df, feature_cols):
        """
        Transform a DataFrame of features using rolling Z-score.
        
        For batch processing (backtest), uses pandas rolling.
        """
        df = df.copy()
        for col in feature_cols:
            if col in df.columns:
                rolling_mean = df[col].rolling(self.window, min_periods=10).mean()
                rolling_std = df[col].rolling(self.window, min_periods=10).std()
                df[col] = (df[col] - rolling_mean) / (rolling_std + 1e-10)
        return df
    
    def seed(self, df, feature_cols):
        """Seed the scaler history from a DataFrame (at startup)."""
        for col in feature_cols:
            if col in df.columns:
                self.history[col] = df[col].tail(self.window).tolist()
        log.info("Adaptive scaler seeded with %d bars", min(len(df), self.window))
    
    def reset(self):
        """Clear history."""
        self.history = {}
