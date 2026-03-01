"""
Learned Regime Detection — K-means clustering on market features.

Replaces hardcoded ADX/ATR threshold-based regime classification with
a data-driven approach that discovers non-obvious market regimes.

Phase 4 upgrade: adaptive regime detection.
"""

import numpy as np
import pandas as pd
import joblib
import os
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config import MODEL_DIR

log = logging.getLogger("learned_regime")

REGIME_MODEL_PATH = os.path.join(MODEL_DIR, "regime_model.pkl")
REGIME_SCALER_PATH = os.path.join(MODEL_DIR, "regime_scaler.pkl")

# Features used for regime clustering
REGIME_FEATURES = [
    "adx_normalized",
    "atr_percentile_rank",
    "ema_slope_magnitude",
    "volatility_zscore",
    "bb_width",
    "rolling_std_20",
]

# Regime labels (assigned after clustering based on feature centroids)
REGIME_LABELS = {
    0: "Trending",
    1: "Ranging",
    2: "High_Volatility",
    3: "Low_Volatility",
    4: "Breakout",
    5: "Choppy",
}

N_CLUSTERS = 4  # start with 4 regimes, can increase


class LearnedRegimeDetector:
    """
    K-means based regime detector.
    
    Learns market regimes from historical data instead of using
    hardcoded thresholds. More adaptive to changing market conditions.
    """

    def __init__(self, n_clusters=N_CLUSTERS):
        self.n_clusters = n_clusters
        self.model = None
        self.scaler = None
        self.label_map = {}  # cluster_id -> regime_name

    def fit(self, df):
        """
        Fit regime detector on historical data.
        
        Args:
            df: DataFrame with REGIME_FEATURES computed
        """
        features = [f for f in REGIME_FEATURES if f in df.columns]
        if len(features) < 3:
            log.warning("Not enough regime features (%d), need at least 3", len(features))
            return False

        X = df[features].dropna().values
        if len(X) < 100:
            log.warning("Not enough data for regime learning (%d rows)", len(X))
            return False

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300,
        )
        self.model.fit(X_scaled)

        # Assign labels based on centroids
        self._assign_labels(features)

        log.info("Learned %d regimes from %d samples", self.n_clusters, len(X))
        return True

    def _assign_labels(self, features):
        """
        Assign human-readable labels to clusters based on centroid characteristics.
        """
        centroids = self.scaler.inverse_transform(self.model.cluster_centers_)

        adx_idx = features.index("adx_normalized") if "adx_normalized" in features else None
        atr_idx = features.index("atr_percentile_rank") if "atr_percentile_rank" in features else None
        vol_idx = features.index("volatility_zscore") if "volatility_zscore" in features else None

        for i in range(self.n_clusters):
            adx = centroids[i][adx_idx] if adx_idx is not None else 0.5
            atr = centroids[i][atr_idx] if atr_idx is not None else 0.5
            vol = centroids[i][vol_idx] if vol_idx is not None else 0.0

            if vol > 1.0:
                self.label_map[i] = "High_Volatility"
            elif adx > 0.35:
                self.label_map[i] = "Trending"
            elif atr < 0.3:
                self.label_map[i] = "Low_Volatility"
            else:
                self.label_map[i] = "Ranging"

        log.info("Regime labels: %s", self.label_map)

    def predict(self, df):
        """
        Predict regime for the latest bar(s).
        
        Returns:
            str: regime name
        """
        if self.model is None or self.scaler is None:
            return "Ranging"  # fallback

        features = [f for f in REGIME_FEATURES if f in df.columns]
        if len(features) < 3:
            return "Ranging"

        row = df[features].iloc[-1:].values
        if np.isnan(row).any():
            return "Ranging"

        row_scaled = self.scaler.transform(row)
        cluster = int(self.model.predict(row_scaled)[0])
        return self.label_map.get(cluster, "Ranging")

    def predict_series(self, df):
        """Predict regime for all rows."""
        if self.model is None or self.scaler is None:
            return pd.Series("Ranging", index=df.index)

        features = [f for f in REGIME_FEATURES if f in df.columns]
        X = df[features].fillna(0).values
        X_scaled = self.scaler.transform(X)
        clusters = self.model.predict(X_scaled)
        return pd.Series([self.label_map.get(c, "Ranging") for c in clusters], index=df.index)

    def save(self, model_path=REGIME_MODEL_PATH, scaler_path=REGIME_SCALER_PATH):
        """Save trained regime model."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "label_map": self.label_map,
            "n_clusters": self.n_clusters,
        }, model_path)
        joblib.dump(self.scaler, scaler_path)
        log.info("Saved regime model -> %s", model_path)

    def load(self, model_path=REGIME_MODEL_PATH, scaler_path=REGIME_SCALER_PATH):
        """Load trained regime model."""
        if not os.path.exists(model_path):
            return False
        data = joblib.load(model_path)
        self.model = data["model"]
        self.label_map = data["label_map"]
        self.n_clusters = data["n_clusters"]
        self.scaler = joblib.load(scaler_path)
        log.info("Loaded regime model (%d clusters)", self.n_clusters)
        return True


# Global singleton
_detector = LearnedRegimeDetector()


def train_regime_model(df):
    """Train and save the learned regime detector."""
    success = _detector.fit(df)
    if success:
        _detector.save()
    return success


def predict_learned_regime(df):
    """Predict regime using learned model (with auto-load)."""
    if _detector.model is None:
        if not _detector.load():
            return "Ranging"  # fallback to rule-based
    return _detector.predict(df)
