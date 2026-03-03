"""
Time-Weighted Training — V3 Layer 4: Exponential recency weights.

Recent data is more predictive than old data in financial markets.
Applies exponential decay weighting so the model focuses on recent patterns.
Combined with Triple Barrier weights for final sample_weight.
"""

import numpy as np
import logging

log = logging.getLogger("time_weight")

DEFAULT_HALF_LIFE_DAYS = 180  # 180 days half-life
BARS_PER_DAY_M5 = 288  # 24 * 60 / 5


def compute_time_weights(n_samples, half_life_days=DEFAULT_HALF_LIFE_DAYS,
                          bars_per_day=BARS_PER_DAY_M5):
    """
    Compute exponential recency weights for n_samples bars.
    
    Most recent bar gets weight 1.0. Weight halves every half_life_days.
    """
    half_life_bars = half_life_days * bars_per_day
    decay = np.log(2) / half_life_bars
    
    # Distance from most recent (index n-1 is most recent)
    distances = np.arange(n_samples - 1, -1, -1, dtype=np.float64)
    weights = np.exp(-decay * distances)
    
    # Normalize so mean weight = 1.0 (preserves effective sample size)
    weights = weights / weights.mean()
    
    log.info("Time weights: n=%d, half_life=%d days, min=%.4f, max=%.4f",
             n_samples, half_life_days, weights.min(), weights.max())
    
    return weights


def combine_weights(time_weights, tb_weights=None):
    """
    Combine time weights with Triple Barrier sample weights.
    
    Final weight = time_weight × tb_weight (if available).
    Normalized so mean = 1.0.
    """
    if tb_weights is not None:
        combined = time_weights * tb_weights
    else:
        combined = time_weights.copy()
    
    # Normalize
    if combined.sum() > 0:
        combined = combined / combined.mean()
    
    return combined
