"""
Purged Walk-Forward CV — V3 Layer 6: Leakage-free cross-validation.

Standard TimeSeriesSplit allows leakage through:
  1. Label leakage: Triple Barrier labels look forward N bars
  2. Feature leakage: Rolling features carry information across folds

This implementation adds:
  - Purge gap: removes bars around train/test boundary
  - Embargo period: adds buffer after test set starts
"""

import numpy as np
import logging

log = logging.getLogger("purged_wfcv")

DEFAULT_PURGE_BARS = 10   # Remove 10 bars before test set
DEFAULT_EMBARGO_BARS = 5  # Remove 5 bars after test start


class PurgedWalkForwardCV:
    """
    Purged Walk-Forward Cross-Validation.
    
    Like TimeSeriesSplit but with purge gap between train and test
    to prevent label and feature leakage.
    
    Parameters:
      n_splits: number of folds
      purge_bars: bars to remove before test boundary
      embargo_bars: bars to remove after test boundary start
    """
    
    def __init__(self, n_splits=5, purge_bars=DEFAULT_PURGE_BARS,
                 embargo_bars=DEFAULT_EMBARGO_BARS):
        self.n_splits = n_splits
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars
    
    def split(self, X, y=None, groups=None):
        """
        Generate purged train/test indices.
        
        Yields (train_indices, test_indices) for each fold.
        """
        n = len(X) if hasattr(X, '__len__') else X.shape[0]
        fold_size = n // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            test_start = fold_size * (i + 1)
            test_end = fold_size * (i + 2) if i < self.n_splits - 1 else n
            
            # Train: everything before test, minus purge gap
            train_end = max(0, test_start - self.purge_bars)
            train_indices = np.arange(0, train_end)
            
            # Test: starts after embargo
            test_start_actual = min(test_start + self.embargo_bars, n)
            test_indices = np.arange(test_start_actual, test_end)
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def purged_walk_forward_split(n_samples, n_splits=5, purge_bars=DEFAULT_PURGE_BARS,
                               embargo_bars=DEFAULT_EMBARGO_BARS):
    """
    Functional interface for purged walk-forward splits.
    
    Returns list of (train_idx, test_idx) tuples.
    """
    cv = PurgedWalkForwardCV(n_splits, purge_bars, embargo_bars)
    return list(cv.split(np.arange(n_samples)))
