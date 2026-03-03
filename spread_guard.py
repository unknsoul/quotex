"""
Spread Guard — V3 Layer 8 Gate 8: Real-time spread monitoring.

Blocks trades when spread exceeds 2.5× the median spread for that symbol.
"""

import numpy as np
import logging
from collections import defaultdict

log = logging.getLogger("spread_guard")

MAX_SPREAD_RATIO = 2.5
HISTORY_SIZE = 200


class SpreadGuard:
    """Spread monitor and gate for signal filtering."""
    
    def __init__(self, max_ratio=MAX_SPREAD_RATIO):
        self.max_ratio = max_ratio
        self.spread_history = defaultdict(list)
    
    def update(self, symbol, spread):
        """Record current spread."""
        self.spread_history[symbol].append(spread)
        if len(self.spread_history[symbol]) > HISTORY_SIZE:
            self.spread_history[symbol] = self.spread_history[symbol][-HISTORY_SIZE:]
    
    def check(self, symbol, current_spread):
        """Check if current spread is acceptable. Returns (pass, ratio)."""
        history = self.spread_history.get(symbol, [])
        if len(history) < 10:
            return True, 1.0
        
        median = np.median(history)
        if median <= 0:
            return True, 1.0
        
        ratio = current_spread / median
        passed = ratio <= self.max_ratio
        
        if not passed:
            log.info("Spread guard BLOCKED %s: spread=%.5f, median=%.5f, ratio=%.1f",
                     symbol, current_spread, median, ratio)
        
        return passed, ratio
    
    def get_median(self, symbol):
        """Get median spread for symbol."""
        history = self.spread_history.get(symbol, [])
        return float(np.median(history)) if history else 0.0
