"""
Correlation Guard — prevents correlated pair conflicts.

When trading multiple currency pairs, this module prevents taking
simultaneous positions on highly correlated pairs in the same direction.

Phase 4 upgrade: portfolio-level risk management.
"""

import logging
from datetime import datetime, timedelta

log = logging.getLogger("correlation_guard")

# Known correlation matrix for major pairs (approximate, updated monthly)
# Positive = move together, Negative = move opposite
PAIR_CORRELATIONS = {
    ("EURUSD", "GBPUSD"):  0.85,
    ("EURUSD", "AUDUSD"):  0.75,
    ("EURUSD", "NZDUSD"):  0.70,
    ("EURUSD", "USDCHF"): -0.90,
    ("EURUSD", "USDJPY"): -0.30,
    ("GBPUSD", "AUDUSD"):  0.70,
    ("GBPUSD", "NZDUSD"):  0.65,
    ("GBPUSD", "USDCHF"): -0.85,
    ("AUDUSD", "NZDUSD"):  0.90,
    ("USDJPY", "USDCHF"):  0.60,
}

# Threshold above which we consider pairs "too correlated" to trade together
CORRELATION_THRESHOLD = 0.75

# Time window to check for recent signals (in minutes)
SIGNAL_WINDOW_MINUTES = 10


class CorrelationGuard:
    """
    Tracks recent signals and prevents correlated pair conflicts.
    
    Usage:
        guard = CorrelationGuard()
        # Before sending a signal:
        if guard.should_allow(symbol="GBPUSD", direction="BUY"):
            # Send signal
            guard.record_signal("GBPUSD", "BUY")
        else:
            # Skip — correlated pair already has a signal
    """

    def __init__(self, correlation_threshold=CORRELATION_THRESHOLD,
                 window_minutes=SIGNAL_WINDOW_MINUTES):
        self.threshold = correlation_threshold
        self.window = timedelta(minutes=window_minutes)
        self.recent_signals = []  # [(symbol, direction, timestamp)]

    def _cleanup_old(self):
        """Remove signals older than the window."""
        cutoff = datetime.utcnow() - self.window
        self.recent_signals = [
            (sym, dir, ts) for sym, dir, ts in self.recent_signals
            if ts > cutoff
        ]

    def _get_correlation(self, sym1, sym2):
        """Get correlation between two pairs (order-independent)."""
        if sym1 == sym2:
            return 1.0
        key1 = (sym1, sym2)
        key2 = (sym2, sym1)
        return PAIR_CORRELATIONS.get(key1, PAIR_CORRELATIONS.get(key2, 0.0))

    def should_allow(self, symbol, direction):
        """
        Check if a new signal should be allowed based on correlation guard.
        
        Returns:
            (allowed: bool, reason: str)
        """
        self._cleanup_old()

        for sig_sym, sig_dir, sig_ts in self.recent_signals:
            corr = self._get_correlation(symbol, sig_sym)

            # Same direction on positively correlated pairs = conflict
            if abs(corr) >= self.threshold:
                if corr > 0 and direction == sig_dir:
                    reason = (f"Correlated pair conflict: {sig_sym} {sig_dir} "
                              f"(corr={corr:.2f}) active")
                    log.warning("CORRELATION GUARD: %s", reason)
                    return False, reason
                elif corr < 0 and direction != sig_dir:
                    # Opposite direction on negatively correlated = same trade
                    reason = (f"Inverse correlated conflict: {sig_sym} {sig_dir} "
                              f"(corr={corr:.2f}) active")
                    log.warning("CORRELATION GUARD: %s", reason)
                    return False, reason

        return True, ""

    def record_signal(self, symbol, direction):
        """Record a signal that was sent."""
        self.recent_signals.append((symbol, direction, datetime.utcnow()))
        log.info("Recorded signal: %s %s", symbol, direction)

    def get_active_signals(self):
        """Get list of currently active signals."""
        self._cleanup_old()
        return [(sym, dir) for sym, dir, _ in self.recent_signals]


# Global singleton
_guard = CorrelationGuard()


def check_correlation(symbol, direction):
    """Module-level convenience function."""
    return _guard.should_allow(symbol, direction)


def record_trade(symbol, direction):
    """Module-level convenience function."""
    _guard.record_signal(symbol, direction)
