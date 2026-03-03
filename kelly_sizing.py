"""
Kelly Sizing — V3 Layer 9: DrawdownAwareKelly with streak multipliers.

Implements fractional Kelly criterion for position sizing:
  kelly_fraction = (p * b - q) / b
  where p = win probability, b = win/loss ratio, q = 1-p

Includes 4-tier streak multiplier to reduce size during losing streaks:
  0-1 losses: 1.0× (full size)
  2-3 losses: 0.50× (half size)
  4+ losses:  0.25× (quarter size)
"""

import numpy as np
import logging

log = logging.getLogger("kelly_sizing")

# Streak multipliers
STREAK_MULTIPLIERS = {
    0: 1.0,   # No recent losses
    1: 1.0,   # 1 loss
    2: 0.50,  # 2 losses
    3: 0.50,  # 3 losses
    4: 0.25,  # 4+ losses
}
MAX_KELLY_FRACTION = 0.25  # Never bet more than 25% of bank
MIN_KELLY_FRACTION = 0.01  # Minimum meaningful bet


class DrawdownAwareKelly:
    """Kelly criterion with drawdown-aware streak multipliers."""
    
    def __init__(self, max_fraction=MAX_KELLY_FRACTION):
        self.max_fraction = max_fraction
        self.loss_streak = 0
        self.results = []  # Track recent results
    
    def compute_fraction(self, win_prob, win_loss_ratio=1.0):
        """
        Compute Kelly fraction adjusted for current loss streak.
        
        Args:
            win_prob: estimated probability of winning (0-1)
            win_loss_ratio: average win / average loss
        
        Returns:
            float: suggested position size as fraction of bankroll
        """
        # Raw Kelly
        p = np.clip(win_prob, 0.01, 0.99)
        q = 1 - p
        b = max(win_loss_ratio, 0.1)
        
        raw_kelly = (p * b - q) / b
        
        # Half Kelly (standard conservative approach)
        half_kelly = raw_kelly / 2.0
        
        # Streak adjustment
        streak_mult = self._get_streak_multiplier()
        adjusted = half_kelly * streak_mult
        
        # Clamp
        fraction = np.clip(adjusted, 0, self.max_fraction)
        
        return float(fraction)
    
    def update(self, was_win):
        """Record trade result and update streak."""
        self.results.append(int(was_win))
        if was_win:
            self.loss_streak = 0
        else:
            self.loss_streak += 1
    
    def _get_streak_multiplier(self):
        """Get multiplier based on current loss streak."""
        streak = min(self.loss_streak, max(STREAK_MULTIPLIERS.keys()))
        return STREAK_MULTIPLIERS.get(streak, 0.25)
    
    def get_suggested_size_pct(self, win_prob, win_loss_ratio=1.0):
        """Get position size as percentage (0-100)."""
        return self.compute_fraction(win_prob, win_loss_ratio) * 100
    
    def get_stats(self):
        """Return current Kelly stats."""
        n = len(self.results)
        wins = sum(self.results) if n > 0 else 0
        return {
            "total_trades": n,
            "win_rate": wins / n if n > 0 else 0.5,
            "loss_streak": self.loss_streak,
            "streak_mult": self._get_streak_multiplier(),
        }
