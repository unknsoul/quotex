"""
Anti-Streak Engine — v11: Prevents directional collapse.

Problem: The model predicted DOWN 96% of the time in production,
creating 41-signal DOWN streaks with degrading accuracy.
Short streaks (<=3) had 55% win rate vs 49.8% for long streaks (>10).

Solution: Exponential penalty on same-direction probability when
the model makes consecutive same-direction predictions.
"""

import math
import logging
import json
import os

log = logging.getLogger("anti_streak")

# ─── Configuration ───────────────────────────────────────────────────────────
STREAK_PENALTY_START = 4        # Start penalizing after N consecutive same-direction
STREAK_PENALTY_RATE = 0.04      # Exponential penalty rate per excess signal
STREAK_MAX_PENALTY = 0.30       # Maximum 30% confidence reduction
STREAK_FORCE_SKIP = 12          # Force HOLD after N consecutive same-direction
DIRECTION_BALANCE_WINDOW = 50   # Window for checking direction balance
DIRECTION_IMBALANCE_THRESHOLD = 0.80  # >80% same direction = imbalanced

# Persistence
_STATE_FILE = os.path.join(os.path.dirname(__file__), "logs", "streak_state.json")


class AntiStreakEngine:
    """Tracks prediction streaks and applies corrective penalties."""
    
    def __init__(self):
        self.direction_history = []  # list of "UP"/"DOWN"
        self.current_streak_dir = None
        self.current_streak_len = 0
        self._load_state()
    
    def _load_state(self):
        """Load streak state from disk (survives restarts)."""
        try:
            if os.path.exists(_STATE_FILE):
                with open(_STATE_FILE, "r") as f:
                    state = json.load(f)
                self.direction_history = state.get("direction_history", [])
                self.current_streak_dir = state.get("current_streak_dir")
                self.current_streak_len = state.get("current_streak_len", 0)
                log.info("Streak state loaded: streak=%d %s, history=%d",
                         self.current_streak_len, self.current_streak_dir or "None",
                         len(self.direction_history))
        except Exception as e:
            log.warning("Failed to load streak state: %s", e)
    
    def _save_state(self):
        """Save streak state to disk."""
        try:
            os.makedirs(os.path.dirname(_STATE_FILE), exist_ok=True)
            state = {
                "direction_history": self.direction_history[-DIRECTION_BALANCE_WINDOW:],
                "current_streak_dir": self.current_streak_dir,
                "current_streak_len": self.current_streak_len,
            }
            with open(_STATE_FILE, "w") as f:
                json.dump(state, f)
        except Exception as e:
            log.warning("Failed to save streak state: %s", e)
    
    def record_prediction(self, direction):
        """Record a new prediction direction ("UP" or "DOWN")."""
        self.direction_history.append(direction)
        if len(self.direction_history) > DIRECTION_BALANCE_WINDOW * 2:
            self.direction_history = self.direction_history[-DIRECTION_BALANCE_WINDOW:]
        
        if direction == self.current_streak_dir:
            self.current_streak_len += 1
        else:
            self.current_streak_dir = direction
            self.current_streak_len = 1
        
        self._save_state()
    
    def get_streak_penalty(self, direction):
        """
        Calculate confidence penalty for predicting the same direction again.
        
        Returns (penalty_multiplier, reason).
        penalty_multiplier: 0.70-1.00 (1.0 = no penalty)
        """
        if direction != self.current_streak_dir:
            return 1.0, ""  # Different direction → no penalty
        
        streak = self.current_streak_len + 1  # +1 because this would be the next signal
        
        if streak >= STREAK_FORCE_SKIP:
            return 0.0, f"Forced HOLD: {streak} consecutive {direction} signals"
        
        if streak <= STREAK_PENALTY_START:
            return 1.0, ""  # Within safe zone
        
        excess = streak - STREAK_PENALTY_START
        penalty = min(STREAK_PENALTY_RATE * excess, STREAK_MAX_PENALTY)
        multiplier = 1.0 - penalty
        
        reason = f"Streak penalty: {streak}x {direction} (-{penalty*100:.0f}% conf)"
        log.info(reason)
        return multiplier, reason
    
    def should_force_hold(self, direction):
        """Check if we should force HOLD due to excessive streak."""
        if direction == self.current_streak_dir and self.current_streak_len >= STREAK_FORCE_SKIP:
            return True, f"Max streak ({STREAK_FORCE_SKIP}) reached for {direction}"
        return False, ""
    
    def get_direction_balance(self):
        """
        Check recent direction balance.
        Returns (balance_ratio, majority_direction).
        balance_ratio: 0.5 = perfectly balanced, 1.0 = all same direction
        """
        if len(self.direction_history) < 10:
            return 0.5, None
        
        recent = self.direction_history[-DIRECTION_BALANCE_WINDOW:]
        up_count = sum(1 for d in recent if d == "UP")
        down_count = len(recent) - up_count
        
        majority = "UP" if up_count > down_count else "DOWN"
        balance = max(up_count, down_count) / len(recent)
        
        return round(balance, 3), majority
    
    def get_imbalance_penalty(self, direction):
        """
        Additional penalty when overall direction is heavily imbalanced.
        
        Returns penalty_multiplier: 0.85-1.00
        """
        balance, majority = self.get_direction_balance()
        
        if balance < DIRECTION_IMBALANCE_THRESHOLD:
            return 1.0, ""  # Balanced enough
        
        if direction == majority:
            # Penalize continuing the imbalanced direction
            excess = balance - DIRECTION_IMBALANCE_THRESHOLD
            penalty = min(excess * 0.75, 0.15)  # max 15%
            return 1.0 - penalty, f"Direction imbalance ({balance:.0%} {majority})"
        
        return 1.0, ""  # Different direction → no penalty


# Global singleton
_engine = AntiStreakEngine()


def get_streak_engine():
    """Get the global AntiStreakEngine instance."""
    return _engine


def apply_streak_correction(direction, confidence_pct):
    """
    Convenience function: Apply streak correction to a confidence value.
    
    Returns (adjusted_confidence, should_hold, reason).
    """
    engine = get_streak_engine()
    
    # Check force hold
    force_hold, hold_reason = engine.should_force_hold(direction)
    if force_hold:
        return 0.0, True, hold_reason
    
    # Streak penalty
    streak_mult, streak_reason = engine.get_streak_penalty(direction)
    
    # Direction imbalance penalty
    imbalance_mult, imbalance_reason = engine.get_imbalance_penalty(direction)
    
    adjusted = confidence_pct * streak_mult * imbalance_mult
    
    reasons = [r for r in [streak_reason, imbalance_reason] if r]
    reason = "; ".join(reasons)
    
    return round(adjusted, 2), False, reason
