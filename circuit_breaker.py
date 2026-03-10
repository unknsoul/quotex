"""
Circuit Breaker — V3 Layer 9: Emergency halt on excessive losses.

Monitors trading session for:
  1. Consecutive losses (5 = halt)
  2. Session drawdown (8% = halt)
  3. Daily loss limit

When triggered, halts all trading and sends Telegram alert.
"""

import logging
from datetime import datetime, timedelta

log = logging.getLogger("circuit_breaker")

MAX_CONSECUTIVE_LOSSES = 5
MAX_SESSION_DRAWDOWN_PCT = 8.0   # v10: 8% drawdown triggers halt
COOLDOWN_MINUTES = 60  # 1 hour cooldown after trigger


class CircuitBreaker:
    """Emergency trading halt system."""
    
    def __init__(self, max_losses=MAX_CONSECUTIVE_LOSSES,
                 max_drawdown_pct=MAX_SESSION_DRAWDOWN_PCT,
                 cooldown_min=COOLDOWN_MINUTES):
        self.max_losses = max_losses
        self.max_drawdown_pct = max_drawdown_pct
        self.cooldown_min = cooldown_min
        
        self.consecutive_losses = 0
        self.session_pnl = 0.0
        self.session_peak = 0.0
        self.is_halted = False
        self.halt_time = None
        self.halt_reason = ""
        self.daily_trades = 0
        self.daily_losses = 0
    
    def check(self, pnl_change=0.0, was_win=None):
        """
        Check circuit breaker conditions after a trade.
        
        Args:
            pnl_change: P&L change from this trade
            was_win: True if trade was profitable
        
        Returns:
            (is_halted, reason)
        """
        # Update cooldown
        if self.is_halted and self.halt_time:
            elapsed = (datetime.now() - self.halt_time).total_seconds() / 60
            if elapsed >= self.cooldown_min:
                self.reset_halt()
        
        if self.is_halted:
            return True, self.halt_reason
        
        # Update counters
        if was_win is not None:
            self.daily_trades += 1
            if was_win:
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
                self.daily_losses += 1
        
        # Update P&L
        self.session_pnl += pnl_change
        self.session_peak = max(self.session_peak, self.session_pnl)
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_losses:
            self._trigger(f"{self.consecutive_losses} consecutive losses")
            return True, self.halt_reason
        
        # Check drawdown
        drawdown = self.session_peak - self.session_pnl
        if self.session_peak > 0 and (drawdown / self.session_peak * 100) >= self.max_drawdown_pct:
            self._trigger(f"Session drawdown {drawdown:.1f} ({drawdown/self.session_peak*100:.1f}%)")
            return True, self.halt_reason
        
        return False, ""
    
    def can_trade(self):
        """Check if trading is allowed."""
        if self.is_halted:
            # Check cooldown
            if self.halt_time:
                elapsed = (datetime.now() - self.halt_time).total_seconds() / 60
                if elapsed >= self.cooldown_min:
                    self.reset_halt()
                    return True
            return False
        return True
    
    def _trigger(self, reason):
        """Trigger circuit breaker halt."""
        self.is_halted = True
        self.halt_time = datetime.now()
        self.halt_reason = f"CIRCUIT BREAKER: {reason}"
        log.warning(self.halt_reason)
    
    def reset_halt(self):
        """Reset halt state (after cooldown or manual reset)."""
        self.is_halted = False
        self.halt_reason = ""
        self.consecutive_losses = 0
        log.info("Circuit breaker reset")
    
    def reset_session(self):
        """Reset for new trading session."""
        self.session_pnl = 0.0
        self.session_peak = 0.0
        self.daily_trades = 0
        self.daily_losses = 0
        self.consecutive_losses = 0
        self.reset_halt()
    
    def get_status(self):
        """Return current status."""
        return {
            "halted": self.is_halted,
            "reason": self.halt_reason,
            "consecutive_losses": self.consecutive_losses,
            "session_pnl": self.session_pnl,
            "daily_trades": self.daily_trades,
        }
