"""
Adaptive Threshold Auto-Tuner — Upgrade 3.

Automatically adjusts MIN_CONFIDENCE_FILTER based on recent
signal performance. If accuracy drops, thresholds tighten.
If accuracy is high, thresholds can loosen slightly to allow
more signals.

This creates a self-correcting feedback loop.
"""

import logging
import json
import os
from datetime import datetime, timezone

log = logging.getLogger("adaptive_threshold")

STATE_FILE = os.path.join(os.path.dirname(__file__), "logs", "adaptive_state.json")

# Bounds — never go below or above these
MIN_BOUND = 38.0    # absolute minimum confidence threshold
MAX_BOUND = 55.0    # absolute maximum
DEFAULT = 42.0      # starting point

# Tuning parameters
EVAL_WINDOW = 30    # evaluate every N outcomes
TARGET_WR = 0.70    # target win rate (70%)
STEP_SIZE = 1.0     # adjustment step in percentage points


class AdaptiveThreshold:
    """
    Self-tuning confidence threshold.

    After every `eval_window` outcomes:
    - If win rate > target + 5%: loosen by step_size (allow more signals)
    - If win rate < target - 5%: tighten by step_size (be more selective)
    - Otherwise: no change
    """

    def __init__(self, initial=DEFAULT, eval_window=EVAL_WINDOW):
        self.threshold = initial
        self.eval_window = eval_window
        self._outcomes = []  # list of bools (True=correct, False=wrong)
        self._adjustments = []  # history of adjustments
        self._load_state()

    def _load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    state = json.load(f)
                self.threshold = state.get("threshold", DEFAULT)
                self._outcomes = state.get("outcomes", [])
                self._adjustments = state.get("adjustments", [])
                log.info("Adaptive threshold loaded: %.1f%% (%d outcomes in buffer)",
                         self.threshold, len(self._outcomes))
            except Exception as e:
                log.warning("Failed to load adaptive state: %s", e)

    def _save_state(self):
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        try:
            with open(STATE_FILE, "w") as f:
                json.dump({
                    "threshold": self.threshold,
                    "outcomes": self._outcomes[-200:],  # keep last 200
                    "adjustments": self._adjustments[-50:],
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                }, f, indent=2)
        except Exception as e:
            log.warning("Failed to save adaptive state: %s", e)

    def record_outcome(self, correct: bool):
        """Record a signal outcome and potentially adjust threshold."""
        self._outcomes.append(correct)

        # Evaluate every N outcomes
        if len(self._outcomes) % self.eval_window == 0:
            self._evaluate()

    def _evaluate(self):
        """Check recent performance and adjust threshold."""
        recent = self._outcomes[-self.eval_window:]
        win_rate = sum(recent) / len(recent)

        old_threshold = self.threshold
        adjustment = None

        if win_rate > TARGET_WR + 0.05:
            # Winning well — can loosen slightly
            self.threshold = max(MIN_BOUND, self.threshold - STEP_SIZE)
            adjustment = f"LOOSEN {old_threshold:.1f} -> {self.threshold:.1f} (WR={win_rate:.1%})"
        elif win_rate < TARGET_WR - 0.05:
            # Losing — tighten
            self.threshold = min(MAX_BOUND, self.threshold + STEP_SIZE)
            adjustment = f"TIGHTEN {old_threshold:.1f} -> {self.threshold:.1f} (WR={win_rate:.1%})"
        else:
            adjustment = f"HOLD {self.threshold:.1f} (WR={win_rate:.1%})"

        log.info("Adaptive threshold: %s", adjustment)
        self._adjustments.append({
            "time": datetime.now(timezone.utc).isoformat(),
            "action": adjustment,
            "win_rate": round(win_rate, 4),
            "threshold": self.threshold,
        })
        self._save_state()

    @property
    def current(self):
        """Current threshold value."""
        return self.threshold

    def status(self):
        """Return current status for monitoring."""
        recent = self._outcomes[-self.eval_window:] if self._outcomes else []
        wr = sum(recent) / len(recent) if recent else 0.0
        return {
            "threshold": self.threshold,
            "recent_win_rate": round(wr, 4),
            "total_outcomes": len(self._outcomes),
            "last_adjustment": self._adjustments[-1] if self._adjustments else None,
        }
