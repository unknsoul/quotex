"""
Production State Manager — tracks model version, rolling metrics, retrain history.

Creates and maintains production_state.json for monitoring system health.
"""

import json
import os
import logging
from datetime import datetime, timezone

from config import MODEL_DIR, LOG_DIR

log = logging.getLogger("production_state")

STATE_FILE = os.path.join(LOG_DIR, "production_state.json")

_DEFAULT_STATE = {
    "model_version": "V3.0",
    "model_architecture": "XGB+LGB+RF ensemble + purged CV + regime routing",
    "feature_count": 65,
    "last_retrain_time": None,
    "last_retrain_type": None,       # "full" or "lite"
    "total_predictions": 0,
    "rolling_accuracy": 0.0,
    "rolling_spearman": 0.0,
    "total_correct": 0,
    "uptime_start": None,
    "last_prediction_time": None,
    "last_error": None,
    "error_count": 0,
    "avg_latency_ms": 0.0,
}


def load_state() -> dict:
    """Load production state from disk (creates default if missing)."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
            # Merge any missing keys from defaults
            for k, v in _DEFAULT_STATE.items():
                if k not in state:
                    state[k] = v
            return state
        except Exception as e:
            log.warning("Corrupt state file, recreating: %s", e)
    return _DEFAULT_STATE.copy()


def save_state(state: dict):
    """Save production state to disk."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def record_prediction(correct: bool, confidence: float, latency_ms: float):
    """Update rolling accuracy and prediction count."""
    state = load_state()
    state["total_predictions"] += 1
    if correct:
        state["total_correct"] += 1
    if state["total_predictions"] > 0:
        state["rolling_accuracy"] = round(
            state["total_correct"] / state["total_predictions"], 4
        )
    # Exponential moving average for latency
    if state["avg_latency_ms"] == 0:
        state["avg_latency_ms"] = latency_ms
    else:
        state["avg_latency_ms"] = round(
            0.9 * state["avg_latency_ms"] + 0.1 * latency_ms, 1
        )
    state["last_prediction_time"] = datetime.now(timezone.utc).isoformat()
    save_state(state)


def record_retrain(retrain_type: str = "full"):
    """Record that a retrain occurred."""
    state = load_state()
    state["last_retrain_time"] = datetime.now(timezone.utc).isoformat()
    state["last_retrain_type"] = retrain_type
    save_state(state)


def record_error(error_msg: str):
    """Record an error event."""
    state = load_state()
    state["error_count"] += 1
    state["last_error"] = f"{datetime.now(timezone.utc).isoformat()}: {error_msg}"
    save_state(state)


def record_startup():
    """Record bot startup time and reset session errors."""
    state = load_state()
    state["uptime_start"] = datetime.now(timezone.utc).isoformat()
    state["error_count"] = 0
    state["last_error"] = None
    save_state(state)


def update_spearman(spearman: float):
    """Update rolling Spearman correlation."""
    state = load_state()
    state["rolling_spearman"] = round(spearman, 4)
    save_state(state)


def get_state_summary() -> str:
    """Return formatted state summary for Telegram."""
    state = load_state()
    lines = [
        f"🏭 *Production State*",
        f"━━━━━━━━━━━━━━━━━━━━━",
        f"📦 Version: *{state['model_version']}*",
        f"📊 Features: *{state['feature_count']}*",
        f"🎯 Rolling Accuracy: *{state['rolling_accuracy']:.1%}*",
        f"📈 Spearman: *{state['rolling_spearman']:.4f}*",
        f"📝 Total Predictions: *{state['total_predictions']}*",
        f"⏱️ Avg Latency: *{state['avg_latency_ms']:.0f}ms*",
        f"🔄 Last Retrain: *{state.get('last_retrain_time', 'Never')}*",
        f"❌ Errors: *{state['error_count']}*",
    ]
    return "\n".join(lines)
