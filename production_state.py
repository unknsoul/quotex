"""Production state persistence for forward deployment monitoring."""

import json
import os
from datetime import datetime, timezone

from config import PRODUCTION_STATE_PATH


def _default_state():
    return {
        "model_version": "unknown",
        "rolling_accuracy": 0.5,
        "rolling_spearman": 0.0,
        "last_retrain_time": None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def load_state():
    if not os.path.exists(PRODUCTION_STATE_PATH):
        return _default_state()
    try:
        with open(PRODUCTION_STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        base = _default_state()
        base.update(data)
        return base
    except Exception:
        return _default_state()


def save_state(state):
    os.makedirs(os.path.dirname(PRODUCTION_STATE_PATH), exist_ok=True)
    state = dict(state)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(PRODUCTION_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def touch_state(model_version="unknown", last_retrain_time=None):
    state = load_state()
    if model_version:
        state["model_version"] = model_version
    if last_retrain_time is not None:
        state["last_retrain_time"] = last_retrain_time
    save_state(state)
    return state
