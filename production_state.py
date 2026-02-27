"""Production state persistence for runtime health metrics."""

import json
import os
from datetime import datetime, timezone

from config import PRODUCTION_STATE_PATH, MODEL_VERSION


DEFAULT_STATE = {
    "model_version": MODEL_VERSION,
    "rolling_accuracy": 0.5,
    "rolling_spearman": 0.0,
    "last_retrain_time": None,
    "updated_at": None,
}


def load_state() -> dict:
    if not os.path.exists(PRODUCTION_STATE_PATH):
        return dict(DEFAULT_STATE)
    try:
        with open(PRODUCTION_STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        state = dict(DEFAULT_STATE)
        state.update(data)
        return state
    except Exception:
        return dict(DEFAULT_STATE)


def save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(PRODUCTION_STATE_PATH), exist_ok=True)
    payload = dict(DEFAULT_STATE)
    payload.update(state)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(PRODUCTION_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def update_runtime_metrics(rolling_accuracy: float, rolling_spearman: float, model_version: str | None = None):
    state = load_state()
    state["rolling_accuracy"] = float(rolling_accuracy)
    state["rolling_spearman"] = float(rolling_spearman)
    state["model_version"] = model_version or state.get("model_version") or MODEL_VERSION
    save_state(state)


def set_last_retrain_time(ts: str | None):
    state = load_state()
    state["last_retrain_time"] = ts
    save_state(state)
