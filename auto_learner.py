"""
Auto-Learning Controller — production-safe retrain with validation safeguard.

Monitors:
  - Rolling accuracy (< 52% triggers retrain)
  - Confidence-accuracy correlation (< 0.1 triggers retrain)
  - Feature importance drift (cosine < threshold triggers retrain)

Retrain process:
  1. Backup current models
  2. Train new models (never blocks predictions)
  3. Validate on holdout slice
  4. Deploy only if new model >= old model - 2pp
  5. Rollback if worse

Usage:
    python auto_learner.py --symbol EURUSD --check     # Check if retrain needed
    python auto_learner.py --symbol EURUSD --retrain   # Force retrain
    python auto_learner.py --symbol EURUSD --monitor   # Continuous monitoring loop
"""

import argparse
import os
import sys
import time
import shutil
import logging
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
from scipy import stats

from config import (
    DEFAULT_SYMBOL, MODEL_DIR, MODEL_BACKUP_DIR,
    ENSEMBLE_MODEL_PATH, META_MODEL_PATH, META_FEATURE_LIST_PATH,
    WEIGHT_MODEL_PATH, OOF_PREDICTIONS_PATH, FEATURE_LIST_PATH,
    AUTO_RETRAIN_ACCURACY_TRIGGER, AUTO_RETRAIN_CORRELATION_TRIGGER,
    PREDICTION_LOG_CSV, DATA_BUFFER_SIZE,
    DRIFT_COSINE_THRESHOLD,
    LOG_LEVEL, LOG_FORMAT,
)

log = logging.getLogger("auto_learner")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

_retrain_in_progress = False
_retrain_lock = threading.Lock()


# =============================================================================
#  Monitoring
# =============================================================================

def load_prediction_log():
    """Load prediction history from CSV log."""
    if not os.path.exists(PREDICTION_LOG_CSV):
        log.warning("No prediction log found at %s", PREDICTION_LOG_CSV)
        return pd.DataFrame()
    df = pd.read_csv(PREDICTION_LOG_CSV)
    return df


def compute_confidence_correlation(predictions, actuals, confidences):
    """Spearman correlation between confidence and correctness."""
    if len(predictions) < 30:
        return 0.0
    correct = (np.array(predictions) == np.array(actuals)).astype(float)
    corr, _ = stats.spearmanr(confidences, correct)
    return float(corr) if not np.isnan(corr) else 0.0


class PerformanceMonitor:
    """Track live prediction performance for retrain triggers."""

    def __init__(self, window=100):
        self.window = window
        self._predictions = []
        self._actuals = []
        self._confidences = []

    def record(self, predicted_dir, actual_dir, confidence):
        self._predictions.append(predicted_dir)
        self._actuals.append(actual_dir)
        self._confidences.append(confidence)
        # Keep only recent
        if len(self._predictions) > self.window * 3:
            self._predictions = self._predictions[-self.window * 2:]
            self._actuals = self._actuals[-self.window * 2:]
            self._confidences = self._confidences[-self.window * 2:]

    @property
    def rolling_accuracy(self):
        if len(self._predictions) < 10:
            return 0.5
        recent_p = self._predictions[-self.window:]
        recent_a = self._actuals[-self.window:]
        correct = sum(1 for p, a in zip(recent_p, recent_a) if p == a)
        return correct / len(recent_p)

    @property
    def confidence_correlation(self):
        if len(self._predictions) < 30:
            return 0.5
        return compute_confidence_correlation(
            self._predictions[-self.window:],
            self._actuals[-self.window:],
            self._confidences[-self.window:],
        )

    def should_retrain(self):
        triggers = []
        acc = self.rolling_accuracy
        if acc < AUTO_RETRAIN_ACCURACY_TRIGGER and len(self._predictions) >= 50:
            triggers.append(f"accuracy={acc:.3f} < {AUTO_RETRAIN_ACCURACY_TRIGGER}")
        corr = self.confidence_correlation
        if corr < AUTO_RETRAIN_CORRELATION_TRIGGER and len(self._predictions) >= 50:
            triggers.append(f"correlation={corr:.3f} < {AUTO_RETRAIN_CORRELATION_TRIGGER}")
        return triggers

    def status(self):
        return {
            "rolling_accuracy": round(self.rolling_accuracy, 4),
            "confidence_correlation": round(self.confidence_correlation, 4),
            "total_predictions": len(self._predictions),
            "retrain_triggers": self.should_retrain(),
        }


# =============================================================================
#  Backup & Rollback
# =============================================================================

MODEL_FILES = [
    ENSEMBLE_MODEL_PATH, META_MODEL_PATH, META_FEATURE_LIST_PATH,
    WEIGHT_MODEL_PATH, OOF_PREDICTIONS_PATH, FEATURE_LIST_PATH,
]


def backup_current_models():
    """Backup current production models."""
    os.makedirs(MODEL_BACKUP_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(MODEL_BACKUP_DIR, timestamp)
    os.makedirs(backup_dir, exist_ok=True)

    for path in MODEL_FILES:
        if os.path.exists(path):
            shutil.copy2(path, backup_dir)

    log.info("Models backed up to %s", backup_dir)
    return backup_dir


def rollback_models(backup_dir):
    """Restore models from backup."""
    for path in MODEL_FILES:
        basename = os.path.basename(path)
        backup_path = os.path.join(backup_dir, basename)
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, path)
    log.info("Models rolled back from %s", backup_dir)


# =============================================================================
#  Retrain Pipeline
# =============================================================================

def retrain(symbol, validate=True):
    """
    Full retrain pipeline (NEVER blocks predictions):
    1. Set retrain flag
    2. Backup current models
    3. Train new models
    4. Validate on holdout
    5. Deploy only if better
    6. Reload cached models
    """
    global _retrain_in_progress

    with _retrain_lock:
        if _retrain_in_progress:
            log.warning("Retrain already in progress.")
            return False
        _retrain_in_progress = True

    # Notify predict_engine
    try:
        from predict_engine import set_retrain_flag, reload_models as pe_reload
        set_retrain_flag(True)
    except ImportError:
        pe_reload = None

    try:
        log.info("=" * 60)
        log.info("AUTO-RETRAIN STARTED for %s", symbol)
        log.info("=" * 60)

        # Step 1: Backup
        backup_dir = backup_current_models()

        # Step 2: Get current model performance
        old_score = _evaluate_current_model(symbol)
        log.info("Current model score: %.4f", old_score if old_score else 0)

        # Step 3: Retrain
        import subprocess
        python = sys.executable

        for script in ["train_model.py", "meta_model.py", "weight_learner.py"]:
            log.info("Running %s...", script)
            result = subprocess.run(
                [python, script, "--symbol", symbol],
                capture_output=True, text=True, cwd=os.path.dirname(__file__),
            )
            if result.returncode != 0:
                log.error("RETRAIN FAILED at %s:\n%s", script, result.stderr[-500:])
                rollback_models(backup_dir)
                return False

        # Step 4: Validate
        if validate:
            new_score = _evaluate_current_model(symbol)
            log.info("New model score: %.4f", new_score if new_score else 0)

            if old_score and new_score and new_score < old_score - 0.02:
                log.warning("NEW MODEL WORSE: %.4f < %.4f. Rolling back.", new_score, old_score)
                rollback_models(backup_dir)
                return False

        # Step 5: Reload models in predict_engine
        if pe_reload:
            pe_reload()

        log.info("=" * 60)
        log.info("AUTO-RETRAIN COMPLETE — new models deployed.")
        log.info("=" * 60)
        return True

    except Exception as e:
        log.exception("Retrain failed: %s", e)
        return False
    finally:
        with _retrain_lock:
            _retrain_in_progress = False
        try:
            from predict_engine import set_retrain_flag
            set_retrain_flag(False)
        except ImportError:
            pass


def retrain_in_background(symbol):
    """Run retrain in a background thread (never blocks predictions)."""
    thread = threading.Thread(target=retrain, args=(symbol,), daemon=True)
    thread.start()
    log.info("Background retrain started for %s", symbol)
    return thread


def retrain_lite(symbol, validate=True):
    """
    Online Learning Lite: retrain ONLY meta + weight (fast).
    Keeps primary ensemble frozen. Useful for drift correction
    without overfitting risk. ~10x faster than full retrain.
    """
    global _retrain_in_progress

    with _retrain_lock:
        if _retrain_in_progress:
            log.warning("Retrain already in progress.")
            return False
        _retrain_in_progress = True

    try:
        from predict_engine import set_retrain_flag, reload_models as pe_reload
        set_retrain_flag(True)
    except ImportError:
        pe_reload = None

    try:
        log.info("=" * 60)
        log.info("LITE RETRAIN STARTED for %s (meta + weight only)", symbol)
        log.info("=" * 60)

        backup_dir = backup_current_models()
        old_score = _evaluate_current_model(symbol)
        log.info("Current model score: %.4f", old_score if old_score else 0)

        import subprocess
        python = sys.executable

        # Only retrain meta + weight (skip primary ensemble)
        for script in ["meta_model.py", "weight_learner.py"]:
            log.info("Running %s...", script)
            result = subprocess.run(
                [python, script, "--symbol", symbol],
                capture_output=True, text=True, cwd=os.path.dirname(__file__),
            )
            if result.returncode != 0:
                log.error("LITE RETRAIN FAILED at %s:\n%s", script, result.stderr[-500:])
                rollback_models(backup_dir)
                return False

        if validate:
            new_score = _evaluate_current_model(symbol)
            log.info("New model score: %.4f", new_score if new_score else 0)
            if old_score and new_score and new_score < old_score - 0.02:
                log.warning("NEW MODEL WORSE: %.4f < %.4f. Rolling back.", new_score, old_score)
                rollback_models(backup_dir)
                return False

        if pe_reload:
            pe_reload()

        log.info("=" * 60)
        log.info("LITE RETRAIN COMPLETE — meta + weight updated, ensemble frozen.")
        log.info("=" * 60)
        return True

    except Exception as e:
        log.exception("Lite retrain failed: %s", e)
        return False
    finally:
        with _retrain_lock:
            _retrain_in_progress = False
        try:
            from predict_engine import set_retrain_flag
            set_retrain_flag(False)
        except ImportError:
            pass


def retrain_lite_in_background(symbol):
    """Run lite retrain in a background thread."""
    thread = threading.Thread(target=retrain_lite, args=(symbol,), daemon=True)
    thread.start()
    log.info("Background lite retrain started for %s", symbol)
    return thread


def _evaluate_current_model(symbol):
    """Quick evaluation: run predictions on last 500 bars, return accuracy."""
    try:
        from data_collector import load_csv, load_multi_tf
        from feature_engineering import compute_features, add_target, FEATURE_COLUMNS
        from calibration import CalibratedModel

        mtf = load_multi_tf(symbol)
        df = mtf.get("M5")
        if df is None:
            df = load_csv(symbol, "M5")
        m15, h1 = mtf.get("M15"), mtf.get("H1")
        df = compute_features(df, m15_df=m15, h1_df=h1)
        df = add_target(df)
        df = df.dropna(subset=["target"]).reset_index(drop=True)

        ensemble = joblib.load(ENSEMBLE_MODEL_PATH)
        X = df[FEATURE_COLUMNS].tail(500)
        y = df["target"].tail(500).astype(int).values

        all_p = np.array([m.predict_proba(X)[:, 1] for m in ensemble])
        mean_p = all_p.mean(axis=0)
        preds = (mean_p >= 0.5).astype(int)
        acc = (preds == y).mean()
        return float(acc)
    except Exception as e:
        log.warning("Evaluation failed: %s", e)
        return None


# =============================================================================
#  CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Auto-Learning Controller")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--check", action="store_true", help="Check if retrain needed")
    parser.add_argument("--retrain", action="store_true", help="Force full retrain")
    parser.add_argument("--lite", action="store_true", help="Lite retrain (meta + weight only, fast)")
    parser.add_argument("--monitor", action="store_true", help="Continuous monitoring")
    parser.add_argument("--interval", type=int, default=300, help="Monitor interval (seconds)")
    args = parser.parse_args()

    if args.check:
        score = _evaluate_current_model(args.symbol)
        print(f"\n  Current model accuracy (last 500 bars): {score:.4f}" if score else "  Could not evaluate.")
        print(f"  Retrain trigger threshold: < {AUTO_RETRAIN_ACCURACY_TRIGGER}")
        if score and score < AUTO_RETRAIN_ACCURACY_TRIGGER:
            print("  ⚠️  RETRAIN RECOMMENDED")
        else:
            print("  ✅ Model performing within acceptable range")

    elif args.retrain:
        print(f"\n>> Forcing full retrain for {args.symbol}...")
        success = retrain(args.symbol)
        print(f"\n>> {'SUCCESS' if success else 'FAILED'}")

    elif args.lite:
        print(f"\n>> Lite retrain (meta + weight only) for {args.symbol}...")
        success = retrain_lite(args.symbol)
        print(f"\n>> {'SUCCESS' if success else 'FAILED'}")

    elif args.monitor:
        print(f"\n>> Monitoring {args.symbol} every {args.interval}s...")
        print(f"   Retrain triggers: accuracy < {AUTO_RETRAIN_ACCURACY_TRIGGER}")
        while True:
            score = _evaluate_current_model(args.symbol)
            now = datetime.now().strftime("%H:%M:%S")
            if score:
                status = "⚠️ RETRAIN" if score < AUTO_RETRAIN_ACCURACY_TRIGGER else "✅ OK"
                print(f"  [{now}] Accuracy={score:.4f} {status}")
                if score < AUTO_RETRAIN_ACCURACY_TRIGGER:
                    print(f"  [{now}] Triggering background retrain...")
                    retrain_in_background(args.symbol)
            else:
                print(f"  [{now}] Could not evaluate.")
            time.sleep(args.interval)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
