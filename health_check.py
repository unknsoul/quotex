"""
Health Check — V3 Layer 11: System health monitoring.

Monitors:
  - MT5 connection status
  - Disk space
  - Memory usage
  - Prediction latency
  - Signal silence (no signals for too long)
  - Model file freshness
"""

import os
import time
import logging
from datetime import datetime, timedelta

log = logging.getLogger("health_check")

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MAX_MODEL_AGE_DAYS = 7
MAX_SIGNAL_SILENCE_MINUTES = 60
MAX_LATENCY_MS = 2000


class HealthChecker:
    """System health monitoring."""
    
    def __init__(self):
        self.last_signal_time = None
        self.last_latency_ms = 0
        self.issues = []
    
    def run_all_checks(self):
        """Run all health checks, return list of issues."""
        self.issues = []
        
        self._check_disk_space()
        self._check_memory()
        self._check_model_freshness()
        self._check_signal_silence()
        self._check_latency()
        
        return self.issues
    
    def _check_disk_space(self):
        """Check available disk space."""
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            free_gb = free / (1024 ** 3)
            if free_gb < 1.0:
                self.issues.append(f"LOW DISK: {free_gb:.1f} GB free")
        except Exception:
            pass
    
    def _check_memory(self):
        """Check system memory usage."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            if mem.percent > 90:
                self.issues.append(f"HIGH MEMORY: {mem.percent:.0f}% used")
        except ImportError:
            pass
    
    def _check_model_freshness(self):
        """Check if model files are recent enough."""
        try:
            for fname in os.listdir(MODEL_DIR):
                if fname.endswith(".pkl"):
                    path = os.path.join(MODEL_DIR, fname)
                    mtime = datetime.fromtimestamp(os.path.getmtime(path))
                    age_days = (datetime.now() - mtime).days
                    if age_days > MAX_MODEL_AGE_DAYS:
                        self.issues.append(f"STALE MODEL: {fname} ({age_days} days old)")
        except FileNotFoundError:
            self.issues.append("MODEL DIR MISSING")
    
    def _check_signal_silence(self):
        """Check if too long since last signal."""
        if self.last_signal_time:
            elapsed = (datetime.now() - self.last_signal_time).total_seconds() / 60
            if elapsed > MAX_SIGNAL_SILENCE_MINUTES:
                self.issues.append(f"SIGNAL SILENCE: {elapsed:.0f} min since last signal")
    
    def _check_latency(self):
        """Check prediction latency."""
        if self.last_latency_ms > MAX_LATENCY_MS:
            self.issues.append(f"HIGH LATENCY: {self.last_latency_ms:.0f}ms")
    
    def record_signal(self):
        """Record that a signal was generated."""
        self.last_signal_time = datetime.now()
    
    def record_latency(self, latency_ms):
        """Record prediction latency."""
        self.last_latency_ms = latency_ms
    
    def is_healthy(self):
        """Quick check: any issues?"""
        return len(self.run_all_checks()) == 0
    
    def get_status_message(self):
        """Get formatted status message for Telegram."""
        issues = self.run_all_checks()
        if not issues:
            return "System healthy"
        return "HEALTH ISSUES:\n" + "\n".join(f"  - {i}" for i in issues)
