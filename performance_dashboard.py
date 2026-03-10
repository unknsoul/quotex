"""
Performance Dashboard — Daily JSON report with rolling metrics.

Generates dashboard data from signal_db:
  - Win rate by symbol, regime, session
  - Rolling accuracy over last N trades
  - Today's P&L summary
  - Confidence calibration data
"""

import json
import logging
import os
from datetime import datetime, timezone

from config import DASHBOARD_DIR, DASHBOARD_ROLLING_WINDOW
from signal_db import SignalDB

log = logging.getLogger("performance_dashboard")


def _ensure_dir():
    os.makedirs(DASHBOARD_DIR, exist_ok=True)


def generate_dashboard(db: SignalDB = None) -> dict:
    """Generate a full dashboard snapshot and save as JSON."""
    if db is None:
        db = SignalDB()

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date": today,
        "today_stats": db.get_today_stats(),
        "rolling_accuracy": db.get_rolling_accuracy(DASHBOARD_ROLLING_WINDOW),
        "symbol_stats": db.get_symbol_stats(),
        "regime_stats": db.get_regime_stats(),
        "session_stats": db.get_session_stats(),
    }

    _ensure_dir()
    path = os.path.join(DASHBOARD_DIR, f"dashboard_{today}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Dashboard saved: %s", path)
    return report


def format_dashboard_text(report: dict) -> str:
    """Format dashboard dict as readable text for Telegram /dashboard command."""
    lines = ["📊 PERFORMANCE DASHBOARD", ""]

    today = report.get("today_stats", {})
    wins = today.get("wins", 0)
    losses = today.get("losses", 0)
    total = wins + losses
    wr = today.get("win_rate", 0)
    lines.append(f"Today: {wins}W / {losses}L ({wr:.0%}) | Pending: {today.get('pending', 0)}")

    rolling = report.get("rolling_accuracy", 0.5)
    n = DASHBOARD_ROLLING_WINDOW
    lines.append(f"Rolling {n}: {rolling:.1%}")
    lines.append("")

    # Symbol breakdown
    sym_stats = report.get("symbol_stats", {})
    if sym_stats:
        lines.append("By Symbol:")
        for sym, s in sorted(sym_stats.items()):
            lines.append(f"  {sym}: {s['wins']}/{s['total']} ({s['win_rate']:.0%})")
        lines.append("")

    # Regime breakdown
    regime_stats = report.get("regime_stats", {})
    if regime_stats:
        lines.append("By Regime:")
        for reg, s in sorted(regime_stats.items()):
            lines.append(f"  {reg}: {s['wins']}/{s['total']} ({s['win_rate']:.0%})")
        lines.append("")

    # Session breakdown
    sess_stats = report.get("session_stats", {})
    if sess_stats:
        lines.append("By Session:")
        for sess, s in sorted(sess_stats.items()):
            lines.append(f"  {sess}: {s['wins']}/{s['total']} ({s['win_rate']:.0%})")

    return "\n".join(lines)
