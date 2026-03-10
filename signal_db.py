"""
Signal Database — v10+: SQLite storage for all signals and outcomes.

Stores every signal dispatched with full context:
  - Features, gate results, model probabilities
  - Signal strength, reason chain, quality score
  - Outcome (win/loss), P&L
  - Queryable by symbol, regime, session for analysis and dashboards
"""

import sqlite3
import json
import os
import logging
from datetime import datetime, timezone

log = logging.getLogger("signal_db")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signals.db")

# New columns added in v10+ upgrade — used for safe schema migration
_V10_EXTRA_COLUMNS = [
    ("signal_strength", "TEXT DEFAULT ''"),
    ("reason_chain", "TEXT DEFAULT ''"),
    ("quality_score", "REAL DEFAULT 0.0"),
    ("unanimity", "REAL DEFAULT 0.0"),
    ("ensemble_variance", "REAL DEFAULT 0.0"),
]


class SignalDB:
    """SQLite database for signal storage and analysis."""

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist, then migrate schema."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                probability REAL,
                confidence REAL,
                meta_score REAL,
                regime TEXT,
                session TEXT,
                ensemble_std REAL,
                gates_passed INTEGER,
                gates_total INTEGER,
                gate_details TEXT,
                kelly_fraction REAL,
                outcome TEXT DEFAULT 'pending',
                pnl REAL DEFAULT 0.0,
                was_correct INTEGER DEFAULT -1,
                features_json TEXT,
                model_version TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                signal_strength TEXT DEFAULT '',
                reason_chain TEXT DEFAULT '',
                quality_score REAL DEFAULT 0.0,
                unanimity REAL DEFAULT 0.0,
                ensemble_variance REAL DEFAULT 0.0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_regime ON signals(regime)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_session ON signals(session)")
        conn.commit()

        # Safe migration: add new columns if they don't exist (for existing DBs)
        existing = {r[1] for r in conn.execute("PRAGMA table_info(signals)").fetchall()}
        for col_name, col_def in _V10_EXTRA_COLUMNS:
            if col_name not in existing:
                conn.execute(f"ALTER TABLE signals ADD COLUMN {col_name} {col_def}")
                log.info("Migrated signal_db: added column %s", col_name)
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    #  Insert / Update
    # ------------------------------------------------------------------

    def insert_signal(self, symbol, direction, probability, confidence,
                      meta_score=0.0, regime="", session="",
                      ensemble_std=0.0, gates_passed=0, gates_total=10,
                      gate_details=None, kelly_fraction=0.0,
                      features=None, model_version="v10",
                      signal_strength="", reason_chain="",
                      quality_score=0.0, unanimity=0.0,
                      ensemble_variance=0.0):
        """Insert a new signal record."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO signals (timestamp, symbol, direction, probability,
                                 confidence, meta_score, regime, session,
                                 ensemble_std, gates_passed, gates_total,
                                 gate_details, kelly_fraction,
                                 features_json, model_version,
                                 signal_strength, reason_chain,
                                 quality_score, unanimity, ensemble_variance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(), symbol, direction, probability,
            confidence, meta_score, regime, session,
            ensemble_std, gates_passed, gates_total,
            json.dumps(gate_details or {}),
            kelly_fraction,
            json.dumps(features or {}),
            model_version,
            signal_strength,
            reason_chain if isinstance(reason_chain, str) else json.dumps(reason_chain),
            quality_score, unanimity, ensemble_variance,
        ))
        signal_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.commit()
        conn.close()
        return signal_id

    def update_outcome(self, signal_id, was_correct, pnl=0.0):
        """Update signal with outcome."""
        conn = sqlite3.connect(self.db_path)
        outcome = "win" if was_correct else "loss"
        conn.execute("""
            UPDATE signals SET outcome=?, was_correct=?, pnl=?
            WHERE id=?
        """, (outcome, int(was_correct), pnl, signal_id))
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    #  Query helpers
    # ------------------------------------------------------------------

    def get_win_rate(self, symbol=None, regime=None, session=None, n_last=100):
        """Get win rate with optional filters."""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT was_correct FROM signals WHERE was_correct >= 0"
        params = []
        if symbol:
            query += " AND symbol=?"
            params.append(symbol)
        if regime:
            query += " AND regime=?"
            params.append(regime)
        if session:
            query += " AND session=?"
            params.append(session)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(n_last)

        rows = conn.execute(query, params).fetchall()
        conn.close()

        if not rows:
            return 0.5
        return sum(r[0] for r in rows) / len(rows)

    def get_recent_signals(self, n=20):
        """Get most recent signals."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT id, timestamp, symbol, direction, probability, confidence,
                   regime, outcome, was_correct, pnl
            FROM signals ORDER BY id DESC LIMIT ?
        """, (n,)).fetchall()
        conn.close()
        return rows

    def get_regime_stats(self):
        """Get win rate breakdown by regime."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT regime, COUNT(*) as total,
                   SUM(CASE WHEN was_correct=1 THEN 1 ELSE 0 END) as wins
            FROM signals WHERE was_correct >= 0
            GROUP BY regime
        """).fetchall()
        conn.close()
        return {r[0]: {"total": r[1], "wins": r[2],
                "win_rate": r[2]/r[1] if r[1] > 0 else 0} for r in rows}

    def get_session_stats(self):
        """Get win rate breakdown by session."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT session, COUNT(*) as total,
                   SUM(CASE WHEN was_correct=1 THEN 1 ELSE 0 END) as wins
            FROM signals WHERE was_correct >= 0
            GROUP BY session
        """).fetchall()
        conn.close()
        return {r[0]: {"total": r[1], "wins": r[2],
                "win_rate": r[2]/r[1] if r[1] > 0 else 0} for r in rows}

    def get_today_stats(self):
        """Get today's signal stats (total, wins, losses, win_rate)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN was_correct=1 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN was_correct=0 THEN 1 ELSE 0 END) as losses,
                   SUM(CASE WHEN was_correct=-1 THEN 1 ELSE 0 END) as pending
            FROM signals WHERE timestamp LIKE ?
        """, (f"{today}%",)).fetchone()
        conn.close()
        total, wins, losses, pending = row
        wins = wins or 0
        losses = losses or 0
        pending = pending or 0
        return {
            "total": total,
            "wins": wins,
            "losses": losses,
            "pending": pending,
            "win_rate": wins / max(wins + losses, 1),
        }

    def get_rolling_accuracy(self, n=50):
        """Get rolling accuracy of the last n resolved signals."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT was_correct FROM signals
            WHERE was_correct >= 0
            ORDER BY id DESC LIMIT ?
        """, (n,)).fetchall()
        conn.close()
        if not rows:
            return 0.5
        return sum(r[0] for r in rows) / len(rows)

    def get_signals_by_session(self, session_name, n_last=50):
        """Get recent signals for a specific session."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT id, timestamp, symbol, direction, probability, confidence,
                   regime, outcome, was_correct, pnl, signal_strength
            FROM signals WHERE session=?
            ORDER BY id DESC LIMIT ?
        """, (session_name, n_last)).fetchall()
        conn.close()
        return rows

    def get_symbol_stats(self):
        """Get win rate breakdown by symbol."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT symbol, COUNT(*) as total,
                   SUM(CASE WHEN was_correct=1 THEN 1 ELSE 0 END) as wins,
                   AVG(confidence) as avg_confidence,
                   AVG(quality_score) as avg_quality
            FROM signals WHERE was_correct >= 0
            GROUP BY symbol
        """).fetchall()
        conn.close()
        return {r[0]: {"total": r[1], "wins": r[2],
                "win_rate": r[2]/r[1] if r[1] > 0 else 0,
                "avg_confidence": r[3] or 0,
                "avg_quality": r[4] or 0} for r in rows}

    def get_duplicate_check(self, symbol, direction, window_sec=300):
        """Check if a duplicate signal exists within the time window."""
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=window_sec)).isoformat()
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("""
            SELECT COUNT(*) FROM signals
            WHERE symbol=? AND direction=? AND timestamp > ?
        """, (symbol, direction, cutoff)).fetchone()
        conn.close()
        return row[0] > 0

    def count_signals_last_hour(self):
        """Count signals dispatched in the last hour."""
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT COUNT(*) FROM signals WHERE timestamp > ?", (cutoff,)
        ).fetchone()
        conn.close()
        return row[0]
