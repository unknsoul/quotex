"""
Signal Database — V3 Layer 11: SQLite storage for all signals and outcomes.

Stores every signal dispatched with full context:
  - Features, gate results, model probabilities
  - Outcome (win/loss), P&L
  - Queryable for analysis, error attribution, pair scoring
"""

import sqlite3
import json
import os
import logging
from datetime import datetime

log = logging.getLogger("signal_db")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signals.db")


class SignalDB:
    """SQLite database for signal storage and analysis."""
    
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Create tables if they don't exist."""
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
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_regime ON signals(regime)
        """)
        conn.commit()
        conn.close()
    
    def insert_signal(self, symbol, direction, probability, confidence,
                       meta_score=0.0, regime="", session="",
                       ensemble_std=0.0, gates_passed=0, gates_total=10,
                       gate_details=None, kelly_fraction=0.0,
                       features=None, model_version="v3"):
        """Insert a new signal record."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO signals (timestamp, symbol, direction, probability,
                                  confidence, meta_score, regime, session,
                                  ensemble_std, gates_passed, gates_total,
                                  gate_details, kelly_fraction,
                                  features_json, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(), symbol, direction, probability,
            confidence, meta_score, regime, session,
            ensemble_std, gates_passed, gates_total,
            json.dumps(gate_details or {}),
            kelly_fraction,
            json.dumps(features or {}),
            model_version,
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
    
    def get_win_rate(self, symbol=None, regime=None, n_last=100):
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
