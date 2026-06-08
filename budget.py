"""
src/autopilot/budget.py

Tracks Claude credit spend (USD) and Copilot premium-request usage for the
current billing month, persisted in SQLite so it survives restarts.

The router reads BudgetState on every request to decide whether a pool is
healthy enough to route to. This module also writes spend after each call.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

from models import BudgetPool, BudgetSnapshot


# ── Schema ─────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS budget (
    month_key            TEXT PRIMARY KEY,   -- "YYYY-MM"
    claude_spent_usd     REAL NOT NULL DEFAULT 0.0,
    copilot_requests     REAL NOT NULL DEFAULT 0.0,
    last_updated         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS request_log (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp            REAL NOT NULL,
    prompt_hash          TEXT NOT NULL,
    complexity_tier      INTEGER,
    router_confidence    REAL,
    backend_id           TEXT NOT NULL,
    budget_pool          TEXT NOT NULL,
    input_tokens         INTEGER NOT NULL DEFAULT 0,
    output_tokens        INTEGER NOT NULL DEFAULT 0,
    latency_ms           REAL NOT NULL DEFAULT 0.0,
    cost_usd             REAL NOT NULL DEFAULT 0.0,
    premium_requests     REAL NOT NULL DEFAULT 0.0,
    verifier_score       REAL,
    was_escalated        INTEGER NOT NULL DEFAULT 0   -- SQLite bool
);

CREATE TABLE IF NOT EXISTS verification_log (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    request_log_id          INTEGER NOT NULL REFERENCES request_log(id),
    verified_at             REAL NOT NULL,
    original_backend_id     TEXT NOT NULL,
    verifier_backend_id     TEXT NOT NULL,
    agreement_score         REAL NOT NULL,       -- 0.0–1.0
    is_mis_route            INTEGER NOT NULL DEFAULT 0,
    added_to_training       INTEGER NOT NULL DEFAULT 0,
    original_tier           INTEGER,
    corrected_tier          INTEGER              -- set when is_mis_route=1
);
"""


# ── BudgetState ────────────────────────────────────────────────────────────────

class BudgetState:
    """
    Thread-safe budget tracker backed by SQLite.

    Usage:
        budget = BudgetState("data/autopilot.db", claude_limit=20.0, copilot_limit=300)
        snap = budget.snapshot()
        budget.record_spend(pool=BudgetPool.CLAUDE_CREDIT, cost_usd=0.0012)
        budget.record_spend(pool=BudgetPool.COPILOT_PREMIUM, premium_requests=3.0)
    """

    def __init__(
        self,
        db_path: str | Path,
        claude_monthly_limit_usd: float = 20.0,
        copilot_monthly_requests_limit: float = 300.0,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.claude_limit = claude_monthly_limit_usd
        self.copilot_limit = copilot_monthly_requests_limit
        self._init_db()

    # ── Internal helpers ───────────────────────────────────────────────────────

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @staticmethod
    def _month_key() -> str:
        return datetime.now().strftime("%Y-%m")

    def _ensure_month_row(self, conn: sqlite3.Connection) -> None:
        month = self._month_key()
        conn.execute(
            "INSERT OR IGNORE INTO budget (month_key, last_updated) VALUES (?, ?)",
            (month, datetime.now().isoformat()),
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def snapshot(self) -> BudgetSnapshot:
        """Return the current month's spend across both pools."""
        month = self._month_key()
        with self._conn() as conn:
            self._ensure_month_row(conn)
            row = conn.execute(
                "SELECT claude_spent_usd, copilot_requests FROM budget WHERE month_key = ?",
                (month,),
            ).fetchone()
        return BudgetSnapshot(
            claude_spent_usd=row["claude_spent_usd"],
            claude_limit_usd=self.claude_limit,
            copilot_requests_used=row["copilot_requests"],
            copilot_requests_limit=self.copilot_limit,
            month_key=month,
        )

    def record_spend(
        self,
        pool: BudgetPool,
        cost_usd: float = 0.0,
        premium_requests: float = 0.0,
    ) -> None:
        """Increment the spend counter for the given pool."""
        month = self._month_key()
        with self._conn() as conn:
            self._ensure_month_row(conn)
            if pool == BudgetPool.CLAUDE_CREDIT:
                conn.execute(
                    "UPDATE budget SET claude_spent_usd = claude_spent_usd + ?, "
                    "last_updated = ? WHERE month_key = ?",
                    (cost_usd, datetime.now().isoformat(), month),
                )
            elif pool == BudgetPool.COPILOT_PREMIUM:
                conn.execute(
                    "UPDATE budget SET copilot_requests = copilot_requests + ?, "
                    "last_updated = ? WHERE month_key = ?",
                    (premium_requests, datetime.now().isoformat(), month),
                )
            # BudgetPool.FREE: nothing to track

    def log_request(
        self,
        *,
        timestamp: float,
        prompt_hash: str,
        complexity_tier: int | None,
        router_confidence: float | None,
        backend_id: str,
        budget_pool: BudgetPool,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cost_usd: float,
        premium_requests: float,
        verifier_score: float | None = None,
        was_escalated: bool = False,
    ) -> int:
        """Write one row to the request audit log. Returns the new row id."""
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO request_log (
                    timestamp, prompt_hash, complexity_tier, router_confidence,
                    backend_id, budget_pool, input_tokens, output_tokens,
                    latency_ms, cost_usd, premium_requests,
                    verifier_score, was_escalated
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    timestamp, prompt_hash, complexity_tier, router_confidence,
                    backend_id, budget_pool.value, input_tokens, output_tokens,
                    latency_ms, cost_usd, premium_requests,
                    verifier_score, int(was_escalated),
                ),
            )
            return cur.lastrowid

    def log_verification(
        self,
        *,
        request_log_id: int,
        verified_at: float,
        original_backend_id: str,
        verifier_backend_id: str,
        agreement_score: float,
        is_mis_route: bool,
        added_to_training: bool,
        original_tier: int | None = None,
        corrected_tier: int | None = None,
    ) -> None:
        """Write one row to the verification audit log."""
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO verification_log (
                    request_log_id, verified_at, original_backend_id,
                    verifier_backend_id, agreement_score, is_mis_route,
                    added_to_training, original_tier, corrected_tier
                ) VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    request_log_id, verified_at, original_backend_id,
                    verifier_backend_id, agreement_score, int(is_mis_route),
                    int(added_to_training), original_tier, corrected_tier,
                ),
            )

    def is_pool_healthy(self, pool: BudgetPool, thresholds: dict) -> bool:
        """
        Returns False if a pool is running low per the routing.yaml thresholds.
        Used by the router to avoid routing to a depleted pool.
        """
        if pool == BudgetPool.FREE:
            return True   # local is always "healthy"

        snap = self.snapshot()
        if pool == BudgetPool.CLAUDE_CREDIT:
            floor = thresholds.get("claude_usd_remaining", 5.0)
            return snap.claude_remaining_usd >= floor

        if pool == BudgetPool.COPILOT_PREMIUM:
            floor = thresholds.get("copilot_requests_remaining", 30)
            return snap.copilot_remaining_requests >= floor

        return True
