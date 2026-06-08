"""
src/autopilot/budget.py

Tracks Claude credit spend (USD) and Copilot premium-request usage for the
current billing month, persisted in SQLite so it survives restarts.

The router reads BudgetState on every request to decide whether a pool is
healthy enough to route to. This module also writes spend after each call.

All hot-path methods are async (aiosqlite) so they don't block the event loop
under concurrent load. __init__ keeps a synchronous sqlite3 call only for the
one-time schema setup.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

import aiosqlite
import httpx

from autopilot.models import BudgetPool, BudgetSnapshot

logger = logging.getLogger(__name__)


# ── Schema & migrations ────────────────────────────────────────────────────────

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
    was_escalated        INTEGER NOT NULL DEFAULT 0,  -- SQLite bool
    routing_reason       TEXT
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

CREATE TABLE IF NOT EXISTS schema_migrations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT UNIQUE NOT NULL,
    applied_at TEXT NOT NULL
);
"""

# Additive migrations for databases created before the full schema above.
# Each entry is (name, sql). Applied once; recorded in schema_migrations.
_MIGRATIONS: list[tuple[str, str]] = [
    ("001_add_routing_reason", "ALTER TABLE request_log ADD COLUMN routing_reason TEXT"),
]


# ── BudgetState ────────────────────────────────────────────────────────────────

class BudgetState:
    """
    Budget tracker backed by SQLite.

    All public methods are async so they don't block the FastAPI event loop.
    Schema init uses synchronous sqlite3 (runs once at startup before the loop
    is live).

    Usage:
        budget = BudgetState("data/autopilot.db", claude_limit=20.0, copilot_limit=300)
        snap = await budget.snapshot()
        await budget.record_spend(pool=BudgetPool.CLAUDE_CREDIT, cost_usd=0.0012)
        await budget.record_spend(pool=BudgetPool.COPILOT_PREMIUM, premium_requests=3.0)
    """

    def __init__(
        self,
        db_path: str | Path,
        claude_monthly_limit_usd: float = 20.0,
        copilot_monthly_requests_limit: float = 300.0,
        alert_webhook_url: str = "",
        alert_threshold_usd: float | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.claude_limit = claude_monthly_limit_usd
        self.copilot_limit = copilot_monthly_requests_limit
        self._alert_webhook_url = alert_webhook_url or os.environ.get("BUDGET_ALERT_WEBHOOK_URL", "")
        raw_threshold = alert_threshold_usd or float(os.environ.get("BUDGET_ALERT_THRESHOLD_USD", "0"))
        self._alert_threshold_usd = raw_threshold if raw_threshold > 0 else None
        self._last_alert_fired_at_usd: float | None = None
        self._month_end_notified: str | None = None  # month_key of last pre-notification
        self._init_db()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _init_db(self) -> None:
        """One-time synchronous schema setup. Runs before the event loop starts."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
            conn.executescript(_SCHEMA)
            for name, sql in _MIGRATIONS:
                already = conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE name = ?", (name,)
                ).fetchone()
                if already:
                    continue
                try:
                    conn.execute(sql)
                except sqlite3.OperationalError:
                    pass  # column already present in a fresh DB
                conn.execute(
                    "INSERT OR IGNORE INTO schema_migrations (name, applied_at) VALUES (?, ?)",
                    (name, datetime.now().isoformat()),
                )
            conn.commit()
        finally:
            conn.close()

    @asynccontextmanager
    async def _aconn(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            yield conn

    @staticmethod
    def _month_key() -> str:
        return datetime.now().strftime("%Y-%m")

    async def _ensure_month_row(self, conn: aiosqlite.Connection) -> None:
        month = self._month_key()
        await conn.execute(
            "INSERT OR IGNORE INTO budget (month_key, last_updated) VALUES (?, ?)",
            (month, datetime.now().isoformat()),
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    async def snapshot(self) -> BudgetSnapshot:
        """Return the current month's spend across both pools."""
        month = self._month_key()
        async with self._aconn() as conn:
            await self._ensure_month_row(conn)
            await conn.commit()
            cursor = await conn.execute(
                "SELECT claude_spent_usd, copilot_requests FROM budget WHERE month_key = ?",
                (month,),
            )
            row = await cursor.fetchone()
        return BudgetSnapshot(
            claude_spent_usd=row["claude_spent_usd"],
            claude_limit_usd=self.claude_limit,
            copilot_requests_used=row["copilot_requests"],
            copilot_requests_limit=self.copilot_limit,
            month_key=month,
        )

    async def _fire_alert(self, spent_usd: float, limit_usd: float) -> None:
        """POST a JSON alert to the configured webhook URL."""
        if not self._alert_webhook_url:
            return
        payload = {
            "event": "budget_threshold_crossed",
            "claude_spent_usd": round(spent_usd, 4),
            "claude_limit_usd": limit_usd,
            "month": self._month_key(),
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(self._alert_webhook_url, json=payload)
            self._last_alert_fired_at_usd = spent_usd
            logger.warning(
                "budget alert fired",
                extra={"claude_spent_usd": spent_usd, "threshold_usd": self._alert_threshold_usd},
            )
        except Exception as exc:
            logger.error("budget alert webhook failed", extra={"error": str(exc)})

    async def check_month_end_notification(self) -> None:
        """
        Fire a pre-notification webhook if we are within 3 days of month-end
        and some Claude budget has been used this month.
        Fires at most once per month.
        """
        if not self._alert_webhook_url:
            return

        now = datetime.now()
        import calendar
        days_in_month = calendar.monthrange(now.year, now.month)[1]
        days_remaining = days_in_month - now.day
        if days_remaining > 3:
            return

        month = self._month_key()
        if self._month_end_notified == month:
            return

        snap = await self.snapshot()
        if snap.claude_spent_usd <= 0 and snap.copilot_requests_used <= 0:
            return

        payload = {
            "event": "month_end_approaching",
            "days_remaining": days_remaining,
            "claude_spent_usd": round(snap.claude_spent_usd, 4),
            "claude_limit_usd": snap.claude_limit_usd,
            "copilot_requests_used": snap.copilot_requests_used,
            "copilot_requests_limit": snap.copilot_requests_limit,
            "month": month,
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(self._alert_webhook_url, json=payload)
            self._month_end_notified = month
            logger.info("month-end pre-notification sent", extra={"days_remaining": days_remaining, "month": month})
        except Exception as exc:
            logger.error("month-end notification webhook failed", extra={"error": str(exc)})

    async def record_spend(
        self,
        pool: BudgetPool,
        cost_usd: float = 0.0,
        premium_requests: float = 0.0,
    ) -> None:
        """Increment the spend counter for the given pool."""
        month = self._month_key()
        async with self._aconn() as conn:
            await self._ensure_month_row(conn)
            if pool == BudgetPool.CLAUDE_CREDIT:
                await conn.execute(
                    "UPDATE budget SET claude_spent_usd = claude_spent_usd + ?, "
                    "last_updated = ? WHERE month_key = ?",
                    (cost_usd, datetime.now().isoformat(), month),
                )
                if self._alert_threshold_usd is not None:
                    cursor = await conn.execute(
                        "SELECT claude_spent_usd FROM budget WHERE month_key = ?", (month,)
                    )
                    row = await cursor.fetchone()
                    new_spent = row["claude_spent_usd"] if row else 0.0
                    prev_alert = self._last_alert_fired_at_usd or 0.0
                    if new_spent >= self._alert_threshold_usd > prev_alert:
                        await conn.commit()
                        await self._fire_alert(new_spent, self.claude_limit)
                        return
            elif pool == BudgetPool.COPILOT_PREMIUM:
                await conn.execute(
                    "UPDATE budget SET copilot_requests = copilot_requests + ?, "
                    "last_updated = ? WHERE month_key = ?",
                    (premium_requests, datetime.now().isoformat(), month),
                )
            await conn.commit()

    async def log_request(
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
        routing_reason: str | None = None,
    ) -> int:
        """Write one row to the request audit log. Returns the new row id."""
        async with self._aconn() as conn:
            cursor = await conn.execute(
                """INSERT INTO request_log (
                    timestamp, prompt_hash, complexity_tier, router_confidence,
                    backend_id, budget_pool, input_tokens, output_tokens,
                    latency_ms, cost_usd, premium_requests,
                    verifier_score, was_escalated, routing_reason
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    timestamp, prompt_hash, complexity_tier, router_confidence,
                    backend_id, budget_pool.value, input_tokens, output_tokens,
                    latency_ms, cost_usd, premium_requests,
                    verifier_score, int(was_escalated), routing_reason,
                ),
            )
            await conn.commit()
            return cursor.lastrowid

    async def log_verification(
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
        async with self._aconn() as conn:
            await conn.execute(
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
            await conn.commit()

    async def is_pool_healthy(self, pool: BudgetPool, thresholds: dict) -> bool:
        """
        Returns False if a pool is running low per the routing.yaml thresholds.
        Used by the router to avoid routing to a depleted pool.
        """
        if pool == BudgetPool.FREE:
            return True

        snap = await self.snapshot()
        if pool == BudgetPool.CLAUDE_CREDIT:
            floor = thresholds.get("claude_usd_remaining", 5.0)
            return snap.claude_remaining_usd >= floor

        if pool == BudgetPool.COPILOT_PREMIUM:
            floor = thresholds.get("copilot_requests_remaining", 30)
            return snap.copilot_remaining_requests >= floor

        return True
