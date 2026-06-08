"""
dashboard_data.py

All SQL queries that power the Phase 4 dashboard (and Phase 5 /v1/stats endpoint).
Returns plain dicts / lists — no Streamlit dependency, fully testable.

The "what-if all Claude Opus" baseline uses a fixed reference price so the
savings number is stable and meaningful as a portfolio metric.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

# Reference price for "what-if everything went to a premium model" comparison.
# Using claude-opus-4 public pricing as the baseline — update if pricing changes.
BASELINE_COST_PER_INPUT_TOKEN  = 0.000015   # $15 / 1M input tokens
BASELINE_COST_PER_OUTPUT_TOKEN = 0.000075   # $75 / 1M output tokens


@contextmanager
def _conn(db_path: str | Path) -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _month_key() -> str:
    return datetime.now().strftime("%Y-%m")


# ── Headline metrics ───────────────────────────────────────────────────────────

def get_headline_metrics(db_path: str | Path, month: str | None = None) -> dict:
    """
    Returns the top-line numbers shown in the hero section of the dashboard.

    Keys:
        total_requests, free_requests, free_pct,
        copilot_requests_used, claude_spent_usd,
        actual_cost_usd, baseline_cost_usd, savings_usd, savings_pct,
        escalation_rate, avg_confidence, avg_latency_ms
    """
    month = month or _month_key()

    with _conn(db_path) as conn:
        # Filter to current month using timestamp range
        rows = conn.execute(
            """
            SELECT
                COUNT(*)                                    AS total,
                SUM(CASE WHEN budget_pool='FREE' THEN 1 ELSE 0 END) AS free_count,
                SUM(cost_usd)                               AS actual_cost,
                SUM(premium_requests)                       AS copilot_used,
                SUM(input_tokens)                           AS total_input,
                SUM(output_tokens)                          AS total_output,
                AVG(router_confidence)                      AS avg_conf,
                AVG(latency_ms)                             AS avg_latency,
                SUM(was_escalated)                          AS escalated
            FROM request_log
            WHERE strftime('%Y-%m', datetime(timestamp, 'unixepoch')) = ?
            """,
            (month,),
        ).fetchone()

        claude_row = conn.execute(
            "SELECT claude_spent_usd FROM budget WHERE month_key = ?",
            (month,),
        ).fetchone()

    total        = rows["total"] or 0
    free_count   = rows["free_count"] or 0
    actual_cost  = (rows["actual_cost"] or 0.0) + (claude_row["claude_spent_usd"] if claude_row else 0.0)
    total_input  = rows["total_input"] or 0
    total_output = rows["total_output"] or 0
    escalated    = rows["escalated"] or 0

    baseline = (
        total_input  * BASELINE_COST_PER_INPUT_TOKEN +
        total_output * BASELINE_COST_PER_OUTPUT_TOKEN
    )
    savings     = max(0.0, baseline - actual_cost)
    savings_pct = (savings / baseline * 100) if baseline > 0 else 0.0

    return {
        "total_requests":      total,
        "free_requests":       free_count,
        "free_pct":            (free_count / total * 100) if total else 0.0,
        "copilot_requests_used": rows["copilot_used"] or 0.0,
        "claude_spent_usd":    claude_row["claude_spent_usd"] if claude_row else 0.0,
        "actual_cost_usd":     actual_cost,
        "baseline_cost_usd":   baseline,
        "savings_usd":         savings,
        "savings_pct":         savings_pct,
        "escalation_rate":     (escalated / total * 100) if total else 0.0,
        "avg_confidence":      rows["avg_conf"] or 0.0,
        "avg_latency_ms":      rows["avg_latency"] or 0.0,
    }


# ── Budget burn-down ───────────────────────────────────────────────────────────

def get_budget_status(
    db_path: str | Path,
    claude_limit_usd: float = 20.0,
    copilot_limit_requests: float = 300.0,
    month: str | None = None,
) -> dict:
    """
    Returns remaining budget for both pools and projected end-of-month usage.

    Keys:
        claude_spent, claude_limit, claude_remaining, claude_pct,
        copilot_used, copilot_limit, copilot_remaining, copilot_pct,
        claude_projected_eom, copilot_projected_eom,
        days_elapsed, days_in_month
    """
    month = month or _month_key()
    now   = datetime.now()
    days_in_month  = (datetime(now.year, now.month % 12 + 1, 1) - datetime(now.year, now.month, 1)).days if now.month < 12 else 31
    days_elapsed   = max(now.day, 1)

    with _conn(db_path) as conn:
        row = conn.execute(
            "SELECT claude_spent_usd, copilot_requests FROM budget WHERE month_key = ?",
            (month,),
        ).fetchone()

    claude_spent  = row["claude_spent_usd"]  if row else 0.0
    copilot_used  = row["copilot_requests"]  if row else 0.0

    # Linear projection to end of month
    rate = days_elapsed / days_in_month
    claude_proj  = (claude_spent  / rate) if rate > 0 else 0.0
    copilot_proj = (copilot_used  / rate) if rate > 0 else 0.0

    return {
        "claude_spent":        claude_spent,
        "claude_limit":        claude_limit_usd,
        "claude_remaining":    max(0.0, claude_limit_usd - claude_spent),
        "claude_pct":          min(100.0, claude_spent / claude_limit_usd * 100) if claude_limit_usd else 0.0,
        "claude_projected_eom": claude_proj,
        "copilot_used":        copilot_used,
        "copilot_limit":       copilot_limit_requests,
        "copilot_remaining":   max(0.0, copilot_limit_requests - copilot_used),
        "copilot_pct":         min(100.0, copilot_used / copilot_limit_requests * 100) if copilot_limit_requests else 0.0,
        "copilot_projected_eom": copilot_proj,
        "days_elapsed":        days_elapsed,
        "days_in_month":       days_in_month,
    }


# ── Routing distribution ───────────────────────────────────────────────────────

def get_routing_distribution(db_path: str | Path, month: str | None = None) -> list[dict]:
    """
    Returns per-backend request counts for the pie chart.
    Each row: {backend_id, budget_pool, count, pct}
    """
    month = month or _month_key()
    with _conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT backend_id, budget_pool, COUNT(*) AS cnt
            FROM request_log
            WHERE strftime('%Y-%m', datetime(timestamp, 'unixepoch')) = ?
            GROUP BY backend_id, budget_pool
            ORDER BY cnt DESC
            """,
            (month,),
        ).fetchall()

    total = sum(r["cnt"] for r in rows)
    return [
        {
            "backend_id":  r["backend_id"],
            "budget_pool": r["budget_pool"],
            "count":       r["cnt"],
            "pct":         r["cnt"] / total * 100 if total else 0.0,
        }
        for r in rows
    ]


# ── Quality score distribution ────────────────────────────────────────────────

def get_quality_distribution(db_path: str | Path, month: str | None = None) -> dict:
    """
    Returns verifier score buckets and mis-route stats.
    Keys: scored_count, avg_score, mis_route_count, mis_route_pct,
          training_examples_added, score_buckets (list of {label, count})
    """
    month = month or _month_key()
    with _conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                COUNT(*)                                            AS scored,
                AVG(agreement_score)                                AS avg_score,
                SUM(is_mis_route)                                   AS mis_routes,
                SUM(added_to_training)                              AS training_added,
                SUM(CASE WHEN agreement_score >= 0.75 THEN 1 ELSE 0 END) AS good,
                SUM(CASE WHEN agreement_score >= 0.50
                         AND agreement_score <  0.75 THEN 1 ELSE 0 END) AS degraded,
                SUM(CASE WHEN agreement_score <  0.50 THEN 1 ELSE 0 END) AS poor
            FROM verification_log vl
            JOIN request_log rl ON vl.request_log_id = rl.id
            WHERE strftime('%Y-%m', datetime(rl.timestamp, 'unixepoch')) = ?
            """,
            (month,),
        ).fetchone()

    scored = rows["scored"] or 0
    return {
        "scored_count":           scored,
        "avg_score":              rows["avg_score"] or 0.0,
        "mis_route_count":        rows["mis_routes"] or 0,
        "mis_route_pct":          (rows["mis_routes"] / scored * 100) if scored else 0.0,
        "training_examples_added": rows["training_added"] or 0,
        "score_buckets": [
            {"label": "Good (≥0.75)",      "count": rows["good"]     or 0},
            {"label": "Degraded (0.5–0.75)", "count": rows["degraded"] or 0},
            {"label": "Poor (<0.5)",        "count": rows["poor"]     or 0},
        ],
    }


# ── Requests over time ─────────────────────────────────────────────────────────

def get_requests_over_time(db_path: str | Path, month: str | None = None) -> list[dict]:
    """
    Daily request counts split by budget pool for the line chart.
    Each row: {date, FREE, COPILOT_PREMIUM, CLAUDE_CREDIT, total}
    """
    month = month or _month_key()
    with _conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                date(timestamp, 'unixepoch')    AS day,
                budget_pool,
                COUNT(*)                        AS cnt
            FROM request_log
            WHERE strftime('%Y-%m', datetime(timestamp, 'unixepoch')) = ?
            GROUP BY day, budget_pool
            ORDER BY day
            """,
            (month,),
        ).fetchall()

    # Pivot into one row per day
    days: dict[str, dict] = {}
    for r in rows:
        d = r["day"]
        if d not in days:
            days[d] = {"date": d, "FREE": 0, "COPILOT_PREMIUM": 0, "CLAUDE_CREDIT": 0}
        days[d][r["budget_pool"]] = r["cnt"]

    result = []
    for row in days.values():
        row["total"] = row["FREE"] + row["COPILOT_PREMIUM"] + row["CLAUDE_CREDIT"]
        result.append(row)
    return result


# ── Escalation rate over time ─────────────────────────────────────────────────

def get_escalation_trend(db_path: str | Path, month: str | None = None) -> list[dict]:
    """
    Daily escalation rate (%) for the trend line.
    Each row: {date, total, escalated, rate_pct}
    """
    month = month or _month_key()
    with _conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                date(timestamp, 'unixepoch') AS day,
                COUNT(*)                     AS total,
                SUM(was_escalated)           AS escalated
            FROM request_log
            WHERE strftime('%Y-%m', datetime(timestamp, 'unixepoch')) = ?
            GROUP BY day
            ORDER BY day
            """,
            (month,),
        ).fetchall()

    return [
        {
            "date":      r["day"],
            "total":     r["total"],
            "escalated": r["escalated"] or 0,
            "rate_pct":  (r["escalated"] / r["total"] * 100) if r["total"] else 0.0,
        }
        for r in rows
    ]


# ── Recent requests table ─────────────────────────────────────────────────────

def get_recent_requests(db_path: str | Path, limit: int = 50) -> list[dict]:
    """Last N requests for the live feed table."""
    with _conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                id, datetime(timestamp,'unixepoch','localtime') AS ts,
                complexity_tier, router_confidence, backend_id,
                budget_pool, input_tokens, output_tokens,
                latency_ms, cost_usd, premium_requests,
                verifier_score, was_escalated
            FROM request_log
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
