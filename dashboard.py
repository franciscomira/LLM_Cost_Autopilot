"""
dashboard.py

Phase 4 — Streamlit cost dashboard.

Run:
    streamlit run dashboard.py

Auto-refreshes every 30 seconds. Pass --db to use a different SQLite file:
    streamlit run dashboard.py -- --db data/autopilot.db
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LLM Cost Autopilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://127.0.0.1:8000"

# Parse --db argument passed after `--` in streamlit run
def _parse_db_path() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--db", default="data/autopilot.db")
    args, _ = parser.parse_known_args()
    return Path(args.db)

DB_PATH = _parse_db_path()

# ── Seed demo data when DB is empty ───────────────────────────────────────────

def _seed_demo_data(db_path: Path) -> None:
    """Insert realistic-looking demo rows so the dashboard looks alive on first run."""
    import random, time as _time
    from budget import BudgetState
    from models import BudgetPool

    budget = BudgetState(db_path)

    # Check if already seeded
    import sqlite3
    with sqlite3.connect(db_path) as c:
        count = c.execute("SELECT COUNT(*) FROM request_log").fetchone()[0]
    if count > 0:
        return

    random.seed(42)
    now = _time.time()
    pools = [
        ("ollama_tier1",  "FREE",             0, 0, 0.0,     0.0),
        ("ollama_tier1",  "FREE",             0, 0, 0.0,     0.0),
        ("ollama_tier1",  "FREE",             0, 0, 0.0,     0.0),
        ("copilot_mid",   "COPILOT_PREMIUM",  0, 0, 0.0,     1.0),
        ("copilot_top",   "COPILOT_PREMIUM",  0, 0, 0.0,     3.0),
        ("claude",        "CLAUDE_CREDIT",    0, 0, 0.00015, 0.0),
    ]

    backend_tiers = {
        "ollama_tier1": 1, "copilot_mid": 2,
        "copilot_top": 3, "claude": 3,
    }

    for i in range(120):
        day_offset = random.randint(0, 6) * 86400
        ts = now - day_offset - random.randint(0, 86400)
        backend_id, pool, _, _, cost_per, prem = random.choices(
            pools, weights=[40, 40, 20, 25, 10, 5], k=1
        )[0]
        inp  = random.randint(50, 800)
        outp = random.randint(20, 400)
        cost = inp * cost_per + outp * cost_per * 5 if cost_per else 0.0
        tier = backend_tiers[backend_id]
        conf = random.uniform(2.5, 5.0)
        escalated = random.random() < 0.08

        from models import BudgetPool as BP
        pool_enum = BP(pool)
        asyncio.run(budget.record_spend(pool=pool_enum, cost_usd=cost, premium_requests=prem if prem else 0.0))
        asyncio.run(budget.log_request(
            timestamp=ts,
            prompt_hash=f"demo{i:04d}",
            complexity_tier=tier,
            router_confidence=conf,
            backend_id=backend_id,
            budget_pool=pool_enum,
            input_tokens=inp,
            output_tokens=outp,
            latency_ms=random.uniform(300, 18000),
            cost_usd=cost,
            premium_requests=prem if prem else 0.0,
            was_escalated=escalated,
        ))


# Seed on first load if DB has no data
if DB_PATH.exists() or True:
    try:
        _seed_demo_data(DB_PATH)
    except Exception:
        pass

# ── Data loading (cached, refreshes every 30s) ────────────────────────────────

from dashboard_data import (
    get_budget_status,
    get_escalation_trend,
    get_headline_metrics,
    get_quality_distribution,
    get_recent_requests,
    get_requests_over_time,
    get_routing_distribution,
)

@st.cache_data(ttl=30)
def load_all(db: str, month: str) -> dict:
    return {
        "headline":     get_headline_metrics(db, month),
        "budget":       get_budget_status(db, month=month),
        "routing":      get_routing_distribution(db, month),
        "quality":      get_quality_distribution(db, month),
        "over_time":    get_requests_over_time(db, month),
        "escalation":   get_escalation_trend(db, month),
        "recent":       get_recent_requests(db, limit=50),
    }

# ── Sidebar ────────────────────────────────────────────────────────────────────

from datetime import datetime
import yaml

with st.sidebar:
    st.title("⚙️ Controls")

    # Month picker
    available_months = [datetime.now().strftime("%Y-%m")]
    month = st.selectbox("Month", available_months)

    # Budget limits (editable)
    st.divider()
    st.subheader("Budget limits")
    cfg_path = Path(__file__).parent / "routing.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    claude_limit   = st.number_input("Claude limit ($)",    value=float(cfg["budgets"]["claude_monthly_usd"]),                   step=1.0)
    copilot_limit  = st.number_input("Copilot requests",    value=float(cfg["budgets"]["copilot_monthly_premium_requests"]),     step=10.0)

    st.divider()
    st.caption(f"DB: `{DB_PATH}`")
    if st.button("🔄 Refresh now"):
        st.cache_data.clear()
        st.rerun()

# ── Load data ─────────────────────────────────────────────────────────────────

data = load_all(str(DB_PATH), month)
h    = data["headline"]
b    = data["budget"]
q    = data["quality"]

# Override limits from sidebar
b["claude_limit"]       = claude_limit
b["claude_remaining"]   = max(0.0, claude_limit - b["claude_spent"])
b["claude_pct"]         = min(100.0, b["claude_spent"] / claude_limit * 100) if claude_limit else 0.0
b["copilot_limit"]      = copilot_limit
b["copilot_remaining"]  = max(0.0, copilot_limit - b["copilot_used"])
b["copilot_pct"]        = min(100.0, b["copilot_used"] / copilot_limit * 100) if copilot_limit else 0.0

# ── Tabs ───────────────────────────────────────────────────────────────────────

st.title("🤖 LLM Cost Autopilot")
tab_dash, tab_play = st.tabs(["📊 Dashboard", "🧪 Playground"])

# ═══════════════════════════════════════════════════════════════════════════════
# PLAYGROUND TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_play:
    import httpx as _httpx
    import time as _time

    st.caption("Type a prompt and see which backend the router chooses — and why.")

    EXAMPLES = {
        "— pick an example —": "",
        "Tier 1 · Format a date": "Convert this date to ISO format: March 15, 2024",
        "Tier 1 · Extract emails": "Extract all email addresses from: contact alice@example.com or bob@corp.io for help.",
        "Tier 2 · Summarise": "Summarise in two sentences: Machine learning is a subset of AI that enables systems to learn from data. Common applications include image recognition, NLP, and recommendation systems.",
        "Tier 2 · Sentiment": "Classify the sentiment: 1) 'Fast shipping, exactly as described!' 2) 'Product broke after two days.' 3) 'It's fine, nothing special.'",
        "Tier 3 · Logic puzzle": "A snail climbs a 10-metre wall. Each day it climbs 3 metres but slides back 2 metres at night. How many days to reach the top? Show your reasoning.",
        "Tier 3 · Ethics": "A self-driving car's brakes fail. It can hit one pedestrian or swerve and hit three. What should it do? Analyse from utilitarian and deontological perspectives.",
    }

    col_input, col_ex = st.columns([3, 1])
    with col_ex:
        st.markdown("**Quick examples**")
        example = st.selectbox("Load example", list(EXAMPLES.keys()), label_visibility="collapsed")
    with col_input:
        prompt = st.text_area("Your prompt", value=EXAMPLES.get(example, ""), height=150, placeholder="Type any prompt here…")

    send = st.button("▶ Send", type="primary", disabled=not prompt.strip())

    if send and prompt.strip():
        with st.spinner("Routing and generating…"):
            try:
                resp = _httpx.post(
                    f"{API_BASE}/v1/completions",
                    json={"messages": [{"role": "user", "content": prompt}]},
                    timeout=120.0,
                )
                resp.raise_for_status()
                pdata = resp.json()
            except _httpx.ConnectError:
                st.error("Cannot reach the API at http://127.0.0.1:8000 — is uvicorn running?")
                st.stop()
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        r = pdata["routing"]
        POOL_EMOJI = {"FREE": "🟢", "COPILOT_PREMIUM": "🔵", "CLAUDE_CREDIT": "🟡"}
        TIER_LABEL = {1: "Tier 1 · Simple", 2: "Tier 2 · Moderate", 3: "Tier 3 · Complex"}
        REASON_HELP = {
            "primary":                  "Normal path — primary backend for this tier.",
            "low_confidence":           "Router confidence was low; bumped up one tier.",
            "claude_reserve_threshold": "High confidence on a complex task → reserved Claude.",
            "budget_spill_to_claude":   "Copilot pool running low; spilled to Claude.",
            "budget_exhausted":         "Both pools low; best-effort choice.",
            "fallback":                 "Primary pool low; using fallback backend.",
        }

        st.divider()
        st.markdown("### Routing decision")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Tier",       TIER_LABEL.get(r["complexity_tier"], f"Tier {r['complexity_tier']}"))
        c2.metric("Backend",    r["backend_id"])
        c3.metric("Pool",       f"{POOL_EMOJI.get(r['budget_pool'], '')} {r['budget_pool']}")
        c4.metric("Confidence", f"{r['router_confidence']:.1f} / 5.0")
        c5.metric("Latency",    f"{r['latency_ms']:.0f} ms")

        reason_color = {"primary": "green", "low_confidence": "orange",
                        "claude_reserve_threshold": "orange", "budget_spill_to_claude": "red",
                        "budget_exhausted": "red", "fallback": "orange"}.get(r["routing_reason"], "gray")
        st.markdown(f"**Routing reason:** :{reason_color}[{r['routing_reason']}] — {REASON_HELP.get(r['routing_reason'], '')}")
        if r["was_escalated"]:
            st.warning("⬆ This request was escalated from its initial tier.")
        with st.expander("Full routing metadata"):
            st.json(r)

        st.divider()
        st.markdown("### Response")
        st.markdown(pdata["text"])

        st.divider()
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Input tokens",  r["input_tokens"])
        t2.metric("Output tokens", r["output_tokens"])
        t3.metric("Cost (USD)",    f"${r['cost_usd']:.6f}" if r["cost_usd"] else "—")
        t4.metric("Premium reqs",  r["premium_requests_used"] or "—")
        st.caption("💡 Switch to the Dashboard tab to see this request appear in the stats.")

# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    st.caption(f"Month: **{month}** · Auto-refreshes every 30s · {h['total_requests']} requests logged")

    import pandas as pd
    import altair as alt

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🆓 Served free locally", f"{h['free_pct']:.1f}%", help="% of requests handled by local Ollama (zero cloud cost)")
    with col2:
        st.metric("💰 Savings vs all-premium", f"${h['savings_usd']:.2f}", delta=f"{h['savings_pct']:.0f}% reduction",
                  help=f"Baseline (all Claude Opus): ${h['baseline_cost_usd']:.2f} · Actual: ${h['actual_cost_usd']:.4f}")
    with col3:
        st.metric("📊 Total requests", f"{h['total_requests']:,}")
    with col4:
        st.metric("⚡ Avg latency", f"{h['avg_latency_ms'] / 1000:.1f}s")

    st.divider()
    st.subheader("💳 Budget pools")
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        st.markdown("**Claude Pro credit**")
        st.progress(min(b["claude_pct"] / 100, 1.0))
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Spent",     f"${b['claude_spent']:.2f}")
        mc2.metric("Remaining", f"${b['claude_remaining']:.2f}")
        mc3.metric("Proj. EOM", f"${b['claude_projected_eom']:.2f}",
                   delta=f"{'over' if b['claude_projected_eom'] > claude_limit else 'under'} limit",
                   delta_color="inverse" if b['claude_projected_eom'] > claude_limit else "normal")
    with bcol2:
        st.markdown("**Copilot premium requests**")
        st.progress(min(b["copilot_pct"] / 100, 1.0))
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Used",      f"{b['copilot_used']:.0f}")
        mc2.metric("Remaining", f"{b['copilot_remaining']:.0f}")
        mc3.metric("Proj. EOM", f"{b['copilot_projected_eom']:.0f}",
                   delta=f"{'over' if b['copilot_projected_eom'] > copilot_limit else 'under'} limit",
                   delta_color="inverse" if b['copilot_projected_eom'] > copilot_limit else "normal")
    st.caption(f"Day {b['days_elapsed']} of {b['days_in_month']} · Projection assumes current daily rate continues")

    st.divider()
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("🗂️ Routing distribution")
        dist = data["routing"]
        if dist:
            df_dist = pd.DataFrame(dist)
            pool_colors = {"FREE": "#22c55e", "COPILOT_PREMIUM": "#3b82f6", "CLAUDE_CREDIT": "#f59e0b"}
            df_dist["color"] = df_dist["budget_pool"].map(pool_colors)
            pie = (alt.Chart(df_dist).mark_arc(innerRadius=60)
                   .encode(theta=alt.Theta("count:Q"),
                           color=alt.Color("backend_id:N",
                               scale=alt.Scale(domain=df_dist["backend_id"].tolist(), range=df_dist["color"].tolist()),
                               legend=alt.Legend(title="Backend")),
                           tooltip=["backend_id", "count", alt.Tooltip("pct:Q", format=".1f")])
                   .properties(height=260))
            st.altair_chart(pie, use_container_width=True)
            pool_summary = df_dist.groupby("budget_pool")["pct"].sum().reset_index()
            for _, row in pool_summary.iterrows():
                label = {"FREE": "🟢 Free (local)", "COPILOT_PREMIUM": "🔵 Copilot", "CLAUDE_CREDIT": "🟡 Claude"}.get(row["budget_pool"], row["budget_pool"])
                st.caption(f"{label}: {row['pct']:.1f}%")
        else:
            st.info("No routing data yet for this month.")
    with chart_col2:
        st.subheader("📈 Requests over time")
        ot = data["over_time"]
        if ot:
            df_ot = pd.DataFrame(ot)
            df_melted = df_ot.melt(id_vars=["date"], value_vars=["FREE", "COPILOT_PREMIUM", "CLAUDE_CREDIT"], var_name="pool", value_name="requests")
            color_scale = alt.Scale(domain=["FREE", "COPILOT_PREMIUM", "CLAUDE_CREDIT"], range=["#22c55e", "#3b82f6", "#f59e0b"])
            line = (alt.Chart(df_melted).mark_area(opacity=0.7)
                    .encode(x=alt.X("date:T", title="Date"),
                            y=alt.Y("requests:Q", title="Requests", stack="zero"),
                            color=alt.Color("pool:N", scale=color_scale, legend=alt.Legend(title="Pool")),
                            tooltip=["date:T", "pool:N", "requests:Q"])
                    .properties(height=260))
            st.altair_chart(line, use_container_width=True)
        else:
            st.info("No time-series data yet.")

    st.divider()
    qcol1, qcol2 = st.columns(2)
    with qcol1:
        st.subheader("✅ Verification quality")
        if q["scored_count"] > 0:
            qm1, qm2, qm3 = st.columns(3)
            qm1.metric("Verified",   f"{q['scored_count']}")
            qm2.metric("Avg score",  f"{q['avg_score']:.2f}")
            qm3.metric("Mis-routes", f"{q['mis_route_count']}", delta=f"{q['mis_route_pct']:.1f}%",
                       delta_color="inverse" if q['mis_route_pct'] > 10 else "normal")
            df_q = pd.DataFrame(q["score_buckets"])
            bar = (alt.Chart(df_q).mark_bar()
                   .encode(x=alt.X("label:N", title=None, sort=None), y=alt.Y("count:Q", title="Count"),
                           color=alt.Color("label:N", scale=alt.Scale(
                               domain=["Good (≥0.75)", "Degraded (0.5–0.75)", "Poor (<0.5)"],
                               range=["#22c55e", "#f59e0b", "#ef4444"]), legend=None),
                           tooltip=["label", "count"])
                   .properties(height=200))
            st.altair_chart(bar, use_container_width=True)
            st.caption(f"🔄 {q['training_examples_added']} mis-routes added to training data (flywheel)")
        else:
            st.info("No verifications run yet this month.")
    with qcol2:
        st.subheader("🚀 Escalation rate")
        esc = data["escalation"]
        if esc:
            df_esc = pd.DataFrame(esc)
            esc_line = (alt.Chart(df_esc).mark_line(point=True, color="#f59e0b")
                        .encode(x=alt.X("date:T", title="Date"),
                                y=alt.Y("rate_pct:Q", title="Escalation %", scale=alt.Scale(domain=[0, 100])),
                                tooltip=["date:T", "total:Q", "escalated:Q", alt.Tooltip("rate_pct:Q", format=".1f")])
                        .properties(height=200))
            st.altair_chart(esc_line, use_container_width=True)
            overall_esc = h["escalation_rate"]
            color = "🔴" if overall_esc > 20 else ("🟡" if overall_esc > 10 else "🟢")
            st.caption(f"{color} Overall escalation rate this month: **{overall_esc:.1f}%**")
        else:
            st.info("No escalation data yet.")

    st.divider()
    st.subheader("💡 What-if: everything routed to Claude Opus")
    wf1, wf2, wf3 = st.columns(3)
    wf1.metric("Hypothetical spend", f"${h['baseline_cost_usd']:.2f}", help="If every token went through Claude Opus at market rate")
    wf2.metric("Actual spend",       f"${h['actual_cost_usd']:.4f}")
    wf3.metric("You saved",          f"${h['savings_usd']:.2f}", delta=f"{h['savings_pct']:.0f}% reduction", delta_color="normal")

    st.divider()
    st.subheader("🕐 Recent requests")
    recent = data["recent"]
    if recent:
        _BACKEND_MODEL = {
            "ollama_tier1":  "local",
            "copilot_mid":   "gpt-4o-mini",
            "copilot_top":   "gpt-4o",
            "claude_haiku":  "haiku-4-5",
            "claude_sonnet": "sonnet-4-6",
        }
        _REASON_LABEL = {
            "primary":                  "primary",
            "low_confidence":           "⬆ low conf",
            "claude_reserve_threshold": "⭐ reserved",
            "budget_spill_to_claude":   "💸 spill",
            "budget_exhausted":         "⚠ exhausted",
            "fallback":                 "↩ fallback",
        }
        df_r = pd.DataFrame(recent)
        df_r["Model"]       = df_r["backend_id"].map(lambda x: _BACKEND_MODEL.get(x, x))
        df_r["Reason"]      = df_r["routing_reason"].map(lambda x: _REASON_LABEL.get(x, x or "—"))
        df_r = df_r.rename(columns={
            "ts": "Time", "complexity_tier": "Tier", "router_confidence": "Conf",
            "backend_id": "Backend", "input_tokens": "In", "output_tokens": "Out",
            "latency_ms": "Latency(ms)", "cost_usd": "Cost($)",
            "was_escalated": "Esc", "verifier_score": "V.Score",
        })
        df_r["Latency(ms)"] = df_r["Latency(ms)"].round(0).astype(int)
        df_r["Cost($)"]     = df_r["Cost($)"].map(lambda x: f"${x:.5f}" if x > 0 else "—")
        df_r["Esc"]         = df_r["Esc"].map(lambda x: "✅" if x else "")
        df_r["V.Score"]     = df_r["V.Score"].map(lambda x: f"{x:.2f}" if x is not None and x == x else "—")
        df_r["Conf"]        = df_r["Conf"].map(lambda x: f"{x:.1f}" if x is not None else "—")
        st.dataframe(
            df_r[["Time", "Tier", "Conf", "Backend", "Model", "Reason", "In", "Out", "Latency(ms)", "Cost($)", "Esc", "V.Score"]],
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No requests logged yet.")

    import time
    st.caption("⏱️ Auto-refreshing every 30 seconds")
    time.sleep(30)
    st.rerun()
