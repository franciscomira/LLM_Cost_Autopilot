"""
pages/1_Playground.py

Interactive prompt tester — type a prompt, see the routing decision live.
Calls POST /v1/completions on the running API and displays the full
routing metadata alongside the response.
"""
from __future__ import annotations

import time

import httpx
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Playground — LLM Cost Autopilot", page_icon="🧪", layout="wide")

st.title("🧪 Playground")
st.caption("Type a prompt and see which backend the router chooses — and why.")

# ── Prompt input ───────────────────────────────────────────────────────────────

EXAMPLES = {
    "— pick an example —": "",
    "Tier 1 · Format a date": "Convert this date to ISO format: March 15, 2024",
    "Tier 1 · Extract emails": "Extract all email addresses from: contact alice@example.com or bob@corp.io for help.",
    "Tier 2 · Summarise": "Summarise in two sentences: Machine learning is a subset of AI that enables systems to learn from data using statistical techniques. Common applications include image recognition, NLP, and recommendation systems.",
    "Tier 2 · Sentiment analysis": "Classify the sentiment of each review: 1) 'Fast shipping, exactly as described!' 2) 'Product broke after two days.' 3) 'It\\'s fine, nothing special.'",
    "Tier 3 · Logic puzzle": "A snail climbs a 10-metre wall. Each day it climbs 3 metres but slides back 2 metres each night. How many days to reach the top? Show your reasoning.",
    "Tier 3 · Ethics": "A self-driving car's brakes fail. It can hit one pedestrian or swerve and hit three. What should it be programmed to do? Analyse from utilitarian and deontological perspectives.",
}

col_input, col_settings = st.columns([3, 1])

with col_settings:
    st.markdown("**Quick examples**")
    example = st.selectbox("Load example", list(EXAMPLES.keys()), label_visibility="collapsed")

with col_input:
    initial = EXAMPLES.get(example, "")
    prompt = st.text_area(
        "Your prompt",
        value=initial,
        height=160,
        placeholder="Type any prompt here…",
    )

send = st.button("▶ Send", type="primary", disabled=not prompt.strip())

# ── Call the API ───────────────────────────────────────────────────────────────

if send and prompt.strip():
    with st.spinner("Routing and generating…"):
        t0 = time.monotonic()
        try:
            resp = httpx.post(
                f"{API_BASE}/v1/completions",
                json={"messages": [{"role": "user", "content": prompt}]},
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = (time.monotonic() - t0) * 1000
        except httpx.ConnectError:
            st.error("Cannot reach the API at http://127.0.0.1:8000 — is uvicorn running?")
            st.stop()
        except httpx.HTTPStatusError as e:
            st.error(f"API error {e.response.status_code}: {e.response.text}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    r = data["routing"]

    # ── Routing decision card ──────────────────────────────────────────────────

    POOL_EMOJI  = {"FREE": "🟢", "COPILOT_PREMIUM": "🔵", "CLAUDE_CREDIT": "🟡"}
    TIER_LABEL  = {1: "Tier 1 · Simple", 2: "Tier 2 · Moderate", 3: "Tier 3 · Complex"}
    REASON_HELP = {
        "primary":                 "Normal path — primary backend for this tier.",
        "low_confidence":          "Router confidence was low; bumped up one tier.",
        "claude_reserve_threshold":"High confidence on a complex task → reserved Claude slot.",
        "budget_spill_to_claude":  "Copilot pool running low; spilled to Claude.",
        "budget_exhausted":        "Both pools low; best-effort choice.",
        "fallback":                "Primary pool low; using fallback backend.",
    }

    pool_emoji  = POOL_EMOJI.get(r["budget_pool"], "⚪")
    tier_label  = TIER_LABEL.get(r["complexity_tier"], f"Tier {r['complexity_tier']}")
    reason_help = REASON_HELP.get(r["routing_reason"], r["routing_reason"])

    st.divider()
    st.markdown("### Routing decision")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tier",       tier_label)
    c2.metric("Backend",    r["backend_id"])
    c3.metric("Pool",       f"{pool_emoji} {r['budget_pool']}")
    c4.metric("Confidence", f"{r['router_confidence']:.1f} / 5.0")
    c5.metric("Latency",    f"{r['latency_ms']:.0f} ms")

    reason_color = {
        "primary": "green",
        "low_confidence": "orange",
        "claude_reserve_threshold": "orange",
        "budget_spill_to_claude": "red",
        "budget_exhausted": "red",
        "fallback": "orange",
    }.get(r["routing_reason"], "gray")

    st.markdown(
        f"**Routing reason:** :{reason_color}[{r['routing_reason']}] — {reason_help}"
    )

    if r["was_escalated"]:
        st.warning("⬆ This request was escalated from its initial tier.")

    with st.expander("Full routing metadata"):
        st.json(r)

    # ── Response ───────────────────────────────────────────────────────────────

    st.divider()
    st.markdown("### Response")
    st.markdown(data["text"])

    # ── Token & cost breakdown ─────────────────────────────────────────────────

    st.divider()
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Input tokens",  r["input_tokens"])
    t2.metric("Output tokens", r["output_tokens"])
    t3.metric("Cost (USD)",    f"${r['cost_usd']:.6f}" if r["cost_usd"] else "—")
    t4.metric("Premium reqs",  r["premium_requests_used"] if r["premium_requests_used"] else "—")

    st.caption("💡 Refresh the Dashboard page to see this request appear in the stats.")
