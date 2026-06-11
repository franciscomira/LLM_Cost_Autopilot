# Feature Roadmap — LLM Cost Autopilot

Ideas discussed in the AI Architect session. Prioritised roughly by ROI / effort.

---

## Tier 1 — High-value, low-effort

### ✅ #1 · Router learning from verifier feedback (close the loop)
Mis-routes are already appended to `data/routing_dataset.jsonl`, but the router prompt
never sees them. Inject the most recent verifier-corrected examples as dynamic few-shot
examples at call time — the simplest form of online learning without fine-tuning.

**Status:** Implemented (Sprint 7). See `src/autopilot/router_feedback.py`.

---

### ✅ #3 · Budget alerts — proactive depletion forecasting
The dashboard shows projected EOM usage, but not "you'll run out in N days."
Compute per-pool `days_until_exhausted` from the current daily burn rate and
surface `st.warning` / `st.error` banners at the top of the Dashboard tab.

**Status:** Implemented (Sprint 7). Added to `dashboard_data.get_budget_status()` and `dashboard.py`.

---

### #2 · Per-task-type routing profiles
Add task-type detection (code / summarise / extract / converse) as a second routing axis.
A coding task and a summarisation task at the same complexity level should route differently
(Copilot is better at code; Claude is better at nuanced reasoning).

**Approach:** Extend the router prompt and JSON output to include a `task_type` field.
Add per-type override rules to `routing.yaml`.

---

## Tier 2 — Medium lift, high strategic value

### #4 · Multi-turn conversation routing
Currently routing is per-prompt. Long conversations should be pinned to a tier once
established — re-routing mid-conversation breaks coherence and wastes tokens on context
re-injection.

**Approach:** Track a `conversation_id` on the client. Once a conversation is pinned to
a tier, bypass the classifier and route directly to the same backend.

---

### #5 · Prompt caching awareness
Claude's prompt caching (5-min TTL) can cut repeat-context costs ~90%.
The router should detect cacheable prefixes (system prompts, long docs) and prefer Claude
specifically when caching applies — it may actually be cheaper than Copilot for those patterns.

**Approach:** Hash the first N tokens of each message. If the same prefix appeared in the
last 5 minutes and was sent to Claude, prefer Claude again and pass the `cache_control` header.

---

### #6 · A/B evaluation harness
Formalise the quality sampling as a configurable eval harness. Allow comparing two routing
configurations (e.g. current vs candidate) over the same prompt set before deploying changes.

**Approach:** Add a `/v1/eval` endpoint that runs a prompt against two configurations and
records both scores in `verification_log`. Dashboard shows A vs B comparison panel.

---

## Tier 3 — Bigger bets

### #7 · Cost-per-token accounting across all tiers
Real cost modelling: Copilot quota is not literally free (it has an opportunity cost).
Model "what would this have cost on pay-per-token Anthropic/OpenAI" gives a more honest
ROI story and makes the system useful for teams who don't have Copilot.

**Approach:** Add a `shadow_cost_usd` column to `request_log` that records the
pay-per-token equivalent for every request, regardless of pool.

---

### #8 · Self-hosted router upgrade path
The local Ollama router is the bottleneck for accuracy (~82%). Allow a user to swap in a
fine-tuned classifier (trained on their own verifier data) as a drop-in.

**Approach:** Expose a `router_backend` config option in `routing.yaml` alongside the
existing Ollama router. The fine-tuning script reads `data/routing_dataset.jsonl` and
produces a LoRA adapter for the base router model.
