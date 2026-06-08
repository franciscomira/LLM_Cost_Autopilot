# Engineering Journal — LLM Cost Autopilot

Design decisions, trade-offs, and build notes written while building the project. Kept here because the reasoning behind the choices is as important as the choices themselves.

---

## The core reframe

Most LLM cost-optimisation guides optimise for *cheapest cost-per-token*. This project optimises across **three budget pools**: free local compute, Copilot premium requests, and Claude credit. Those are fundamentally different resources with different scarcity properties.

The autopilot's mission: **maximise free local inference, and when it must escalate, spend from whichever pool is healthiest.**

**Why Copilot as the default escalation target, not Claude:** Copilot Pro+ gives ~300 premium requests/month versus $20 of API-rate Claude credit with no rollover. Copilot carries the workhorse tier; Claude is reserved for the genuinely hardest ~5% of requests. This bias stretches both pools furthest.

---

## Phase 0 — Hardware profiling ✅

**Decision: hardware detection at setup time, not runtime.**

The router model and Tier-1 generator are chosen based on available RAM/VRAM. Rather than hardcoding models, `hardware_profile.py` runs once and writes a profile that the registry reads at startup. This means the same codebase adapts to different machines automatically.

**Sizing rule:** a 4-bit quantised model needs roughly `params_in_billions × 0.7 GB` for weights, plus ~3–4 GB for KV cache and OS headroom. The profiler picks the largest model that fits with a safety margin and falls back to a tiny model if resources are tight. Model list is in `models_by_hardware.yaml` (not code) so it can be updated as better small models are released.

---

## Phase 1 — Unified interface + budget registry ✅

**Decision: one `send_request()` function, three backends behind it.**

All three backends (Ollama HTTP, GitHub Models API, Anthropic SDK) return the same `Response` dataclass: text, input/output tokens, latency, cost, backend ID. The router never touches provider-specific code. This made testing much easier and kept the routing logic clean.

**Decision: budget tracking in SQLite, not in-memory.**

An in-memory counter resets on restart and can't be shared across processes. SQLite survives restarts, is cheap, and avoids introducing a dependency on Redis or Postgres for what is effectively a pair of counters. The schema tracks spend at the month level; a new row is inserted each month automatically.

**Later evolution (Sprint 3):** migrated the hot-path methods from synchronous `sqlite3` to `aiosqlite` after identifying that per-call connection setup was serialising concurrent requests on the event loop.

---

## Phase 2 — The routing brain ✅

**Decision: LLM classification, not a rules-based classifier.**

Two options were considered:
1. A rules-based classifier (token count, keyword matching, instruction verb detection).
2. A small local LLM with a few-shot prompt that outputs `{tier, confidence}` as JSON.

Option 2 wins for generalisation. A 3B-class model running locally is fast enough (adds ~200ms) and handles novel prompt patterns that rules would miss. The cost is near-zero since the router model runs on local hardware.

The prompt is versioned in `prompts/router_classify.txt`. The JSON output schema is strict; `_parse_classification()` has a regex fallback for when the model outputs extra tokens around the JSON object.

**Decision: confidence score (1–5) alongside tier.**

Tier alone isn't enough. A Tier-1 classification with confidence 2 should escalate to Tier 2 — the router is uncertain. A Tier-3 with confidence 5 should go straight to Claude, bypassing the Copilot-first default. The `resolve_backend()` function implements these rules explicitly, and every response includes a `routing_reason` string so the system explains its own decisions.

**Routing reasons emitted:**
- `"primary"` — normal path
- `"low_confidence"` — bumped one tier up
- `"claude_reserve_threshold"` — high confidence on Tier 3, sent to Claude
- `"budget_spill_to_claude"` — Copilot pool low, spilled to Claude
- `"budget_exhausted"` — both pools low, best-effort
- `"fallback"` — primary backend failed, using fallback

---

## Phase 3 — Sampled async verification ✅

**Decision: sample ~10%, not verify everything.**

Verifying every cheap response against a premium model would spend the budget the router just saved. The economics only work if verification is cheap. The implementation:
- 10% of requests are sampled randomly.
- Every low-confidence route (below threshold) is always verified.
- Verification runs out-of-band after the user receives their response.

**Decision: fire-and-forget queue, not await.**

The verification queue never blocks the HTTP response. If the queue fills (>256 items), new jobs are silently dropped. This is intentional: a flooded verifier would itself burn the Copilot budget. Budget protection is the priority.

**The feedback loop:** every verified mis-route is written to `verification_log` and flagged for the few-shot prompt refresh cycle. This creates a flywheel — routing mistakes become training signal.

---

## Phase 4 — Logging + dashboard ✅

**Decision: SQLite as the analytics store, not a time-series DB.**

The dashboard queries are straightforward aggregations (spend by month, routing distribution, escalation rate). SQLite handles these fine at the volumes this system generates. Adding InfluxDB or TimescaleDB would be over-engineering.

The Streamlit dashboard runs as a separate process (separate Docker service) that reads directly from the same SQLite file. It doesn't import `api.py` — clean separation of concerns.

`pages/1_Playground.py` adds an interactive prompt tester that calls the live API and shows the full routing metadata in real time. Useful for demos.

---

## Phase 5 — FastAPI service ✅

**Decision: singletons on `_AppState`, not module globals.**

`BudgetState`, `ModelRegistry`, and `VerificationQueue` are expensive to construct and must survive across requests. They live on a `_AppState` object initialized in FastAPI's `lifespan` context manager. Module-level globals would work but are harder to test; `_AppState` makes the dependency explicit and mockable.

**Decision: `PUT /v1/routing-config` writes to disk.**

When the API updates thresholds, it writes the change back to `routing.yaml`. This means the file stays the single source of truth even after a container restart. The alternative (in-memory only) would silently lose threshold changes on restart.

**Endpoints:**
- `POST /v1/completions` — full pipeline: classify → resolve → send → verify (async)
- `GET /v1/models` — registered backends with pool and quality tier
- `GET /v1/stats` — live budget snapshot + headline savings metrics
- `PUT /v1/routing-config` — partial threshold update, live-reloads registry
- `POST /v1/routing-config/reload` — hot-reload `routing.yaml` after hand edits
- `GET /health` — Docker healthcheck

---

## Sprint 1 — Security hardening ✅

Items deferred from Phase 5 that block real use:

**`X-API-Key` middleware.** All `/v1/*` routes require a key when `API_KEY` is set. If unset, auth is disabled for local dev — no behaviour change for existing users.

**Input caps.** Message content capped at 32,000 chars; message list capped at 50 items (Pydantic `Field` validation). Without these, a single malformed request could exhaust the Claude monthly budget in one call.

---

## Sprint 2 — Observability ✅

**Structured JSON logging.** Every log line is one JSON object on stdout. Log aggregators (Datadog, CloudWatch, Loki) parse JSON natively. Human-readable logs with `LOG_LEVEL=DEBUG`.

**Request-ID correlation.** A UUID is set per request via a `ContextVar`, injected into every log field emitted during that request's lifetime, and echoed as `X-Request-Id` in the response header. This makes it possible to reconstruct the full lifecycle of any request from logs alone — without a distributed tracing system.

**Budget alert webhook.** When Claude spend crosses `BUDGET_ALERT_THRESHOLD_USD`, the service POSTs a JSON alert to `BUDGET_ALERT_WEBHOOK_URL`. Fires once per crossing (tracked with `_last_alert_fired_at_usd`) to prevent alert spam.

---

## Sprint 3 — Resilience ✅

**Tenacity retries on Ollama.** `_call_ollama_router` retries up to 3× on `TransportError` or `TimeoutException`, with exponential back-off (1–8 s). Per-attempt timeout tightened from 120 s to 30 s — a 120-second hang with no retry was a silent failure mode.

**Generalised fallback.** The original fallback only triggered for `provider == "ollama"`. Copilot and Claude failures propagated as raw 502s. Any provider can fail transiently; the fallback logic is now provider-agnostic.

**aiosqlite migration.** The synchronous `sqlite3` calls in `BudgetState` blocked the event loop on every request. `aiosqlite` moves the I/O off the loop. Schema init (`_init_db`) stays synchronous — it runs once before the loop starts, so there's no reason to complicate it.

---

## Deferred / future work

| Item | Rationale for deferring |
|---|---|
| SSE streaming for `/v1/completions` | Adds complexity; buffered responses are fine for the current use case |
| Versioned DB migrations (Alembic) | The current `ALTER TABLE` migration hack is fragile at scale, but there's only one migration so far |
| Monthly budget-reset pre-notification | Nice-to-have; webhook alert covers the critical case |
| Sprint 4: proper test suite | `test_smoke.py` covers the budget layer; router accuracy and verifier tests exist in `tests/` but integration tests against the full API are missing |
| Embedding + logistic-regression router | The LLM router hits ≥82% accuracy; a second approach would make a good A/B comparison but isn't needed for the core product |
