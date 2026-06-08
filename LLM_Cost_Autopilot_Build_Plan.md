# LLM Cost Autopilot — Build Plan

**Project 2 from the AI Engineering Projects Guide, adapted to your stack.**

A routing layer that scores each request's complexity with a *local* model, then
routes it to the cheapest capable backend, and continuously verifies the routing
was correct.

---

## Your configuration

| Thing | Your setup | Implication |
|---|---|---|
| Goal | Portfolio **and** personal use | Keep dashboard + case study; make it actually runnable |
| Routing brain | Small **local** LLM (Ollama) | 3B-class model; classification, not generation |
| Tier 1 (free) | Local model via Ollama | Costs nothing; handles simple tasks only |
| Tier 2 + default escalation | **GitHub Copilot SDK** | Spends Copilot *premium requests* — the workhorse for everything that leaves local |
| Tier 3 (reserved top tier) | **Claude Agent SDK** (Claude Pro) | $20/mo credit, API-rate metered, no rollover — kept for the genuinely hardest ~5% |
| Hardware | Modest (small models only) | Auto-detected by a profiler that picks the local models; quantized; sampled verification |

### The core reframe
The original guide optimizes for *cheapest cost-per-token*. You're optimizing
across **three budget pools**: free local compute, Copilot premium requests, and
the Claude credit. The autopilot's mission: **maximize free local inference, and
when it must escalate, spend from whichever pool is healthiest.** That is a
stronger interview story than the textbook version.

**Starting bias for your two subscriptions:** Copilot Pro+ premium requests give
more monthly escalation headroom than $20 of API-rate Claude credit, so seed the
policy with **Copilot as the default escalation target** and **Claude reserved for
the hardest Tier-3 requests** (and as Copilot's fallback if its budget runs low).
The budget-aware router adapts from there, but this bias stretches both pools furthest.

### Two key adaptations vs. the guide
1. **Budget-aware routing**, not price-per-token. The registry tracks *which pool*
   a backend draws from and how much of each pool remains this month.
2. **Sampled verification**, not verify-everything. Verifying every cheap response
   against a premium model would burn the budget you're saving. Verify a configurable
   % (start ~10%) plus every low-confidence route.

---

## Phase 0 — Prep (Day 0)

1. **Hardware profiler (`hardware_profile.py`)** — run once at setup (and re-run on
   any new machine). It detects total RAM (`psutil`), GPU + VRAM (NVIDIA via
   `nvidia-smi`/`pynvml`, Apple Silicon via unified-memory detection, AMD via ROCm),
   CPU cores, and OS/architecture, then maps those to a recommended **router model**
   and **Tier-1 generator** from a `config/models_by_hardware.yaml` table, with a
   manual override.
   - Sizing rule of thumb: a 4-bit model needs roughly `params_in_billions × ~0.7 GB`
     for weights plus headroom for KV cache + OS (≈3–4 GB free for a 3B, ≈6 GB for a
     7–8B). The profiler picks the largest model that fits with a safety margin and
     falls back to a tiny model if resources are tight.
   - Keep the model list in config (not code) so you can swap in newer models without
     edits — the "best small model" changes every few months.
2. **Install Ollama** and pull the two models the profiler recommends:
   - Router: a small instruct model (typically 3B-class on your hardware). Routing is
     classification, so small is fine.
   - Tier-1 generator: the largest model that fits per the profiler.
   Smoke-test with `ollama run`. Ollama exposes an OpenAI-compatible endpoint at
   `http://localhost:11434/v1`.
3. **Copilot SDK**: install the official `@github/copilot-sdk` (Node) or the Python
   SDK. Authenticate with your Copilot Pro+ account. Confirm you can list available
   models and make one completion. (The official Copilot SDK is the supported path —
   avoid reverse-engineered Copilot-as-API wrappers; they're ToS-risky.)
4. **Claude Agent SDK**: until June 15 you have two options:
   - Develop against a temporary **Anthropic API key** (a few dollars) so you're not
     blocked, then flip the auth to subscription mode on June 15; **or**
   - Build Tiers 1–2 first (local + Copilot) and wire in Claude last.
   Decide which and note it.
5. Repo skeleton: `src/`, `prompts/`, `data/`, `config/`, `tests/`, `docker/`.
   Set up Python 3.11+, `uv`/`poetry`, pre-commit, `.env` for keys.

---

## Phase 1 — Unified Model Interface + Budget Registry (Day 1–3)

1. **`ModelConfig` dataclass** — provider, model ID, **budget_pool**
   (`FREE` / `COPILOT_PREMIUM` / `CLAUDE_CREDIT`), premium-request multiplier or
   $/token, avg latency, quality tier (low/med/high).
2. **`BudgetState`** — tracks remaining Claude credit ($) and Copilot premium
   requests for the month, persisted to SQLite. The router reads this live.
   The local backend is registered straight from the **Phase 0 profiler output**, so
   the registry always reflects the actual machine it's running on.
3. **`send_request(prompt, model_config) -> Response`** — one function, three
   backends behind it (Ollama HTTP, Copilot SDK, Claude Agent SDK). Every call
   returns a standardized `Response`: text, input/output tokens, latency,
   estimated cost (or premium-request units), backend id.
4. **Smoke test** — send the same 10 prompts to every backend; log outputs, latency,
   and budget consumed. This validates the abstraction and gives baseline data.

**Done when:** one function call hits any of the three backends and returns a
uniform `Response` with budget accounting.

---

## Phase 2 — The Local Routing Brain (Day 3–6)

1. **Define 3 complexity tiers** (from the guide):
   - Tier 1: reformatting, extraction, basic Q&A from provided context.
   - Tier 2: summarization, classification, structured analysis.
   - Tier 3: multi-step reasoning, creative generation, nuanced judgment.
2. **Labeled dataset** — write/collect **200+ prompts** across the tiers, label each
   by hand. Capture features: token count, instruction verbs ("analyze", "compare"),
   number of constraints, whether context is provided, output-format complexity.
3. **Build the router** — your stated approach: a **small local LLM** does the
   classification via a tight few-shot prompt that outputs a tier + a confidence
   score (1–5) as JSON. Keep the prompt versioned in `/prompts`.
   - *Optional near-free alternative / fallback:* an embedding + logistic-regression
     classifier on the extracted features. It's almost free to run and makes a great
     "I compared two routing strategies" talking point. Build it if time allows.
4. **Evaluate the router** — held-out set, target ≥80% tier accuracy for V1. Track a
   confusion matrix (mis-routing *down* is a quality risk; mis-routing *up* is a
   cost risk — note which your matrix shows).
5. **Routing map (`config/routing.yaml`)** — tier → backend, **budget-aware**, with
   the bias set for your subscriptions:
   - Tier 1 → local Ollama (free).
   - Tier 2 → Copilot mid model (the default workhorse for anything leaving local).
   - Tier 3 → Copilot top model **by default**; **Claude (SDK) reserved** for the
     hardest cases (very high complexity score, or where the verifier has flagged
     Copilot weak on this task type).
   - **Budget guardrails:** if the Copilot premium-request pool runs low, spill the
     overflow to Claude; if the Claude credit runs low, hold the line at Copilot. Make
     the pool thresholds and the "reserve Claude" complexity cutoff configurable.

**Done when:** a prompt goes in, the local model emits tier + confidence, and the
YAML policy resolves it to a concrete backend respecting budget state.

---

## Phase 3 — Sampled Async Verification Loop (Day 6–9)

1. **Quality thresholds per task type** — extraction: got all key fields?
   summarization: LLM-as-judge ≥4/5? classification: matches what a Tier-3 model
   would say?
2. **Sampled async verifier** — after returning the response to the user, queue a
   background job for **(a)** a configurable sample (~10%) and **(b)** every route
   the router flagged low-confidence. The job re-runs the prompt on a higher tier and
   scores agreement. This protects your small budgets.
3. **Auto-escalation** — if the verifier finds a significant divergence, re-run on the
   higher tier and return/replace with the better result (if latency permits). Log:
   original backend, escalated backend, budget delta, the quality gap that triggered it.
4. **Feedback to the router** — every verified mis-route becomes a new labeled example.
   Weekly (or on-demand) refresh the few-shot set / retrain the optional classifier.
   This is the flywheel that makes routing smarter over time.

**Done when:** low-confidence and sampled requests get verified out-of-band,
mis-routes auto-escalate, and failures accumulate as new training data.

---

## Phase 4 — Logging + Budget Dashboard (Day 9–11)

1. **Log everything** — one SQLite row per request: timestamp, prompt hash, tier,
   confidence, backend, budget pool + amount consumed, latency, verifier score,
   escalated?. This is your audit trail.
2. **Streamlit dashboard** — the money shots:
   - **% of requests served free** on local (headline efficiency metric).
   - **Budget burn-down** per pool: Claude credit ($ used / $20) and Copilot premium
     requests used, with projected end-of-month consumption.
   - **"What if everything went to Claude Opus"** comparison — the savings number.
   - Routing distribution (pie), quality-score distribution, escalation rate over time.
3. **The headline metric** — prominently display cost reduction % vs. an
   all-premium baseline. That number is your portfolio centerpiece.

**Done when:** the dashboard tells the cost story at a glance and the numbers
reconcile with the logs.

---

## Phase 5 — Expose as an API ✅ DONE (2026-06-08)

### What was built

| File | Purpose |
|---|---|
| `api.py` | FastAPI app with lifespan-managed singletons |
| `Dockerfile` | Python 3.11-slim image; mounts `data/` + `routing.yaml` |
| `docker-compose.yml` | Three services: `api`, `dashboard`, `ollama` |

**Endpoints delivered:**
- `POST /v1/completions` — full pipeline: classify → resolve → send → queue verification. Response carries `routing` metadata block (tier, confidence, backend, pool, cost, latency).
- `GET /v1/models` — all registered backends with pool + quality tier.
- `GET /v1/stats` — live budget snapshot + headline savings from `dashboard_data.py`.
- `PUT /v1/routing-config` — partial update of `routing.yaml` thresholds (budgets, low-budget floors, sample rate, `claude_reserve_threshold`). Reloads registry in-process; no redeploy needed.
- `GET /health` — Docker healthcheck.

**Architecture decisions made:**
- All singletons (`BudgetState`, `ModelRegistry`, `VerificationQueue`) live on a module-level `_AppState` object, initialised in the FastAPI `lifespan` context manager. This keeps them off global scope while surviving across requests.
- `PUT /v1/routing-config` writes directly to `routing.yaml` on disk, so the file stays the single source of truth even if the container restarts. The `ModelRegistry` is swapped atomically on the `_state` object; no lock is needed because Python's GIL makes the reassignment atomic for the dict-backed singletons we have.
- The dashboard runs as a separate compose service sharing the same SQLite volume — it is read-only from the data perspective and doesn't need to import `api.py`.
- Verification is fire-and-forget: `vq.enqueue(job)` never blocks the HTTP response. If the queue is full (>256 items) the job is silently dropped — correct behaviour for a budget-protection system where a flooded verifier would itself burn budget.

**Known limitations / deferred:**
- Streaming (`stream: true`) is accepted in the schema but ignored — all responses are buffered. Add SSE in a follow-up if needed.
- `PUT /v1/routing-config` rewrites the entire YAML file on each call; fine at this scale, but a concurrent write race is possible under load. Not a problem until the service is truly multi-process.
- The `docker-compose.yml` does not pull Ollama models automatically. After first `docker-compose up`, run: `docker exec <ollama-container> ollama pull <model>` for both the router and Tier-1 models.

---

## Phase 6 — Polish for Portfolio (Day 13–14)

1. **Realistic load test** — push **500–1,000 diverse prompts** through it. Capture the
   final savings report and dashboard screenshots. These are your portfolio artifacts.
2. **Case study** — lead with the number: *"Built a router that kept X% of requests on
   free local inference and cut effective LLM spend by Y% while maintaining Z% quality
   parity, verified by an async sampling loop."* Explain the budget-pool routing logic
   and the feedback flywheel.
3. **README as onboarding docs** (not a tutorial): one-paragraph summary, setup, how
   the budget-aware routing works, how to tune thresholds, architecture decisions with
   rationale. Optional 3-min Loom: a Tier-1 prompt served locally (instant, free), a
   Tier-3 prompt escalating to Claude, and the dashboard updating live.

---

## Architect's improvement notes

These are not blockers for Phase 6 but are worth doing before calling the project finished.
Ranked by impact-to-effort ratio.

### 1. Add a `POST /v1/routing-config/reload` endpoint (high value, 15 min)
Right now `PUT /v1/routing-config` only accepts the specific fields defined in
`RoutingConfigUpdate`. If you hand-edit `routing.yaml` (e.g. to swap a backend model),
those changes are invisible until the container restarts. A bare `POST /v1/routing-config/reload`
that just calls `ModelRegistry(ROUTING_YAML)` and swaps `_state.registry` gives you
a full hot-reload escape hatch for free.

### 2. Make the SQLite path configurable via env var (medium value, 10 min)
`api.py` currently hard-codes `_DB_PATH = _ROOT / "data" / "autopilot.db"`. The
`docker-compose.yml` passes `DB_PATH=/app/data/autopilot.db` as an env var but nothing
reads it. Wire `os.environ.get("DB_PATH", str(_DB_PATH))` so the path is actually
overridable — this matters if you ever run two api instances pointing at different
databases (e.g. a test environment).

### 3. Expose escalation reason in the routing metadata (medium value, 20 min)
The `RoutingMeta` response tells the caller *which* backend was chosen but not *why* it
was escalated. A `routing_reason` string field (`"low_confidence"`, `"budget_spill"`,
`"claude_reserve_threshold"`, `"primary"`) would make the logs and the dashboard much
more actionable and is a strong portfolio talking point — it shows the system is
self-aware about its routing decisions.

### 4. Add an Ollama model-pull step to docker-compose (low effort, high friction saved)
The current `ollama` service starts Ollama but leaves models un-pulled. A small
`ollama-init` init-container that runs `ollama pull <router_model> && ollama pull
<tier1_model>` (reading model names from env vars that mirror `routing.yaml`) makes
`docker-compose up` a true one-command setup. Without it, first-time users hit a
confusing 404 from the Ollama backend.

### 5. Guard `PUT /v1/routing-config` with a write lock (low value now, good hygiene)
Two concurrent PUT requests could interleave YAML reads and writes, producing a
corrupted file. Add an `asyncio.Lock` on `_AppState` and acquire it inside the
endpoint. One line of setup, one `async with` in the handler — negligible overhead,
eliminates the race entirely.

---

## Open decisions — status

| Decision | Status |
|---|---|
| Claude dev path (API key vs. subscription) | Resolved: Anthropic API key via `ANTHROPIC_API_KEY`; flip `USE_CLAUDE_SUBSCRIPTION=true` on June 15 |
| Optional second router (embedding + sklearn) | Deferred — build after Phase 6 load test if time allows |
| Copilot models (Tier 2 vs Tier 3) | Set: `gpt-4o-mini` (Tier 2), `gpt-4o` (Tier 3 default) |
| "Reserve Claude" cutoff | Set: confidence ≥ 4.5 on Tier-3 → Claude |
| Verification sampling rate | Set: 10% + always-verify below confidence 3 |

## Watch-outs
- Claude Pro Agent SDK credit is **$20/mo, no rollover, API-rate metered** — small.
  It's the **reserved** top tier, not the workhorse; Copilot carries most escalations.
  Verify which models Pro exposes via the SDK in your account.
- Copilot is a **coding** assistant; keep volume modest and don't expose the
  Copilot-backed route as a public service.
- Use **official SDKs only** (Copilot SDK, Claude Agent SDK). Skip reverse-engineered
  Copilot-as-OpenAI wrappers.
