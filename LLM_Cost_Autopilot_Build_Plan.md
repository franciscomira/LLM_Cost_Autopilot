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

## Phase 5 — Expose as an API (Day 11–13)

1. **FastAPI service** — `POST /v1/completions` accepts a standard chat request; the
   **router** chooses the backend (caller doesn't). Response includes metadata: which
   backend was chosen and why (tier + confidence + budget reason).
2. **Config endpoints** — `GET /v1/models` (backends + budget pools), `GET /v1/stats`
   (savings + budget burn-down), `PUT /v1/routing-config` (update thresholds without
   redeploy).
3. **Containerize** — `docker-compose`: api + background verification worker + Ollama
   + SQLite volume. Document env vars (Copilot auth, Claude auth, sampling %, thresholds).

**Done when:** `docker-compose up` gives a working router API with the dashboard live.

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

## Open decisions to make as you go
- **Claude dev path before June 15:** temporary API key, or build Claude last?
- **Optional second router** (embedding + sklearn) for the comparison story — yes/no?
- **Copilot models:** which model is your Tier-2 workhorse vs. your Tier-3 default top
  model (check what your Pro+ plan exposes and the premium-request multipliers).
- **"Reserve Claude" cutoff:** the complexity score above which Tier 3 goes to Claude
  instead of Copilot's top model.
- **Sampling rate** for verification (start 10%, tune against budget burn).

## Watch-outs
- Claude Pro Agent SDK credit is **$20/mo, no rollover, API-rate metered** — small.
  It's the **reserved** top tier, not the workhorse; Copilot carries most escalations.
  Verify which models Pro exposes via the SDK in your account.
- Copilot is a **coding** assistant; keep volume modest and don't expose the
  Copilot-backed route as a public service.
- Use **official SDKs only** (Copilot SDK, Claude Agent SDK). Skip reverse-engineered
  Copilot-as-OpenAI wrappers.
