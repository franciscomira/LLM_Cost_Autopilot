I got tired of paying premium API prices for prompts like "convert this date to ISO format".
This gateway routes each request to the cheapest model that can handle it — local models for
simple tasks, GPT-4o or Claude for harder ones. A background verifier samples responses and
feeds mis-routes back as training data, so the router improves from live traffic.

Across my own usage: ~92% cost reduction at ~94% quality parity.

If you want to check my demo: www.linkedin.com/in/francisco-mira

# LLM Gateway

**A production-grade LLM gateway that reduced API costs by ~92% while maintaining ~94% quality parity** — by routing each request to the cheapest capable model, verified by an async sampling loop that feeds mis-routes back as training data.

Drop-in compatible with the Anthropic SDK: point `ANTHROPIC_BASE_URL=http://localhost:8000` at the gateway and existing tools (Claude Code, any Anthropic SDK client) route transparently through it — no code changes required.

Built as a portfolio project and personal tool; the same routing problem every company running LLMs at scale faces.

---

## The headline result

> Routing 500 diverse prompts: **~60% served free on local Ollama**, ~35% via Copilot, ~5% via Claude — achieving **~92% cost reduction** vs an all-Claude-Opus baseline while an async sampling loop verifies quality.

```
  ★  Cost saved vs all-premium baseline:  ~92%
  ★  Quality parity with premium models:  ~94%
  ★  Requests served free (local):         ~60%

  FREE  (local Ollama):      ~300  (60%)
  COPILOT_PREMIUM (GitHub):  ~175  (35%)
  CLAUDE_CREDIT (Anthropic):  ~25  ( 5%)
  Avg router confidence:     3.8 / 5.0
  Router tier accuracy:      ~82%
```

---

## How it works

```
User prompt
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Local router model (Ollama)                                     │
│  Classifies: tier (1/2/3) + confidence (1–5)                    │
│  Retries up to 3× on network failure (tenacity)                 │
└─────────────────┬───────────────────────────────────────────────┘
                  │
          Budget-aware policy (routing.yaml)
                  │
        ┌─────────┼───────────┐
        ▼         ▼           ▼
   Tier 1     Tier 2      Tier 3
   Ollama     Copilot     Copilot top
   (FREE)     gpt-4o-mini  gpt-4o
                           │
                    confidence ≥ 4.5
                    OR Copilot low
                           │
                           ▼
                         Claude
                    (reserved top tier)
                  │
                  ▼
    Response → User
                  │
          10% sampled async
                  │
                  ▼
         Verifier (background)
         Mis-routes → training data
```

**Three budget pools:**

| Pool | Backend | Cost |
|---|---|---|
| `FREE` | Local Ollama | $0 — bounded by hardware |
| `COPILOT_PREMIUM` | GitHub Copilot (gpt-4o-mini / gpt-4o) | Premium requests (~300/mo on Pro+) |
| `CLAUDE_CREDIT` | Anthropic Claude | $20/mo credit — reserved for the hardest ~5% |

The router maximises `FREE` first, spends `COPILOT_PREMIUM` for the workhorse tier, and only draws on `CLAUDE_CREDIT` when confidence is high on a genuinely complex task — or as a Copilot fallback when Copilot requests run low.

---

## Architecture

| Component | File(s) | Role |
|---|---|---|
| Hardware profiler | `hardware_profile.py` | Detects RAM/GPU once at setup, recommends Ollama models |
| Model interface | `interface.py` | `send_request()` — one function, three backends |
| Backend adapters | `ollama.py`, `github_models.py`, `claude.py` | Provider-specific HTTP/SDK calls |
| Budget registry | `budget.py` | Async (aiosqlite) monthly spend tracking per pool |
| Model registry | `registry.py` | Loads `routing.yaml`, maps tier → backend config |
| Router | `router.py` | LLM classification + budget-aware `resolve_backend()` |
| Verifier | `verifier.py` + `verification_queue.py` | Async background quality check + training data collection |
| Dashboard data | `dashboard_data.py` | SQL queries powering the dashboard and `/v1/stats` |
| Streamlit UI | `dashboard.py` + `pages/1_Playground.py` | Cost dashboard + interactive prompt tester |
| API | `api.py` | FastAPI service with auth middleware and structured logging |
| Logging | `logging_config.py` | JSON log formatter with per-request `request_id` correlation |

### Why these design choices

**Local router, not a rules-based classifier.** A 3B-class instruct model running on Ollama classifies prompts well at near-zero cost. The alternative (embedding + logistic regression) would need frequent retraining; the LLM generalises better to new prompt patterns.

**Budget pools, not price-per-token.** You own Copilot Pro+ and Claude Pro — you've already paid the fixed cost. The right optimisation is "spend these pre-paid pools most efficiently," not "minimise marginal price."

**Sampled verification, not verify-everything.** Verifying every cheap response against a premium model would spend the budget you just saved. Verifying 10% + every low-confidence route catches systematic failures without breaking the economics.

**Routing reason in every response.** The `routing_reason` field (`"primary"`, `"low_confidence"`, `"claude_reserve_threshold"`, `"budget_spill_to_claude"`, `"budget_exhausted"`, `"fallback"`) makes the system self-aware and gives the dashboard actionable signal.

---

## Production hardening

Beyond the core routing logic, the service has been hardened across three sprints:

### Security (Sprint 1)
- **API key authentication** — `X-API-Key` HTTP middleware on all `/v1/*` routes. Set `API_KEY` in `.env`; if unset, auth is disabled for local dev.
- **Input size caps** — message content capped at 32 000 chars; message list capped at 50 items (Pydantic validation), preventing runaway Claude spend from malformed requests.

### Observability (Sprint 2)
- **Structured JSON logging** — every log line is one JSON object on stdout, ready for log aggregators (Datadog, CloudWatch, Loki). Configurable via `LOG_LEVEL`.
- **Request-ID correlation** — a `request_id` UUID is set per request, echoed as `X-Request-Id` in the response header, and present in every log field — making distributed tracing possible with no additional infrastructure.
- **Budget alert webhook** — when Claude spend crosses a configurable threshold, the service POSTs a JSON alert to `BUDGET_ALERT_WEBHOOK_URL` (Slack, PagerDuty, etc.). Fires once per crossing to avoid spam.

### Resilience (Sprint 3)
- **Tenacity retries on Ollama** — `_call_ollama_router` retries up to 3× on network/timeout errors with exponential back-off (1–8 s). Per-attempt timeout tightened from 120 s → 30 s.
- **Generalised fallback** — any provider failure (Ollama, Copilot, or Claude) now attempts the tier's fallback backend, not just Ollama failures. Guards against retrying the same backend when primary and fallback coincide.
- **Non-blocking SQLite** — `BudgetState` migrated from synchronous `sqlite3` to `aiosqlite`, removing the serialisation bottleneck under concurrent load.

### Test coverage (Sprint 4)
- **Unit tests for `resolve_backend`** — covers budget-exhaustion paths, low-confidence escalation, and all Tier-3 branching via `AsyncMock` snapshots; no live backends required.
- **Integration tests via `ASGITransport`** — full `/v1/completions` path tested in-process: 200 golden path, fallback on provider error, 502 propagation, 422 input validation, and auth rejection.
- **Parser unit tests** — `_parse_classification` tested against JSON, embedded JSON, regex fallback, garbage input, and out-of-range tier values.

### Docker hardening (Sprint 5)
- **Non-root container** — `Dockerfile` creates a locked-down `appuser` system account and switches to it before the process starts, eliminating root exposure.
- **`HEALTHCHECK` in image** — Docker can probe `GET /health` directly on the image (not just via compose), so `docker ps` reports accurate health state and compose `depends_on: condition: service_healthy` gates work correctly.
- **Named volume for SQLite** — `autopilot_data` is a Docker-managed named volume; data survives `docker-compose down` and container replacement without depending on a host-side `./data` folder.

### Operational polish (Sprint 6)
- **SSE streaming** — `POST /v1/completions` accepts `"stream": true` and returns `text/event-stream`. Clients receive three event types: `routing` (backend decision, sent before generation begins), `chunk` (one text fragment per frame), and `done` (full metadata — tokens, cost, latency — after generation completes). Spend and audit logging happen at stream end, not after buffering the full response. Each backend adapter (`ollama.py`, `github_models.py`, `claude.py`) has a dedicated `send_stream()` async generator.
- **Versioned migration table** — `schema_migrations` table replaces the silent `try/except ALTER TABLE` hack. Migrations are declared as a named list; each runs once and is recorded. The only caught exception is `sqlite3.OperationalError` (column already exists on a fresh DB), so real errors are no longer swallowed.
- **Month-end pre-notification** — a daily background task calls `BudgetState.check_month_end_notification()`, which POSTs a `month_end_approaching` webhook payload to `BUDGET_ALERT_WEBHOOK_URL` when ≤3 days remain in the billing month and any budget was consumed. Fires at most once per month, reusing the existing alert webhook.

---

## Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running
- GitHub account with Copilot Pro+ (for Tier 2/3 via GitHub Models)
- Anthropic API key or Claude Pro account (for Tier 3 top tier)

### 1. Hardware profile

```bash
python hardware_profile.py
```

Detects your RAM and GPU, recommends which Ollama models to pull. Output written to `config/hardware_profile.json`.

### 2. Pull Ollama models

```bash
ollama pull phi3:mini          # router model
ollama pull phi3:mini          # Tier-1 generator (or larger if you have VRAM)
```

### 3. Configure credentials

```bash
cp .env.example .env
```

Edit `.env` — see [Environment variables](#environment-variables) below.

### 4. Run locally

```bash
pip install -e .
uvicorn autopilot.api:app --host 0.0.0.0 --port 8000 --reload

# Dashboard (separate terminal):
streamlit run dashboard.py
```

### 5. Run with Docker

```bash
docker-compose up
```

Services:
- `ollama` — local LLM backend on port 11434
- `ollama-init` — one-shot container that pulls required models, then exits
- `api` — FastAPI router on port 8000
- `dashboard` — Streamlit on port 8501

> **First run:** after `docker-compose up`, the `ollama-init` container pulls the router and Tier-1 models automatically.

---

## Using the API

### Drop-in Anthropic SDK compatibility

Set one environment variable and any tool that uses the Anthropic SDK routes through the gateway automatically:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=any-value   # gateway ignores this; auth uses X-API-Key

# Claude Code — every prompt now routes through the gateway
claude

# Python SDK
from anthropic import Anthropic
client = Anthropic()   # picks up ANTHROPIC_BASE_URL automatically
response = client.messages.create(
    model="claude-opus-4-8",   # ignored — the gateway's router decides the actual model
    max_tokens=1024,
    messages=[{"role": "user", "content": "Summarise the CAP theorem."}]
)
```

The `model` field is accepted but ignored — the router classifies each request and picks the cheapest capable backend. The response is returned in the standard Anthropic format, so the caller never knows it was rerouted.

---

### Route a request

```bash
curl -s -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"messages": [{"role": "user", "content": "Summarise the key points of the CAP theorem."}]}' \
  | jq '.routing'
```

Example response `routing` block:

```json
{
  "backend_id": "copilot_mid",
  "provider": "github_models",
  "model": "gpt-4o-mini",
  "budget_pool": "COPILOT_PREMIUM",
  "complexity_tier": 2,
  "router_confidence": 4.0,
  "routing_reason": "primary",
  "was_escalated": false,
  "latency_ms": 843.2,
  "input_tokens": 48,
  "output_tokens": 142,
  "cost_usd": 0.0,
  "premium_requests_used": 1.0
}
```

The response header also includes `X-Request-Id` for log correlation.

### Stream a response (SSE)

```bash
curl -s -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"messages": [{"role": "user", "content": "Explain backpressure in distributed systems."}], "stream": true}'
```

The response is a `text/event-stream` with three event types:

```
data: {"event": "routing", "backend_id": "copilot_mid", "provider": "github_models", "model": "gpt-4o-mini", "routing_reason": "primary"}

data: {"event": "chunk", "text": "Backpressure is a "}

data: {"event": "chunk", "text": "flow-control mechanism..."}

data: {"event": "done", "routing": {"input_tokens": 22, "output_tokens": 187, "cost_usd": 0.0, "latency_ms": 1243.1, ...}}

data: [DONE]
```

### Check budget and savings

```bash
curl http://localhost:8000/v1/stats | jq
```

### Live-tune routing thresholds (no restart needed)

```bash
# Lower the Claude-reserve confidence cutoff
curl -X PUT http://localhost:8000/v1/routing-config \
  -H "Content-Type: application/json" \
  -d '{"claude_reserve_threshold": 5.0}'

# Reload after hand-editing routing.yaml
curl -X POST http://localhost:8000/v1/routing-config/reload
```

### Run the load test

```bash
python scripts/load_test.py --count 500 --concurrency 20
```

Results are saved to `results/load_test_<timestamp>.json`.

---

## Tuning the routing policy

All thresholds live in `routing.yaml` and can be updated live via `PUT /v1/routing-config`.

| Setting | Default | Effect |
|---|---|---|
| `budgets.copilot_monthly_premium_requests` | 300 | Set to your actual monthly allowance from GitHub |
| `budgets.claude_monthly_usd` | 20.0 | Your Claude credit limit |
| `low_budget_thresholds.copilot_requests_remaining` | 30 | Below this, stop routing to Copilot |
| `low_budget_thresholds.claude_usd_remaining` | 5.0 | Below this, stop routing to Claude |
| `tiers.3.claude_reserve_threshold` | 4.5 | Confidence ≥ this on Tier 3 → always use Claude |
| `verification.sample_rate` | 0.10 | Fraction of requests verified async (0.0–1.0) |
| `verification.always_verify_confidence_below` | 3 | Always verify when confidence < this |

**To maximise local inference:** raise `confidence_min` for Tier 1 so more prompts stay local.

**To protect Copilot budget:** lower `low_budget_thresholds.copilot_requests_remaining` and raise `claude_reserve_threshold` so complex requests spill to Claude sooner.

**To minimise Claude spend:** set `claude_reserve_threshold: 5.0` (effectively disable the Claude reservation) and raise `claude_usd_remaining` floor.

---

## Case study

### The problem

Using a single premium LLM for every request is expensive and wasteful. Most LLM workloads follow a power law: ~60% are simple reformatting, extraction, or basic Q&A that don't need GPT-4-level capability. Routing everything to the same model is like hiring a senior consultant to file your expense reports.

### The solution

A three-tier routing layer that classifies each request locally (free) and routes to the cheapest capable backend:

| Tier | Task type | Backend | Pool |
|---|---|---|---|
| 1 | Reformatting, extraction, basic Q&A | Local Ollama | FREE |
| 2 | Summarisation, analysis, classification | Copilot gpt-4o-mini | COPILOT_PREMIUM |
| 3 | Multi-step reasoning, nuanced judgment | Copilot gpt-4o → Claude | COPILOT_PREMIUM / CLAUDE_CREDIT |

### Key engineering decisions

**Budget pools over price-per-token.** The system owner holds pre-paid subscriptions (Copilot Pro+ and Claude Pro). The right objective is not "minimise marginal cost" but "maximise value from what you've already paid." The router tracks pool health and adapts — if Copilot requests run low mid-month, Tier-3 requests automatically spill to Claude.

**Sampled async verification.** Verifying every cheap response against a premium model would defeat the purpose. The system verifies ~10% of requests plus every low-confidence route, out-of-band after the user receives their response. Mis-routed examples automatically become training data for the router, closing the feedback loop.

**Routing reason transparency.** Every response carries a `routing_reason` explaining the decision: `"primary"`, `"low_confidence"`, `"claude_reserve_threshold"`, `"budget_spill_to_claude"`, `"budget_exhausted"`, `"fallback"`. This makes the system debuggable and the dashboard actionable.

**Production-grade foundations.** Auth middleware, structured JSON logging with request-ID correlation, budget alert webhooks, tenacity retries with bounded timeouts, and an async SQLite layer (aiosqlite) — the same concerns a real service would face.

### The feedback flywheel

Every verified mis-route is logged to the `verification_log` table. Replaying these against the router shows where the few-shot prompt is weak. Periodic refreshes of `prompts/router_classify.txt` improve routing accuracy over time without model retraining.

---

## Project structure

```
LLM_Cost_Autopilot/
├── src/autopilot/            # Core application package
│   ├── api.py                #   FastAPI service — auth, /v1/messages shim, lifespan
│   ├── router.py             #   LLM classification + budget-aware resolve_backend()
│   ├── router_feedback.py    #   Injects verifier corrections into the router prompt
│   ├── interface.py          #   Unified send_request() across all backends
│   ├── budget.py             #   Async monthly budget tracking (aiosqlite)
│   ├── registry.py           #   routing.yaml → ModelConfig objects
│   ├── models.py             #   Shared dataclasses (ModelConfig, Response, BudgetSnapshot …)
│   ├── logging_config.py     #   JSON log formatter + request_id ContextVar
│   ├── verifier.py           #   Quality verification logic (LLM-as-judge)
│   ├── verification_queue.py #   Async background queue (fire-and-forget)
│   ├── dashboard_data.py     #   SQL queries for dashboard + /v1/stats
│   ├── hardware_profile.py   #   Hardware detection → recommends Ollama models
│   ├── ollama.py             #   Ollama backend adapter
│   ├── github_models.py      #   GitHub Models (Copilot) backend adapter
│   ├── claude.py             #   Anthropic Claude backend adapter
│   ├── routing.yaml          #   Live-editable routing policy
│   ├── models_by_hardware.yaml  # Hardware size → recommended model mapping
│   └── prompts/
│       └── router_classify.txt  # Few-shot classification prompt (versioned)
├── dashboard.py              # Streamlit entrypoint (must stay at root for Streamlit)
├── scripts/
│   ├── check_models.py       # Utility to verify Ollama models are available
│   └── load_test.py          # 500-prompt load test + savings report
├── tests/
│   ├── test_smoke.py            # Budget, routing, and API smoke tests
│   ├── test_routing.py          # Unit + integration tests (resolve_backend, API, parser)
│   ├── test_router_accuracy.py  # Router tier accuracy against labeled dataset
│   └── test_verifier.py         # Verifier unit tests
├── docs/
│   └── engineering_journal.md   # Design decisions and trade-offs per phase
├── data/
│   └── routing_dataset.jsonl    # Labeled prompts for router evaluation
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Engineering journal

[`docs/engineering_journal.md`](docs/engineering_journal.md) covers the design decisions and trade-offs made during each phase — why LLM classification instead of rules, why budget pools instead of price-per-token, why aiosqlite, how the verification sampling rate was chosen, and what's deferred and why.

---

## Development

```bash
# Install in editable mode
pip install -e ".[dev]"

# Run all unit + integration tests (no live backends required)
pytest tests/test_smoke.py tests/test_routing.py -v

# Run router accuracy evaluation (requires Ollama running)
pytest tests/test_router_accuracy.py -v -s --timeout=300

# Run verifier tests
pytest tests/test_verifier.py -v

# Dry-run the load test (validates prompts, no API calls)
python scripts/load_test.py --dry-run --count 500
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `GITHUB_TOKEN` | — | PAT with `models:read` for GitHub Models (Copilot) |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `USE_CLAUDE_SUBSCRIPTION` | `false` | `true` to use Claude Pro SDK credit |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `DB_PATH` | `./data/autopilot.db` | SQLite database path |
| `API_KEY` | — | If set, all `/v1/*` routes require `X-API-Key: <value>` |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `BUDGET_ALERT_WEBHOOK_URL` | — | URL to POST JSON alerts when Claude spend crosses threshold |
| `BUDGET_ALERT_THRESHOLD_USD` | — | Claude spend (USD) that triggers the webhook alert |
