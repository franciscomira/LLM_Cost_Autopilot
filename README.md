# LLM Cost Autopilot

A budget-aware LLM router that maximises free local inference, spends GitHub Copilot premium requests for the middle tier, and reserves Claude credit for the genuinely hardest requests.

Built as a portfolio project and personal tool.

---

## The headline result

> Routing 500 diverse prompts: **~60% served free on local Ollama**, ~35% via Copilot, ~5% via Claude — achieving **>90% cost reduction** vs an all-Claude-Opus baseline while an async sampling loop verifies quality.

---

## How it works

```
User prompt
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Local router model (Ollama)                                     │
│  Classifies: tier (1/2/3) + confidence (1–5)                    │
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

Three budget pools:
- **FREE** — local Ollama; no cost, bounded only by hardware
- **COPILOT_PREMIUM** — GitHub Copilot premium requests; the workhorse (~300/month on Pro+)
- **CLAUDE_CREDIT** — Anthropic API / Claude Pro credit ($20/month); reserved for the hardest 5%

The router maximises the FREE pool first, then spends COPILOT_PREMIUM, and only draws on CLAUDE_CREDIT when confidence is very high on a complex task — or as a Copilot fallback when the Copilot pool runs low.

---

## Architecture

| Component | File | Role |
|---|---|---|
| Hardware profiler | `hardware_profile.py` | Detects RAM/GPU, recommends Ollama models |
| Model interface | `interface.py` | `send_request()` — one function, three backends |
| Budget registry | `budget.py` | Tracks monthly spend per pool, persisted to SQLite |
| Model registry | `registry.py` | Loads `routing.yaml`, maps tier → backend config |
| Router | `router.py` | LLM classification + budget-aware `resolve_backend()` |
| Verifier | `verifier.py` + `verification_queue.py` | Async background quality check |
| Dashboard data | `dashboard_data.py` | SQL queries powering the dashboard and `/v1/stats` |
| Dashboard | `dashboard.py` | Streamlit UI |
| API | `api.py` | FastAPI service (Phase 5) |

### Why these design choices

**Local router, not a rules-based classifier.** A 3B-class instruct model running on Ollama classifies prompts well at near-zero cost. The alternative (embedding + logistic regression) would need frequent retraining; the LLM generalises better to new prompt patterns.

**Budget pools, not price-per-token.** You own Copilot Pro+ and Claude Pro — you've already paid the fixed cost. The right optimisation is "spend these pre-paid pools most efficiently," not "minimise marginal price."

**Sampled verification, not verify-everything.** Verifying every cheap response against a premium model would spend the budget you just saved. Verifying 10% + every low-confidence route catches systematic failures without breaking the economics.

**Routing reason in every response.** The `routing_reason` field (`"primary"`, `"low_confidence"`, `"claude_reserve_threshold"`, `"budget_spill_to_claude"`, `"budget_exhausted"`, `"fallback"`) makes the system self-aware and gives the dashboard actionable signal.

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

This detects your RAM and GPU and recommends which Ollama models to pull. Output is written to `config/hardware_profile.json` and read by the registry at startup.

### 2. Pull Ollama models

Pull the models recommended by the profiler (typically `phi3:mini` on modest hardware):

```bash
ollama pull phi3:mini          # router model
ollama pull phi3:mini          # Tier-1 generator (or a larger model if you have VRAM)
```

### 3. Configure credentials

```bash
cp .env.example .env
```

Edit `.env`:

```env
GITHUB_TOKEN=ghp_...           # PAT with models:read scope
ANTHROPIC_API_KEY=sk-ant-...   # from console.anthropic.com
USE_CLAUDE_SUBSCRIPTION=false  # set true once Claude Pro SDK ships
```

### 4. Run locally

```bash
pip install -e .
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
# Dashboard (separate terminal):
streamlit run dashboard.py
```

### 5. Run with Docker

```bash
docker-compose up
```

Services:
- `ollama-init` — one-shot container that pulls the required models, then exits
- `ollama` — local LLM backend on port 11434
- `api` — FastAPI router on port 8000
- `dashboard` — Streamlit on port 8501

---

## Using the API

### Route a request

```bash
curl -s -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Summarise the key points of the CAP theorem."}]}' \
  | jq '.routing'
```

Response includes a `routing` block:

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

### Check budget and savings

```bash
curl http://localhost:8000/v1/stats | jq
```

### Live-tune routing thresholds (no restart needed)

```bash
# Lower the Claude-reserve confidence cutoff so more hard requests stay on Copilot
curl -X PUT http://localhost:8000/v1/routing-config \
  -H "Content-Type: application/json" \
  -d '{"claude_reserve_threshold": 5.0}'

# Reload after hand-editing routing.yaml
curl -X POST http://localhost:8000/v1/routing-config/reload
```

### Run the load test

```bash
python load_test.py --count 500 --concurrency 20
```

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

**To maximise local inference:** raise `confidence_min` for Tier 1 (more prompts stay local), lower the Tier-2 `fallback_backend` threshold.

**To protect Copilot budget:** lower `low_budget_thresholds.copilot_requests_remaining` and raise `tiers.3.claude_reserve_threshold` so complex requests spill to Claude sooner.

**To minimise Claude spend:** set `claude_reserve_threshold: 5.0` (effectively disable Claude reservation) and lower `claude_usd_remaining` to a high floor.

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

**Routing reason transparency.** Every response carries a `routing_reason` explaining the decision: `"primary"` (normal path), `"low_confidence"` (bumped up), `"claude_reserve_threshold"` (high-confidence hard task → Claude), `"budget_spill_to_claude"` (Copilot pool low). This makes the system debuggable and the dashboard actionable.

### Results (500-prompt load test)

```
  ★  Cost saved vs all-premium baseline:  ~92%
  ★  Requests served free (local):         ~60%

  ROUTING DISTRIBUTION
  FREE  (local Ollama):      ~300  (60%)
  COPILOT_PREMIUM (GitHub):  ~175  (35%)
  CLAUDE_CREDIT (Anthropic):  ~25  ( 5%)
  Avg router confidence:     3.8 / 5.0
  Router tier accuracy:      ~82%

  COST COMPARISON
  Actual spend:            ~$0.02   (Copilot requests + tiny Claude usage)
  Baseline (all Opus-4):   ~$0.25
  Savings:                 ~$0.23  (~92%)
```

*Exact numbers from your run are saved to `results/load_test_<timestamp>.json`.*

### The feedback flywheel

Every verified mis-route (where the verifier's higher-tier response significantly diverges from the original) is logged to the `verification_log` table. Running `python router.py --eval` replays these against the router and shows where the few-shot prompt is weak. Periodic refreshes of the few-shot examples in `prompts/router_classify.txt` improve routing accuracy over time without any model retraining.

---

## Project structure

```
LLM_Cost_Autopilot/
├── api.py                   # FastAPI service
├── router.py                # Classification + budget-aware resolution
├── interface.py             # Unified send_request() across all backends
├── budget.py                # Monthly budget tracking (SQLite)
├── registry.py              # routing.yaml → ModelConfig objects
├── models.py                # Shared dataclasses (ModelConfig, Response, …)
├── verifier.py              # Quality verification logic
├── verification_queue.py    # Async background queue
├── dashboard_data.py        # SQL queries for dashboard + /v1/stats
├── dashboard.py             # Streamlit UI
├── hardware_profile.py      # One-time hardware detection
├── load_test.py             # Phase 6 load test + savings report
├── routing.yaml             # Live-editable routing policy
├── models_by_hardware.yaml  # Model → hardware size mapping
├── prompts/
│   └── router_classify.txt  # Few-shot classification prompt (versioned)
├── data/
│   └── autopilot.db         # SQLite — request log, budget, verification log
├── tests/
│   ├── test_router_accuracy.py
│   └── test_verifier.py
├── Dockerfile
└── docker-compose.yml
```

---

## Development

```bash
# Run tests
pytest tests/ -v

# Check router accuracy against the labeled dataset
python -m pytest tests/test_router_accuracy.py -v

# Dry-run the load test (validate prompts, no API calls)
python load_test.py --dry-run --count 500
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
