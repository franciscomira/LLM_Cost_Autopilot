"""
api.py

Phase 5 — FastAPI service.

Endpoints:
  POST /v1/completions              — route a chat request; response includes routing metadata
  GET  /v1/models                   — registered backends + their budget pool and status
  GET  /v1/stats                    — savings summary + budget burn-down for the current month
  PUT  /v1/routing-config           — update thresholds in routing.yaml without redeploying
  POST /v1/routing-config/reload    — hot-reload routing.yaml (picks up hand-edits)
  GET  /health                      — Docker healthcheck

Start with:
  uvicorn api:app --host 0.0.0.0 --port 8000

Env vars (all optional, have defaults):
  DB_PATH                   — path to SQLite database (default: ./data/autopilot.db)
  OLLAMA_BASE_URL           — Ollama endpoint (default: http://localhost:11434)
  GITHUB_TOKEN              — PAT with models:read for GitHub Models
  ANTHROPIC_API_KEY         — Anthropic API key (when USE_CLAUDE_SUBSCRIPTION=false)
  USE_CLAUDE_SUBSCRIPTION   — "true" to use Claude Pro SDK credit
"""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from budget import BudgetState
from dashboard_data import get_headline_metrics
from hardware_profile import profile_hardware, recommend_models
from interface import AutopilotSettings, send_request
from registry import ModelRegistry
from router import route
from verification_queue import VerificationQueue
from verifier import VerificationJob, should_verify

load_dotenv()

# ── Auth ───────────────────────────────────────────────────────────────────────

# Set API_KEY in your .env to require X-API-Key on all /v1/* requests.
# If unset, auth is disabled (useful for local development only).
_API_KEY = os.environ.get("API_KEY", "")

# ── Paths ──────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent
_ROUTING_YAML = _ROOT / "routing.yaml"
# DB_PATH env var lets docker-compose (or tests) override the database location.
_DB_PATH = Path(os.environ.get("DB_PATH", str(_ROOT / "data" / "autopilot.db")))


# ── App-level singletons ───────────────────────────────────────────────────────

class _AppState:
    settings: AutopilotSettings
    budget: BudgetState
    registry: ModelRegistry
    vq: VerificationQueue
    # Serialises concurrent writes to routing.yaml
    config_lock: asyncio.Lock


_state = _AppState()


def _load_registry_and_budget() -> tuple[ModelRegistry, BudgetState]:
    """Read routing.yaml and construct fresh Registry + BudgetState objects."""
    with open(_ROUTING_YAML, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    budgets = cfg.get("budgets", {})
    budget = BudgetState(
        db_path=_DB_PATH,
        claude_monthly_limit_usd=budgets.get("claude_monthly_usd", 20.0),
        copilot_monthly_requests_limit=budgets.get("copilot_monthly_premium_requests", 300),
    )
    hw = profile_hardware()
    models = recommend_models(hw)
    registry = ModelRegistry(_ROUTING_YAML, hardware_profile=hw, recommended_models=models)
    return registry, budget


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state.settings = AutopilotSettings.from_env()
    _state.config_lock = asyncio.Lock()
    _state.registry, _state.budget = _load_registry_and_budget()

    _state.vq = VerificationQueue(
        registry=_state.registry,
        budget=_state.budget,
        settings=_state.settings,
    )
    await _state.vq.start()

    yield

    await _state.vq.stop()


app = FastAPI(
    title="LLM Cost Autopilot",
    version="1.0.0",
    description="Budget-aware LLM router: local Ollama → GitHub Copilot → Claude",
    lifespan=lifespan,
)


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if _API_KEY and request.url.path.startswith("/v1/"):
        key = request.headers.get("X-API-Key", "")
        if key != _API_KEY:
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing X-API-Key"})
    return await call_next(request)


# ── Schemas ────────────────────────────────────────────────────────────────────

_MAX_CONTENT_CHARS = 32_000   # ~8 k tokens; prevents runaway Claude spend
_MAX_MESSAGES = 50


class Message(BaseModel):
    role: str
    content: str = Field(..., max_length=_MAX_CONTENT_CHARS)


class CompletionRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1, max_length=_MAX_MESSAGES)
    stream: bool = False  # reserved; streaming not yet implemented


class RoutingMeta(BaseModel):
    backend_id: str
    provider: str
    model: str
    budget_pool: str
    complexity_tier: int | None
    router_confidence: float | None
    routing_reason: str          # why this backend was chosen
    was_escalated: bool
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    premium_requests_used: float


class CompletionResponse(BaseModel):
    text: str
    routing: RoutingMeta


class BackendStatus(BaseModel):
    backend_id: str
    provider: str
    model: str
    budget_pool: str
    quality_tier: str


class BudgetStats(BaseModel):
    month: str
    claude_spent_usd: float
    claude_limit_usd: float
    claude_remaining_usd: float
    copilot_requests_used: float
    copilot_requests_limit: float
    copilot_remaining_requests: float


class StatsResponse(BaseModel):
    budget: BudgetStats
    headline: dict[str, Any]


class RoutingConfigUpdate(BaseModel):
    """
    Partial update for routing.yaml thresholds. Only supplied keys are changed.
    All values are validated before the file is written.
    """
    claude_monthly_usd: float | None = None
    copilot_monthly_premium_requests: float | None = None
    claude_usd_remaining: float | None = None
    copilot_requests_remaining: float | None = None
    sample_rate: float | None = None
    claude_reserve_threshold: float | None = None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(req: CompletionRequest):
    """Route a chat request through the autopilot and return the response."""
    messages = [m.model_dump() for m in req.messages]
    prompt_text = " ".join(m["content"] for m in messages if m["role"] == "user")

    selected_config, tier, confidence, routing_reason = await route(
        prompt_text=prompt_text,
        budget=_state.budget,
        registry=_state.registry,
        settings=_state.settings,
    )

    try:
        response, log_id = await send_request(
            messages=messages,
            config=selected_config,
            budget=_state.budget,
            settings=_state.settings,
            complexity_tier=tier,
            router_confidence=confidence,
            routing_reason=routing_reason,
        )
    except Exception as exc:
        # Ollama timed out or failed — fall back to the tier's fallback backend
        if selected_config.provider == "ollama":
            try:
                fallback_config = _state.registry.fallback_backend(tier)
                response, log_id = await send_request(
                    messages=messages,
                    config=fallback_config,
                    budget=_state.budget,
                    settings=_state.settings,
                    complexity_tier=tier,
                    router_confidence=confidence,
                    routing_reason="fallback",
                )
            except Exception as exc2:
                raise HTTPException(status_code=502, detail=str(exc2)) from exc2
        else:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    # Queue verification if needed (non-blocking — never delays the HTTP response)
    ver_cfg = _state.registry.verification_config
    if should_verify(response, ver_cfg):
        job = VerificationJob(
            request_log_id=log_id,
            messages=messages,
            original_response=response,
            tier=tier,
            confidence=confidence,
        )
        _state.vq.enqueue(job)

    return CompletionResponse(
        text=response.text,
        routing=RoutingMeta(
            backend_id=response.backend_id,
            provider=selected_config.provider,
            model=response.model,
            budget_pool=response.budget_pool.value,
            complexity_tier=response.complexity_tier,
            router_confidence=response.router_confidence,
            routing_reason=routing_reason,
            was_escalated=response.was_escalated,
            latency_ms=round(response.latency_ms, 1),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
            premium_requests_used=response.premium_requests_used,
        ),
    )


@app.get("/v1/models", response_model=list[BackendStatus])
async def list_models():
    """List all registered backends with their budget pool and quality tier."""
    return [
        BackendStatus(
            backend_id=cfg.backend_id,
            provider=cfg.provider,
            model=cfg.model,
            budget_pool=cfg.budget_pool.value,
            quality_tier=cfg.quality_tier.value,
        )
        for cfg in _state.registry.all().values()
    ]


@app.get("/v1/stats", response_model=StatsResponse)
async def stats():
    """Return current-month budget burn-down and headline savings metrics."""
    snap = _state.budget.snapshot()
    headline = get_headline_metrics(_DB_PATH)

    return StatsResponse(
        budget=BudgetStats(
            month=snap.month_key,
            claude_spent_usd=round(snap.claude_spent_usd, 4),
            claude_limit_usd=snap.claude_limit_usd,
            claude_remaining_usd=round(snap.claude_remaining_usd, 4),
            copilot_requests_used=snap.copilot_requests_used,
            copilot_requests_limit=snap.copilot_requests_limit,
            copilot_remaining_requests=snap.copilot_remaining_requests,
        ),
        headline=headline,
    )


@app.put("/v1/routing-config")
async def update_routing_config(update: RoutingConfigUpdate):
    """
    Apply partial updates to routing.yaml and reload the registry live.
    No redeploy needed — changes take effect on the next request.
    Protected by an asyncio.Lock so concurrent PUTs cannot corrupt the file.
    """
    async with _state.config_lock:
        with open(_ROUTING_YAML, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        changed: list[str] = []

        if update.claude_monthly_usd is not None:
            cfg["budgets"]["claude_monthly_usd"] = update.claude_monthly_usd
            changed.append("budgets.claude_monthly_usd")

        if update.copilot_monthly_premium_requests is not None:
            cfg["budgets"]["copilot_monthly_premium_requests"] = update.copilot_monthly_premium_requests
            changed.append("budgets.copilot_monthly_premium_requests")

        if update.claude_usd_remaining is not None:
            cfg["low_budget_thresholds"]["claude_usd_remaining"] = update.claude_usd_remaining
            changed.append("low_budget_thresholds.claude_usd_remaining")

        if update.copilot_requests_remaining is not None:
            cfg["low_budget_thresholds"]["copilot_requests_remaining"] = update.copilot_requests_remaining
            changed.append("low_budget_thresholds.copilot_requests_remaining")

        if update.sample_rate is not None:
            if not (0.0 <= update.sample_rate <= 1.0):
                raise HTTPException(status_code=422, detail="sample_rate must be between 0.0 and 1.0")
            cfg["verification"]["sample_rate"] = update.sample_rate
            changed.append("verification.sample_rate")

        if update.claude_reserve_threshold is not None:
            if not (1.0 <= update.claude_reserve_threshold <= 5.0):
                raise HTTPException(status_code=422, detail="claude_reserve_threshold must be between 1.0 and 5.0")
            cfg["tiers"][3]["claude_reserve_threshold"] = update.claude_reserve_threshold
            changed.append("tiers.3.claude_reserve_threshold")

        if not changed:
            return {"status": "no_change", "changed": []}

        with open(_ROUTING_YAML, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

        _apply_config(cfg)

    return {"status": "updated", "changed": changed}


@app.post("/v1/routing-config/reload")
async def reload_routing_config():
    """
    Hot-reload routing.yaml from disk.
    Use this after hand-editing the file — no restart required.
    Protected by the same lock as PUT /v1/routing-config.
    """
    async with _state.config_lock:
        with open(_ROUTING_YAML, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        _apply_config(cfg)

    return {"status": "reloaded"}


def _apply_config(cfg: dict) -> None:
    """
    Swap in a freshly loaded config. Called inside the config_lock by both
    PUT /v1/routing-config and POST /v1/routing-config/reload.
    """
    _state.registry = ModelRegistry(_ROUTING_YAML)
    budgets = cfg.get("budgets", {})
    _state.budget.claude_limit = budgets.get("claude_monthly_usd", _state.budget.claude_limit)
    _state.budget.copilot_limit = budgets.get(
        "copilot_monthly_premium_requests", _state.budget.copilot_limit
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
