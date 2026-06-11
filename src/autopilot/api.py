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
  uvicorn autopilot.api:app --host 0.0.0.0 --port 8000

Env vars (all optional, have defaults):
  DB_PATH                   — path to SQLite database (default: ./data/autopilot.db)
  OLLAMA_BASE_URL           — Ollama endpoint (default: http://localhost:11434)
  GITHUB_TOKEN              — PAT with models:read for GitHub Models
  ANTHROPIC_API_KEY         — Anthropic API key (when USE_CLAUDE_SUBSCRIPTION=false)
  USE_CLAUDE_SUBSCRIPTION   — "true" to use Claude Pro SDK credit
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from autopilot.logging_config import REQUEST_ID, configure_logging

from autopilot import claude as claude_backend
from autopilot import github_models
from autopilot import ollama as ollama_backend
from autopilot.budget import BudgetState
from autopilot.dashboard_data import get_headline_metrics
from autopilot.hardware_profile import profile_hardware, recommend_models
from autopilot.interface import AutopilotSettings, _hash_prompt, send_request
from autopilot.models import BudgetPool
from autopilot.registry import ModelRegistry
from autopilot.router import route
from autopilot.verification_queue import VerificationQueue
from autopilot.verifier import VerificationJob, should_verify

load_dotenv()

configure_logging(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# ── Auth ───────────────────────────────────────────────────────────────────────

# Set API_KEY in your .env to require X-API-Key on all /v1/* requests.
# If unset, auth is disabled (useful for local development only).
_API_KEY = os.environ.get("API_KEY", "")

# ── Paths ──────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).parent
_ROUTING_YAML = _ROOT / "routing.yaml"
# DB_PATH env var lets docker-compose (or tests) override the database location.
_DB_PATH = Path(os.environ.get("DB_PATH", str(_ROOT.parent.parent / "data" / "autopilot.db")))


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


async def _monthly_notifier_loop() -> None:
    """Check once per day whether a month-end pre-notification should fire."""
    while True:
        await asyncio.sleep(86_400)
        try:
            await _state.budget.check_month_end_notification()
        except Exception as exc:
            logger.error("monthly notifier error", extra={"error": str(exc)})


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

    notifier_task = asyncio.create_task(_monthly_notifier_loop())

    yield

    notifier_task.cancel()
    await _state.vq.stop()


app = FastAPI(
    title="LLM Cost Autopilot",
    version="1.0.0",
    description="Budget-aware LLM router: local Ollama → GitHub Copilot → Claude",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    REQUEST_ID.set(request_id)

    if _API_KEY and request.url.path.startswith("/v1/"):
        key = request.headers.get("X-API-Key", "")
        if key != _API_KEY:
            logger.warning("unauthorized request", extra={"path": request.url.path})
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing X-API-Key"},
                headers={"X-Request-Id": request_id},
            )

    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response


# ── Schemas ────────────────────────────────────────────────────────────────────

_MAX_CONTENT_CHARS = 32_000   # ~8 k tokens; prevents runaway Claude spend
_MAX_MESSAGES = 50


class Message(BaseModel):
    role: str
    content: str = Field(..., max_length=_MAX_CONTENT_CHARS)


class CompletionRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1, max_length=_MAX_MESSAGES)
    stream: bool = False


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


# ── Anthropic Messages API compatibility shim ──────────────────────────────────
# Clients that set ANTHROPIC_BASE_URL=http://localhost:8000 (Claude Code, the
# Anthropic Python SDK, etc.) will hit this endpoint. The `model` field is
# accepted but ignored — the router decides the actual backend.

class _TextBlock(BaseModel):
    type: str = "text"
    text: str = ""

class _AnthropicInboundMessage(BaseModel):
    role: str
    # Anthropic allows content as a plain string or a list of typed blocks
    content: str | list[_TextBlock]

class MessagesRequest(BaseModel):
    model: str = "claude-haiku-4-5-20251001"   # ignored — router decides
    max_tokens: int = 1024
    system: str | list[_TextBlock] | None = None
    messages: list[_AnthropicInboundMessage] = Field(..., min_length=1, max_length=_MAX_MESSAGES)
    stream: bool = False

    def to_internal_messages(self) -> list[dict]:
        """Flatten Anthropic request shape to the internal {role, content: str} list."""
        out: list[dict] = []

        if self.system:
            sys_text = (
                self.system if isinstance(self.system, str)
                else " ".join(b.text for b in self.system if b.type == "text")
            )
            if sys_text:
                out.append({"role": "system", "content": sys_text[:_MAX_CONTENT_CHARS]})

        for m in self.messages:
            text = (
                m.content if isinstance(m.content, str)
                else " ".join(b.text for b in m.content if b.type == "text")
            )
            out.append({"role": m.role, "content": text[:_MAX_CONTENT_CHARS]})

        return out


# ── Endpoints ──────────────────────────────────────────────────────────────────

async def _stream_completions(
    messages: list[dict],
    selected_config,
    tier: int | None,
    confidence: float | None,
    routing_reason: str,
) -> AsyncGenerator[str, None]:
    """SSE generator for streaming completions."""
    routing_event = {
        "event": "routing",
        "backend_id": selected_config.backend_id,
        "provider": selected_config.provider,
        "model": selected_config.model,
        "routing_reason": routing_reason,
    }
    yield f"data: {json.dumps(routing_event)}\n\n"

    usage_out: dict = {}
    t0 = time.perf_counter()

    try:
        if selected_config.provider == "ollama":
            gen = ollama_backend.send_stream(
                messages, selected_config, usage_out, _state.settings.ollama_base_url
            )
        elif selected_config.provider == "github_models":
            gen = github_models.send_stream(
                messages, selected_config, _state.settings.github_token, usage_out
            )
        elif selected_config.provider == "anthropic":
            gen = claude_backend.send_stream(messages, selected_config, usage_out)
        else:
            yield f"data: {json.dumps({'event': 'error', 'detail': f'Unknown provider: {selected_config.provider}'})}\n\n"
            return

        async for chunk in gen:
            yield f"data: {json.dumps({'event': 'chunk', 'text': chunk})}\n\n"

    except Exception as exc:
        logger.error("streaming error", extra={"error": str(exc), "backend_id": selected_config.backend_id})
        yield f"data: {json.dumps({'event': 'error', 'detail': str(exc)})}\n\n"
        return

    input_tokens = usage_out.get("input_tokens", 0)
    output_tokens = usage_out.get("output_tokens", 0)
    latency_ms = usage_out.get("latency_ms", (time.perf_counter() - t0) * 1000)
    cost_usd = selected_config.estimate_cost_usd(input_tokens, output_tokens)
    premium_requests = (
        selected_config.premium_request_multiplier
        if selected_config.budget_pool == BudgetPool.COPILOT_PREMIUM
        else 0.0
    )

    await _state.budget.record_spend(
        pool=selected_config.budget_pool, cost_usd=cost_usd, premium_requests=premium_requests
    )
    prompt_text = " ".join(m["content"] for m in messages if m["role"] == "user")
    await _state.budget.log_request(
        timestamp=t0,
        prompt_hash=_hash_prompt(prompt_text),
        complexity_tier=tier,
        router_confidence=confidence,
        backend_id=selected_config.backend_id,
        budget_pool=selected_config.budget_pool,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=round(latency_ms, 1),
        cost_usd=cost_usd,
        premium_requests=premium_requests,
        routing_reason=routing_reason,
    )

    done_event = {
        "event": "done",
        "routing": {
            "backend_id": selected_config.backend_id,
            "provider": selected_config.provider,
            "model": selected_config.model,
            "budget_pool": selected_config.budget_pool.value,
            "complexity_tier": tier,
            "router_confidence": confidence,
            "routing_reason": routing_reason,
            "latency_ms": round(latency_ms, 1),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "premium_requests_used": premium_requests,
        },
    }
    yield f"data: {json.dumps(done_event)}\n\n"
    yield "data: [DONE]\n\n"


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

    if req.stream:
        return StreamingResponse(
            _stream_completions(messages, selected_config, tier, confidence, routing_reason),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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
        # Primary backend failed — try the tier's fallback regardless of provider
        try:
            fallback_config = _state.registry.fallback_backend(tier)
            if fallback_config.backend_id == selected_config.backend_id:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            response, log_id = await send_request(
                messages=messages,
                config=fallback_config,
                budget=_state.budget,
                settings=_state.settings,
                complexity_tier=tier,
                router_confidence=confidence,
                routing_reason="fallback",
            )
        except HTTPException:
            raise
        except Exception as exc2:
            raise HTTPException(status_code=502, detail=str(exc2)) from exc2

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

    logger.info(
        "completion served",
        extra={
            "backend_id": response.backend_id,
            "provider": selected_config.provider,
            "tier": tier,
            "cost_usd": response.cost_usd,
            "latency_ms": round(response.latency_ms, 1),
            "routing_reason": routing_reason,
        },
    )

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


async def _anthropic_stream(
    messages: list[dict],
    selected_config,
    tier: int | None,
    confidence: float | None,
    routing_reason: str,
    input_tokens_hint: int,
) -> AsyncGenerator[str, None]:
    """SSE generator that speaks the Anthropic streaming wire protocol."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    usage_out: dict = {}
    t0 = time.perf_counter()

    # message_start
    yield (
        f"event: message_start\n"
        f"data: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': selected_config.model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': input_tokens_hint, 'output_tokens': 0}}})}\n\n"
    )
    yield "event: content_block_start\ndata: " + json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}) + "\n\n"
    yield "event: ping\ndata: {\"type\": \"ping\"}\n\n"

    try:
        if selected_config.provider == "ollama":
            gen = ollama_backend.send_stream(messages, selected_config, usage_out, _state.settings.ollama_base_url)
        elif selected_config.provider == "github_models":
            gen = github_models.send_stream(messages, selected_config, _state.settings.github_token, usage_out)
        elif selected_config.provider == "anthropic":
            gen = claude_backend.send_stream(messages, selected_config, usage_out)
        else:
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': f'Unknown provider: {selected_config.provider}'}})}\n\n"
            return

        async for chunk in gen:
            yield "event: content_block_delta\ndata: " + json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": chunk}}) + "\n\n"

    except Exception as exc:
        logger.error("anthropic stream error", extra={"error": str(exc), "backend_id": selected_config.backend_id})
        yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': str(exc)}})}\n\n"
        return

    output_tokens = usage_out.get("output_tokens", 0)
    latency_ms = usage_out.get("latency_ms", (time.perf_counter() - t0) * 1000)
    cost_usd = selected_config.estimate_cost_usd(input_tokens_hint, output_tokens)
    premium_requests = (
        selected_config.premium_request_multiplier
        if selected_config.budget_pool == BudgetPool.COPILOT_PREMIUM else 0.0
    )

    await _state.budget.record_spend(pool=selected_config.budget_pool, cost_usd=cost_usd, premium_requests=premium_requests)
    prompt_text = " ".join(m["content"] for m in messages if m["role"] == "user")
    await _state.budget.log_request(
        timestamp=t0, prompt_hash=_hash_prompt(prompt_text),
        complexity_tier=tier, router_confidence=confidence,
        backend_id=selected_config.backend_id, budget_pool=selected_config.budget_pool,
        input_tokens=input_tokens_hint, output_tokens=output_tokens,
        latency_ms=round(latency_ms, 1), cost_usd=cost_usd,
        premium_requests=premium_requests, routing_reason=routing_reason,
    )

    yield "event: content_block_stop\ndata: " + json.dumps({"type": "content_block_stop", "index": 0}) + "\n\n"
    yield "event: message_delta\ndata: " + json.dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": output_tokens}}) + "\n\n"
    yield "event: message_stop\ndata: " + json.dumps({"type": "message_stop"}) + "\n\n"


@app.post("/v1/messages")
async def messages(req: MessagesRequest):
    """
    Anthropic Messages API-compatible endpoint.
    Set ANTHROPIC_BASE_URL=http://localhost:8000 in any Anthropic SDK client
    (Claude Code, Python SDK, etc.) to route transparently through this gateway.
    The `model` field is accepted but ignored — the router decides the backend.
    """
    internal_messages = req.to_internal_messages()
    prompt_text = " ".join(m["content"] for m in internal_messages if m["role"] == "user")

    selected_config, tier, confidence, routing_reason = await route(
        prompt_text=prompt_text,
        budget=_state.budget,
        registry=_state.registry,
        settings=_state.settings,
    )

    # Rough input token estimate for streaming (actual usage filled in by backend)
    input_tokens_hint = sum(len(m["content"].split()) * 4 // 3 for m in internal_messages)

    if req.stream:
        return StreamingResponse(
            _anthropic_stream(internal_messages, selected_config, tier, confidence, routing_reason, input_tokens_hint),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        response, log_id = await send_request(
            messages=internal_messages,
            config=selected_config,
            budget=_state.budget,
            settings=_state.settings,
            complexity_tier=tier,
            router_confidence=confidence,
            routing_reason=routing_reason,
        )
    except Exception as exc:
        try:
            fallback_config = _state.registry.fallback_backend(tier)
            if fallback_config.backend_id == selected_config.backend_id:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            response, log_id = await send_request(
                messages=internal_messages,
                config=fallback_config,
                budget=_state.budget,
                settings=_state.settings,
                complexity_tier=tier,
                router_confidence=confidence,
                routing_reason="fallback",
            )
        except HTTPException:
            raise
        except Exception as exc2:
            raise HTTPException(status_code=502, detail=str(exc2)) from exc2

    ver_cfg = _state.registry.verification_config
    if should_verify(response, ver_cfg):
        _state.vq.enqueue(VerificationJob(
            request_log_id=log_id, messages=internal_messages,
            original_response=response, tier=tier, confidence=confidence,
        ))

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    return JSONResponse(content={
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": response.text}],
        "model": response.model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        },
        # Non-standard extension: routing metadata so callers can inspect decisions
        "x_gateway_routing": {
            "backend_id": response.backend_id,
            "routing_reason": routing_reason,
            "complexity_tier": tier,
            "router_confidence": confidence,
            "cost_usd": response.cost_usd,
        },
    })


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
    snap = await _state.budget.snapshot()
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
