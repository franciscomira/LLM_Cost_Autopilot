"""
src/autopilot/interface.py

The unified send_request() function.

This is the only function the rest of the system (router, API, tests) calls.
It dispatches to the correct backend based on the ModelConfig, handles errors,
and records spend in BudgetState automatically.

Usage:
    response = await send_request(
        messages=[{"role": "user", "content": "Summarise this text: ..."}],
        config=registry.get("copilot_mid"),
        budget=budget_state,
        settings=settings,
    )
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass

import claude
import github_models
import ollama as ollama_backend
from budget import BudgetState
from models import BudgetPool, ModelConfig, Response


# ── Settings bag (read from env at startup, passed around) ─────────────────────

@dataclass
class AutopilotSettings:
    ollama_base_url: str = "http://localhost:11434"
    github_token: str = ""

    @classmethod
    def from_env(cls) -> "AutopilotSettings":
        return cls(
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            github_token=os.environ.get("GITHUB_TOKEN", ""),
        )


# ── Core dispatch ──────────────────────────────────────────────────────────────

async def send_request(
    messages: list[dict[str, str]],
    config: ModelConfig,
    budget: BudgetState,
    settings: AutopilotSettings,
    log: bool = True,
    complexity_tier: int | None = None,
    router_confidence: float | None = None,
    was_escalated: bool = False,
    routing_reason: str | None = None,
) -> tuple[Response, int | None]:
    """
    Send `messages` to the backend described by `config`, record spend in
    `budget`, and return a fully populated Response.

    Args:
        messages:          OpenAI-style message list.
        config:            Which backend to use (from the registry).
        budget:            Live BudgetState — updated after every call.
        settings:          Auth credentials loaded from env.
        log:               Write to request_log table (True for normal calls;
                           False for internal smoke tests).
        complexity_tier:   Tier assigned by the router (1/2/3).
        router_confidence: Confidence score from the routing step.
        was_escalated:     True if this call was triggered by the verifier.
    """
    provider = config.provider

    if provider == "ollama":
        response = await ollama_backend.send(
            messages=messages,
            config=config,
            base_url=settings.ollama_base_url,
        )

    elif provider == "github_models":
        if not settings.github_token:
            raise EnvironmentError(
                "GITHUB_TOKEN is not set. "
                "Generate a PAT with models:read at https://github.com/settings/tokens"
            )
        response = await github_models.send(
            messages=messages,
            config=config,
            github_token=settings.github_token,
        )

    elif provider == "anthropic":
        response = await claude.send(
            messages=messages,
            config=config,
        )

    else:
        raise ValueError(f"Unknown provider '{provider}' in config for {config.backend_id}")

    # Attach routing metadata
    response.complexity_tier = complexity_tier
    response.router_confidence = router_confidence
    response.was_escalated = was_escalated

    # Record spend in budget
    budget.record_spend(
        pool=config.budget_pool,
        cost_usd=response.cost_usd,
        premium_requests=response.premium_requests_used,
    )

    # Audit log
    log_id: int | None = None
    if log:
        prompt_text = " ".join(m.get("content", "") for m in messages)
        log_id = budget.log_request(
            timestamp=response.timestamp,
            prompt_hash=_hash_prompt(prompt_text),
            complexity_tier=complexity_tier,
            router_confidence=router_confidence,
            backend_id=config.backend_id,
            budget_pool=config.budget_pool,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
            cost_usd=response.cost_usd,
            premium_requests=response.premium_requests_used,
            was_escalated=was_escalated,
            routing_reason=routing_reason,
        )

    return response, log_id


def _hash_prompt(text: str) -> str:
    """SHA-256 prefix — used as a pseudonymous audit key, not for caching."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ── Smoke test helper ──────────────────────────────────────────────────────────

async def smoke_test_all_backends(
    registry,
    budget: BudgetState,
    settings: AutopilotSettings,
) -> dict[str, dict]:
    """
    Send a single test prompt to every registered backend.
    Returns a dict of backend_id → result summary.
    Used by scripts/setup.py to validate the full stack.
    """
    test_messages = [{"role": "user", "content": "Say the word 'OK' and nothing else."}]
    results = {}

    for backend_id, config in registry.all().items():
        if config.backend_id == "ollama_router":
            continue   # tested separately in Phase 2
        try:
            t0 = time.perf_counter()
            resp, _ = await send_request(
                messages=test_messages,
                config=config,
                budget=budget,
                settings=settings,
                log=False,
            )
            results[backend_id] = {
                "status": "ok",
                "latency_ms": round(resp.latency_ms, 1),
                "tokens": resp.total_tokens,
                "text_preview": resp.text[:60],
            }
        except Exception as e:
            results[backend_id] = {
                "status": "error",
                "error": str(e),
            }

    return results
