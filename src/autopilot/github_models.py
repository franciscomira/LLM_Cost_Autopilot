"""
src/autopilot/github_models.py

Calls the GitHub Models inference API — an OpenAI-compatible endpoint that
is included with GitHub Copilot Pro/Pro+ subscriptions.

Auth: a GitHub Personal Access Token (classic or fine-grained) with the
      "models:read" scope. Generate at: https://github.com/settings/tokens

Endpoint: https://models.inference.ai.azure.com/chat/completions
Available models (as of mid-2026): gpt-4o, gpt-4o-mini, o1-mini, o1, and others.
Check current availability at: https://github.com/marketplace/models

NOTE ON CLAUDE VIA COPILOT
──────────────────────────
The GitHub Models REST API currently hosts OpenAI and Azure AI models only.
To use Claude models via your Copilot subscription programmatically, the
official path is the GitHub Copilot SDK (github/copilot-sdk, GA June 2026).
That SDK wraps the full Copilot agent runtime; its Python API surface is
evolving — see https://github.com/github/copilot-sdk for the latest docs.

For the Tier-2 workhorse (gpt-4o-mini, gpt-4o) this GitHub Models backend
is clean, stable, and fully documented today.
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from autopilot.models import BudgetPool, ModelConfig, Response


_GITHUB_MODELS_URL = "https://models.inference.ai.azure.com/chat/completions"


async def send(
    messages: list[dict[str, str]],
    config: ModelConfig,
    github_token: str,
    timeout: float = 60.0,
) -> Response:
    """
    Send a chat completion request to GitHub Models.

    Args:
        messages:     OpenAI-style message list.
        config:       ModelConfig (provider="github_models").
        github_token: GitHub PAT with models:read scope.
        timeout:      Request timeout in seconds.
    """
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "stream": False,
    }

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(_GITHUB_MODELS_URL, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    latency_ms = (time.perf_counter() - t0) * 1000

    choice = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    premium_requests = config.premium_request_multiplier

    return Response(
        text=choice,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        backend_id=config.backend_id,
        model=config.model,
        budget_pool=BudgetPool.COPILOT_PREMIUM,
        cost_usd=0.0,
        premium_requests_used=premium_requests,
    )


async def list_available_models(github_token: str) -> list[dict]:
    """
    List models available through the GitHub Models API.
    Returns a list of model metadata dicts.
    """
    url = "https://models.inference.ai.azure.com/models"
    headers = {"Authorization": f"Bearer {github_token}"}
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        return r.json()
