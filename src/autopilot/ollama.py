"""
src/autopilot/ollama.py

Async wrapper around Ollama's OpenAI-compatible HTTP endpoint.
Works for both the router model (classification) and the Tier-1 generator.

Ollama must be running locally: https://ollama.com
Start with: ollama serve
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from autopilot.models import BudgetPool, ModelConfig, Response


_DEFAULT_BASE_URL = "http://localhost:11434"


async def send(
    messages: list[dict[str, str]],
    config: ModelConfig,
    base_url: str = _DEFAULT_BASE_URL,
    timeout: float = 30.0,
) -> Response:
    """
    Send a chat completion request to Ollama.

    Args:
        messages: OpenAI-style message list, e.g.
                  [{"role": "user", "content": "Hello"}]
        config:   ModelConfig for this Ollama backend.
        base_url: Ollama server URL (default: localhost).
        timeout:  Request timeout in seconds (local models can be slow on CPU).
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": 1024},  # cap output to prevent infinite generation
    }

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
    latency_ms = (time.perf_counter() - t0) * 1000

    choice = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    return Response(
        text=choice,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        backend_id=config.backend_id,
        model=config.model,
        budget_pool=BudgetPool.FREE,
        cost_usd=0.0,
        premium_requests_used=0.0,
    )


async def list_local_models(base_url: str = _DEFAULT_BASE_URL) -> list[str]:
    """Return the names of all models currently pulled in Ollama."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{base_url.rstrip('/')}/api/tags")
        r.raise_for_status()
        data = r.json()
    return [m["name"] for m in data.get("models", [])]


async def pull_model(model: str, base_url: str = _DEFAULT_BASE_URL) -> None:
    """
    Pull a model into Ollama (equivalent to `ollama pull <model>`).
    Streams progress; logs completion when done.
    """
    async with httpx.AsyncClient(timeout=600.0) as client:
        async with client.stream(
            "POST",
            f"{base_url.rstrip('/')}/api/pull",
            json={"name": model},
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if '"status":"success"' in line:
                    break
