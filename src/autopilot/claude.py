"""
src/autopilot/claude.py

Calls the Anthropic API using the standard `anthropic` Python SDK.
This is the Tier-3 reserved backend — used only for the hardest requests.

Auth: set ANTHROPIC_API_KEY in .env.
"""
from __future__ import annotations

import os
import time

import anthropic

from autopilot.models import BudgetPool, ModelConfig, Response


def _build_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set. Add it to .env.")
    return anthropic.Anthropic(api_key=api_key)


def _build_async_client() -> anthropic.AsyncAnthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set. Add it to .env.")
    return anthropic.AsyncAnthropic(api_key=api_key)


async def send(
    messages: list[dict[str, str]],
    config: ModelConfig,
    max_tokens: int = 2048,
) -> Response:
    """
    Send a chat completion request to Claude.

    Args:
        messages:   OpenAI-style message list. System messages are extracted
                    and passed separately as Anthropic requires.
        config:     ModelConfig (provider="anthropic").
        max_tokens: Max output tokens (default 2048 — keeps cost down).
    """
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    system_prompt = "\n\n".join(system_parts) if system_parts else None
    conversation = [m for m in messages if m["role"] != "system"]

    client = _build_client()

    t0 = time.perf_counter()
    kwargs: dict = {
        "model": config.model,
        "max_tokens": max_tokens,
        "messages": conversation,
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    import asyncio
    loop = asyncio.get_running_loop()
    message = await loop.run_in_executor(
        None,
        lambda: client.messages.create(**kwargs),
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    text = message.content[0].text
    input_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens
    cost_usd = config.estimate_cost_usd(input_tokens, output_tokens)

    return Response(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        backend_id=config.backend_id,
        model=config.model,
        budget_pool=BudgetPool.CLAUDE_CREDIT,
        cost_usd=cost_usd,
        premium_requests_used=0.0,
    )


async def send_stream(
    messages: list[dict[str, str]],
    config: ModelConfig,
    usage_out: dict,
    max_tokens: int = 2048,
):
    """
    Async generator that yields text chunks from the Anthropic streaming API.
    Populates *usage_out* with input_tokens, output_tokens, latency_ms after exhaustion.
    """
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    system_prompt = "\n\n".join(system_parts) if system_parts else None
    conversation = [m for m in messages if m["role"] != "system"]

    client = _build_async_client()
    kwargs: dict = {"model": config.model, "max_tokens": max_tokens, "messages": conversation}
    if system_prompt:
        kwargs["system"] = system_prompt

    t0 = time.perf_counter()
    async with client.messages.stream(**kwargs) as stream:
        async for text in stream.text_stream:
            yield text
        final = await stream.get_final_message()

    usage_out["input_tokens"] = final.usage.input_tokens
    usage_out["output_tokens"] = final.usage.output_tokens
    usage_out["latency_ms"] = (time.perf_counter() - t0) * 1000
