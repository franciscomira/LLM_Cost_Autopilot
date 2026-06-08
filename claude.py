"""
src/autopilot/backends/claude.py

Calls the Anthropic API using the standard `anthropic` Python SDK.
This is the Tier-3 reserved backend — used only for the hardest requests.

── AUTH MODE ───────────────────────────────────────────────────────────────────
BEFORE June 15 2026 (dev mode):
    Set ANTHROPIC_API_KEY in your .env.
    The standard client uses this key with pay-per-token billing.

AFTER June 15 2026 (subscription mode):
    Set USE_CLAUDE_SUBSCRIPTION=true in your .env.
    The client authenticates via your Claude Code OAuth session, drawing
    from your Pro plan's $20 Agent SDK credit (API-rate metered, no rollover).
    Make sure you're logged in: run `claude` in your terminal at least once.

    The subscription auth path uses the Claude Agent SDK's OAuth mechanism.
    If the SDK's Python auth API changes, check:
    https://docs.anthropic.com/en/docs/claude-code/sdk
────────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
import time

import anthropic

from models import BudgetPool, ModelConfig, Response


def _build_client(use_subscription: bool) -> anthropic.Anthropic:
    """
    Build the Anthropic client.

    In subscription mode, we omit the api_key so the SDK falls back to its
    OAuth credential chain (Claude Code CLI session → ~/.claude credentials).
    This is the supported path for Agent SDK usage on subscription plans.
    """
    if use_subscription:
        # No api_key → SDK uses OAuth from your `claude` CLI session.
        # If this raises an AuthenticationError, run `claude` in terminal first.
        return anthropic.Anthropic()
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. Either set it in .env for dev mode, "
                "or set USE_CLAUDE_SUBSCRIPTION=true to use your Pro plan credit."
            )
        return anthropic.Anthropic(api_key=api_key)


async def send(
    messages: list[dict[str, str]],
    config: ModelConfig,
    use_subscription: bool = False,
    max_tokens: int = 2048,
) -> Response:
    """
    Send a chat completion request to Claude.

    Args:
        messages:         OpenAI-style message list. System messages are extracted
                          and passed separately as Anthropic requires.
        config:           ModelConfig (provider="anthropic").
        use_subscription: True → OAuth/subscription auth; False → API key.
        max_tokens:       Max output tokens (default 2048 — keeps cost down).
    """
    # Separate system prompt from conversation (Anthropic API format)
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    system_prompt = "\n\n".join(system_parts) if system_parts else None
    conversation = [m for m in messages if m["role"] != "system"]

    client = _build_client(use_subscription)

    t0 = time.perf_counter()
    kwargs: dict = {
        "model": config.model,
        "max_tokens": max_tokens,
        "messages": conversation,
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    # anthropic SDK is sync; wrap in asyncio executor for the async interface
    import asyncio
    loop = asyncio.get_event_loop()
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
