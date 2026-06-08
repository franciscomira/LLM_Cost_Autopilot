"""
router.py

Phase 2 routing brain.

Flow:
  1. classify_prompt()  — asks the local Ollama router model for (tier, confidence)
  2. resolve_backend()  — applies routing.yaml policy + live BudgetSnapshot
  3. route()            — combines both; this is what callers use

Usage:
    selected_config, tier, confidence = await route(
        prompt_text=user_message,
        budget=budget_state,
        registry=registry,
        settings=settings,
    )
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from budget import BudgetState
from models import BudgetPool, BudgetSnapshot, ModelConfig
from registry import ModelRegistry
from interface import AutopilotSettings


# ── Prompt template ────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "router_classify.txt"

def _load_prompt_template() -> str:
    return _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")


# ── Ollama call (direct HTTP — keeps the router dependency minimal) ────────────

@retry(
    retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
async def _call_ollama_router(
    prompt_text: str,
    router_config: ModelConfig,
    base_url: str,
    timeout: float = 30.0,
) -> str:
    """Return the raw text response from the local router model."""
    template = _load_prompt_template()
    full_prompt = template.replace("{PROMPT}", prompt_text.strip())

    payload = {
        "model": router_config.model,
        "messages": [{"role": "user", "content": full_prompt}],
        "stream": False,
        # Keep the response tight — we only need a small JSON object.
        "options": {"num_predict": 32, "temperature": 0.0},
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{base_url}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()

    return data["message"]["content"]


# ── JSON extraction with regex fallback ───────────────────────────────────────

def _parse_classification(raw: str) -> tuple[int, float]:
    """
    Extract (tier, confidence) from the model output.
    Tries strict JSON parse first; falls back to regex extraction.
    Returns (2, 2.0) as a safe default if both fail.
    """
    # Try to find a JSON object anywhere in the output
    match = re.search(r'\{[^{}]*"tier"[^{}]*\}', raw, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            tier = int(obj["tier"])
            confidence = float(obj["confidence"])
            if tier not in (1, 2, 3):
                raise ValueError(f"tier out of range: {tier}")
            if not (1.0 <= confidence <= 5.0):
                raise ValueError(f"confidence out of range: {confidence}")
            return tier, confidence
        except (KeyError, ValueError, json.JSONDecodeError):
            pass

    # Regex fallback — extract numbers directly
    tier_match = re.search(r'"tier"\s*:\s*([123])', raw)
    conf_match = re.search(r'"confidence"\s*:\s*([1-5](?:\.\d+)?)', raw)
    if tier_match and conf_match:
        return int(tier_match.group(1)), float(conf_match.group(1))

    # Cannot parse — default to Tier 2 with low confidence so it escalates safely
    return 2, 2.0


# ── Classification entry point ─────────────────────────────────────────────────

async def classify_prompt(
    prompt_text: str,
    router_config: ModelConfig,
    settings: AutopilotSettings,
    raise_on_error: bool = False,
) -> tuple[int, float]:
    """
    Ask the local Ollama router model to classify prompt_text.
    Returns (tier: 1|2|3, confidence: 1.0–5.0).
    """
    try:
        raw = await _call_ollama_router(
            prompt_text=prompt_text,
            router_config=router_config,
            base_url=settings.ollama_base_url,
        )
        return _parse_classification(raw)
    except Exception as exc:
        if raise_on_error:
            raise
        # Router unavailable — default to Tier 2 so we never silently under-serve
        return 2, 1.0


# ── Budget-aware backend resolution ───────────────────────────────────────────

async def resolve_backend(
    tier: int,
    confidence: float,
    budget: BudgetState,
    registry: ModelRegistry,
) -> tuple[ModelConfig, str]:
    """
    Map (tier, confidence) to a concrete ModelConfig, applying:
      1. Low-confidence escalation  — if confidence < policy minimum, bump tier up.
      2. Tier-3 Claude reservation  — confidence ≥ claude_reserve_threshold → Claude.
      3. Budget guardrails          — if the preferred pool is exhausted, fall back.

    Returns (ModelConfig, routing_reason) where routing_reason is one of:
      "primary"                  — normal path, primary backend for the resolved tier
      "low_confidence"           — bumped one tier up because confidence was too low
      "claude_reserve_threshold" — Tier 3 sent to Claude (confidence ≥ cutoff)
      "budget_spill_to_claude"   — Copilot pool low; spilled to Claude
      "budget_exhausted"         — both pools low; best-effort choice
      "fallback"                 — primary pool low; using fallback backend
    """
    snapshot: BudgetSnapshot = await budget.snapshot()
    policy = registry.tier_policy(tier)
    thresholds = registry.low_budget_thresholds

    reason = "primary"

    # Low-confidence escalation: bump to the next tier
    if confidence < policy.get("confidence_min", 3) and tier < 3:
        tier += 1
        policy = registry.tier_policy(tier)
        reason = "low_confidence"

    # Tier 3: check if this should go to Claude instead of Copilot top
    if tier == 3:
        claude_cutoff = registry.claude_reserve_threshold()
        sonnet_cutoff = registry.claude_sonnet_threshold()
        haiku_cfg = registry.get("claude_haiku")
        sonnet_cfg = registry.get("claude_sonnet")
        copilot_top_cfg = registry.primary_backend(3)

        copilot_low = (
            snapshot.copilot_remaining_requests
            <= thresholds.get("copilot_requests_remaining", 30)
        )
        claude_low = (
            snapshot.claude_remaining_usd
            <= thresholds.get("claude_usd_remaining", 5.0)
        )

        if confidence >= claude_cutoff and not claude_low:
            claude_cfg = sonnet_cfg if confidence >= sonnet_cutoff else haiku_cfg
            return claude_cfg, "claude_reserve_threshold"
        if copilot_low and not claude_low:
            return haiku_cfg, "budget_spill_to_claude"
        if copilot_low and claude_low:
            if snapshot.copilot_remaining_requests > 0:
                return copilot_top_cfg, "budget_exhausted"
            return haiku_cfg, "budget_exhausted"
        return copilot_top_cfg, reason

    # Tier 1 or 2 — use primary unless its pool is exhausted
    primary = registry.primary_backend(tier)
    fallback = registry.fallback_backend(tier)

    if _pool_is_healthy(primary.budget_pool, snapshot, thresholds):
        return primary, reason

    if _pool_is_healthy(fallback.budget_pool, snapshot, thresholds):
        return fallback, "fallback"

    return primary, "budget_exhausted"


def _pool_is_healthy(
    pool: BudgetPool,
    snapshot: BudgetSnapshot,
    thresholds: dict,
) -> bool:
    if pool == BudgetPool.FREE:
        return True
    if pool == BudgetPool.COPILOT_PREMIUM:
        return (
            snapshot.copilot_remaining_requests
            > thresholds.get("copilot_requests_remaining", 30)
        )
    if pool == BudgetPool.CLAUDE_CREDIT:
        return (
            snapshot.claude_remaining_usd
            > thresholds.get("claude_usd_remaining", 5.0)
        )
    return True


# ── Main entry point ───────────────────────────────────────────────────────────

async def route(
    prompt_text: str,
    budget: BudgetState,
    registry: ModelRegistry,
    settings: AutopilotSettings,
) -> tuple[ModelConfig, int, float, str]:
    """
    Full routing pipeline: classify then resolve.
    Returns (selected_backend_config, tier, confidence, routing_reason).
    """
    router_cfg = registry.router_config()

    tier, confidence = await classify_prompt(
        prompt_text=prompt_text,
        router_config=router_cfg,
        settings=settings,
    )

    selected, reason = await resolve_backend(
        tier=tier,
        confidence=confidence,
        budget=budget,
        registry=registry,
    )

    return selected, tier, confidence, reason
