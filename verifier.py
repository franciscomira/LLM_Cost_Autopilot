"""
verifier.py

Phase 3 — Sampled Async Verification Loop.

Responsibilities:
  1. Decide whether a completed request should be verified (sampling + low-confidence).
  2. Re-run the prompt on the verifier backend and score agreement via LLM-as-judge.
  3. If agreement is below threshold, record it as a mis-route and append a corrected
     example to the training dataset (the feedback flywheel).
  4. Write the result to verification_log in SQLite.

This module is pure logic — it does not own a queue or background task.
See verification_queue.py for the async worker that calls run_verification_job().
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

from budget import BudgetState
from interface import AutopilotSettings, send_request
from models import Response
from registry import ModelRegistry

DATASET_PATH = Path(__file__).parent / "data" / "routing_dataset.jsonl"

# ── Judge prompt ───────────────────────────────────────────────────────────────

_JUDGE_PROMPT = """\
You are an impartial quality judge for LLM responses.

USER REQUEST:
{user_request}

RESPONSE A (the response being evaluated):
{response_a}

REFERENCE RESPONSE B (a higher-quality reference):
{response_b}

Score how well Response A answers the user request compared to Response B.
Consider: factual accuracy, completeness, relevance, and absence of harmful errors.
Ignore stylistic differences.

Output ONLY valid JSON with a single key:
{{"score": <integer 1-5>}}

Where:
  5 = Response A is equally good or better than B
  4 = Response A is slightly worse but still useful
  3 = Response A is noticeably worse — missing key points
  2 = Response A is substantially wrong or incomplete
  1 = Response A is harmful, hallucinated, or completely off
"""


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class VerificationJob:
    request_log_id: int
    messages: list[dict[str, str]]
    original_response: Response
    tier: int
    confidence: float


@dataclass
class VerificationResult:
    job: VerificationJob
    verifier_response: Response
    agreement_score: float      # 0.0–1.0 (normalised from 1–5 judge score)
    is_mis_route: bool
    added_to_training: bool


# ── Sampling decision ──────────────────────────────────────────────────────────

def should_verify(response: Response, verification_config: dict) -> bool:
    """
    Return True if this response should be queued for out-of-band verification.

    Two triggers from routing.yaml:
      - random sample_rate (default 10%)
      - router_confidence below always_verify_confidence_below (default 3)
    """
    if response.router_confidence is not None:
        threshold = verification_config.get("always_verify_confidence_below", 3)
        if response.router_confidence < threshold:
            return True

    sample_rate = verification_config.get("sample_rate", 0.10)
    return random.random() < sample_rate


# ── Judge score parsing ────────────────────────────────────────────────────────

def _parse_judge_score(raw: str) -> float:
    """
    Extract the judge score from raw LLM output.
    Returns a normalised 0.0–1.0 value (judge outputs 1–5).
    Falls back to 1.0 (assume good) if unparseable — erring on the side of
    not falsely flagging mis-routes when the judge itself fails.
    """
    match = re.search(r'"score"\s*:\s*([1-5])', raw)
    if match:
        raw_score = int(match.group(1))
        return (raw_score - 1) / 4.0   # map 1→0.0, 5→1.0
    return 1.0


# ── LLM-as-judge call ─────────────────────────────────────────────────────────

async def _score_agreement(
    messages: list[dict[str, str]],
    original_text: str,
    reference_text: str,
    verifier_config: ModelRegistry,
    budget: BudgetState,
    settings: AutopilotSettings,
) -> float:
    """
    Ask the verifier backend to score how well the original response compares
    to the reference. Returns a normalised agreement score 0.0–1.0.
    """
    user_request = " ".join(m["content"] for m in messages if m.get("role") == "user")

    judge_messages = [
        {
            "role": "user",
            "content": _JUDGE_PROMPT.format(
                user_request=user_request[:2000],
                response_a=original_text[:2000],
                response_b=reference_text[:2000],
            ),
        }
    ]

    verifier_cfg = verifier_config.get(
        verifier_config.verification_config.get("verifier_backend", "copilot_top")
    )

    try:
        resp, _ = await send_request(
            messages=judge_messages,
            config=verifier_cfg,
            budget=budget,
            settings=settings,
            log=False,
        )
        return _parse_judge_score(resp.text)
    except Exception:
        return 1.0   # assume good if the judge itself errors


# ── Training data flywheel ─────────────────────────────────────────────────────

def _corrected_tier(original_tier: int) -> int:
    """A mis-routed request should have gone one tier higher."""
    return min(original_tier + 1, 3)


def _append_training_example(
    prompt: str,
    predicted_tier: int,
    corrected_tier: int,
    dataset_path: Path = DATASET_PATH,
) -> None:
    """
    Append a new labeled example to the JSONL training dataset.
    The note marks it as verifier-generated so it can be audited separately.
    """
    example = {
        "prompt": prompt,
        "tier": corrected_tier,
        "notes": f"verifier-generated: was routed to tier {predicted_tier}, "
                 f"corrected to tier {corrected_tier}",
    }
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(example) + "\n")


# ── Main verification job ─────────────────────────────────────────────────────

async def run_verification_job(
    job: VerificationJob,
    registry: ModelRegistry,
    budget: BudgetState,
    settings: AutopilotSettings,
) -> VerificationResult:
    """
    Full verification pipeline for one request:
      1. Re-run the prompt on the verifier backend (one tier up).
      2. Score agreement between the original and reference responses.
      3. If it's a mis-route, add to training data and log the result.
      4. Write to verification_log in SQLite.
    """
    verification_cfg = registry.verification_config
    min_agreement = verification_cfg.get("min_agreement_score", 0.75)

    # Re-run on the verifier backend to get a reference response
    verifier_backend_id = verification_cfg.get("verifier_backend", "copilot_top")
    verifier_cfg = registry.get(verifier_backend_id)

    try:
        ref_response, _ = await send_request(
            messages=job.messages,
            config=verifier_cfg,
            budget=budget,
            settings=settings,
            log=False,
            was_escalated=True,
        )
    except Exception:
        # Verifier backend unavailable — skip gracefully
        return VerificationResult(
            job=job,
            verifier_response=job.original_response,
            agreement_score=1.0,
            is_mis_route=False,
            added_to_training=False,
        )

    # Score agreement via LLM-as-judge
    agreement = await _score_agreement(
        messages=job.messages,
        original_text=job.original_response.text,
        reference_text=ref_response.text,
        verifier_config=registry,
        budget=budget,
        settings=settings,
    )

    is_mis_route = agreement < min_agreement
    added_to_training = False
    corrected = _corrected_tier(job.tier)

    if is_mis_route:
        user_prompt = " ".join(
            m["content"] for m in job.messages if m.get("role") == "user"
        )
        _append_training_example(
            prompt=user_prompt,
            predicted_tier=job.tier,
            corrected_tier=corrected,
        )
        added_to_training = True

    # Persist to verification_log
    budget.log_verification(
        request_log_id=job.request_log_id,
        verified_at=time.time(),
        original_backend_id=job.original_response.backend_id,
        verifier_backend_id=verifier_backend_id,
        agreement_score=agreement,
        is_mis_route=is_mis_route,
        added_to_training=added_to_training,
        original_tier=job.tier,
        corrected_tier=corrected if is_mis_route else None,
    )

    return VerificationResult(
        job=job,
        verifier_response=ref_response,
        agreement_score=agreement,
        is_mis_route=is_mis_route,
        added_to_training=added_to_training,
    )
