"""
tests/test_verifier.py

Unit tests for verifier.py (no network) and an integration test gated behind
RUN_VERIFIER_EVAL=1 that exercises a real verification job against live backends.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models import BudgetPool, Response
from verifier import (
    VerificationJob,
    _append_training_example,
    _corrected_tier,
    _parse_judge_score,
    should_verify,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_response(
    tier: int = 2,
    confidence: float = 4.0,
    backend_id: str = "copilot_mid",
    text: str = "test response",
) -> Response:
    return Response(
        text=text,
        input_tokens=10,
        output_tokens=20,
        latency_ms=100.0,
        backend_id=backend_id,
        model="gpt-4o-mini",
        budget_pool=BudgetPool.COPILOT_PREMIUM,
        complexity_tier=tier,
        router_confidence=confidence,
    )


def _make_job(response: Response | None = None) -> VerificationJob:
    resp = response or _make_response()
    return VerificationJob(
        request_log_id=42,
        messages=[{"role": "user", "content": "Summarise this text."}],
        original_response=resp,
        tier=resp.complexity_tier or 2,
        confidence=resp.router_confidence or 4.0,
    )


# ── should_verify ──────────────────────────────────────────────────────────────

class TestShouldVerify:
    def test_low_confidence_always_triggers(self):
        cfg = {"always_verify_confidence_below": 3, "sample_rate": 0.0}
        resp = _make_response(confidence=2.9)
        assert should_verify(resp, cfg) is True

    def test_high_confidence_below_threshold_does_not_trigger_at_zero_rate(self):
        cfg = {"always_verify_confidence_below": 3, "sample_rate": 0.0}
        resp = _make_response(confidence=4.0)
        assert should_verify(resp, cfg) is False

    def test_sample_rate_100_always_triggers(self):
        cfg = {"always_verify_confidence_below": 3, "sample_rate": 1.0}
        resp = _make_response(confidence=5.0)
        assert should_verify(resp, cfg) is True

    def test_none_confidence_falls_through_to_sample_rate(self):
        cfg = {"always_verify_confidence_below": 3, "sample_rate": 1.0}
        resp = _make_response(confidence=5.0)
        resp.router_confidence = None
        assert should_verify(resp, cfg) is True

    def test_confidence_exactly_at_threshold_does_not_trigger(self):
        # threshold is "below X", so exactly X should NOT trigger
        cfg = {"always_verify_confidence_below": 3, "sample_rate": 0.0}
        resp = _make_response(confidence=3.0)
        assert should_verify(resp, cfg) is False


# ── _parse_judge_score ─────────────────────────────────────────────────────────

class TestParseJudgeScore:
    def test_parses_clean_json(self):
        assert _parse_judge_score('{"score": 5}') == 1.0
        assert _parse_judge_score('{"score": 1}') == 0.0
        assert _parse_judge_score('{"score": 3}') == pytest.approx(0.5)

    def test_parses_score_embedded_in_prose(self):
        raw = 'Here is my assessment: {"score": 4} based on the criteria.'
        assert _parse_judge_score(raw) == pytest.approx(0.75)

    def test_fallback_on_malformed(self):
        assert _parse_judge_score("no score here") == 1.0
        assert _parse_judge_score("") == 1.0
        assert _parse_judge_score('{"rating": 3}') == 1.0


# ── _corrected_tier ────────────────────────────────────────────────────────────

class TestCorrectedTier:
    def test_tier1_corrects_to_2(self):
        assert _corrected_tier(1) == 2

    def test_tier2_corrects_to_3(self):
        assert _corrected_tier(2) == 3

    def test_tier3_stays_at_3(self):
        assert _corrected_tier(3) == 3


# ── _append_training_example ───────────────────────────────────────────────────

class TestAppendTrainingExample:
    def test_appends_valid_jsonl(self):
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w"
        ) as f:
            path = Path(f.name)

        try:
            _append_training_example("Summarise this.", 1, 2, dataset_path=path)
            lines = path.read_text().strip().splitlines()
            assert len(lines) == 1
            obj = json.loads(lines[0])
            assert obj["tier"] == 2
            assert obj["prompt"] == "Summarise this."
            assert "verifier-generated" in obj["notes"]
        finally:
            path.unlink(missing_ok=True)

    def test_appends_multiple_examples(self):
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w"
        ) as f:
            path = Path(f.name)

        try:
            _append_training_example("Prompt A", 1, 2, dataset_path=path)
            _append_training_example("Prompt B", 2, 3, dataset_path=path)
            lines = path.read_text().strip().splitlines()
            assert len(lines) == 2
        finally:
            path.unlink(missing_ok=True)


# ── Integration test (live backends) ──────────────────────────────────────────

@pytest.mark.skipif(
    not os.environ.get("RUN_VERIFIER_EVAL"),
    reason="Set RUN_VERIFIER_EVAL=1 to run live backend verification test",
)
def test_verification_job_live() -> None:
    """
    Runs a single verification job against live backends.
    Checks that a deliberately weak response gets flagged as a mis-route.
    """
    from budget import BudgetState
    from hardware_profile import profile_hardware, recommend_models
    from interface import AutopilotSettings
    from registry import ModelRegistry
    from verifier import run_verification_job

    settings = AutopilotSettings.from_env()
    hw = profile_hardware()
    models = recommend_models(hw)
    registry = ModelRegistry(recommended_models=models)
    budget = BudgetState(
        "data/test_verifier.db",
        claude_monthly_limit_usd=20.0,
        copilot_monthly_requests_limit=300.0,
    )

    # Craft a deliberately poor response to a complex prompt
    poor_response = _make_response(
        tier=1,
        confidence=2.0,
        text="Yes.",   # clearly inadequate for any real question
        backend_id="ollama_tier1",
    )
    poor_response.budget_pool = BudgetPool.FREE

    job = VerificationJob(
        request_log_id=999,
        messages=[{
            "role": "user",
            "content": (
                "Explain the trade-offs between eventual consistency and "
                "strong consistency for a distributed checkout service."
            ),
        }],
        original_response=poor_response,
        tier=1,
        confidence=2.0,
    )

    result = asyncio.run(
        run_verification_job(job=job, registry=registry, budget=budget, settings=settings)
    )

    print(f"\nagreement_score: {result.agreement_score:.2f}")
    print(f"is_mis_route:    {result.is_mis_route}")
    print(f"added_to_training: {result.added_to_training}")
    print(f"verifier text preview: {result.verifier_response.text[:200]}")

    # A one-word answer to a complex question should score below 0.75
    assert result.agreement_score < 0.75, (
        f"Expected poor response to be flagged but got score {result.agreement_score:.2f}"
    )
    assert result.is_mis_route is True
