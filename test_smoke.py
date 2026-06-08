"""
tests/test_smoke.py

Phase 1 smoke tests — run with: pytest tests/test_smoke.py -v

These tests use real backends by default (integration style).
Set SMOKE_MOCK=true to skip actual API calls (CI mode).
"""
from __future__ import annotations

import os
import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

# Make src importable
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from autopilot.hardware_profile import profile_hardware, recommend_models
from autopilot.models import BudgetPool, ModelConfig, QualityTier, BudgetSnapshot
from autopilot.budget import BudgetState
from autopilot.interface import AutopilotSettings, send_request

MOCK_MODE = os.environ.get("SMOKE_MOCK", "false").lower() == "true"


# ── Hardware profiler ──────────────────────────────────────────────────────────

def test_hardware_profiler_returns_profile():
    hw = profile_hardware()
    assert hw.ram_gb > 0
    assert hw.cpu_cores > 0
    assert hw.gpu_type in ("nvidia", "apple_silicon", "amd", "cpu_only")
    assert hw.effective_memory_gb > 0
    assert hw.effective_memory_gb <= hw.ram_gb + hw.vram_gb + 1   # sanity


def test_hardware_profiler_effective_memory_is_positive():
    hw = profile_hardware()
    assert hw.effective_memory_gb > 0.5   # even the tiniest machines have some


def test_recommend_models_returns_valid_names(tmp_path):
    hw = profile_hardware()
    # Point at the real config file
    config = Path(__file__).parents[1] / "config" / "models_by_hardware.yaml"
    models = recommend_models(hw, config_path=config)
    assert models.router_model, "Router model should not be empty"
    assert models.tier1_model, "Tier-1 model should not be empty"
    assert ":" in models.router_model, "Ollama model names should include a tag"


# ── BudgetState ────────────────────────────────────────────────────────────────

def test_budget_state_fresh_snapshot(tmp_path):
    budget = BudgetState(tmp_path / "test.db", claude_monthly_limit_usd=20.0)
    snap = budget.snapshot()
    assert snap.claude_spent_usd == 0.0
    assert snap.copilot_requests_used == 0.0
    assert snap.claude_remaining_usd == 20.0


def test_budget_state_record_claude_spend(tmp_path):
    budget = BudgetState(tmp_path / "test.db", claude_monthly_limit_usd=20.0)
    budget.record_spend(pool=BudgetPool.CLAUDE_CREDIT, cost_usd=1.50)
    budget.record_spend(pool=BudgetPool.CLAUDE_CREDIT, cost_usd=0.25)
    snap = budget.snapshot()
    assert abs(snap.claude_spent_usd - 1.75) < 0.001
    assert abs(snap.claude_remaining_usd - 18.25) < 0.001


def test_budget_state_record_copilot_spend(tmp_path):
    budget = BudgetState(tmp_path / "test.db", copilot_monthly_requests_limit=300)
    budget.record_spend(pool=BudgetPool.COPILOT_PREMIUM, premium_requests=3.0)
    snap = budget.snapshot()
    assert snap.copilot_requests_used == 3.0
    assert snap.copilot_remaining_requests == 297.0


def test_budget_state_free_pool_no_spend(tmp_path):
    budget = BudgetState(tmp_path / "test.db")
    budget.record_spend(pool=BudgetPool.FREE)   # should be a no-op
    snap = budget.snapshot()
    assert snap.claude_spent_usd == 0.0
    assert snap.copilot_requests_used == 0.0


def test_budget_pool_healthy_checks(tmp_path):
    budget = BudgetState(
        tmp_path / "test.db",
        claude_monthly_limit_usd=20.0,
        copilot_monthly_requests_limit=300,
    )
    thresholds = {"claude_usd_remaining": 5.0, "copilot_requests_remaining": 30}

    assert budget.is_pool_healthy(BudgetPool.FREE, thresholds) is True
    assert budget.is_pool_healthy(BudgetPool.CLAUDE_CREDIT, thresholds) is True
    assert budget.is_pool_healthy(BudgetPool.COPILOT_PREMIUM, thresholds) is True

    # Burn near the limit
    budget.record_spend(BudgetPool.CLAUDE_CREDIT, cost_usd=16.0)  # $4 left < $5 floor
    assert budget.is_pool_healthy(BudgetPool.CLAUDE_CREDIT, thresholds) is False

    budget2 = BudgetState(tmp_path / "test2.db", copilot_monthly_requests_limit=300)
    budget2.record_spend(BudgetPool.COPILOT_PREMIUM, premium_requests=275)  # 25 left < 30
    assert budget2.is_pool_healthy(BudgetPool.COPILOT_PREMIUM, thresholds) is False


def test_budget_log_request_writes_row(tmp_path):
    import time, sqlite3
    budget = BudgetState(tmp_path / "test.db")
    budget.log_request(
        timestamp=time.time(),
        prompt_hash="abc123",
        complexity_tier=2,
        router_confidence=3.5,
        backend_id="copilot_mid",
        budget_pool=BudgetPool.COPILOT_PREMIUM,
        input_tokens=100,
        output_tokens=50,
        latency_ms=340.0,
        cost_usd=0.0,
        premium_requests=1.0,
    )
    conn = sqlite3.connect(tmp_path / "test.db")
    rows = conn.execute("SELECT * FROM request_log").fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0][5] == "copilot_mid"   # backend_id column


# ── ModelConfig ────────────────────────────────────────────────────────────────

def test_model_config_cost_estimate():
    cfg = ModelConfig(
        backend_id="claude",
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        budget_pool=BudgetPool.CLAUDE_CREDIT,
        quality_tier=QualityTier.HIGH,
        cost_per_input_token=0.000001,
        cost_per_output_token=0.000005,
    )
    cost = cfg.estimate_cost_usd(input_tokens=1000, output_tokens=200)
    assert abs(cost - 0.002) < 0.0001   # 1000×0.000001 + 200×0.000005


# ── send_request (mock mode) ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_send_request_ollama_mock(tmp_path):
    """Verify send_request dispatches to Ollama and records budget (mocked)."""
    from autopilot.models import Response
    from unittest.mock import patch

    cfg = ModelConfig(
        backend_id="ollama_tier1",
        provider="ollama",
        model="llama3.2:3b",
        budget_pool=BudgetPool.FREE,
        quality_tier=QualityTier.LOW,
    )
    budget = BudgetState(tmp_path / "test.db")
    settings = AutopilotSettings(ollama_base_url="http://localhost:11434")

    fake_response = Response(
        text="OK",
        input_tokens=10,
        output_tokens=1,
        latency_ms=120.0,
        backend_id="ollama_tier1",
        model="llama3.2:3b",
        budget_pool=BudgetPool.FREE,
    )

    with patch("autopilot.backends.ollama.send", new=AsyncMock(return_value=fake_response)):
        resp = await send_request(
            messages=[{"role": "user", "content": "Say OK"}],
            config=cfg,
            budget=budget,
            settings=settings,
            log=False,
        )

    assert resp.text == "OK"
    assert resp.budget_pool == BudgetPool.FREE
    snap = budget.snapshot()
    assert snap.claude_spent_usd == 0.0   # FREE pool — no charges


@pytest.mark.asyncio
async def test_send_request_records_copilot_spend(tmp_path):
    from autopilot.models import Response
    from unittest.mock import patch

    cfg = ModelConfig(
        backend_id="copilot_mid",
        provider="github_models",
        model="gpt-4o-mini",
        budget_pool=BudgetPool.COPILOT_PREMIUM,
        quality_tier=QualityTier.MID,
        premium_request_multiplier=1.0,
    )
    budget = BudgetState(tmp_path / "test.db", copilot_monthly_requests_limit=300)
    settings = AutopilotSettings(github_token="fake-token")

    fake_response = Response(
        text="OK",
        input_tokens=10,
        output_tokens=1,
        latency_ms=200.0,
        backend_id="copilot_mid",
        model="gpt-4o-mini",
        budget_pool=BudgetPool.COPILOT_PREMIUM,
        premium_requests_used=1.0,
    )

    with patch("autopilot.backends.github_models.send", new=AsyncMock(return_value=fake_response)):
        await send_request(
            messages=[{"role": "user", "content": "Say OK"}],
            config=cfg,
            budget=budget,
            settings=settings,
            log=False,
        )

    snap = budget.snapshot()
    assert snap.copilot_requests_used == 1.0
    assert snap.claude_spent_usd == 0.0
