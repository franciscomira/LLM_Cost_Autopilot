"""
tests/test_routing.py

Sprint 4 — Test Coverage + Sprint 6 — Operational Polish

Unit tests for resolve_backend (mocked BudgetState.snapshot) and integration
tests for POST /v1/completions via httpx.AsyncClient(app=app).
Sprint 6 adds: SSE streaming, versioned migration table, month-end notification.

Run with: pytest tests/test_routing.py -v
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime as _real_datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from autopilot.hardware_profile import RecommendedModels
from autopilot.models import BudgetPool, BudgetSnapshot, ModelConfig, QualityTier, Response
from autopilot.budget import BudgetState
from autopilot.registry import ModelRegistry
from autopilot.router import resolve_backend, _parse_classification

_ROUTING_YAML = Path(__file__).parents[1] / "src" / "autopilot" / "routing.yaml"
_FAKE_MODELS = RecommendedModels(
    router_model="llama3.2:1b",
    tier1_model="llama3.2:3b",
    hardware_tier_name="small",
    effective_memory_gb=8.0,
)


def _registry() -> ModelRegistry:
    return ModelRegistry(
        routing_config_path=_ROUTING_YAML,
        recommended_models=_FAKE_MODELS,
    )


def _snapshot(
    claude_remaining: float = 15.0,
    copilot_remaining: float = 200.0,
) -> BudgetSnapshot:
    return BudgetSnapshot(
        claude_spent_usd=20.0 - claude_remaining,
        claude_limit_usd=20.0,
        copilot_requests_used=300.0 - copilot_remaining,
        copilot_requests_limit=300.0,
        month_key="2026-06",
    )


def _mock_budget(snapshot: BudgetSnapshot) -> BudgetState:
    budget = MagicMock(spec=BudgetState)
    budget.snapshot = AsyncMock(return_value=snapshot)
    return budget


# ── _parse_classification ──────────────────────────────────────────────────────

def test_parse_classification_valid_json():
    raw = '{"tier": 2, "confidence": 3.5}'
    assert _parse_classification(raw) == (2, 3.5)


def test_parse_classification_embedded_json():
    raw = 'Here is the answer: {"tier": 1, "confidence": 4.0} done.'
    assert _parse_classification(raw) == (1, 4.0)


def test_parse_classification_regex_fallback():
    raw = '"tier": 3, "confidence": 4.7'
    assert _parse_classification(raw) == (3, 4.7)


def test_parse_classification_defaults_on_garbage():
    assert _parse_classification("no json here at all") == (2, 2.0)


def test_parse_classification_out_of_range_tier_falls_back():
    # tier=9 is invalid; falls back to regex or default
    raw = '{"tier": 9, "confidence": 3.0}'
    tier, _ = _parse_classification(raw)
    assert tier in (1, 2, 3)


# ── resolve_backend — normal paths ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_resolve_backend_tier1_primary():
    registry = _registry()
    budget = _mock_budget(_snapshot())
    cfg, reason = await resolve_backend(
        tier=1, confidence=4.0, budget=budget, registry=registry
    )
    assert cfg.backend_id == "ollama_tier1"
    assert reason == "primary"


@pytest.mark.asyncio
async def test_resolve_backend_tier2_primary():
    registry = _registry()
    budget = _mock_budget(_snapshot())
    cfg, reason = await resolve_backend(
        tier=2, confidence=4.0, budget=budget, registry=registry
    )
    assert cfg.backend_id == "copilot_mid"
    assert reason == "primary"


@pytest.mark.asyncio
async def test_resolve_backend_tier3_copilot_when_confidence_below_threshold():
    registry = _registry()
    # claude_reserve_threshold is 4.5; confidence 3.0 stays on Copilot
    budget = _mock_budget(_snapshot())
    cfg, reason = await resolve_backend(
        tier=3, confidence=3.0, budget=budget, registry=registry
    )
    assert cfg.backend_id == "copilot_top"
    assert reason == "primary"


@pytest.mark.asyncio
async def test_resolve_backend_tier3_routes_to_claude_haiku():
    registry = _registry()
    # confidence >= 4.5 and < 4.8 -> claude_haiku
    budget = _mock_budget(_snapshot(claude_remaining=15.0))
    cfg, reason = await resolve_backend(
        tier=3, confidence=4.6, budget=budget, registry=registry
    )
    assert cfg.backend_id == "claude_haiku"
    assert reason == "claude_reserve_threshold"


@pytest.mark.asyncio
async def test_resolve_backend_tier3_routes_to_claude_sonnet():
    registry = _registry()
    # confidence >= 4.8 -> claude_sonnet
    budget = _mock_budget(_snapshot(claude_remaining=15.0))
    cfg, reason = await resolve_backend(
        tier=3, confidence=4.9, budget=budget, registry=registry
    )
    assert cfg.backend_id == "claude_sonnet"
    assert reason == "claude_reserve_threshold"


# ── resolve_backend — low-confidence escalation ───────────────────────────────

@pytest.mark.asyncio
async def test_resolve_backend_low_confidence_tier1_escalates_to_tier2():
    registry = _registry()
    budget = _mock_budget(_snapshot())
    # confidence_min for tier 1 is 3; passing 2.0 forces escalation
    cfg, reason = await resolve_backend(
        tier=1, confidence=2.0, budget=budget, registry=registry
    )
    assert cfg.backend_id == "copilot_mid"
    assert reason == "low_confidence"


@pytest.mark.asyncio
async def test_resolve_backend_low_confidence_tier2_escalates_to_tier3():
    registry = _registry()
    budget = _mock_budget(_snapshot())
    cfg, reason = await resolve_backend(
        tier=2, confidence=1.5, budget=budget, registry=registry
    )
    # tier 3 primary is copilot_top (confidence 1.5 < 4.5 threshold)
    assert cfg.backend_id == "copilot_top"
    assert reason == "low_confidence"


# ── resolve_backend — budget guardrails (the critical safety invariant) ───────

@pytest.mark.asyncio
async def test_resolve_backend_copilot_exhausted_spills_to_claude():
    """When Copilot is low and Claude has budget, Tier 3 spills to Claude Haiku."""
    registry = _registry()
    # copilot_remaining <= 30 triggers low-budget path
    budget = _mock_budget(_snapshot(claude_remaining=15.0, copilot_remaining=10.0))
    cfg, reason = await resolve_backend(
        tier=3, confidence=3.0, budget=budget, registry=registry
    )
    assert cfg.backend_id == "claude_haiku"
    assert reason == "budget_spill_to_claude"


@pytest.mark.asyncio
async def test_resolve_backend_both_pools_low_prefers_copilot_if_requests_remain():
    """When both pools are low but Copilot still has some requests, use it."""
    registry = _registry()
    budget = _mock_budget(_snapshot(claude_remaining=1.0, copilot_remaining=10.0))
    cfg, reason = await resolve_backend(
        tier=3, confidence=3.0, budget=budget, registry=registry
    )
    assert cfg.backend_id == "copilot_top"
    assert reason == "budget_exhausted"


@pytest.mark.asyncio
async def test_resolve_backend_both_pools_exhausted_falls_back_to_haiku():
    """When both pools are fully gone, best-effort fallback is Claude Haiku."""
    registry = _registry()
    budget = _mock_budget(_snapshot(claude_remaining=1.0, copilot_remaining=0.0))
    cfg, reason = await resolve_backend(
        tier=3, confidence=3.0, budget=budget, registry=registry
    )
    assert cfg.backend_id == "claude_haiku"
    assert reason == "budget_exhausted"


@pytest.mark.asyncio
async def test_resolve_backend_tier1_copilot_pool_low_uses_free_fallback(tmp_path):
    """Tier 1 primary is FREE (Ollama), so it's always healthy — no fallback needed."""
    registry = _registry()
    # Even with Copilot exhausted, Tier 1 primary (FREE pool) is fine
    budget = _mock_budget(_snapshot(copilot_remaining=0.0))
    cfg, reason = await resolve_backend(
        tier=1, confidence=4.0, budget=budget, registry=registry
    )
    assert cfg.backend_id == "ollama_tier1"
    assert reason == "primary"


@pytest.mark.asyncio
async def test_resolve_backend_tier2_copilot_exhausted_both_exhausted():
    """Both Tier 2 backends share COPILOT_PREMIUM — both unhealthy -> budget_exhausted on primary."""
    registry = _registry()
    # copilot_remaining=10 <= threshold 30 -> both copilot_mid and copilot_top are unhealthy
    budget = _mock_budget(_snapshot(copilot_remaining=10.0))
    cfg, reason = await resolve_backend(
        tier=2, confidence=4.0, budget=budget, registry=registry
    )
    # Primary is returned with budget_exhausted (both pools are on COPILOT_PREMIUM)
    assert cfg.backend_id == "copilot_mid"
    assert reason == "budget_exhausted"


# ── Integration tests — POST /v1/completions ──────────────────────────────────

def _make_fake_response(backend_id: str = "copilot_mid") -> Response:
    return Response(
        text="Hello!",
        input_tokens=10,
        output_tokens=5,
        latency_ms=200.0,
        backend_id=backend_id,
        model="gpt-4o-mini",
        budget_pool=BudgetPool.COPILOT_PREMIUM,
        complexity_tier=2,
        router_confidence=3.5,
        cost_usd=0.0,
        premium_requests_used=1.0,
    )


def _make_app_state(tmp_path: Path):
    """Return a minimal _AppState-like namespace for injection."""
    import autopilot.api as api_module
    import asyncio

    registry = _registry()
    budget = BudgetState(tmp_path / "test.db", claude_monthly_limit_usd=20.0)

    api_module._state.registry = registry
    api_module._state.budget = budget
    api_module._state.settings = MagicMock()
    api_module._state.config_lock = asyncio.Lock()

    vq = MagicMock()
    vq.enqueue = MagicMock()
    api_module._state.vq = vq

    return registry, budget


@pytest.mark.asyncio
async def test_completions_returns_200_and_routing_meta(tmp_path):
    import autopilot.api as api_module
    from autopilot.api import app

    _make_app_state(tmp_path)

    fake_cfg = _registry().get("copilot_mid")
    fake_resp = _make_fake_response("copilot_mid")

    with (
        patch("autopilot.api.route", new=AsyncMock(return_value=(fake_cfg, 2, 3.5, "primary"))),
        patch("autopilot.api.send_request", new=AsyncMock(return_value=(fake_resp, 1))),
        patch("autopilot.api.should_verify", return_value=False),
    ):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/v1/completions",
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )

    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "Hello!"
    assert body["routing"]["backend_id"] == "copilot_mid"
    assert body["routing"]["routing_reason"] == "primary"


@pytest.mark.asyncio
async def test_completions_uses_fallback_on_primary_failure(tmp_path):
    import autopilot.api as api_module
    from autopilot.api import app

    _make_app_state(tmp_path)

    fake_cfg_primary = _registry().get("copilot_mid")
    fake_cfg_fallback = _registry().get("copilot_top")
    fake_resp = _make_fake_response("copilot_top")

    call_count = 0

    async def _send_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("primary backend down")
        return fake_resp, 2

    with (
        patch("autopilot.api.route", new=AsyncMock(return_value=(fake_cfg_primary, 2, 3.5, "primary"))),
        patch("autopilot.api.send_request", side_effect=_send_side_effect),
        patch("autopilot.api.should_verify", return_value=False),
    ):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/v1/completions",
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )

    assert r.status_code == 200
    assert call_count == 2


@pytest.mark.asyncio
async def test_completions_returns_502_when_both_backends_fail(tmp_path):
    import autopilot.api as api_module
    from autopilot.api import app

    _make_app_state(tmp_path)

    fake_cfg = _registry().get("copilot_mid")

    with (
        patch("autopilot.api.route", new=AsyncMock(return_value=(fake_cfg, 2, 3.5, "primary"))),
        patch("autopilot.api.send_request", side_effect=RuntimeError("all backends down")),
        patch("autopilot.api.should_verify", return_value=False),
    ):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/v1/completions",
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )

    assert r.status_code == 502


@pytest.mark.asyncio
async def test_completions_rejects_oversized_message(tmp_path):
    from autopilot.api import app
    _make_app_state(tmp_path)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/v1/completions",
            json={"messages": [{"role": "user", "content": "x" * 33_000}]},
        )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_completions_rejects_empty_messages(tmp_path):
    from autopilot.api import app
    _make_app_state(tmp_path)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/v1/completions", json={"messages": []})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_health_endpoint():
    from autopilot.api import app

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_auth_middleware_blocks_missing_key(tmp_path, monkeypatch):
    import autopilot.api as api_module
    from autopilot.api import app

    monkeypatch.setattr(api_module, "_API_KEY", "secret")
    _make_app_state(tmp_path)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/v1/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_auth_middleware_passes_correct_key(tmp_path, monkeypatch):
    import autopilot.api as api_module
    from autopilot.api import app

    monkeypatch.setattr(api_module, "_API_KEY", "secret")
    _make_app_state(tmp_path)

    fake_cfg = _registry().get("copilot_mid")
    fake_resp = _make_fake_response("copilot_mid")

    with (
        patch("autopilot.api.route", new=AsyncMock(return_value=(fake_cfg, 2, 3.5, "primary"))),
        patch("autopilot.api.send_request", new=AsyncMock(return_value=(fake_resp, 1))),
        patch("autopilot.api.should_verify", return_value=False),
    ):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/v1/completions",
                headers={"X-API-Key": "secret"},
                json={"messages": [{"role": "user", "content": "Hi"}]},
            )
    assert r.status_code == 200


# ── Sprint 6: SSE streaming ────────────────────────────────────────────────────

def _parse_sse(text: str) -> list[dict]:
    """Return all parsed JSON objects from an SSE response body (skips [DONE])."""
    events = []
    for line in text.splitlines():
        if line.startswith("data: ") and line[6:].strip() != "[DONE]":
            events.append(json.loads(line[6:]))
    return events


@pytest.mark.asyncio
async def test_streaming_emits_routing_chunks_done(tmp_path):
    """stream:true returns text/event-stream with routing -> chunk(s) -> done."""
    from autopilot.api import app

    _make_app_state(tmp_path)
    fake_cfg = _registry().get("copilot_mid")  # github_models provider

    async def _fake_stream(messages, config, github_token, usage_out, timeout=60.0):
        yield "Hello "
        yield "world"
        usage_out.update({"input_tokens": 10, "output_tokens": 2, "latency_ms": 120.0})

    with (
        patch("autopilot.api.route", new=AsyncMock(return_value=(fake_cfg, 2, 3.5, "primary"))),
        patch("autopilot.api.github_models.send_stream", new=_fake_stream),
    ):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/v1/completions",
                json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
            )

    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]

    events = _parse_sse(r.text)
    event_types = [e["event"] for e in events]

    assert event_types[0] == "routing"
    assert "chunk" in event_types
    assert event_types[-1] == "done"

    assert events[0]["backend_id"] == "copilot_mid"
    assert "".join(e["text"] for e in events if e["event"] == "chunk") == "Hello world"

    done = events[-1]["routing"]
    assert done["input_tokens"] == 10
    assert done["output_tokens"] == 2
    assert done["backend_id"] == "copilot_mid"


@pytest.mark.asyncio
async def test_streaming_records_spend_after_stream(tmp_path):
    """Spend and audit log are written after the stream completes."""
    from autopilot.api import app

    _, budget = _make_app_state(tmp_path)
    fake_cfg = _registry().get("copilot_mid")

    async def _fake_stream(messages, config, github_token, usage_out, timeout=60.0):
        yield "OK"
        usage_out.update({"input_tokens": 5, "output_tokens": 1, "latency_ms": 50.0})

    with (
        patch("autopilot.api.route", new=AsyncMock(return_value=(fake_cfg, 2, 3.5, "primary"))),
        patch("autopilot.api.github_models.send_stream", new=_fake_stream),
    ):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/v1/completions",
                json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
            )

    snap = await budget.snapshot()
    assert snap.copilot_requests_used == 1.0  # premium_request_multiplier=1.0 for copilot_mid

    conn = sqlite3.connect(tmp_path / "test.db")
    rows = conn.execute("SELECT * FROM request_log").fetchall()
    conn.close()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_streaming_emits_error_event_on_provider_failure(tmp_path):
    """If the provider generator raises, the client receives an error SSE event."""
    from autopilot.api import app

    _make_app_state(tmp_path)
    fake_cfg = _registry().get("copilot_mid")

    async def _failing_stream(messages, config, github_token, usage_out, timeout=60.0):
        raise RuntimeError("upstream exploded")
        yield  # make it a generator

    with (
        patch("autopilot.api.route", new=AsyncMock(return_value=(fake_cfg, 2, 3.5, "primary"))),
        patch("autopilot.api.github_models.send_stream", new=_failing_stream),
    ):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/v1/completions",
                json={"messages": [{"role": "user", "content": "Hi"}], "stream": True},
            )

    assert r.status_code == 200  # HTTP layer is 200; error is carried in SSE body
    events = _parse_sse(r.text)
    assert any(e["event"] == "error" for e in events)


# ── Sprint 6: versioned migration table ───────────────────────────────────────

def test_migration_table_created_and_seeded(tmp_path):
    """Fresh DB has schema_migrations table with 001 recorded."""
    from autopilot.budget import BudgetState

    BudgetState(tmp_path / "test.db")

    conn = sqlite3.connect(tmp_path / "test.db")
    names = [r[0] for r in conn.execute("SELECT name FROM schema_migrations").fetchall()]
    conn.close()

    assert "001_add_routing_reason" in names


def test_migration_idempotent_on_reinit(tmp_path):
    """Re-initialising on the same DB records each migration exactly once."""
    from autopilot.budget import BudgetState

    BudgetState(tmp_path / "test.db")
    BudgetState(tmp_path / "test.db")  # second init on same file

    conn = sqlite3.connect(tmp_path / "test.db")
    count = conn.execute(
        "SELECT COUNT(*) FROM schema_migrations WHERE name = '001_add_routing_reason'"
    ).fetchone()[0]
    conn.close()

    assert count == 1


def test_migration_handles_column_already_present(tmp_path):
    """Fresh DB already has routing_reason column; second init must not raise."""
    from autopilot.budget import BudgetState

    # _SCHEMA creates routing_reason, so ALTER TABLE in the migration would fail
    # with OperationalError if not guarded — this test catches that regression.
    BudgetState(tmp_path / "test.db")
    b2 = BudgetState(tmp_path / "test.db")
    assert b2 is not None


# ── Sprint 6: month-end pre-notification ──────────────────────────────────────

class _FakeDatetime(_real_datetime):
    """Subclass of datetime that returns a fixed value from now()."""
    _fixed: "_real_datetime | None" = None

    @classmethod
    def now(cls, tz=None):
        return cls._fixed or _real_datetime.now(tz)


@pytest.mark.asyncio
async def test_month_end_notification_fires_within_3_days(tmp_path):
    """Notification fires when <= 3 days remain in the month and budget was used."""
    from autopilot.budget import BudgetState, BudgetPool

    budget = BudgetState(tmp_path / "test.db", alert_webhook_url="http://fake-webhook")
    await budget.record_spend(BudgetPool.CLAUDE_CREDIT, cost_usd=5.0)

    # June has 30 days; day 28 -> 2 days remaining -> should fire
    _FakeDatetime._fixed = _real_datetime(2026, 6, 28)

    mock_client = AsyncMock()
    mock_client.post = AsyncMock()

    with (
        patch("autopilot.budget.datetime", _FakeDatetime),
        patch("autopilot.budget.httpx.AsyncClient") as mock_cls,
    ):
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        await budget.check_month_end_notification()

    mock_client.post.assert_called_once()
    payload = mock_client.post.call_args[1]["json"]
    assert payload["event"] == "month_end_approaching"
    assert payload["days_remaining"] == 2
    assert payload["claude_spent_usd"] == 5.0


@pytest.mark.asyncio
async def test_month_end_notification_fires_only_once_per_month(tmp_path):
    """Calling check twice in the same month fires the webhook exactly once."""
    from autopilot.budget import BudgetState, BudgetPool

    budget = BudgetState(tmp_path / "test.db", alert_webhook_url="http://fake-webhook")
    await budget.record_spend(BudgetPool.CLAUDE_CREDIT, cost_usd=3.0)

    _FakeDatetime._fixed = _real_datetime(2026, 6, 28)

    mock_client = AsyncMock()
    mock_client.post = AsyncMock()

    with (
        patch("autopilot.budget.datetime", _FakeDatetime),
        patch("autopilot.budget.httpx.AsyncClient") as mock_cls,
    ):
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        await budget.check_month_end_notification()
        await budget.check_month_end_notification()  # second call same month

    assert mock_client.post.call_count == 1


@pytest.mark.asyncio
async def test_month_end_notification_does_not_fire_mid_month(tmp_path):
    """No notification when more than 3 days remain in the month."""
    from autopilot.budget import BudgetState, BudgetPool

    budget = BudgetState(tmp_path / "test.db", alert_webhook_url="http://fake-webhook")
    await budget.record_spend(BudgetPool.CLAUDE_CREDIT, cost_usd=5.0)

    _FakeDatetime._fixed = _real_datetime(2026, 6, 15)  # 15 days remaining

    mock_client = AsyncMock()
    mock_client.post = AsyncMock()

    with (
        patch("autopilot.budget.datetime", _FakeDatetime),
        patch("autopilot.budget.httpx.AsyncClient") as mock_cls,
    ):
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        await budget.check_month_end_notification()

    mock_client.post.assert_not_called()


@pytest.mark.asyncio
async def test_month_end_notification_silent_when_no_budget_used(tmp_path):
    """No notification when both budget pools are at zero."""
    from autopilot.budget import BudgetState

    budget = BudgetState(tmp_path / "test.db", alert_webhook_url="http://fake-webhook")
    # deliberately no spend recorded

    _FakeDatetime._fixed = _real_datetime(2026, 6, 29)  # 1 day remaining

    mock_client = AsyncMock()
    mock_client.post = AsyncMock()

    with (
        patch("autopilot.budget.datetime", _FakeDatetime),
        patch("autopilot.budget.httpx.AsyncClient") as mock_cls,
    ):
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        await budget.check_month_end_notification()

    mock_client.post.assert_not_called()
