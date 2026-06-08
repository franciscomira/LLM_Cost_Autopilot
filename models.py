"""
src/autopilot/models.py

Core data structures shared across the whole system.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class BudgetPool(str, Enum):
    FREE = "FREE"                    # local Ollama — no monetary cost
    COPILOT_PREMIUM = "COPILOT_PREMIUM"   # GitHub Copilot premium requests
    CLAUDE_CREDIT = "CLAUDE_CREDIT"  # Claude Pro Agent SDK monthly credit


class QualityTier(str, Enum):
    LOW = "low"       # Tier 1 — local
    MID = "mid"       # Tier 2 — Copilot mid
    HIGH = "high"     # Tier 3 — Copilot top / Claude


@dataclass
class ModelConfig:
    """Everything the system needs to know about one backend model."""
    backend_id: str               # key in routing.yaml, e.g. "copilot_mid"
    provider: str                 # "ollama" | "github_models" | "anthropic"
    model: str                    # provider-specific model name/ID
    budget_pool: BudgetPool
    quality_tier: QualityTier

    # Cost tracking — use whichever applies to this pool
    cost_per_input_token: float = 0.0      # USD (CLAUDE_CREDIT only)
    cost_per_output_token: float = 0.0     # USD (CLAUDE_CREDIT only)
    premium_request_multiplier: float = 1.0  # Copilot premium-request weight

    # Performance baseline (populated after smoke test in Phase 1)
    avg_latency_ms: float = 0.0

    def estimate_cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.cost_per_input_token
            + output_tokens * self.cost_per_output_token
        )

    def estimate_premium_requests(self) -> float:
        """One call = multiplier premium requests from the Copilot pool."""
        return self.premium_request_multiplier


@dataclass
class Response:
    """Uniform return value from every backend, regardless of provider."""
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    backend_id: str
    model: str
    budget_pool: BudgetPool

    # Populated by the router after the call
    complexity_tier: Optional[int] = None        # 1 / 2 / 3
    router_confidence: Optional[float] = None    # 1.0–5.0
    cost_usd: float = 0.0                        # for CLAUDE_CREDIT calls
    premium_requests_used: float = 0.0           # for COPILOT_PREMIUM calls
    was_escalated: bool = False
    timestamp: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class BudgetSnapshot:
    """Point-in-time view of both budget pools for the current month."""
    claude_spent_usd: float
    claude_limit_usd: float
    copilot_requests_used: float
    copilot_requests_limit: float
    month_key: str                   # "YYYY-MM"

    @property
    def claude_remaining_usd(self) -> float:
        return max(0.0, self.claude_limit_usd - self.claude_spent_usd)

    @property
    def copilot_remaining_requests(self) -> float:
        return max(0.0, self.copilot_requests_limit - self.copilot_requests_used)

    @property
    def claude_pct_used(self) -> float:
        if self.claude_limit_usd == 0:
            return 0.0
        return self.claude_spent_usd / self.claude_limit_usd * 100

    @property
    def copilot_pct_used(self) -> float:
        if self.copilot_requests_limit == 0:
            return 0.0
        return self.copilot_requests_used / self.copilot_requests_limit * 100
