"""
src/autopilot/registry.py

Loads config/routing.yaml and the hardware profiler output and produces a
live dict of ModelConfig objects keyed by backend_id.

Also exposes the routing policy (tier → backend, thresholds, budget guardrails).
"""
from __future__ import annotations

from pathlib import Path

import yaml

from hardware_profile import HardwareProfile, RecommendedModels
from models import BudgetPool, ModelConfig, QualityTier


# ── Tier → quality mapping ─────────────────────────────────────────────────────

_TIER_TO_QUALITY = {
    "ollama_router": QualityTier.LOW,
    "ollama_tier1":  QualityTier.LOW,
    "copilot_mid":   QualityTier.MID,
    "copilot_top":   QualityTier.HIGH,
    "claude_haiku":  QualityTier.HIGH,
    "claude_sonnet": QualityTier.HIGH,
}

_POOL_MAP = {
    "FREE":             BudgetPool.FREE,
    "COPILOT_PREMIUM":  BudgetPool.COPILOT_PREMIUM,
    "CLAUDE_CREDIT":    BudgetPool.CLAUDE_CREDIT,
}


# ── Registry ───────────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Populated at startup from routing.yaml + hardware profile.
    Thread-safe for reads (dicts are read-only after init).
    """

    def __init__(
        self,
        routing_config_path: Path | str | None = None,
        hardware_profile: HardwareProfile | None = None,
        recommended_models: RecommendedModels | None = None,
    ) -> None:
        if routing_config_path is None:
            routing_config_path = (
                Path(__file__).parent / "routing.yaml"
            )
        with open(routing_config_path) as f:
            self._cfg = yaml.safe_load(f)

        self._backends: dict[str, ModelConfig] = {}
        self._routing = self._cfg["tiers"]
        self._budgets = self._cfg["budgets"]
        self._thresholds = self._cfg["low_budget_thresholds"]
        self._verification = self._cfg["verification"]

        self._load_backends(hardware_profile, recommended_models)

    def _load_backends(
        self,
        hw: HardwareProfile | None,
        models: RecommendedModels | None,
    ) -> None:
        raw_backends: dict = self._cfg.get("backends", {})
        for backend_id, cfg in raw_backends.items():
            pool = _POOL_MAP[cfg["budget_pool"]]
            quality = _TIER_TO_QUALITY.get(backend_id, QualityTier.MID)

            # Ollama models come from the hardware profiler, not static YAML
            if cfg["provider"] == "ollama":
                if models is None:
                    raise RuntimeError(
                        "HardwareProfile / RecommendedModels required to resolve "
                        "Ollama model names. Run hardware_profile.py first."
                    )
                model = (
                    models.router_model
                    if cfg["role"] == "router"
                    else models.tier1_model
                )
            else:
                model = cfg["model"]

            self._backends[backend_id] = ModelConfig(
                backend_id=backend_id,
                provider=cfg["provider"],
                model=model,
                budget_pool=pool,
                quality_tier=quality,
                cost_per_input_token=cfg.get("cost_per_input_token", 0.0),
                cost_per_output_token=cfg.get("cost_per_output_token", 0.0),
                premium_request_multiplier=cfg.get("premium_request_multiplier", 1.0),
            )

    # ── Accessors ──────────────────────────────────────────────────────────────

    def get(self, backend_id: str) -> ModelConfig:
        try:
            return self._backends[backend_id]
        except KeyError:
            raise KeyError(
                f"Backend '{backend_id}' not found. "
                f"Available: {list(self._backends)}"
            )

    def all(self) -> dict[str, ModelConfig]:
        return dict(self._backends)

    def router_config(self) -> ModelConfig:
        return self.get("ollama_router")

    def tier_policy(self, tier: int) -> dict:
        """Return the full routing policy dict for a given tier (1/2/3)."""
        return self._routing[tier]

    def primary_backend(self, tier: int) -> ModelConfig:
        return self.get(self._routing[tier]["primary_backend"])

    def fallback_backend(self, tier: int) -> ModelConfig:
        return self.get(self._routing[tier]["fallback_backend"])

    def claude_reserve_threshold(self) -> float:
        """Confidence ≥ this → route Tier 3 to Claude (Haiku)."""
        return self._routing[3].get("claude_reserve_threshold", 4.5)

    def judge_backend_id(self) -> str:
        """Backend used as the LLM judge — should differ from the verifier backend."""
        return self._verification.get("judge_backend", "copilot_mid")

    def claude_sonnet_threshold(self) -> float:
        """Confidence ≥ this → upgrade Tier 3 Claude call to Sonnet."""
        return self._routing[3].get("claude_sonnet_threshold", 4.8)

    @property
    def budget_config(self) -> dict:
        return self._budgets

    @property
    def low_budget_thresholds(self) -> dict:
        return self._thresholds

    @property
    def verification_config(self) -> dict:
        return self._verification

    def summary(self) -> str:
        lines = ["Registered backends:"]
        for bid, cfg in self._backends.items():
            lines.append(
                f"  {bid:<20} {cfg.provider:<16} {cfg.model:<35} "
                f"pool={cfg.budget_pool.value}"
            )
        return "\n".join(lines)
