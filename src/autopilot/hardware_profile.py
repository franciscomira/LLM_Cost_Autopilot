"""
src/autopilot/hardware_profile.py

Detects the local machine's memory/GPU capabilities and recommends the
best Ollama models to use for the router and Tier-1 generator.

Run directly to see what your machine qualifies for:
    python -m autopilot.hardware_profile
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psutil
import yaml


# ── Hardware snapshot ──────────────────────────────────────────────────────────

@dataclass
class HardwareProfile:
    ram_gb: float
    gpu_type: str                    # "nvidia" | "apple_silicon" | "amd" | "cpu_only"
    gpu_name: Optional[str]
    vram_gb: float                   # 0.0 if no discrete GPU
    cpu_cores: int
    os_name: str
    arch: str
    effective_memory_gb: float       # what Ollama can actually use (see below)

    def summary(self) -> str:
        gpu_line = (
            f"{self.gpu_name} ({self.vram_gb:.1f} GB VRAM)"
            if self.gpu_name else "None (CPU inference)"
        )
        return (
            f"OS: {self.os_name} {self.arch}\n"
            f"RAM: {self.ram_gb:.1f} GB\n"
            f"GPU: {gpu_line}\n"
            f"Effective memory for Ollama: {self.effective_memory_gb:.1f} GB"
        )


# ── GPU detection helpers ──────────────────────────────────────────────────────

def _detect_nvidia() -> tuple[Optional[str], float]:
    """Returns (gpu_name, vram_gb) or (None, 0.0) if no NVIDIA GPU found."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            # Take first GPU if multiple
            line = out.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            name = parts[0]
            vram_mb = float(parts[1])
            return name, vram_mb / 1024.0
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None, 0.0


def _detect_apple_silicon() -> tuple[Optional[str], float]:
    """
    Returns (chip_name, effective_vram_gb) for Apple Silicon.
    On M-series Macs, GPU and CPU share unified memory — Ollama can use
    most of the total RAM for model weights via Metal.
    """
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None, 0.0

    chip_name = "Apple Silicon"
    try:
        out = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-json"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            data = json.loads(out.stdout)
            hw_list = data.get("SPHardwareDataType", [])
            if hw_list:
                chip_name = hw_list[0].get("chip_type", "Apple Silicon")
    except Exception:
        pass

    # Unified memory: Ollama/Metal can use ~75% of total RAM for model weights
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    return chip_name, total_ram_gb * 0.75


def _detect_amd() -> tuple[Optional[str], float]:
    """ROCm-based AMD GPU detection (Linux only)."""
    try:
        out = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            data = json.loads(out.stdout)
            # rocm-smi JSON structure varies; grab first card's VRAM total
            for card_key, card_data in data.items():
                if "card" in card_key.lower():
                    vram_bytes = int(card_data.get("VRAM Total Memory (B)", 0))
                    if vram_bytes > 0:
                        return f"AMD GPU ({card_key})", vram_bytes / (1024 ** 3)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError,
            ValueError, KeyError):
        pass
    return None, 0.0


# ── Main profiler ──────────────────────────────────────────────────────────────

def profile_hardware() -> HardwareProfile:
    """Detect the current machine's hardware and return a HardwareProfile."""
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
    os_name = platform.system()
    arch = platform.machine()

    # Try GPU detection in priority order
    nvidia_name, nvidia_vram = _detect_nvidia()
    if nvidia_name:
        effective = nvidia_vram * 0.90   # leave 10% VRAM headroom for driver
        return HardwareProfile(
            ram_gb=ram_gb, gpu_type="nvidia", gpu_name=nvidia_name,
            vram_gb=nvidia_vram, cpu_cores=cpu_cores,
            os_name=os_name, arch=arch, effective_memory_gb=effective,
        )

    apple_name, apple_vram = _detect_apple_silicon()
    if apple_name:
        return HardwareProfile(
            ram_gb=ram_gb, gpu_type="apple_silicon", gpu_name=apple_name,
            vram_gb=apple_vram, cpu_cores=cpu_cores,
            os_name=os_name, arch=arch, effective_memory_gb=apple_vram,
        )

    amd_name, amd_vram = _detect_amd()
    if amd_name:
        effective = amd_vram * 0.90
        return HardwareProfile(
            ram_gb=ram_gb, gpu_type="amd", gpu_name=amd_name,
            vram_gb=amd_vram, cpu_cores=cpu_cores,
            os_name=os_name, arch=arch, effective_memory_gb=effective,
        )

    # CPU-only: Ollama can use RAM, but leave 50% for OS + KV cache overhead
    effective = ram_gb * 0.50
    return HardwareProfile(
        ram_gb=ram_gb, gpu_type="cpu_only", gpu_name=None,
        vram_gb=0.0, cpu_cores=cpu_cores,
        os_name=os_name, arch=arch, effective_memory_gb=effective,
    )


# ── Model selector ─────────────────────────────────────────────────────────────

@dataclass
class RecommendedModels:
    router_model: str
    tier1_model: str
    hardware_tier_name: str          # "tiny" / "small" / "medium" / "large"
    effective_memory_gb: float

    def to_env_lines(self) -> str:
        return (
            f"OLLAMA_ROUTER_MODEL={self.router_model}\n"
            f"OLLAMA_TIER1_MODEL={self.tier1_model}\n"
        )


def recommend_models(
    profile: HardwareProfile,
    config_path: Path | str | None = None,
) -> RecommendedModels:
    """
    Read config/models_by_hardware.yaml and return the best models for
    this machine. Manual env-var overrides always take precedence.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "models_by_hardware.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Sort tiers descending by min_memory_gb, pick the first one we fit
    tiers = sorted(config["tiers"], key=lambda t: t["min_memory_gb"], reverse=True)
    chosen = tiers[-1]   # fallback: smallest tier
    for tier in tiers:
        if profile.effective_memory_gb >= tier["min_memory_gb"]:
            chosen = tier
            break

    # Env-var overrides let the user skip auto-detection
    router_model = (
        os.environ.get("OLLAMA_ROUTER_MODEL")
        or chosen["router"]["model"]
    )
    tier1_model = (
        os.environ.get("OLLAMA_TIER1_MODEL")
        or chosen["tier1_generator"]["model"]
    )

    return RecommendedModels(
        router_model=router_model,
        tier1_model=tier1_model,
        hardware_tier_name=chosen["name"],
        effective_memory_gb=profile.effective_memory_gb,
    )


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print("\n[bold cyan]LLM Cost Autopilot — Hardware Profiler[/bold cyan]\n")

    hw = profile_hardware()
    console.print(Panel(hw.summary(), title="Detected Hardware", expand=False))

    models = recommend_models(hw)
    console.print(
        Panel(
            f"Hardware tier : [bold]{models.hardware_tier_name}[/bold]\n"
            f"Router model  : [green]{models.router_model}[/green]\n"
            f"Tier-1 model  : [green]{models.tier1_model}[/green]\n\n"
            f"Add these to your .env to lock in the selection:\n"
            f"[dim]{models.to_env_lines().strip()}[/dim]",
            title="Recommended Models",
            expand=False,
        )
    )
