#!/usr/bin/env python3
"""
scripts/setup.py

First-run setup for the LLM Cost Autopilot.

What it does:
  1. Profiles your hardware and recommends Ollama models.
  2. Writes the recommended model names to .env (with your approval).
  3. Pulls the required Ollama models (if not already present).
  4. Runs a smoke test against every backend to confirm auth + connectivity.

Run with:
    python scripts/setup.py

Or to skip Ollama pull (e.g. models already downloaded):
    python scripts/setup.py --skip-pull
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Make the src package importable from the scripts/ directory
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from autopilot.budget import BudgetState
from autopilot.hardware_profile import profile_hardware, recommend_models
from autopilot.interface import AutopilotSettings, smoke_test_all_backends
from autopilot.registry import ModelRegistry
from autopilot.ollama import list_local_models, pull_model

console = Console()
load_dotenv()


# ── Step 1: Hardware profiling ──────────────────────────────────────────────────

def step_hardware() -> tuple:
    console.rule("[bold cyan]Step 1 — Hardware Profile[/bold cyan]")
    hw = profile_hardware()
    console.print(Panel(hw.summary(), title="Detected Hardware", expand=False))

    models = recommend_models(hw)
    console.print(
        Panel(
            f"Hardware tier : [bold]{models.hardware_tier_name}[/bold]  "
            f"({models.effective_memory_gb:.1f} GB effective memory)\n"
            f"Router model  : [green]{models.router_model}[/green]\n"
            f"Tier-1 model  : [green]{models.tier1_model}[/green]",
            title="Recommended Ollama Models",
            expand=False,
        )
    )
    return hw, models


# ── Step 2: Write .env ─────────────────────────────────────────────────────────

def step_write_env(models) -> None:
    console.rule("[bold cyan]Step 2 — Update .env[/bold cyan]")
    env_path = Path(".env")

    if not env_path.exists():
        example = Path(".env.example")
        if example.exists():
            env_path.write_text(example.read_text())
            console.print("[dim]Created .env from .env.example[/dim]")
        else:
            env_path.write_text("")

    content = env_path.read_text()
    lines = content.splitlines(keepends=True)

    def set_or_add(key: str, value: str) -> None:
        nonlocal lines
        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[i] = f"{key}={value}\n"
                return
        lines.append(f"{key}={value}\n")

    set_or_add("OLLAMA_ROUTER_MODEL", models.router_model)
    set_or_add("OLLAMA_TIER1_MODEL", models.tier1_model)
    env_path.write_text("".join(lines))
    console.print(f"[green]✓[/green] Wrote model names to .env")
    console.print(f"  OLLAMA_ROUTER_MODEL=[green]{models.router_model}[/green]")
    console.print(f"  OLLAMA_TIER1_MODEL=[green]{models.tier1_model}[/green]")
    console.print()
    console.print(
        "[yellow]⚠[/yellow]  Make sure you've also set "
        "[bold]GITHUB_TOKEN[/bold] and either "
        "[bold]ANTHROPIC_API_KEY[/bold] or "
        "[bold]USE_CLAUDE_SUBSCRIPTION=true[/bold] in .env"
    )


# ── Step 3: Pull Ollama models ─────────────────────────────────────────────────

async def step_pull_models(models) -> None:
    console.rule("[bold cyan]Step 3 — Pull Ollama Models[/bold cyan]")
    needed = list({models.router_model, models.tier1_model})

    try:
        local = await list_local_models()
    except Exception as e:
        console.print(f"[red]✗[/red] Cannot reach Ollama at localhost:11434")
        console.print(f"  → Start Ollama first: [bold]ollama serve[/bold]")
        console.print(f"  Error: {e}")
        return

    for model in needed:
        if model in local:
            console.print(f"[green]✓[/green] {model} already pulled — skipping")
        else:
            console.print(f"[cyan]↓[/cyan] Pulling {model} (this may take a while)…")
            try:
                await pull_model(model)
                console.print(f"[green]✓[/green] {model} pulled successfully")
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to pull {model}: {e}")
                console.print(f"  → Try manually: [bold]ollama pull {model}[/bold]")


# ── Step 4: Smoke test all backends ───────────────────────────────────────────

async def step_smoke_test(registry, budget, settings) -> None:
    console.rule("[bold cyan]Step 4 — Backend Smoke Test[/bold cyan]")
    console.print("Sending a one-word test prompt to every backend…\n")

    results = await smoke_test_all_backends(registry, budget, settings)

    table = Table(show_header=True, header_style="bold")
    table.add_column("Backend", style="cyan")
    table.add_column("Status")
    table.add_column("Latency")
    table.add_column("Tokens")
    table.add_column("Response preview")

    for backend_id, result in results.items():
        if result["status"] == "ok":
            table.add_row(
                backend_id,
                "[green]✓ ok[/green]",
                f"{result['latency_ms']} ms",
                str(result["tokens"]),
                result["text_preview"],
            )
        else:
            table.add_row(
                backend_id,
                "[red]✗ error[/red]",
                "—",
                "—",
                f"[red]{result['error'][:80]}[/red]",
            )

    console.print(table)


# ── Main ───────────────────────────────────────────────────────────────────────

async def main(skip_pull: bool) -> None:
    console.print(
        "\n[bold]LLM Cost Autopilot — First-run Setup[/bold]\n",
        justify="center",
    )

    hw, models = step_hardware()
    step_write_env(models)

    if not skip_pull:
        await step_pull_models(models)
    else:
        console.print("[dim]Skipping Ollama pull (--skip-pull)[/dim]")

    # Re-load env after writing model names
    load_dotenv(override=True)
    settings = AutopilotSettings.from_env()

    db_path = os.environ.get("DB_PATH", "data/autopilot.db")
    budget = BudgetState(db_path=db_path)

    try:
        registry = ModelRegistry(
            hardware_profile=hw,
            recommended_models=models,
        )
        console.print("\n" + registry.summary() + "\n")
        await step_smoke_test(registry, budget, settings)
    except Exception as e:
        console.print(f"[red]Registry/smoke-test error: {e}[/red]")
        raise

    console.print(
        "\n[bold green]Setup complete![/bold green] "
        "Your stack is wired up. Next: Phase 2 — build the routing brain.\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Cost Autopilot setup")
    parser.add_argument("--skip-pull", action="store_true",
                        help="Skip Ollama model pull (if already downloaded)")
    args = parser.parse_args()
    asyncio.run(main(skip_pull=args.skip_pull))
