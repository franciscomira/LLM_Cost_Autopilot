"""
tests/test_router_accuracy.py

Live accuracy evaluation for the Phase 2 router.
Requires Ollama running with the router model loaded.

Run:
    pytest tests/test_router_accuracy.py -v -s --timeout=300

Target: ≥ 80% tier accuracy on the held-out split.
The test prints a confusion matrix so you can see the direction of mis-routes
(down = quality risk, up = cost risk).
"""
from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path

import pytest

DATASET_PATH = Path(__file__).parents[1] / "data" / "routing_dataset.jsonl"
OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
ACCURACY_TARGET = 0.80
HELD_OUT_FRACTION = 0.20
RANDOM_SEED = 42


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_dataset(path: Path) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def held_out_split(examples: list[dict], fraction: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * fraction))
    return shuffled[:cut]


def print_confusion(matrix: dict[int, dict[int, int]]) -> None:
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print(f"{'':8} {'T1':>6} {'T2':>6} {'T3':>6}")
    for actual in (1, 2, 3):
        row = "  ".join(f"{matrix[actual].get(p, 0):>6}" for p in (1, 2, 3))
        print(f"  T{actual}:  {row}")
    print()


# ── Test ───────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not os.environ.get("RUN_ROUTER_EVAL"),
    reason="Set RUN_ROUTER_EVAL=1 to run live Ollama accuracy test",
)
def test_router_accuracy() -> None:
    """
    Runs the held-out evaluation set through the live Ollama router.
    Asserts >= 80% tier accuracy and prints the confusion matrix.
    """
    from autopilot.router import classify_prompt
    from autopilot.registry import ModelRegistry
    from autopilot.interface import AutopilotSettings
    from autopilot.hardware_profile import profile_hardware, recommend_models

    settings = AutopilotSettings.from_env()
    hw = profile_hardware()
    models = recommend_models(hw)
    registry = ModelRegistry(recommended_models=models)
    router_cfg = registry.router_config()

    examples = load_dataset(DATASET_PATH)
    held_out = held_out_split(examples, HELD_OUT_FRACTION, RANDOM_SEED)

    confusion: dict[int, dict[int, int]] = {1: {1:0,2:0,3:0}, 2: {1:0,2:0,3:0}, 3: {1:0,2:0,3:0}}
    correct = 0

    async def run_all() -> None:
        nonlocal correct
        for ex in held_out:
            predicted_tier, confidence = await classify_prompt(
                prompt_text=ex["prompt"],
                router_config=router_cfg,
                settings=settings,
                raise_on_error=True,
            )
            actual = ex["tier"]
            confusion[actual][predicted_tier] = confusion[actual].get(predicted_tier, 0) + 1
            if predicted_tier == actual:
                correct += 1
            print(
                f"  actual={actual} pred={predicted_tier} conf={confidence:.1f} "
                f"| {ex['prompt'][:60]}"
            )

    asyncio.run(run_all())

    total = len(held_out)
    accuracy = correct / total if total else 0.0

    print_confusion(confusion)
    print(f"Accuracy: {correct}/{total} = {accuracy:.1%}  (target ≥ {ACCURACY_TARGET:.0%})")

    # Flag mis-route direction in output
    down_routes = sum(confusion[a].get(p, 0) for a in (2, 3) for p in range(1, a))
    up_routes = sum(confusion[a].get(p, 0) for a in (1, 2) for p in range(a + 1, 4))
    print(f"Down-routes (quality risk): {down_routes}")
    print(f"Up-routes   (cost risk):    {up_routes}")

    assert accuracy >= ACCURACY_TARGET, (
        f"Router accuracy {accuracy:.1%} is below the {ACCURACY_TARGET:.0%} target. "
        "Add more few-shot examples to prompts/router_classify.txt or expand the dataset."
    )
