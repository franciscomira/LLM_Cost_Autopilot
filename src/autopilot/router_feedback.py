"""
router_feedback.py

Loads verifier-generated mis-route corrections from the training dataset and
formats them as few-shot examples to inject into the router prompt at call time.

This is the "close the loop" mechanism: every time the verifier catches a mis-route
it appends a corrected example to routing_dataset.jsonl. The router then reads the
most recent N of those corrections and includes them in its prompt so it learns from
past mistakes without requiring a full fine-tune cycle.

Only verifier-generated entries are used (identified by "verifier-generated:" in the
notes field). Hand-labeled examples in the dataset are excluded to keep the feedback
signal clean — we don't want static examples competing with dynamic corrections.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Matches the path used in verifier.py
_DEFAULT_DATASET = Path(__file__).parent.parent.parent / "data" / "routing_dataset.jsonl"

# Injected examples use confidence 4 (not 5): they are corrections, not canonical ground truth.
_CORRECTION_CONFIDENCE = 4


def load_feedback_examples(
    dataset_path: Path = _DEFAULT_DATASET,
    max_examples: int = 8,
) -> str:
    """
    Read the training dataset and return a formatted string of the most recent
    verifier-generated corrections, ready to splice into the router prompt.

    Returns an empty string if the dataset doesn't exist or has no corrections yet,
    so the router degrades gracefully before any verifications have run.
    """
    if not dataset_path.exists():
        return ""

    corrections: list[dict] = []
    try:
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                notes = entry.get("notes", "")
                if "verifier-generated:" in notes:
                    corrections.append(entry)
    except OSError as exc:
        log.warning("Could not read training dataset: %s", exc)
        return ""

    if not corrections:
        return ""

    # Use the most recent N entries (file is append-only so last = newest)
    recent = corrections[-max_examples:]

    lines: list[str] = [
        "",
        "CORRECTIONS FROM LIVE TRAFFIC (router mis-routes caught by the verifier):",
        "Use these to adjust your tier assignments for similar prompts.",
    ]
    for entry in recent:
        prompt = entry.get("prompt", "").strip()
        tier   = entry.get("tier")
        if not prompt or tier is None:
            continue
        # Truncate very long prompts so they don't inflate the context
        if len(prompt) > 200:
            prompt = prompt[:200] + "…"
        lines.append(f'Prompt: "{prompt}"')
        lines.append(json.dumps({"tier": tier, "confidence": _CORRECTION_CONFIDENCE}))

    if len(lines) <= 3:  # only the header, no usable examples
        return ""

    return "\n".join(lines)
