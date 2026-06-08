"""
load_test.py

Phase 6 load test — pushes a large, diverse prompt set through the running API
and prints a formatted savings report at the end.

Usage:
    # Make sure the API is running first:
    #   uvicorn api:app --host 0.0.0.0 --port 8000
    #   OR: docker-compose up

    python load_test.py                         # 500 prompts, default concurrency 20
    python load_test.py --count 1000            # 1 000 prompts
    python load_test.py --concurrency 10        # reduce if Ollama is slow
    python load_test.py --base-url http://localhost:8000
    python load_test.py --dry-run               # validate prompts only, no API calls

The script writes results/load_test_<timestamp>.json on completion so you can
replay the numbers for the portfolio case study.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

# ── Prompt corpus ──────────────────────────────────────────────────────────────
# ~150 hand-crafted prompts covering all three complexity tiers plus edge cases.
# The load test samples from this pool (with replacement) to hit the target count.

TIER1_PROMPTS = [
    # Reformatting
    "Convert this date to ISO format: March 15, 2024",
    "Format this phone number with dashes: 5551234567",
    "Convert Fahrenheit to Celsius: 98.6°F",
    "Rewrite this in lowercase: THE QUICK BROWN FOX",
    "Remove all punctuation from: Hello, world! How are you?",
    "Convert these miles to kilometres: 26.2 miles",
    "Format this list as bullet points: apples oranges bananas",
    "Capitalize the first letter of each word: hello world from python",
    "Convert 24-hour time to 12-hour: 14:35",
    "Sort this list alphabetically: zebra, apple, mango, carrot",
    "Extract all numbers from this text: I have 3 cats and 12 fish and 1 dog.",
    "Convert this JSON key from camelCase to snake_case: firstName, lastName, dateOfBirth",
    "Reformat this address: 123 main st springfield il 62701",
    "Convert these seconds to minutes and seconds: 745 seconds",
    "Reverse this string: programming",
    # Basic Q&A from context
    "According to this text: 'Python was created in 1991 by Guido van Rossum.' When was Python created?",
    "The capital of France is Paris. What is the capital of France?",
    "This document says the meeting is on Friday at 3pm. When is the meeting?",
    "Based on the sentence 'Water boils at 100°C at sea level', at what temperature does water boil?",
    "The receipt shows: Total $47.50, Tax $3.25. What is the total?",
    # Extraction
    "Extract the email address from: Please contact support@example.com for help.",
    "What is the product name in: 'Introducing the new Widget Pro 3000 today!'",
    "List the ingredients mentioned: The recipe calls for flour, eggs, butter, and sugar.",
    "Extract the version number from: 'Updated to version 2.4.1 on Tuesday.'",
    "What URL is in this text: Visit https://docs.example.com/api for documentation.",
    # Simple classification
    "Is this sentence positive, negative, or neutral? 'The weather today is okay.'",
    "Classify this email as spam or not spam: 'Your invoice #1234 is attached.'",
    "Is this a question or a statement? 'The sky is blue.'",
    "What language is this written in? 'Bonjour, comment allez-vous?'",
    "Is this number even or odd? 47",
]

TIER2_PROMPTS = [
    # Summarisation
    "Summarise this in 2 sentences: The Python programming language was created by Guido van Rossum and first released in 1991. It emphasizes code readability and uses significant whitespace. Python supports multiple programming paradigms, including structured, object-oriented, and functional programming.",
    "Write a one-paragraph summary of the key points: Machine learning is a subset of artificial intelligence that enables systems to learn from data. It uses statistical techniques to give computers the ability to 'learn' from experience without being explicitly programmed. Common applications include image recognition, natural language processing, and recommendation systems.",
    "Condense this to the main idea: Studies show that regular exercise reduces the risk of chronic diseases, improves mental health, boosts energy levels, and contributes to a longer life. Even 30 minutes of moderate activity most days of the week can produce significant health benefits.",
    "Summarise the pros and cons from this text: Remote work offers flexibility and eliminates commute time, but can lead to isolation and blurred work-life boundaries. Productivity varies—some workers thrive while others struggle with distractions at home.",
    "Extract the key action items from this meeting notes excerpt: We discussed the Q3 roadmap. John will update the design specs by Thursday. Sarah needs to review the API documentation. The team agreed to move the sprint planning to Wednesday mornings.",
    # Classification
    "Classify the sentiment of each review and explain: 1) 'Fast shipping, exactly as described!' 2) 'Product broke after two days, very disappointed.' 3) 'It's fine, nothing special.'",
    "Categorise these support tickets by urgency (high/medium/low): A) 'Site is completely down' B) 'Button colour looks off' C) 'Login fails for premium users'",
    "Classify these transactions by category (food, transport, utilities, entertainment): $45 Uber, $120 electricity bill, $32 Netflix, $78 restaurant dinner",
    "Label each news headline by topic (politics, tech, sports, science): 'New AI model breaks benchmark', 'Senate votes on spending bill', 'Team wins championship', 'Scientists discover new planet'",
    # Structured analysis
    "Identify the main argument and supporting evidence in: 'We should switch to renewable energy. First, it reduces carbon emissions. Second, prices have dropped 80% in a decade. Third, it creates more jobs than fossil fuels per dollar invested.'",
    "Analyse the SWOT for a new coffee shop: small local brand, prime downtown location, experienced baristas, limited marketing budget, growing coffee culture in the area, three existing competitors nearby.",
    "Compare these two approaches and identify trade-offs: Microservices architecture vs monolithic architecture for a startup's first product.",
    "What are the main risks in this plan? We want to launch in 6 weeks, with a team of 3, targeting enterprise customers, with no existing customer relationships.",
    "Identify potential biases in this survey question: 'Don't you agree that our product is excellent?'",
    "Analyse the tone and audience of this paragraph: 'Leverage our synergistic solutions to optimise your workflow paradigm and drive stakeholder value across verticals.'",
    # Information extraction / structured output
    "Extract all dates and events from: The company was founded in 2010. It launched its first product in 2012. The IPO happened in 2018. A major acquisition was completed in 2021.",
    "Parse this address into components (street, city, state, zip): 742 Evergreen Terrace, Springfield, IL 62704",
    "Extract the named entities (people, organisations, locations) from: Elon Musk founded SpaceX in Hawthorne, California. The company's Falcon 9 rocket is manufactured there.",
    "Turn this paragraph into a structured table with columns (name, role, action): Alice is the project manager and she approved the budget. Bob is the developer and he submitted the code. Carol is the designer and she created the mockups.",
    "List all the technical requirements mentioned: The system must handle 10,000 concurrent users, respond in under 200ms, store data encrypted at rest, and provide 99.9% uptime.",
]

TIER3_PROMPTS = [
    # Multi-step reasoning
    "A snail climbs a 10-metre wall. Each day it climbs 3 metres, but each night it slides back 2 metres. How many days does it take to reach the top? Show your reasoning step by step.",
    "I have three boxes. Box A contains only apples. Box B contains only oranges. Box C contains both. All three labels are wrong. I pick one fruit from Box C and it's an apple. Now label all boxes correctly. Explain your reasoning.",
    "A train leaves Chicago at 9am travelling at 60mph. Another train leaves New York at 10am travelling at 80mph toward Chicago. The cities are 780 miles apart. Where and when do they meet? Show your work.",
    "You have 12 balls, one is slightly heavier. You have a balance scale and can only use it 3 times. How do you find the heavy ball? Provide a step-by-step decision tree.",
    "Analyse the flaw in this argument: 'All swans I have ever seen are white. Therefore all swans are white.' What type of reasoning error is this and what are the implications for scientific reasoning?",
    # Nuanced judgment
    "A self-driving car's brakes fail. It can either hit one pedestrian or swerve onto the pavement and hit three pedestrians. From utilitarian, deontological, and virtue ethics perspectives, what should the car be programmed to do? Are there limits to programming ethics into machines?",
    "A startup founder asks: should I take VC funding or bootstrap? Consider cash needs, control, growth speed, founder stress, and exit options. Give a nuanced recommendation and explain what additional information would change your answer.",
    "Evaluate the long-term strategic risks and opportunities for a traditional newspaper company that wants to pivot to digital. Consider revenue models, audience shifts, content differentiation, and platform dependency.",
    "A manager discovers a high-performer is quietly interviewing elsewhere. What are the ethical considerations in how the manager handles this, and what are the strategic options? Consider power dynamics and team morale.",
    "Is it ethical for social media platforms to algorithmically amplify emotionally charged content to increase engagement, even when they know it contributes to polarisation? Analyse from multiple stakeholder perspectives.",
    # Creative generation
    "Write a short product description (3 sentences) for an AI-powered coffee mug that learns your preferred temperature. Make it sound premium but not pretentious.",
    "Draft a professional but warm rejection email for a job applicant who made it to the final round but wasn't selected. The role was a senior engineer position.",
    "Write three distinct taglines for a cybersecurity company targeting small businesses. Each should evoke a different emotion (reassurance, urgency, empowerment).",
    "Create a metaphor that explains how a neural network learns, suitable for explaining to a non-technical executive.",
    "Write a brief narrative (150 words) from the perspective of a router in a corporate network during a DDoS attack.",
    # Complex analysis
    "Critically analyse the business model of a free-to-play mobile game. Who are the actual customers? What are the principal-agent problems? How does this affect product design decisions?",
    "Compare the CAP theorem trade-offs for a distributed database used in: (a) a financial transactions system, (b) a social media feed, (c) a collaborative document editor. For each, explain which guarantees you'd prioritise and why.",
    "Explain why technical debt compounds non-linearly over time and propose a prioritisation framework for addressing it in a legacy codebase with a small team.",
    "Design a metrics system for measuring the ROI of developer tooling improvements. What would you measure, how would you attribute causation, and what are the pitfalls of this kind of measurement?",
    "A company has two product lines that share infrastructure. Product A is growing fast; Product B is declining. What are the organisational, technical, and financial considerations in deciding whether to split the infrastructure, and how would you sequence the decision?",
]

EDGE_CASE_PROMPTS = [
    # Very short — should be Tier 1
    "Hi",
    "Yes",
    "What?",
    "2+2",
    # Code-related — tends toward Tier 2/3
    "What does this Python code do? x = [i**2 for i in range(10)]",
    "Explain the difference between == and is in Python.",
    "What is a race condition?",
    "When should you use an index in a database?",
    # Ambiguous
    "Tell me about machine learning.",
    "Explain blockchain.",
    "What is REST?",
    "Describe agile.",
]

ALL_PROMPTS = [
    *[(p, 1) for p in TIER1_PROMPTS],
    *[(p, 2) for p in TIER2_PROMPTS],
    *[(p, 3) for p in TIER3_PROMPTS],
    *[(p, None) for p in EDGE_CASE_PROMPTS],
]


# ── Result tracking ────────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    prompt_preview: str
    expected_tier: int | None
    actual_tier: int | None
    routing_reason: str
    backend_id: str
    budget_pool: str
    cost_usd: float
    premium_requests: float
    latency_ms: float
    confidence: float
    was_escalated: bool
    success: bool
    error: str | None = None


@dataclass
class LoadTestSummary:
    total: int = 0
    success: int = 0
    failure: int = 0
    results: list[RequestResult] = field(default_factory=list)
    duration_s: float = 0.0

    # Pool distribution
    free_count: int = 0
    copilot_count: int = 0
    claude_count: int = 0

    # Cost
    total_cost_usd: float = 0.0
    total_premium_requests: float = 0.0
    baseline_cost_usd: float = 0.0

    # Quality
    escalated_count: int = 0
    avg_confidence: float = 0.0
    avg_latency_ms: float = 0.0

    # Routing accuracy (where expected_tier is known)
    tier_matches: int = 0
    tier_total: int = 0


# ── Worker ─────────────────────────────────────────────────────────────────────

BASELINE_INPUT  = 0.000015   # $15 / 1M — claude-opus-4 reference
BASELINE_OUTPUT = 0.000075   # $75 / 1M

async def _send_one(
    client: httpx.AsyncClient,
    base_url: str,
    prompt: str,
    expected_tier: int | None,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    preview = (prompt[:60] + "…") if len(prompt) > 60 else prompt
    async with semaphore:
        try:
            resp = await client.post(
                f"{base_url}/v1/completions",
                json={"messages": [{"role": "user", "content": prompt}]},
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
            r = data["routing"]
            return RequestResult(
                prompt_preview=preview,
                expected_tier=expected_tier,
                actual_tier=r.get("complexity_tier"),
                routing_reason=r.get("routing_reason", ""),
                backend_id=r.get("backend_id", ""),
                budget_pool=r.get("budget_pool", ""),
                cost_usd=r.get("cost_usd", 0.0),
                premium_requests=r.get("premium_requests_used", 0.0),
                latency_ms=r.get("latency_ms", 0.0),
                confidence=r.get("router_confidence") or 0.0,
                was_escalated=r.get("was_escalated", False),
                success=True,
            )
        except Exception as exc:
            return RequestResult(
                prompt_preview=preview,
                expected_tier=expected_tier,
                actual_tier=None,
                routing_reason="",
                backend_id="",
                budget_pool="",
                cost_usd=0.0,
                premium_requests=0.0,
                latency_ms=0.0,
                confidence=0.0,
                was_escalated=False,
                success=False,
                error=str(exc)[:120],
            )


# ── Stats endpoint ─────────────────────────────────────────────────────────────

async def _fetch_stats(client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
    try:
        r = await client.get(f"{base_url}/v1/stats", timeout=10.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


# ── Main runner ────────────────────────────────────────────────────────────────

async def run_load_test(
    base_url: str,
    count: int,
    concurrency: int,
    seed: int = 42,
) -> LoadTestSummary:
    rng = random.Random(seed)
    # Sample `count` prompts from the pool (with replacement for large counts)
    samples = rng.choices(ALL_PROMPTS, k=count)

    semaphore = asyncio.Semaphore(concurrency)
    summary = LoadTestSummary()

    print(f"\nLLM Cost Autopilot — Load Test")
    print(f"{'─' * 50}")
    print(f"Target:      {base_url}")
    print(f"Prompts:     {count} (sampled from {len(ALL_PROMPTS)} unique)")
    print(f"Concurrency: {concurrency}")
    print()

    t0 = time.monotonic()
    async with httpx.AsyncClient() as client:
        tasks = [
            _send_one(client, base_url, prompt, tier, semaphore)
            for prompt, tier in samples
        ]

        done = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            summary.results.append(result)
            done += 1
            if done % 50 == 0 or done == count:
                pct = done / count * 100
                print(f"  {done:>5}/{count}  ({pct:4.0f}%)  "
                      f"errors so far: {sum(1 for r in summary.results if not r.success)}")

        # Pull final stats from the API
        api_stats = await _fetch_stats(client, base_url)

    summary.duration_s = time.monotonic() - t0
    summary.total = count

    # Aggregate
    confidences, latencies = [], []
    for r in summary.results:
        if r.success:
            summary.success += 1
            pool = r.budget_pool
            if pool == "FREE":
                summary.free_count += 1
            elif pool == "COPILOT_PREMIUM":
                summary.copilot_count += 1
            elif pool == "CLAUDE_CREDIT":
                summary.claude_count += 1
            summary.total_cost_usd += r.cost_usd
            summary.total_premium_requests += r.premium_requests
            if r.was_escalated:
                summary.escalated_count += 1
            if r.confidence:
                confidences.append(r.confidence)
            if r.latency_ms:
                latencies.append(r.latency_ms)
            if r.expected_tier is not None and r.actual_tier is not None:
                summary.tier_total += 1
                if r.expected_tier == r.actual_tier:
                    summary.tier_matches += 1
        else:
            summary.failure += 1

    summary.avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    summary.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

    # Use API stats for the authoritative baseline (it has token counts)
    if api_stats and "headline" in api_stats:
        h = api_stats["headline"]
        summary.baseline_cost_usd = h.get("baseline_cost_usd", 0.0)
    else:
        # Rough fallback: assume avg 200 input + 150 output tokens per request
        avg_tokens_in, avg_tokens_out = 200, 150
        summary.baseline_cost_usd = (
            summary.success * avg_tokens_in * BASELINE_INPUT +
            summary.success * avg_tokens_out * BASELINE_OUTPUT
        )

    return summary, api_stats


# ── Report ─────────────────────────────────────────────────────────────────────

def print_report(summary: LoadTestSummary, api_stats: dict) -> None:
    s = summary
    sep = "─" * 60

    savings_usd = max(0.0, s.baseline_cost_usd - s.total_cost_usd)
    savings_pct = (savings_usd / s.baseline_cost_usd * 100) if s.baseline_cost_usd else 0.0
    free_pct    = (s.free_count    / s.success * 100) if s.success else 0.0
    copilot_pct = (s.copilot_count / s.success * 100) if s.success else 0.0
    claude_pct  = (s.claude_count  / s.success * 100) if s.success else 0.0
    error_pct   = (s.failure / s.total * 100) if s.total else 0.0
    escalation_pct = (s.escalated_count / s.success * 100) if s.success else 0.0
    router_accuracy = (s.tier_matches / s.tier_total * 100) if s.tier_total else 0.0
    throughput   = s.success / s.duration_s if s.duration_s else 0.0

    print(f"\n{'═' * 60}")
    print(f"  LLM COST AUTOPILOT — LOAD TEST RESULTS")
    print(f"{'═' * 60}")
    print()
    print(f"  ★  Cost saved vs all-premium baseline:  {savings_pct:.1f}%  (${savings_usd:.4f})")
    print(f"  ★  Requests served free (local):         {free_pct:.1f}%  ({s.free_count}/{s.success})")
    print()
    print(f"{sep}")
    print(f"  THROUGHPUT & RELIABILITY")
    print(f"{sep}")
    print(f"  Total requests:      {s.total}")
    print(f"  Succeeded:           {s.success}  ({100 - error_pct:.1f}%)")
    print(f"  Failed:              {s.failure}  ({error_pct:.1f}%)")
    print(f"  Duration:            {s.duration_s:.1f}s")
    print(f"  Throughput:          {throughput:.1f} req/s")
    print(f"  Avg latency:         {s.avg_latency_ms:.0f} ms")
    print()
    print(f"{sep}")
    print(f"  ROUTING DISTRIBUTION")
    print(f"{sep}")
    print(f"  FREE  (local Ollama):      {s.free_count:>5}  ({free_pct:5.1f}%)")
    print(f"  COPILOT_PREMIUM (GitHub):  {s.copilot_count:>5}  ({copilot_pct:5.1f}%)")
    print(f"  CLAUDE_CREDIT (Anthropic): {s.claude_count:>5}  ({claude_pct:5.1f}%)")
    print(f"  Escalated:                 {s.escalated_count:>5}  ({escalation_pct:5.1f}%)")
    print(f"  Avg router confidence:     {s.avg_confidence:.2f} / 5.0")
    if s.tier_total > 0:
        print(f"  Router accuracy:           {router_accuracy:.1f}%  ({s.tier_matches}/{s.tier_total} labeled)")
    print()
    print(f"{sep}")
    print(f"  COST COMPARISON")
    print(f"{sep}")
    print(f"  Actual spend:            ${s.total_cost_usd:.4f}")
    print(f"  Copilot premium requests: {s.total_premium_requests:.1f}")
    print(f"  Baseline (all Opus-4):   ${s.baseline_cost_usd:.4f}")
    print(f"  Savings:                 ${savings_usd:.4f}  ({savings_pct:.1f}%)")

    if api_stats and "budget" in api_stats:
        b = api_stats["budget"]
        print()
        print(f"{sep}")
        print(f"  LIVE BUDGET SNAPSHOT  (month: {b.get('month', '?')})")
        print(f"{sep}")
        print(f"  Claude credit:   ${b.get('claude_spent_usd', 0):.4f} spent / "
              f"${b.get('claude_limit_usd', 0):.2f} limit  "
              f"(${b.get('claude_remaining_usd', 0):.4f} left)")
        print(f"  Copilot:         {b.get('copilot_requests_used', 0):.0f} used / "
              f"{b.get('copilot_requests_limit', 0):.0f} limit  "
              f"({b.get('copilot_remaining_requests', 0):.0f} left)")

    print(f"\n{'═' * 60}\n")


# ── Persist results ────────────────────────────────────────────────────────────

def save_results(summary: LoadTestSummary, api_stats: dict, out_dir: Path = Path("results")) -> Path:
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"load_test_{ts}.json"

    s = summary
    savings_usd = max(0.0, s.baseline_cost_usd - s.total_cost_usd)
    savings_pct = (savings_usd / s.baseline_cost_usd * 100) if s.baseline_cost_usd else 0.0

    payload = {
        "timestamp": ts,
        "summary": {
            "total": s.total,
            "success": s.success,
            "failure": s.failure,
            "duration_s": round(s.duration_s, 2),
            "throughput_rps": round(s.success / s.duration_s, 2) if s.duration_s else 0,
            "free_count": s.free_count,
            "copilot_count": s.copilot_count,
            "claude_count": s.claude_count,
            "free_pct": round(s.free_count / s.success * 100, 1) if s.success else 0,
            "total_cost_usd": round(s.total_cost_usd, 6),
            "total_premium_requests": s.total_premium_requests,
            "baseline_cost_usd": round(s.baseline_cost_usd, 6),
            "savings_usd": round(savings_usd, 6),
            "savings_pct": round(savings_pct, 1),
            "escalated_count": s.escalated_count,
            "avg_confidence": round(s.avg_confidence, 3),
            "avg_latency_ms": round(s.avg_latency_ms, 1),
            "router_accuracy_pct": round(
                s.tier_matches / s.tier_total * 100, 1
            ) if s.tier_total else None,
        },
        "api_stats": api_stats,
        "results": [
            {
                "prompt_preview": r.prompt_preview,
                "expected_tier": r.expected_tier,
                "actual_tier": r.actual_tier,
                "routing_reason": r.routing_reason,
                "backend_id": r.backend_id,
                "budget_pool": r.budget_pool,
                "cost_usd": r.cost_usd,
                "premium_requests": r.premium_requests,
                "latency_ms": r.latency_ms,
                "confidence": r.confidence,
                "was_escalated": r.was_escalated,
                "success": r.success,
                "error": r.error,
            }
            for r in summary.results
        ],
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Results saved to: {out_path}")
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM Cost Autopilot load test")
    p.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    p.add_argument("--count", type=int, default=500, help="Number of prompts to send (default: 500)")
    p.add_argument("--concurrency", type=int, default=20, help="Max concurrent requests (default: 20)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for prompt sampling")
    p.add_argument("--dry-run", action="store_true", help="Print prompts without sending")
    p.add_argument("--no-save", action="store_true", help="Don't write results JSON")
    return p.parse_args()


async def _async_main() -> None:
    args = _parse_args()

    if args.dry_run:
        rng = random.Random(args.seed)
        samples = rng.choices(ALL_PROMPTS, k=args.count)
        tier_dist = {1: 0, 2: 0, 3: 0, None: 0}
        for _, tier in samples:
            tier_dist[tier] = tier_dist.get(tier, 0) + 1
        print(f"Dry run — {args.count} prompts sampled:")
        for tier, count in sorted(tier_dist.items(), key=lambda x: (x[0] is None, x[0])):
            label = f"Tier {tier}" if tier else "edge case"
            print(f"  {label}: {count}")
        return

    summary, api_stats = await run_load_test(
        base_url=args.base_url,
        count=args.count,
        concurrency=args.concurrency,
        seed=args.seed,
    )
    print_report(summary, api_stats)
    if not args.no_save:
        save_results(summary, api_stats)


def main() -> None:
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
