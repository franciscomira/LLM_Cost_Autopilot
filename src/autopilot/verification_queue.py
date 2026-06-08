"""
verification_queue.py

Async background worker that drains the verification queue without blocking
the caller's request/response cycle.

Usage (wire into your API or main entry point):

    queue = VerificationQueue(registry, budget, settings)
    await queue.start()             # launch worker task

    # After sending a response to the user:
    if should_verify(response, registry.verification_config):
        queue.enqueue(VerificationJob(...))

    await queue.stop()              # graceful drain on shutdown
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from autopilot.budget import BudgetState
from autopilot.interface import AutopilotSettings
from autopilot.registry import ModelRegistry
from autopilot.verifier import VerificationJob, VerificationResult, run_verification_job

log = logging.getLogger(__name__)


@dataclass
class VerificationQueue:
    registry: ModelRegistry
    budget: BudgetState
    settings: AutopilotSettings
    max_queue_size: int = 256       # drop oldest if full, never block the caller
    concurrency: int = 2            # parallel verifier calls

    _queue: asyncio.Queue = field(init=False)
    _worker_tasks: list[asyncio.Task] = field(init=False, default_factory=list)
    _started: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._queue = asyncio.Queue(maxsize=self.max_queue_size)
        self._worker_tasks = []

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Spawn worker coroutines. Call once at application startup."""
        if self._started:
            return
        self._started = True
        for _ in range(self.concurrency):
            task = asyncio.create_task(self._worker())
            self._worker_tasks.append(task)
        log.info(
            "VerificationQueue started (%d workers, queue size %d)",
            self.concurrency, self.max_queue_size,
        )

    async def stop(self) -> None:
        """
        Drain the queue then cancel workers.
        Call during graceful shutdown (e.g. FastAPI lifespan shutdown).
        """
        if not self._started:
            return
        # Signal each worker to exit
        for _ in self._worker_tasks:
            await self._queue.put(None)
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        self._started = False
        log.info("VerificationQueue stopped")

    # ── Enqueue ────────────────────────────────────────────────────────────────

    def enqueue(self, job: VerificationJob) -> bool:
        """
        Non-blocking enqueue. Returns True if queued, False if queue is full
        (the job is dropped rather than blocking the caller).
        """
        try:
            self._queue.put_nowait(job)
            return True
        except asyncio.QueueFull:
            log.warning(
                "VerificationQueue full (%d items) — dropping job for log_id=%s",
                self.max_queue_size, job.request_log_id,
            )
            return False

    # ── Worker ─────────────────────────────────────────────────────────────────

    async def _worker(self) -> None:
        """Drain jobs from the queue until a sentinel None is received."""
        while True:
            job = await self._queue.get()
            if job is None:
                self._queue.task_done()
                break
            try:
                result: VerificationResult = await run_verification_job(
                    job=job,
                    registry=self.registry,
                    budget=self.budget,
                    settings=self.settings,
                )
                if result.is_mis_route:
                    log.info(
                        "Mis-route detected: log_id=%s original_tier=%s "
                        "agreement=%.2f added_to_training=%s",
                        job.request_log_id,
                        job.tier,
                        result.agreement_score,
                        result.added_to_training,
                    )
            except Exception:
                log.exception(
                    "Verification job failed for log_id=%s", job.request_log_id
                )
            finally:
                self._queue.task_done()

    # ── Diagnostics ────────────────────────────────────────────────────────────

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        return self._started and any(not t.done() for t in self._worker_tasks)
