"""Background execution of research jobs.

``JobRunner`` drives the pipeline orchestrator, publishes per-stage progress
:class:`JobEvent`s to subscribers (consumed by the SSE endpoint), persists
paused pipeline state for the human-review gate, and writes the final
:class:`ResearchJob` record. Same external contract as the LangGraph version;
only the engine changed.
"""

import asyncio
import json
import time
from datetime import UTC, datetime
from typing import Any

import structlog.contextvars

from app.core.logging import get_logger
from app.core.metrics import JOB_COST, RESEARCH_DURATION, RESEARCH_JOBS
from app.core.schemas import JobEvent, JobStatus, ResearchJob
from app.pipeline.context import PipelineContext, initial_context
from app.pipeline.deps import PipelineDeps
from app.pipeline.runner import resume_pipeline, run_pipeline
from app.services.job_store import JobStore
from app.services.usage import UsageTracker, reset_tracker, set_tracker

_JOB_NODE = "__job__"
_log = get_logger(__name__)


class JobRunner:
    """Runs research pipelines in the background and fans out progress events."""

    def __init__(
        self,
        deps: PipelineDeps,
        store: JobStore,
        *,
        max_critic_rounds: int = 2,
        llm_model: str = "gpt-4o-mini",
    ) -> None:
        self._deps = deps
        self._store = store
        self._max_critic_rounds = max_critic_rounds
        self._llm_model = llm_model
        self._listeners: dict[str, set[asyncio.Queue[JobEvent]]] = {}

    # -- subscription (SSE) --------------------------------------------------

    def subscribe(self, job_id: str) -> asyncio.Queue[JobEvent]:
        queue: asyncio.Queue[JobEvent] = asyncio.Queue()
        self._listeners.setdefault(job_id, set()).add(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue[JobEvent]) -> None:
        listeners = self._listeners.get(job_id)
        if listeners is not None:
            listeners.discard(queue)
            if not listeners:
                self._listeners.pop(job_id, None)

    async def _publish(self, event: JobEvent) -> None:
        for queue in self._listeners.get(event.job_id, ()):
            queue.put_nowait(event)

    # -- execution -----------------------------------------------------------

    async def _finish(
        self,
        job: ResearchJob,
        status: JobStatus,
        error: str | None = None,
        *,
        elapsed: float | None = None,
    ) -> None:
        job.status = status
        job.error = error
        job.updated_at = datetime.now(UTC)
        await self._store.upsert(job)
        RESEARCH_JOBS.labels(status=status.value).inc()
        if elapsed is not None:
            RESEARCH_DURATION.observe(elapsed)
        JOB_COST.observe(job.cost_usd)
        _log.info(
            "job.finished",
            status=status.value,
            cost_usd=job.cost_usd,
            elapsed_seconds=elapsed,
        )
        await self._publish(
            JobEvent(
                job_id=job.id,
                node=_JOB_NODE,
                status="completed" if status is JobStatus.COMPLETED else "failed",
                detail=error or "",
            )
        )

    async def run(self, job: ResearchJob) -> None:
        """Execute the research pipeline, emitting an event per stage."""
        job.status = JobStatus.RUNNING
        job.updated_at = datetime.now(UTC)
        await self._store.upsert(job)
        await self._publish(JobEvent(job_id=job.id, node=_JOB_NODE, status="started"))

        ctx = initial_context(
            job.request, job_id=job.id, max_critic_rounds=self._max_critic_rounds
        )
        await self._drive(job, ctx)

    async def resume(self, job: ResearchJob, decision: dict[str, Any]) -> None:
        """Resume a job paused at the HITL gate with the reviewer's decision."""
        state_json = await self._store.load_paused_state(job.id)
        if state_json is None:
            await self._finish(
                job, JobStatus.FAILED, "resume unsupported: research job failed"
            )
            return
        job.status = JobStatus.RUNNING
        job.updated_at = datetime.now(UTC)
        await self._store.upsert(job)
        await self._publish(JobEvent(job_id=job.id, node=_JOB_NODE, status="started"))

        ctx = PipelineContext.model_validate_json(state_json)
        await self._drive(job, ctx, decision=decision)

    async def _drive(
        self,
        job: ResearchJob,
        ctx: PipelineContext,
        decision: dict[str, Any] | None = None,
    ) -> None:
        """Run (or resume) the pipeline for ``job`` and settle its status."""
        started = time.monotonic()
        tracker = UsageTracker()
        token = set_tracker(tracker)
        structlog.contextvars.bind_contextvars(job_id=job.id)
        _log.info("job.drive_started")

        async def emit(event: JobEvent) -> None:
            await self._publish(event)

        outcome = None
        try:
            if decision is None:
                outcome = await run_pipeline(ctx, self._deps, emit)
                final_ctx = outcome.context
            else:
                final_ctx = await resume_pipeline(ctx, decision, emit)
        except Exception as exc:  # noqa: BLE001 — a failed job must not crash the app
            reset_tracker(token)
            _log.warning("job.failed", error=type(exc).__name__)
            await self._finish(
                job,
                JobStatus.FAILED,
                f"{type(exc).__name__}: research job failed",
                elapsed=time.monotonic() - started,
            )
            return

        job.cost_usd = round(job.cost_usd + tracker.cost_usd(self._llm_model), 6)
        job.timings["total_seconds"] = round(
            job.timings.get("total_seconds", 0.0) + (time.monotonic() - started), 3
        )
        reset_tracker(token)

        if outcome is not None and outcome.paused:
            await self._await_review(job, outcome)
            return

        job.report = final_ctx.report
        job.citations = final_ctx.citations
        job.sources = final_ctx.sources
        job.sub_questions = final_ctx.sub_questions
        await self._finish(
            job, JobStatus.COMPLETED, elapsed=job.timings["total_seconds"]
        )

    async def _await_review(self, job: ResearchJob, outcome: Any) -> None:
        """Park the job at AWAITING_REVIEW and notify SSE subscribers."""
        await self._store.save_paused_state(
            job.id, outcome.context.model_dump_json()
        )
        job.status = JobStatus.AWAITING_REVIEW
        job.updated_at = datetime.now(UTC)
        try:
            detail = json.dumps(outcome.review_payload, default=str)
        except TypeError:
            detail = "{}"
        await self._store.upsert(job)
        await self._publish(
            JobEvent(
                job_id=job.id,
                node=_JOB_NODE,
                status="awaiting_review",
                detail=detail,
            )
        )
