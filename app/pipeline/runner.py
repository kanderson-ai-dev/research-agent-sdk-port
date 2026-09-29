"""Research pipeline orchestrator — the analog of the LangGraph ``StateGraph``.

Same stages, same order, same observable behavior as the graph version:

``input_guardrail → planner → (search × N → scrape × M → parse × K) →
critic ⇄ re-plan (bounded) → writer → output_guardrail → human_review →
report_assembler``

Differences are mechanical, not semantic:

- ``Send()`` fan-out → ``asyncio.gather`` over per-branch coroutines.
- Reducer channels → explicit ``ctx.*`` merges.
- ``interrupt()`` → a returned pause payload; the caller persists
  ``PipelineContext`` and resumes via :func:`resume_pipeline`.
- Node completion events → identical ``JobEvent.node`` vocabulary emitted
  through ``emit`` so SSE consumers (the console) see the same stream.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

from agents import Runner

from app.agents import prompts
from app.agents.definitions import (
    PlanOutput,
    ReportOutput,
    build_critic,
    build_planner,
    build_writer,
)
from app.core.schemas import (
    CriticVerdict,
    JobEvent,
    SubQuestion,
    SubQuestionStatus,
)
from app.guardrails import (
    sanitize_scraped_content,
    screen_topic,
    verify_citations,
)
from app.pipeline.context import PipelineContext
from app.pipeline.deps import PipelineDeps
from app.services.usage import record_usage

_DEPTH_TARGETS = {"quick": 3, "standard": 5, "deep": 8}

# Per-depth cap on search results per sub-question — "quick" trades breadth
# for latency: fewer sources means fewer serialized scrape rounds.
_DEPTH_SEARCH_CAP = {"quick": 2}

Emit = Callable[[JobEvent], Awaitable[None]]


@dataclass
class PipelineOutcome:
    """Result of :func:`run_pipeline`.

    ``review_payload`` set → the run paused at the human-review gate; the
    caller persists ``context`` and resumes with :func:`resume_pipeline`.
    """

    context: PipelineContext
    review_payload: dict[str, Any] | None = None

    @property
    def paused(self) -> bool:
        return self.review_payload is not None


def _usage_totals(result: Any) -> tuple[int, int]:
    """Sum token usage across the run's model responses."""
    input_tokens = sum(r.usage.input_tokens for r in result.raw_responses)
    output_tokens = sum(r.usage.output_tokens for r in result.raw_responses)
    return input_tokens, output_tokens


async def run_pipeline(
    ctx: PipelineContext, deps: PipelineDeps, emit: Emit
) -> PipelineOutcome:
    """Run the pipeline until completion or the human-review pause."""

    async def stage(node: str) -> None:
        await emit(JobEvent(job_id=ctx.job_id, node=node, status="completed"))

    # -- input guardrail -----------------------------------------------------
    screened = screen_topic(ctx.topic)
    if not screened.allowed:
        ctx.blocked = True
        ctx.rejection_reason = screened.reason
    else:
        ctx.topic = screened.sanitized_topic
    await stage("input_guardrail")
    if ctx.blocked:
        # rejection_output: generic refusal, no detail leakage
        ctx.report = "This research request could not be processed."
        ctx.citations = []
        await stage("rejection_output")
        return PipelineOutcome(ctx)

    # -- planner ---------------------------------------------------------------
    target = min(_DEPTH_TARGETS[ctx.request.depth], deps.max_sub_questions)
    plan_result = await Runner.run(
        build_planner(deps, max_questions=target, language=ctx.request.language),
        ctx.topic,
        max_turns=2,
    )
    record_usage("planner", *_usage_totals(plan_result))
    plan = cast(PlanOutput, plan_result.final_output)
    questions = [
        SubQuestion(id=f"q{i + 1}", question=q)
        for i, q in enumerate(plan.questions[:target])
    ]
    ctx.sub_questions = questions
    ctx.pending_questions = list(questions)
    await stage("planner")

    # -- evidence fan-out + bounded critic loop ---------------------------------
    while True:
        await _evidence_fanout(ctx, deps, emit)
        ctx.pending_questions = []

        evidence = prompts.format_evidence(
            ctx.sources, per_source_chars=prompts.CRITIC_EVIDENCE_CHARS
        )
        critic_result = await Runner.run(
            build_critic(deps),
            prompts.research_prompt(
                topic=ctx.topic,
                sub_questions=ctx.sub_questions,
                evidence=evidence,
            ),
            max_turns=2,
        )
        record_usage("critic", *_usage_totals(critic_result))
        verdict = cast(CriticVerdict, critic_result.final_output)
        ctx.critic_verdict = verdict
        ctx.critic_rounds += 1
        # Escalate to the human gate when the critic still wants more research
        # but the bounded loop has run out of rounds.
        ctx.escalated = bool(
            verdict.needs_more_research and ctx.critic_rounds >= ctx.max_critic_rounds
        )
        await stage("critic")

        if not (
            verdict.needs_more_research
            and ctx.critic_rounds < ctx.max_critic_rounds
        ):
            break
        new_questions = _replan(ctx)
        await stage("replan")
        ctx.pending_questions = new_questions

    # -- writer -----------------------------------------------------------------
    evidence = prompts.format_evidence(
        ctx.sources, per_source_chars=prompts.WRITER_EVIDENCE_CHARS
    )
    write_result = await Runner.run(
        build_writer(deps, language=ctx.request.language),
        prompts.research_prompt(
            topic=ctx.topic, sub_questions=ctx.sub_questions, evidence=evidence
        ),
        context=ctx,
        max_turns=2,
    )
    record_usage("writer", *_usage_totals(write_result))
    report = cast(ReportOutput, write_result.final_output)
    ctx.report = report.report
    await stage("writer")

    # -- output guardrail (citation verification is a filter, not a veto) --------
    # The SDK output_guardrail on the writer agent already wrote the verified
    # sets into ``ctx``; fall back to a direct check if it did not run.
    if write_result.output_guardrail_results:
        dropped = ctx.dropped_citations
    else:  # pragma: no cover - defensive
        kept, dropped_list = verify_citations(report.citations, ctx.sources)
        ctx.citations = kept
        ctx.dropped_citations = dropped = len(dropped_list)
    if dropped:
        ctx.errors.append(
            f"output guardrail dropped {dropped} unsupported citations"
        )
    await stage("output_guardrail")

    # -- human review gate --------------------------------------------------------
    if deps.require_human_review:
        payload: dict[str, Any] = {
            "job_id": ctx.job_id,
            "report": ctx.report,
            "citations": [c.model_dump(mode="json") for c in ctx.citations],
            "escalated": ctx.escalated,
            "missing_aspects": (
                ctx.critic_verdict.missing_aspects if ctx.critic_verdict else []
            ),
        }
        return PipelineOutcome(ctx, review_payload=payload)
    ctx.human_decision = "auto"
    await stage("human_review")

    _assemble(ctx)
    await stage("report_assembler")
    return PipelineOutcome(ctx)


def _replan(ctx: PipelineContext) -> list[SubQuestion]:
    """Turn the critic's missing aspects into new pending sub-questions."""
    verdict = ctx.critic_verdict
    if verdict is None or not verdict.missing_aspects:
        return []
    existing = {sq.question for sq in ctx.sub_questions}
    new_questions: list[SubQuestion] = []
    for aspect in verdict.missing_aspects:
        if aspect in existing:
            continue
        existing.add(aspect)
        new_questions.append(
            SubQuestion(
                id=f"r{ctx.critic_rounds}q{len(new_questions) + 1}",
                question=aspect,
            )
        )
    ctx.sub_questions = [*ctx.sub_questions, *new_questions]
    return new_questions


def _assemble(ctx: PipelineContext) -> None:
    """Finalize sub-question statuses (answered when a source backs them)."""
    answered_ids = {s.sub_question_id for s in ctx.sources}
    ctx.sub_questions = [
        sq.model_copy(
            update={
                "status": (
                    SubQuestionStatus.ANSWERED
                    if sq.id in answered_ids
                    else SubQuestionStatus.UNRESOLVED
                )
            }
        )
        for sq in ctx.sub_questions
    ]


# ---------------------------------------------------------------------------
# Fan-out stages (Send() -> asyncio.gather)
# ---------------------------------------------------------------------------


async def _evidence_fanout(
    ctx: PipelineContext, deps: PipelineDeps, emit: Emit
) -> None:
    """search × pending → scrape × unique URLs → parse × raw docs.

    ``to_critic`` fires only when a fan-out produces nothing — the same
    early-route the graph took; on the happy path document workers edge
    straight into the critic.
    """
    to_critic = JobEvent(job_id=ctx.job_id, node="to_critic", status="completed")
    if not ctx.pending_questions:
        await emit(to_critic)
        return
    await _search_fanout(ctx, deps, emit)
    if not any(r.url for r in ctx.search_results):
        await emit(to_critic)
        return
    await _scrape_fanout(ctx, deps, emit)
    if not ctx.raw_documents:
        await emit(to_critic)
        return
    await _parse_fanout(ctx, deps, emit)


async def _search_fanout(
    ctx: PipelineContext, deps: PipelineDeps, emit: Emit
) -> None:
    """One search worker per pending sub-question."""

    async def worker(sq: SubQuestion) -> None:
        try:
            results = await deps.search(sq)
        except Exception as exc:  # noqa: BLE001 — record and continue
            ctx.errors.append(f"search failed for {sq.id}: {exc}")
        else:
            cap = min(
                _DEPTH_SEARCH_CAP.get(ctx.request.depth, deps.max_search_results),
                deps.max_search_results,
            )
            capped = results[:cap]
            for result in capped:
                result.sub_question_id = sq.id
            ctx.search_results.extend(capped)
        await emit(JobEvent(job_id=ctx.job_id, node="search_worker", status="completed"))

    await asyncio.gather(*(worker(sq) for sq in ctx.pending_questions))
    await emit(JobEvent(job_id=ctx.job_id, node="aggregate_search", status="completed"))


async def _scrape_fanout(
    ctx: PipelineContext, deps: PipelineDeps, emit: Emit
) -> None:
    """One scrape worker per unique accumulated search-result URL."""
    seen: set[str] = set()
    unique = []
    for result in ctx.search_results:
        if result.url not in seen:
            seen.add(result.url)
            unique.append(result)
    if not unique:
        return

    async def worker(result: Any) -> None:
        try:
            document = await deps.scrape(result)
        except Exception as exc:  # noqa: BLE001 — record and continue
            ctx.errors.append(f"scrape failed for {result.url}: {exc}")
        else:
            if document is None:
                ctx.errors.append(f"scrape skipped for {result.url}")
            else:
                ctx.raw_documents.append(document)
        await emit(JobEvent(job_id=ctx.job_id, node="scrape_worker", status="completed"))

    await asyncio.gather(*(worker(r) for r in unique))
    await emit(
        JobEvent(job_id=ctx.job_id, node="aggregate_documents", status="completed")
    )


async def _parse_fanout(
    ctx: PipelineContext, deps: PipelineDeps, emit: Emit
) -> None:
    """One document worker per raw document: parse, sanitize, Source."""

    async def worker(document: Any) -> None:
        try:
            source = await deps.parse(document)
        except Exception as exc:  # noqa: BLE001 — record and continue
            ctx.errors.append(f"parse failed for {document.url}: {exc}")
        else:
            if source is None:
                ctx.errors.append(f"parse produced no source for {document.url}")
            else:
                ctx.sources.append(
                    source.model_copy(
                        update={
                            "extracted_text": sanitize_scraped_content(
                                source.extracted_text
                            )
                        }
                    )
                )
        await emit(
            JobEvent(job_id=ctx.job_id, node="document_worker", status="completed")
        )

    await asyncio.gather(*(worker(doc) for doc in ctx.raw_documents))


# ---------------------------------------------------------------------------
# Resume after the human-review gate
# ---------------------------------------------------------------------------


async def resume_pipeline(
    ctx: PipelineContext, decision: dict[str, Any], emit: Emit
) -> PipelineContext:
    """Apply the reviewer's decision and finish the pipeline.

    Mirrors the original ``human_review`` node: ``reject`` replaces the
    report, ``edit`` publishes the edited text, ``approve`` keeps it.
    """
    action = decision.get("action", "approve")
    if action == "reject":
        ctx.report = "Report rejected by human reviewer."
        ctx.human_decision = "reject"
    elif action == "edit" and isinstance(decision.get("report"), str):
        ctx.report = decision["report"]
        ctx.human_decision = "edit"
    else:
        ctx.human_decision = "approve"
    await emit(JobEvent(job_id=ctx.job_id, node="human_review", status="completed"))
    _assemble(ctx)
    await emit(JobEvent(job_id=ctx.job_id, node="report_assembler", status="completed"))
    return ctx
