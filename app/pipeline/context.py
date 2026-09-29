"""Serializable pipeline context — the analog of LangGraph's ``ResearchState``.

Where LangGraph threaded a ``TypedDict`` through checkpointed graph nodes,
the port keeps a single Pydantic model mutated by the orchestrator
(:mod:`app.pipeline.runner`). Serializability matters for exactly one
reason: the human-review gate persists this object so a paused run resumes
later — the SDK has no checkpointer, so the job store plays that role.
"""

from pydantic import BaseModel, Field

from app.core.schemas import (
    Citation,
    CriticVerdict,
    RawDocument,
    ResearchRequest,
    SearchResult,
    Source,
    SubQuestion,
)


class PipelineContext(BaseModel):
    """Full state threaded through the research pipeline."""

    job_id: str
    request: ResearchRequest
    topic: str

    blocked: bool = False
    rejection_reason: str | None = None

    sub_questions: list[SubQuestion] = Field(default_factory=list)
    pending_questions: list[SubQuestion] = Field(default_factory=list)

    search_results: list[SearchResult] = Field(default_factory=list)
    raw_documents: list[RawDocument] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)

    critic_verdict: CriticVerdict | None = None
    critic_rounds: int = 0
    max_critic_rounds: int = 2

    report: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    dropped_citations: int = 0
    errors: list[str] = Field(default_factory=list)

    escalated: bool = False
    human_decision: str | None = None


def initial_context(
    request: ResearchRequest, *, job_id: str, max_critic_rounds: int = 2
) -> PipelineContext:
    """Build the starting context for a research job.

    ``quick`` depth is a single-pass best-effort run: the critic still grades
    coverage and may escalate to the human gate, but no re-planning round is
    performed — latency stays bounded for demo-style requests.
    """
    if request.depth == "quick":
        max_critic_rounds = 0
    return PipelineContext(
        job_id=job_id,
        request=request,
        topic=request.topic,
        max_critic_rounds=max_critic_rounds,
    )
