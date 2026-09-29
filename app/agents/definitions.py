"""Agent definitions for the LLM-driven roles: planner, critic, writer.

These replace the LangGraph version's ``with_structured_output`` calls:
each role is an ``Agent`` whose ``output_type`` enforces the same Pydantic
contract — prompts shape behavior, schemas enforce shape.

Workers (search/scrape/parse) are intentionally **not** agents — they are
deterministic service calls fanned out by the orchestrator, matching the
original worker nodes exactly.
"""

from typing import Any

from agents import (
    Agent,
    GuardrailFunctionOutput,
    ModelSettings,
    RunContextWrapper,
    input_guardrail,
    output_guardrail,
)
from agents.items import TResponseInputItem
from pydantic import BaseModel, Field

from app.agents import prompts
from app.core.schemas import Citation, CriticVerdict
from app.guardrails import screen_topic, verify_citations
from app.pipeline.context import PipelineContext
from app.pipeline.deps import PipelineDeps


class PlanOutput(BaseModel):
    """Structured output for the planner agent."""

    questions: list[str] = Field(min_length=1)


class ReportOutput(BaseModel):
    """Structured output for the writer agent."""

    report: str
    citations: list[Citation] = Field(default_factory=list)


_ZERO_TEMP = ModelSettings(temperature=0)


# ---------------------------------------------------------------------------
# SDK-native guardrails — the ported pure functions are the enforcement logic;
# the SDK primitives are the boundary hooks. ``run_in_parallel=False`` keeps
# the input check guardrail-first (never start the model call on bad input).
# ---------------------------------------------------------------------------


def _input_text(input: str | list[TResponseInputItem]) -> str:
    """Flatten a Responses-format input into plain text for screening."""
    if isinstance(input, str):
        return input
    parts: list[str] = []
    for item in input:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for piece in content:
                if isinstance(piece, dict) and "text" in piece:
                    parts.append(str(piece["text"]))
    return "\n".join(parts)


@input_guardrail(run_in_parallel=False)
async def topic_injection_guardrail(
    ctx: RunContextWrapper[Any],
    agent: Agent[Any],
    input: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:
    """Agent-boundary input guardrail: screen the topic before the planner
    runs. The pipeline screens first (to sanitize + emit the stage event);
    this hook protects the agent wherever it is invoked."""
    result = screen_topic(_input_text(input))
    return GuardrailFunctionOutput(
        output_info={"reason": result.reason},
        tripwire_triggered=not result.allowed,
    )


@output_guardrail
async def citation_support_guardrail(
    ctx: RunContextWrapper[PipelineContext],
    agent: Agent[Any],
    output: "ReportOutput",
) -> GuardrailFunctionOutput:
    """Verify every citation's quote against the collected sources.

    Filter semantics, same as the graph version: unsupported citations are
    dropped, the report still ships — so the tripwire stays False. The
    filtered sets are written into the pipeline context passed via
    ``Runner.run(context=...)``.
    """
    kept, dropped = verify_citations(output.citations, ctx.context.sources)
    ctx.context.citations = kept
    ctx.context.dropped_citations = len(dropped)
    return GuardrailFunctionOutput(
        output_info={"dropped": len(dropped)}, tripwire_triggered=False
    )


def build_planner(
    deps: PipelineDeps, *, max_questions: int, language: str
) -> Agent[Any]:
    """Planner agent: decompose the topic into sub-questions."""
    return Agent(
        name="planner",
        model=deps.model,
        instructions=prompts.planner_system(
            max_questions=max_questions, language=language
        ),
        output_type=PlanOutput,
        model_settings=_ZERO_TEMP,
        input_guardrails=[topic_injection_guardrail],
    )


def build_critic(deps: PipelineDeps) -> Agent[Any]:
    """Critic agent: grade coverage and citation support, request re-plan."""
    return Agent(
        name="critic",
        model=deps.model,
        instructions=prompts.critic_system(),
        output_type=CriticVerdict,
        model_settings=_ZERO_TEMP,
    )


def build_writer(deps: PipelineDeps, *, language: str) -> Agent[Any]:
    """Writer agent: synthesize the report plus verbatim citations."""
    return Agent(
        name="writer",
        model=deps.model,
        instructions=prompts.writer_system(language=language),
        output_type=ReportOutput,
        model_settings=_ZERO_TEMP,
        output_guardrails=[citation_support_guardrail],
    )
