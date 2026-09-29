"""Agent definitions for the LLM-driven roles: planner, critic, writer.

These replace the LangGraph version's ``with_structured_output`` calls:
each role is an ``Agent`` whose ``output_type`` enforces the same Pydantic
contract — prompts shape behavior, schemas enforce shape.

Workers (search/scrape/parse) are intentionally **not** agents — they are
deterministic service calls fanned out by the orchestrator, matching the
original worker nodes exactly.
"""

from typing import Any

from agents import Agent, ModelSettings
from pydantic import BaseModel, Field

from app.agents import prompts
from app.core.schemas import Citation, CriticVerdict
from app.pipeline.deps import PipelineDeps


class PlanOutput(BaseModel):
    """Structured output for the planner agent."""

    questions: list[str] = Field(min_length=1)


class ReportOutput(BaseModel):
    """Structured output for the writer agent."""

    report: str
    citations: list[Citation] = Field(default_factory=list)


_ZERO_TEMP = ModelSettings(temperature=0)


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
    )
