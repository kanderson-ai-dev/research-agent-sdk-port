"""Model wiring for the planner / critic / writer agents.

Two implementations of the SDK's ``Model`` interface:

- ``OpenAIChatCompletionsModel`` — real calls through OpenAI's chat
  completions API (same surface the LangGraph version used, keeping
  cost/latency comparisons honest).
- ``StubModel`` — deterministic, offline double used by tests and by the
  app when no API key is configured (CI must run with zero secrets).
"""

import json
import re
from collections.abc import AsyncIterator, Sequence
from typing import Any

from agents import (
    AgentOutputSchemaBase,
    Handoff,
    Model,
    ModelResponse,
    ModelSettings,
    Usage,
)
from agents.items import TResponseInputItem, TResponseStreamEvent
from agents.models.interface import ModelTracing
from agents.tool import Tool
from openai import AsyncOpenAI
from openai.types.responses import (
    ResponseOutputMessage,
    ResponseOutputText,
    ResponsePromptParam,
)

from app.core.config import Settings

# ---------------------------------------------------------------------------
# Stub model (deterministic, offline)
# ---------------------------------------------------------------------------

_EVIDENCE_HEADER_RE = re.compile(r"\[(s-[^\]]+)\] (.*) \((https?://[^)\s]+)\)\n")


def _input_text(input: str | list[TResponseInputItem]) -> str:
    """Flatten a Responses-format input into plain text the stub can scan."""
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


def _message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="stub-msg",
        content=[ResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


class _EvidenceSource:
    """One parsed ``[id] title (url) / text`` block from an evidence prompt."""

    def __init__(self, source_id: str, url: str, text: str) -> None:
        self.source_id = source_id
        self.url = url
        self.text = text


def _parse_evidence(text: str) -> list[_EvidenceSource]:
    """Parse the ``Evidence:`` block written by ``prompts.format_evidence``."""
    markers = list(_EVIDENCE_HEADER_RE.finditer(text))
    sources: list[_EvidenceSource] = []
    for i, match in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        body = text[match.end() : end].strip()
        sources.append(
            _EvidenceSource(
                source_id=match.group(1), url=match.group(3), text=body
            )
        )
    return sources


def _parse_sub_questions(text: str) -> list[str]:
    """Extract the ``- question`` lines between ``Sub-questions:`` and ``Evidence:``."""
    if "Sub-questions:" not in text or "Evidence:" not in text:
        return []
    block = text.split("Sub-questions:", 1)[1].split("Evidence:", 1)[0]
    return [
        line[2:].strip() for line in block.splitlines() if line.startswith("- ")
    ]


class StubModel(Model):
    """Deterministic offline ``Model`` for the planner/critic/writer agents.

    Branches on ``output_schema.name`` to produce schema-conforming JSON.
    ``force_needs_more`` makes the critic always request another research
    round, which lets tests prove the critic/re-plan loop is bounded.
    ``aspects`` presets the planner's questions (evaluation harness).
    """

    def __init__(
        self,
        *,
        force_needs_more: bool = False,
        aspects: Sequence[str] | None = None,
    ) -> None:
        self._force_needs_more = force_needs_more
        self._aspects = list(aspects) if aspects is not None else None

    # -- canned responses ---------------------------------------------------

    def _plan(self, system_instructions: str, topic: str) -> dict[str, Any]:
        max_q = 3
        match = re.search(r"at most (\d+)", system_instructions)
        if match:
            max_q = int(match.group(1))
        if self._aspects is not None:
            questions = [f"What are the {a}?" for a in self._aspects[:max_q]]
        else:
            count = min(3, max_q)
            questions = [f"{topic} — aspect {i + 1}" for i in range(count)]
        return {"questions": questions}

    def _critique(self, text: str) -> dict[str, Any]:
        if self._force_needs_more:
            return {
                "coverage_score": 0.4,
                "citation_support_score": 0.4,
                "missing_aspects": ["additional aspect"],
                "needs_more_research": True,
                "feedback": "stub: always needs more research",
            }
        has_sources = "Evidence:\n(none)" not in text
        score = 1.0 if has_sources else 0.0
        return {
            "coverage_score": score,
            "citation_support_score": score,
            "missing_aspects": [],
            "needs_more_research": False,
            "feedback": "stub: satisfied",
        }

    def _report(self, text: str) -> dict[str, Any]:
        topic = text.split("\n", 1)[0].removeprefix("Topic: ").strip()
        questions = _parse_sub_questions(text)
        sources = _parse_evidence(text)
        lines = [f"# Research report: {topic}", ""]
        citations: list[dict[str, Any]] = []
        if not questions:
            lines.append("No sub-questions were planned.")
        for question in questions:
            lines.append(f"## {question}")
            related = sources[:2]
            if not related:
                lines.append("No evidence was collected for this aspect.")
            for source in related:
                quote = source.text[:150].strip() or source.url
                lines.append(f"- {quote} ({source.url})")
                citations.append(
                    {
                        "claim": f"{question}: supported by {source.url}",
                        "source_id": source.source_id,
                        "quote": quote,
                    }
                )
        return {"report": "\n".join(lines), "citations": citations}

    # -- Model interface ------------------------------------------------------

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> ModelResponse:
        del model_settings, tools, handoffs, tracing
        del previous_response_id, conversation_id, prompt
        name = output_schema.name() if output_schema is not None else ""
        text = _input_text(input)
        if "Plan" in name:
            payload: str | dict[str, Any] = self._plan(system_instructions or "", text)
        elif "Critic" in name:
            payload = self._critique(text)
        elif "Report" in name:
            payload = self._report(text)
        else:
            payload = "stub reply"
        body = payload if isinstance(payload, str) else json.dumps(payload)
        return ModelResponse(
            output=[_message(body)],
            usage=Usage(
                requests=1,
                input_tokens=max(1, len(text) // 4),
                output_tokens=max(1, len(body) // 4),
            ),
            response_id="stub-resp",
        )

    def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        raise NotImplementedError("StubModel does not stream")


def get_model(settings: Settings) -> Model:
    """Factory: real OpenAI chat-completions model when configured, else stub."""
    if settings.has_llm_credentials():
        assert settings.openai_api_key is not None
        from agents import OpenAIChatCompletionsModel

        client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        return OpenAIChatCompletionsModel(
            model=settings.llm_model, openai_client=client
        )
    return StubModel()
