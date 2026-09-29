"""Agent tracing wiring: LangSmith when configured, disabled otherwise.

The SDK streams spans to OpenAI's backend by default. Two explicit postures:

- LangSmith configured (``LANGCHAIN_API_KEY`` + flag) → register
  ``OpenAIAgentsTracingProcessor`` via ``set_trace_processors`` so agent
  runs, model calls, tool calls and handoffs land in LangSmith — the same
  observability surface the LangGraph version had.
- Not configured → ``set_tracing_disabled(True)`` so CI and offline runs
  never attempt network calls or need secrets.
"""

import os
from typing import cast

from agents import set_trace_processors, set_tracing_disabled
from agents.tracing.processor_interface import TracingProcessor

from app.core.config import Settings
from app.core.logging import get_logger

_log = get_logger(__name__)


def configure_tracing(settings: Settings) -> None:
    """Register the LangSmith trace processor when credentials are present."""
    if settings.langchain_api_key is None or not settings.langchain_tracing_v2:
        set_tracing_disabled(True)
        return

    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key.get_secret_value()
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

    # Imported lazily so offline installs without the extra still run.
    from langsmith.integrations.openai_agents_sdk import (  # noqa: PLC0415
        OpenAIAgentsTracingProcessor,
    )

    processor = OpenAIAgentsTracingProcessor(  # type: ignore[no-untyped-call]
        project_name=settings.langchain_project
    )
    set_trace_processors([cast(TracingProcessor, processor)])
    _log.info("tracing.configured", backend="langsmith")
