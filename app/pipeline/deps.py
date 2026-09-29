"""Injected dependencies for the research pipeline.

The orchestrator and agent factories never import concrete tool
implementations: they call the callables in :class:`PipelineDeps`, so tests
inject deterministic doubles and production wires the real clients from
``app.services``. The LangGraph version's ``LLMClient`` protocol becomes the
SDK's ``Model`` — real OpenAI model or offline ``StubModel``.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from agents import Model

from app.core.schemas import RawDocument, SearchResult, Source, SubQuestion

SearchFn = Callable[[SubQuestion], Awaitable[list[SearchResult]]]
ScrapeFn = Callable[[SearchResult], Awaitable[RawDocument | None]]
ParseFn = Callable[[RawDocument], Awaitable[Source | None]]


@dataclass(frozen=True)
class PipelineDeps:
    """Everything the pipeline needs beyond its own context."""

    model: Model
    search: SearchFn
    scrape: ScrapeFn
    parse: ParseFn
    max_search_results: int = 3
    max_sub_questions: int = 6
    require_human_review: bool = False
