"""Shared fixtures and deterministic doubles for the test suite."""

import hashlib
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import pytest
from agents import set_tracing_disabled

from app.core.schemas import (
    RawDocument,
    ResearchRequest,
    SearchResult,
    Source,
    SubQuestion,
)
from app.pipeline.deps import PipelineDeps
from app.services.models import StubModel

# The suite must run fully offline: never let SDK tracing or a stray real
# API key turn a unit test into a network call.
set_tracing_disabled(True)
os.environ.pop("OPENAI_API_KEY", None)


@pytest.fixture
def research_request() -> ResearchRequest:
    return ResearchRequest(topic="impact of the EU AI Act on small startups")


@pytest.fixture
def fake_search() -> Callable[[SubQuestion], Awaitable[list[SearchResult]]]:
    async def _search(sub_question: SubQuestion) -> list[SearchResult]:
        return [
            SearchResult(
                url=f"https://example.com/{sub_question.id}/{i}",
                title=f"Result {i} for {sub_question.id}",
                snippet=f"snippet about {sub_question.question}",
                sub_question_id=sub_question.id,
            )
            for i in range(2)
        ]

    return _search


@pytest.fixture
def fake_scrape() -> Callable[[SearchResult], Awaitable[RawDocument | None]]:
    async def _scrape(result: SearchResult) -> RawDocument | None:
        return RawDocument(
            url=result.url,
            content_ref="",
            content_type="text/html",
            status_code=200,
            sub_question_id=result.sub_question_id,
        )

    return _scrape


@pytest.fixture
def fake_parse() -> Callable[[RawDocument], Awaitable[Source | None]]:
    async def _parse(document: RawDocument) -> Source | None:
        digest = hashlib.sha256(document.url.encode()).hexdigest()[:12]
        return Source(
            id=f"s-{digest}",
            url=document.url,
            title=f"Parsed {document.url}",
            fetched_at=datetime.now(UTC),
            content_hash=digest,
            extracted_text=f"Evidence text extracted from {document.url}.",
            sub_question_id=document.sub_question_id,
        )

    return _parse


@pytest.fixture
def stub_deps(
    fake_search: Callable[[SubQuestion], Awaitable[list[SearchResult]]],
    fake_scrape: Callable[[SearchResult], Awaitable[RawDocument | None]],
    fake_parse: Callable[[RawDocument], Awaitable[Source | None]],
) -> PipelineDeps:
    return PipelineDeps(
        model=StubModel(),
        search=fake_search,
        scrape=fake_scrape,
        parse=fake_parse,
        max_search_results=3,
        max_sub_questions=6,
    )


# Re-export the callable types so tests stay annotated consistently.
SearchCallable = Callable[[SubQuestion], Awaitable[list[SearchResult]]]
