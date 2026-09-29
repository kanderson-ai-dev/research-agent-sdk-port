"""Composition root: build :class:`PipelineDeps` from application settings.

When credentials are absent the factory degrades to deterministic doubles
(``StubModel`` / ``NullSearchClient``) so the app — and CI — run fully
offline. Real providers are wired only when keys are configured.
"""

from app.core.config import Settings
from app.core.schemas import RawDocument, SearchResult, Source, SubQuestion
from app.pipeline.deps import PipelineDeps
from app.services.document_parser import parse_raw_document
from app.services.document_store import DocumentStore
from app.services.models import get_model
from app.services.scraper_client import ScraperClient
from app.services.search_client import (
    DuckDuckGoSearchClient,
    NullSearchClient,
    SearchClient,
    TavilySearchClient,
)


def build_pipeline_deps(settings: Settings) -> PipelineDeps:
    """Wire production dependencies for the research pipeline."""
    model = get_model(settings)

    search_client: SearchClient
    provider = settings.search_provider.lower()
    if settings.has_search_credentials() and provider in {"auto", "tavily"}:
        assert settings.search_api_key is not None
        search_client = TavilySearchClient(
            api_key=settings.search_api_key.get_secret_value(),
            base_url=settings.search_api_base_url,
        )
    elif provider in {"auto", "duckduckgo"}:
        search_client = DuckDuckGoSearchClient()
    else:
        search_client = NullSearchClient()

    document_store = DocumentStore()
    scraper = ScraperClient(
        user_agent=settings.scrape_user_agent,
        delay_seconds=settings.scrape_delay_seconds,
        timeout_seconds=settings.scrape_timeout_seconds,
        max_bytes=settings.scrape_max_bytes,
        store=document_store,
    )

    async def search(sub_question: SubQuestion) -> list[SearchResult]:
        return await search_client.search(
            sub_question.question, max_results=settings.max_search_results_per_question
        )

    async def scrape(result: SearchResult) -> RawDocument | None:
        return await scraper.fetch(result.url, sub_question_id=result.sub_question_id)

    async def parse(document: RawDocument) -> Source | None:
        return parse_raw_document(
            document, document_store.get(document.content_ref)
        )

    return PipelineDeps(
        model=model,
        search=search,
        scrape=scrape,
        parse=parse,
        max_search_results=settings.max_search_results_per_question,
        max_sub_questions=settings.max_sub_questions,
        require_human_review=True,
    )
