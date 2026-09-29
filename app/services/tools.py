"""Typed ``@function_tool`` wrappers around the service clients.

These expose search/scrape/parse as SDK tools with inferred JSON schemas
(``Annotated[..., Field(...)]`` constraints play the ``args_schema`` role),
so they can be handed to tool-calling agents. The pipeline's worker stages
call the same underlying functions through :class:`PipelineDeps` — the
deterministic fan-out does not route through the LLM.

The module-level ``do_*`` functions hold the logic so tests exercise the
real code path without the Runner's internal invocation machinery.
"""

import base64
from typing import Annotated, Any

from agents import function_tool
from pydantic import Field

from app.core.schemas import RawDocument, SearchResult, Source
from app.services.document_parser import parse_raw_document
from app.services.scraper_client import ScraperClient
from app.services.search_client import SearchClient

QueryParam = Annotated[str, Field(min_length=1)]
MaxResultsParam = Annotated[int, Field(ge=1, le=10)]
UrlParam = Annotated[str, Field(min_length=1)]


async def do_web_search(
    client: SearchClient, query: str, max_results: int
) -> list[SearchResult]:
    """Search logic behind the ``web_search`` tool."""
    return await client.search(query, max_results=max_results)


async def do_scrape_url(
    scraper: ScraperClient, url: str, sub_question_id: str
) -> RawDocument | None:
    """Fetch logic behind the ``scrape_url`` tool."""
    return await scraper.fetch(url, sub_question_id=sub_question_id)


async def do_parse_document(
    url: str,
    content: str,
    content_bytes_b64: str | None,
    content_type: str,
    sub_question_id: str,
) -> Source | None:
    """Parse logic behind the ``parse_document`` tool."""
    if content_bytes_b64 is not None:
        try:
            payload = base64.b64decode(content_bytes_b64)
        except ValueError:
            return None
    else:
        payload = content.encode("utf-8", errors="replace")
    return parse_raw_document(
        RawDocument(
            url=url,
            content_type=content_type,
            byte_size=len(payload),
            sub_question_id=sub_question_id or "unassigned",
        ),
        payload,
    )


def build_search_tool(client: SearchClient) -> Any:
    """Wrap a :class:`SearchClient` as an SDK function tool."""

    @function_tool
    async def web_search(
        query: QueryParam, max_results: MaxResultsParam = 3
    ) -> list[SearchResult]:
        """Search the web for a query. Returns candidate source URLs with
        titles and snippets."""
        return await do_web_search(client, query, max_results)

    return web_search


def build_scrape_tool(scraper: ScraperClient) -> Any:
    """Wrap a :class:`ScraperClient` as an SDK function tool."""

    @function_tool
    async def scrape_url(
        url: UrlParam, sub_question_id: str = "unassigned"
    ) -> RawDocument | None:
        """Fetch a URL's raw content. Respects robots.txt, rate limits and
        size caps; returns null when the fetch is skipped or fails."""
        return await do_scrape_url(scraper, url, sub_question_id)

    return scrape_url


def build_parse_tool() -> Any:
    """Expose the deterministic document parser as an SDK function tool."""

    @function_tool
    async def parse_document(
        url: UrlParam,
        content: str = "",
        content_bytes_b64: str | None = None,
        content_type: str = "text/html",
        sub_question_id: str = "unassigned",
    ) -> Source | None:
        """Parse raw HTML or base64-encoded PDF content into clean text with
        metadata. Returns null when nothing usable is extracted."""
        return await do_parse_document(
            url, content, content_bytes_b64, content_type, sub_question_id
        )

    return parse_document
