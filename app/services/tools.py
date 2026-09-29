"""Typed ``@function_tool`` wrappers around the service clients.

These expose search/scrape/parse as SDK tools with inferred input schemas,
so they can be handed to tool-calling agents. The pipeline's worker stages
call the same underlying functions through :class:`PipelineDeps` — the
deterministic fan-out does not route through the LLM.
"""

import base64
from typing import Any

from agents import RunContextWrapper, function_tool

from app.core.schemas import RawDocument, SearchResult, Source
from app.services.document_parser import parse_raw_document
from app.services.document_store import DocumentStore
from app.services.scraper_client import ScraperClient
from app.services.search_client import SearchClient


def build_search_tool(client: SearchClient) -> Any:
    """Wrap a :class:`SearchClient` as an SDK function tool."""

    @function_tool
    async def web_search(query: str, max_results: int = 3) -> list[SearchResult]:
        """Search the web for a query. Returns candidate source URLs with
        titles and snippets."""
        return await client.search(query, max_results=max_results)

    return web_search


def build_scrape_tool(scraper: ScraperClient) -> Any:
    """Wrap a :class:`ScraperClient` as an SDK function tool."""

    @function_tool
    async def scrape_url(url: str, sub_question_id: str = "unassigned") -> RawDocument | None:
        """Fetch a URL's raw content. Respects robots.txt, rate limits and
        size caps; returns null when the fetch is skipped or fails."""
        return await scraper.fetch(url, sub_question_id=sub_question_id)

    return scrape_url


def build_parse_tool() -> Any:
    """Expose the deterministic document parser as an SDK function tool."""

    @function_tool
    async def parse_document(
        url: str,
        content: str = "",
        content_bytes_b64: str | None = None,
        content_type: str = "text/html",
        sub_question_id: str = "unassigned",
    ) -> Source | None:
        """Parse raw HTML or base64-encoded PDF content into clean text with
        metadata. Returns null when nothing usable is extracted."""
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

    return parse_document


__all__ = [
    "build_parse_tool",
    "build_scrape_tool",
    "build_search_tool",
    "RunContextWrapper",
    "DocumentStore",
]
