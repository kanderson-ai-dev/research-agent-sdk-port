"""Tests for the SDK function-tool wrappers around service clients."""

from app.core.schemas import RawDocument, SearchResult
from app.services.scraper_client import ScraperClient
from app.services.tools import (
    build_parse_tool,
    build_scrape_tool,
    build_search_tool,
    do_parse_document,
    do_scrape_url,
    do_web_search,
)


class _FakeSearch:
    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        return [
            SearchResult(
                url="https://example.com",
                title="t",
                snippet="s",
                sub_question_id="unassigned",
            )
        ]


async def test_search_tool_invokes_client() -> None:
    tool = build_search_tool(_FakeSearch())
    assert tool.name == "web_search"
    result = await do_web_search(_FakeSearch(), "AI regulation", 2)
    assert len(result) == 1
    assert result[0].url == "https://example.com"


def test_search_tool_schema_carries_validation() -> None:
    """Field constraints land in the JSON schema the model sees."""
    schema = build_search_tool(_FakeSearch()).params_json_schema
    assert schema["properties"]["query"]["minLength"] == 1
    assert schema["properties"]["max_results"]["maximum"] == 10
    assert schema["properties"]["max_results"]["minimum"] == 1
    assert schema["additionalProperties"] is False


async def test_scrape_tool_invokes_scraper() -> None:
    class FakeScraper(ScraperClient):
        def __init__(self) -> None:
            pass

        async def fetch(
            self, url: str, *, sub_question_id: str = ""
        ) -> RawDocument | None:
            return RawDocument(
                url=url, content_ref="d-x", sub_question_id=sub_question_id
            )

    tool = build_scrape_tool(FakeScraper())
    assert tool.name == "scrape_url"
    result = await do_scrape_url(FakeScraper(), "https://example.com", "q1")
    assert result is not None
    assert result.sub_question_id == "q1"


async def test_parse_tool_parses_html() -> None:
    tool = build_parse_tool()
    assert tool.name == "parse_document"
    result = await do_parse_document(
        "https://example.com",
        "<html><head><title>T</title></head>"
        "<body><p>Body text here.</p></body></html>",
        None,
        "text/html",
        "q1",
    )
    assert result is not None
    assert result.title == "T"
    assert "Body text" in result.extracted_text


async def test_parse_tool_bad_base64_returns_none() -> None:
    assert (
        await do_parse_document(
            "https://example.com", "", "!!!not-b64!!!", "application/pdf", "q1"
        )
        is None
    )


def test_scrape_tool_schema_defaults() -> None:
    class _BareScraper(ScraperClient):
        def __init__(self) -> None:
            pass

    tool = build_scrape_tool(_BareScraper())
    # sub_question_id default is part of the schema contract
    assert tool.params_json_schema["properties"]["sub_question_id"]["default"] == (
        "unassigned"
    )
