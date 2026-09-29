"""Offline evaluation harness: fixture-backed deps + deterministic stub model.

The ``StubModel`` is corpus-aware by construction (it plans one sub-question
per expected aspect and writes sections quoting real source text) — the
offline scorecard therefore measures *pipeline* behaviour (retrieval →
parsing → guardrails → citation verification) deterministically. Live-model
quality is measured by the same metrics when real credentials run
``run_eval`` outside CI.
"""

import json
from pathlib import Path
from typing import Any

from app.core.schemas import (
    RawDocument,
    ResearchRequest,
    SearchResult,
    Source,
    SubQuestion,
)
from app.pipeline.deps import PipelineDeps
from app.services.document_parser import parse_raw_document
from app.services.document_store import DocumentStore
from app.services.models import StubModel

CORPUS_DIR = Path(__file__).parent / "corpus"
DATASET_PATH = Path(__file__).parent / "dataset" / "topics.json"


def load_dataset() -> dict[str, Any]:
    """Load the versioned evaluation dataset."""
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def corpus_url(filename: str) -> str:
    """Stable fixture URL for a corpus document."""
    return f"https://eval.local/{filename}"


def build_eval_deps(topic_spec: dict[str, Any]) -> PipelineDeps:
    """Build pipeline deps that serve only the topic's fixture corpus."""
    corpus_files: list[str] = topic_spec["corpus"]

    async def search(sub_question: SubQuestion) -> list[SearchResult]:
        return [
            SearchResult(
                url=corpus_url(name),
                title=name.replace("_", " ").removesuffix(".html"),
                snippet=f"corpus document {name}",
                sub_question_id=sub_question.id,
            )
            for name in corpus_files
        ]

    document_store = DocumentStore()

    async def scrape(result: SearchResult) -> RawDocument | None:
        name = result.url.rsplit("/", 1)[-1]
        path = CORPUS_DIR / name
        if not path.exists():
            return None
        payload = path.read_bytes()
        return RawDocument(
            url=result.url,
            content_ref=document_store.put(payload),
            content_type="text/html",
            byte_size=len(payload),
            sub_question_id=result.sub_question_id,
        )

    async def parse(document: RawDocument) -> Source | None:
        return parse_raw_document(
            document, document_store.get(document.content_ref)
        )

    return PipelineDeps(
        model=StubModel(aspects=topic_spec["expected_aspects"]),
        search=search,
        scrape=scrape,
        parse=parse,
        require_human_review=False,
    )


def topic_request(topic_spec: dict[str, Any]) -> ResearchRequest:
    """Build the job request for a dataset topic."""
    return ResearchRequest(topic=topic_spec["topic"])
