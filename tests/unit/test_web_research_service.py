"""Contracts for composing in-process research over fetched pages."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.embeddings import LocalEmbeddingResult
from chunkhound.services.transient_search_service import TransientSearchService
from chunkhound.services.web_research_service import (
    pages_to_chunks,
    research_web_pages,
)


def test_pages_to_chunks_keeps_source_url_without_writing_files() -> None:
    chunks = pages_to_chunks(
        [("https://example.invalid/docs", ".md", "# Docs\nUseful content")]
    )

    assert chunks
    assert chunks[0].file_path == "https://example.invalid/docs"
    assert chunks[0].code == "# Docs\nUseful content"


@pytest.mark.asyncio
async def test_research_factory_receives_transient_search_service() -> None:
    manager = MagicMock()

    async def embed(texts: list[str]) -> LocalEmbeddingResult:
        return LocalEmbeddingResult(
            embeddings=[[1.0, 0.0] for _ in texts],
            model="test",
            provider="test",
            dims=2,
        )

    manager.embed_texts = AsyncMock(side_effect=embed)
    research = MagicMock()
    research.deep_research = AsyncMock(return_value={"answer": "ANSWER"})
    captured: dict = {}

    def create(**kwargs):
        captured.update(kwargs)
        return research

    with patch(
        "chunkhound.services.web_research_service.ResearchServiceFactory.create",
        side_effect=create,
    ):
        result = await research_web_pages(
            "query",
            [("https://example.invalid/docs", ".md", "# Docs")],
            MagicMock(),
            manager,
            MagicMock(),
        )

    assert result == {"answer": "ANSWER"}
    assert isinstance(captured["db_services"].search_service, TransientSearchService)
