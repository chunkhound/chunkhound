"""Contract tests for the deterministic embedding HTTP fixture."""

import httpx
import pytest

from tests.fixtures.embedding_server import (
    EmbeddingConcept,
    MockEmbeddingServer,
    unit_vector,
)


@pytest.mark.fast
@pytest.mark.asyncio
async def test_embedding_vectors_depend_on_content_not_request_position() -> None:
    """The same text must keep its vector across batches and request shapes."""
    concept = EmbeddingConcept(
        name="target",
        markers=("target_marker",),
        vector=unit_vector(0, 8),
    )

    with MockEmbeddingServer((concept,), dimensions=8) as server:
        async with httpx.AsyncClient(timeout=2.0) as client:
            first = await client.post(
                f"{server.base_url}/embeddings",
                json={"model": "qa", "input": ["target_marker", "other text"]},
            )
            second = await client.post(
                f"{server.base_url}/embeddings",
                json={"model": "qa", "input": ["other text", "target_marker"]},
            )

        assert first.status_code == 200
        assert second.status_code == 200
        first_vectors = {
            item["index"]: item["embedding"] for item in first.json()["data"]
        }
        second_vectors = {
            item["index"]: item["embedding"] for item in second.json()["data"]
        }
        assert first_vectors[0] == second_vectors[1] == list(unit_vector(0, 8))
        assert first_vectors[1] == second_vectors[0]


@pytest.mark.fast
@pytest.mark.asyncio
async def test_embedding_server_scores_specific_marker_before_generic_one() -> None:
    """A more specific marker must win over a generic marker in the same text."""
    concepts = (
        EmbeddingConcept("generic", ("function",), unit_vector(0, 8)),
        EmbeddingConcept("specific", ("modified_function",), unit_vector(1, 8)),
    )

    with MockEmbeddingServer(concepts, dimensions=8) as server:
        assert server.vector_for("def modified_function()") == list(unit_vector(1, 8))
        assert server.matched_concepts["specific"] == 1
        assert server.matched_concepts["generic"] == 0


@pytest.mark.fast
def test_absent_long_marker_cannot_outrank_present_specific_marker() -> None:
    """Only markers present in the text take part in ranking."""
    concepts = (
        EmbeddingConcept(
            "generic",
            ("function", "extremely long absent marker that never occurs"),
            unit_vector(0, 8),
        ),
        EmbeddingConcept("specific", ("modified_function",), unit_vector(1, 8)),
    )

    with MockEmbeddingServer(concepts, dimensions=8) as server:
        assert server.vector_for("modified_function") == list(unit_vector(1, 8))
        assert server.matched_concepts["specific"] == 1
        assert server.matched_concepts["generic"] == 0


@pytest.mark.fast
@pytest.mark.asyncio
async def test_embedding_server_records_unmatched_inputs() -> None:
    """Unmatched content stays visible so tests can audit unexpected requests."""
    with MockEmbeddingServer((), dimensions=4) as server:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.post(
                f"{server.base_url}/embeddings",
                json={"model": "qa", "input": ["unscripted"]},
            )

        assert response.status_code == 200
        assert len(response.json()["data"][0]["embedding"]) == 4
        assert server.fallback_texts == ["unscripted"]
