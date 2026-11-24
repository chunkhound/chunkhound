"""Contract: a blank LLM gap query must not abort gap detection.

Runs the real VoyageAI adapter and gap clustering; only the external Voyage SDK
client and the LLM are faked.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import voyageai

from chunkhound.core.config.research_config import ResearchConfig
from chunkhound.providers.embeddings.voyageai_provider import (
    VoyageAIEmbeddingProvider,
)
from chunkhound.services.research.shared.gap_detection import GapDetectionService
from tests.fixtures.fake_providers import FakeLLMProvider

DIMS = 1024


class _FakeVoyageSDK:
    """External Voyage HTTP boundary: returns one vector per text received."""

    def __init__(self) -> None:
        self._params: dict[str, object] = {}
        self.received: list[str] = []

    def embed(self, texts: list[str], **_: object) -> SimpleNamespace:
        self.received.extend(texts)
        vectors = []
        for text in texts:
            vector = [0.0] * DIMS
            vector[0 if "auth" in text else 1] = 1.0
            vectors.append(vector)
        return SimpleNamespace(embeddings=vectors, total_tokens=1)


class _GapLLM(FakeLLMProvider):
    """LLM returning a mix of valid and whitespace-only gap queries."""

    async def complete_structured(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> dict[str, Any]:
        if "identify semantic gaps" in prompt:
            gaps = [
                {"query": "auth token refresh", "rationale": "r", "confidence": 0.9},
                {"query": "   ", "rationale": "r", "confidence": 0.8},
                {"query": "auth session expiry", "rationale": "r", "confidence": 0.7},
                {"query": "\n\t", "rationale": "r", "confidence": 0.6},
                {"query": "cache eviction", "rationale": "r", "confidence": 0.5},
            ]
            return {"gaps": gaps}
        return {"unified_query": "merged gap"}


class _Manager:
    def __init__(self, provider: Any, llm: Any) -> None:
        self._provider = provider
        self._llm = llm

    def get_provider(self, name: str | None = None) -> Any:
        return self._provider

    def get_utility_provider(self) -> Any:
        return self._llm


@pytest.mark.asyncio
async def test_blank_gap_query_is_dropped_and_valid_gaps_still_cluster() -> None:
    sdk = _FakeVoyageSDK()
    with patch.object(voyageai, "Client", return_value=sdk):
        embedder = VoyageAIEmbeddingProvider(
            api_key="k", model="my-custom-model", output_dims=DIMS
        )
    llm = _GapLLM()
    service = GapDetectionService(
        llm_manager=_Manager(None, llm),  # type: ignore[arg-type]
        embedding_manager=_Manager(embedder, llm),  # type: ignore[arg-type]
        db_services=SimpleNamespace(
            provider=SimpleNamespace(get_base_directory=lambda: Path("/tmp"))
        ),  # type: ignore[arg-type]
        config=ResearchConfig(min_gaps=1, max_gaps=5),
    )

    filled: list[str] = []

    async def fake_fill(root_query: str, gap: Any, *_: Any) -> list[dict]:
        filled.append(gap.query)
        return []

    service._fill_single_gap = fake_fill  # type: ignore[method-assign]

    covered = [{"file_path": "a.py", "content": "def login(): pass", "chunk_id": 1}]
    _, stats = await service.detect_and_fill_gaps(
        root_query="how is auth handled",
        covered_chunks=covered,
        phase1_threshold=0.0,
    )

    assert stats["gaps_found"] == 3
    assert stats["gaps_unified"] == 2  # "auth" pair clusters; "cache" stands alone
    assert all(q.strip() for q in sdk.received)
    assert filled and all(q.strip() for q in filled)
