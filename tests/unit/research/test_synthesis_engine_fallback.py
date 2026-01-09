"""Synthesis engine fallback behavior when rerank results are invalid."""

import pytest

from chunkhound.interfaces.embedding_provider import RerankResult
from chunkhound.llm_manager import LLMManager
from chunkhound.services.research.synthesis_engine import SynthesisEngine
from tests.fixtures.fake_providers import FakeEmbeddingProvider, FakeLLMProvider


class _OutOfBoundsEmbeddingProvider(FakeEmbeddingProvider):
    async def rerank(self, query: str, documents: list[str], top_k=None):  # noqa: ANN001
        return [RerankResult(index=len(documents) + 5, score=1.0)]


class _FakeEmbeddingManager:
    def __init__(self, provider):
        self._provider = provider

    def get_provider(self):  # noqa: ANN001 - test stub
        return self._provider


class _FakeParent:
    def __init__(self, provider):
        self._embedding_manager = _FakeEmbeddingManager(provider)

    async def _emit_event(self, *args, **kwargs):  # noqa: ANN001 - test stub
        return None


@pytest.fixture()
def llm_manager(monkeypatch):
    fake_provider = FakeLLMProvider()

    def _fake_create_provider(self, config):  # noqa: ANN001 - test stub
        return fake_provider

    monkeypatch.setattr(LLMManager, "_create_provider", _fake_create_provider)
    utility_config = {"provider": "fake", "model": "fake-gpt"}
    synthesis_config = {"provider": "fake", "model": "fake-gpt"}
    return LLMManager(utility_config, synthesis_config)


@pytest.mark.asyncio
async def test_rerank_out_of_bounds_falls_back(llm_manager):
    embedding_provider = _OutOfBoundsEmbeddingProvider()
    parent = _FakeParent(embedding_provider)
    engine = SynthesisEngine(llm_manager, database_services=object(), parent_service=parent)

    chunks = [
        {
            "file_path": "a.py",
            "content": "print('hi')",
            "score": 1.0,
            "start_line": 1,
            "end_line": 1,
        }
    ]
    files = {"a.py": "print('hi')"}
    budgets = {"input_tokens": 1000, "output_tokens": 100}

    _prioritized, budgeted_files, _info = await engine._manage_token_budget_for_synthesis(
        chunks=chunks,
        files=files,
        root_query="test query",
        synthesis_budgets=budgets,
    )

    assert "a.py" in budgeted_files
