"""CLI and MCP research report configuration failures as ValueError."""

import pytest

from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.mcp_server.tools import deep_research_impl
from chunkhound.services.deep_research_service import run_deep_research
from tests.fixtures.fake_providers import FakeEmbeddingProvider, FakeLLMProvider


class OfflineLLMProvider(FakeLLMProvider):
    def __init__(self, **kwargs):
        super().__init__(model=kwargs.get("model", "offline"))


class EmbeddingsWithoutReranking(FakeEmbeddingProvider):
    def supports_reranking(self) -> bool:
        return False


@pytest.mark.parametrize("entry_point", [deep_research_impl, run_deep_research])
@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("llm", "LLM not configured"),
        ("embeddings", "No embedding providers available"),
        ("reranker", "requires a provider with reranking support"),
    ],
)
async def test_research_rejects_missing_capabilities_as_validation_errors(
    monkeypatch, entry_point, missing, message
):
    monkeypatch.setitem(LLMManager._providers, "offline", OfflineLLMProvider)
    llm = LLMManager({"provider": "offline"}, {"provider": "offline"})
    embeddings = EmbeddingManager()
    if missing != "embeddings":
        provider = (
            EmbeddingsWithoutReranking()
            if missing == "reranker"
            else FakeEmbeddingProvider()
        )
        embeddings.register_provider(provider, set_default=True)

    with pytest.raises(ValueError, match=message):
        await entry_point(
            services=None,
            embedding_manager=embeddings,
            llm_manager=None if missing == "llm" else llm,
            query="explain the project",
        )
