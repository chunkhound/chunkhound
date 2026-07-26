"""Synthesis engine fallback behavior when rerank results are invalid."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from chunkhound.interfaces.embedding_provider import RerankResult
from chunkhound.llm_manager import LLMManager
from chunkhound.services.clustering_service import ClusterGroup, ClusteringService
from chunkhound.services.research import SynthesisEngine
from chunkhound.services.research.shared.citation_manager import CitationManager
from chunkhound.services.research.v1.pluggable_research_service import (
    PluggableResearchService,
)
from tests.fixtures.fake_providers import FakeEmbeddingProvider, FakeLLMProvider


class _OutOfBoundsEmbeddingProvider(FakeEmbeddingProvider):
    async def rerank(self, query: str, documents: list[str], top_k=None):  # noqa: ANN001
        return [RerankResult(index=len(documents) + 5, score=1.0)]


class _CapturingEmbeddingProvider(FakeEmbeddingProvider):
    def __init__(self):
        super().__init__()
        self.texts: list[str] = []

    async def embed_batch(self, texts: list[str], batch_size: int | None = None):
        self.texts.extend(texts)
        return await super().embed_batch(texts, batch_size)


class _FakeEmbeddingManager:
    def __init__(self, provider):
        self._provider = provider

    def get_provider(self):  # noqa: ANN001 - test stub
        return self._provider


class _FakeParent:
    def __init__(self, provider):
        self._embedding_manager = _FakeEmbeddingManager(provider)
        self._citation_manager = CitationManager()

    async def _emit_event(self, *args, **kwargs):  # noqa: ANN001 - test stub
        return None


class _CapturingFakeLLMProvider(FakeLLMProvider):
    def __init__(self):
        super().__init__()
        self.calls: list[dict[str, object]] = []

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "system": system,
                "max_completion_tokens": max_completion_tokens,
                "timeout": timeout,
            }
        )
        return await super().complete(
            prompt,
            system=system,
            max_completion_tokens=max_completion_tokens,
            timeout=timeout,
        )


@pytest.fixture()
def llm_manager(monkeypatch):
    fake_provider = FakeLLMProvider()

    def _fake_create_provider(self, config):  # noqa: ANN001 - test stub
        return fake_provider

    monkeypatch.setattr(LLMManager, "_create_provider", _fake_create_provider)
    utility_config = {"provider": "fake", "model": "fake-gpt"}
    synthesis_config = {"provider": "fake", "model": "fake-gpt"}
    return LLMManager(utility_config, synthesis_config)


@pytest.fixture()
def capturing_llm_manager(monkeypatch):
    fake_provider = _CapturingFakeLLMProvider()

    def _fake_create_provider(self, config):  # noqa: ANN001 - test stub
        return fake_provider

    monkeypatch.setattr(LLMManager, "_create_provider", _fake_create_provider)
    utility_config = {"provider": "fake", "model": "fake-gpt"}
    synthesis_config = {"provider": "fake", "model": "fake-gpt"}
    return LLMManager(utility_config, synthesis_config), fake_provider


@pytest.mark.asyncio
async def test_rerank_out_of_bounds_falls_back(llm_manager):
    embedding_provider = _OutOfBoundsEmbeddingProvider()
    parent = _FakeParent(embedding_provider)
    engine = SynthesisEngine(
        llm_manager, database_services=object(), parent_service=parent
    )

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

    (
        _prioritized,
        budgeted_files,
        _info,
    ) = await engine._manage_token_budget_for_synthesis(
        chunks=chunks,
        files=files,
        root_query="test query",
        synthesis_budgets=budgets,
    )

    assert "a.py" in budgeted_files


@pytest.mark.asyncio
async def test_map_synthesis_uses_output_budget_for_cluster_allocation(
    capturing_llm_manager,
):
    llm_manager, fake_provider = capturing_llm_manager
    parent = _FakeParent(FakeEmbeddingProvider())
    engine = SynthesisEngine(
        llm_manager, database_services=object(), parent_service=parent
    )

    cluster = ClusterGroup(
        cluster_id=0,
        file_paths=["a.py"],
        files_content={"a.py": "print('hi')"},
        total_tokens=20_000,
    )
    chunks = [
        {
            "file_path": "a.py",
            "content": "print('hi')",
            "start_line": 1,
            "end_line": 1,
        }
    ]

    await engine._map_synthesis_on_cluster(
        cluster=cluster,
        root_query="synthesis test",
        chunks=chunks,
        synthesis_budgets={"output_tokens": 30_000},
        total_input_tokens=100_000,
    )

    assert len(fake_provider.calls) == 1
    call = fake_provider.calls[0]
    assert call["max_completion_tokens"] == 6000
    assert "Target output: ~6,000 tokens" in call["system"]


_WIDGET_SOURCE = "@implementation Widget\n@end"
_WIDGET_FILE = {"Sources/Widget.m": _WIDGET_SOURCE}
_WIDGET_FILES = {**_WIDGET_FILE, "Sources/Helper.py": "def helper(): pass"}
_WIDGET_CHUNKS = [
    {
        "file_path": "Sources/Widget.m",
        "content": _WIDGET_SOURCE,
        "start_line": 1,
        "end_line": 2,
    }
]
_WIDGET_LANGUAGE = {"Sources/Widget.m": "objc"}
_WIDGET_LANGUAGES = {**_WIDGET_LANGUAGE, "Sources/Helper.py": "python"}


def _source_header(file_path: str, language: str, content: str) -> str:
    return f"### [{language}] {file_path}\n{'=' * 80}\n{content}\n{'=' * 80}"


async def _cluster_widget_source():
    clusterer = ClusteringService(FakeEmbeddingProvider(), FakeLLMProvider())
    clusters, _ = await clusterer.cluster_files_hdbscan_bounded(
        _WIDGET_FILES,
        min_tokens_per_cluster=1,
        max_tokens_per_cluster=50_000,
        file_languages=_WIDGET_LANGUAGES,
    )
    return next(
        cluster for cluster in clusters if "Sources/Widget.m" in cluster.file_paths
    )


@pytest.mark.asyncio
async def test_map_synthesis_uses_indexed_language_over_extension_fallback(
    capturing_llm_manager,
):
    llm_manager, fake_provider = capturing_llm_manager
    engine = SynthesisEngine(
        llm_manager, object(), _FakeParent(FakeEmbeddingProvider())
    )
    cluster = await _cluster_widget_source()

    await engine._map_synthesis_on_cluster(
        cluster,
        "widget",
        _WIDGET_CHUNKS,
        {"output_tokens": 30_000},
        100_000,
    )

    assert len(fake_provider.calls) == 1
    prompt = fake_provider.calls[0]["prompt"]
    assert (
        _source_header("Sources/Widget.m", "objc", f"# Lines 1-2\n{_WIDGET_SOURCE}")
        in prompt
    )
    assert "### [matlab] Sources/Widget.m" not in prompt


@pytest.mark.asyncio
async def test_research_flow_preserves_indexed_language_in_embeddings_and_prompt(
    capturing_llm_manager, monkeypatch
):
    llm_manager, fake_provider = capturing_llm_manager
    embedding_provider = _CapturingEmbeddingProvider()
    embedding_manager = _FakeEmbeddingManager(embedding_provider)
    exploration_strategy = MagicMock()
    exploration_strategy.name = "test"
    source = "@implementation Widget\n@end"
    chunks = [
        {
            "chunk_id": 1,
            "file_path": "Sources/Widget.m",
            "content": source,
            "language": "objc",
            "start_line": 1,
            "end_line": 2,
        },
        {
            "chunk_id": 2,
            "file_path": "Sources/Helper.py",
            "content": "def helper(): pass",
            "language": "python",
            "start_line": 1,
            "end_line": 1,
        },
    ]
    files = {chunk["file_path"]: chunk["content"] for chunk in chunks}
    exploration_strategy.explore = AsyncMock(return_value=(chunks, {}, files))
    service = PluggableResearchService(
        database_services=MagicMock(),
        embedding_manager=embedding_manager,
        llm_manager=llm_manager,
        exploration_strategy=exploration_strategy,
    )
    monkeypatch.setattr(service, "_unified_search", AsyncMock(return_value=chunks))
    monkeypatch.setattr(
        service._synthesis_engine,
        "_manage_token_budget_for_synthesis",
        AsyncMock(
            return_value=(
                chunks,
                files,
                {"files_selected": 2, "total_tokens": 10, "chunks_count": 2},
            )
        ),
    )

    await service.deep_research("How does Widget work?")

    assert (
        "# Sources/Widget.m (objc)\n@implementation Widget\n@end"
        in embedding_provider.texts
    )
    prompts = [call["prompt"] for call in fake_provider.calls]
    assert any("### [objc] Sources/Widget.m" in prompt for prompt in prompts)
    assert not any("### [matlab] Sources/Widget.m" in prompt for prompt in prompts)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("file_languages", "expected_language"),
    [
        pytest.param(None, "matlab", id="extension-fallback"),
        pytest.param(_WIDGET_LANGUAGE, "objc", id="indexed-language"),
    ],
)
async def test_single_pass_synthesis_resolves_source_language(
    capturing_llm_manager,
    file_languages,
    expected_language,
):
    llm_manager, fake_provider = capturing_llm_manager
    engine = SynthesisEngine(
        llm_manager, object(), _FakeParent(FakeEmbeddingProvider())
    )
    await engine._single_pass_synthesis(
        "widget",
        _WIDGET_CHUNKS,
        _WIDGET_FILE,
        None,
        {"input_tokens": 50_000, "output_tokens": 5_000},
        file_languages=file_languages,
    )

    assert len(fake_provider.calls) == 1
    prompt = fake_provider.calls[0]["prompt"]
    assert (
        _source_header(
            "Sources/Widget.m", expected_language, f"# Lines 1-2\n{_WIDGET_SOURCE}"
        )
        in prompt
    )
    if file_languages:
        assert "### [matlab] Sources/Widget.m" not in prompt
