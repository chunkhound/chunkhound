"""Tests for scoped deep research via path_filter.

These tests verify that:
1. DeepResearchService respects path_filter when performing unified search.
2. Scoped research only returns chunks from the requested path prefix.
3. Unscoped research can see chunks from multiple prefixes.
"""

import asyncio
from pathlib import Path

import pytest

from chunkhound.database_factory import DatabaseServices
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.mcp_server.tools import deep_research_impl
from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.embedding_service import EmbeddingService
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.services.search_service import SearchService
from tests.fixtures.fake_providers import FakeEmbeddingProvider, FakeLLMProvider


@pytest.mark.asyncio
async def test_scoped_deep_research_uses_path_filter(tmp_path: Path) -> None:
    """Scoped deep research should only return chunks from the requested path."""

    # Create synthetic workspace with two "repos"
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    repo_a.mkdir(parents=True, exist_ok=True)
    repo_b.mkdir(parents=True, exist_ok=True)

    a_file = repo_a / "a.py"
    b_file = repo_b / "b.py"

    a_file.write_text(
        "def alpha():\n"
        "    '''Function in repo-a.'''\n"
        "    return 'A'\n",
        encoding="utf-8",
    )
    b_file.write_text(
        "def beta():\n"
        "    '''Function in repo-b.'''\n"
        "    return 'B'\n",
        encoding="utf-8",
    )

    # Fake embedding + LLM providers to avoid network calls
    embedding_manager = EmbeddingManager()
    embedding_manager.register_provider(FakeEmbeddingProvider(), set_default=True)

    # Monkeypatch LLMManager to use FakeLLMProvider via factory hook
    def _fake_create_provider(self, provider_config):
        return FakeLLMProvider()

    # Patch at class level so both utility and synthesis providers use fakes
    original_create_provider = LLMManager._create_provider
    LLMManager._create_provider = _fake_create_provider  # type: ignore[assignment]
    try:
        llm_manager = LLMManager(
            {"provider": "fake", "model": "fake-gpt"},
            {"provider": "fake", "model": "fake-gpt"},
        )
    finally:
        # Restore original factory to avoid cross-test leakage
        LLMManager._create_provider = original_create_provider  # type: ignore[assignment]

    # Create in-memory DuckDB provider scoped to tmp_path
    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()

    # Create minimal services bundle matching DatabaseServices contract
    parser = create_parser_for_language(Language.PYTHON)
    coordinator = IndexingCoordinator(
        db,
        tmp_path,
        embedding_manager.get_default_provider(),
        {Language.PYTHON: parser},
    )
    search_service = SearchService(db, embedding_manager.get_default_provider())
    embedding_service = EmbeddingService(
        db,
        embedding_manager.get_default_provider(),
    )
    services = DatabaseServices(
        provider=db,
        indexing_coordinator=coordinator,
        search_service=search_service,
        embedding_service=embedding_service,
    )

    # Index both repos by running indexing coordinator directly
    await coordinator.process_file(a_file)
    await coordinator.process_file(b_file)

    stats = services.provider.get_stats()
    assert stats["chunks"] > 0, "Expected chunks after indexing test files"

    # Helper to run deep research with optional scope
    async def run_research(scope: str | None) -> dict:
        # Lower relevance threshold to ensure our tiny test corpus returns results
        import chunkhound.services.deep_research_service as dr_mod

        original_threshold = dr_mod.RELEVANCE_THRESHOLD
        dr_mod.RELEVANCE_THRESHOLD = None  # Disable similarity cutoff for tests
        try:
            return await deep_research_impl(
                services=services,
                embedding_manager=embedding_manager,
                llm_manager=llm_manager,
                query="alpha or beta functions",
                progress=None,
                path=scope,
            )
        finally:
            dr_mod.RELEVANCE_THRESHOLD = original_threshold

    # 1) Unscoped research should be able to see both repos in metadata
    unscoped = await run_research(scope=None)
    assert "metadata" in unscoped
    unscoped_chunks = unscoped["metadata"].get("chunks_analyzed", 0)
    assert unscoped_chunks > 0

    # 2) Scoped research for repo-a should only surface repo-a files
    scoped = await run_research(scope=str(repo_a.relative_to(tmp_path)))
    assert "answer" in scoped
    answer_text = scoped["answer"]

    # The sources footer should reference repo-a but not repo-b
    assert "repo-a/" in answer_text
    assert "a.py" in answer_text
    assert "repo-b/" not in answer_text
    assert "b.py" not in answer_text

    # 3) Sanity: when scope is repo-b, the roles reverse
    scoped_b = await run_research(scope=str(repo_b.relative_to(tmp_path)))
    answer_b = scoped_b["answer"]
    assert "repo-b/" in answer_b
    assert "b.py" in answer_b
    assert "repo-a/" not in answer_b
    assert "a.py" not in answer_b


@pytest.mark.asyncio
async def test_deep_research_propagates_path_filter_to_search_service(
    tmp_path: Path,
) -> None:
    """DeepResearchService must pass path_filter to SearchService calls."""

    # Create synthetic workspace with two repos (same layout as previous test)
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    repo_a.mkdir(parents=True, exist_ok=True)
    repo_b.mkdir(parents=True, exist_ok=True)

    a_file = repo_a / "a.py"
    b_file = repo_b / "b.py"

    a_file.write_text(
        "def alpha():\n"
        "    '''Function in repo-a for path_filter test.'''\n"
        "    return 'A'\n",
        encoding="utf-8",
    )
    b_file.write_text(
        "def beta():\n"
        "    '''Function in repo-b for path_filter test.'''\n"
        "    return 'B'\n",
        encoding="utf-8",
    )

    # Fake embedding + LLM providers
    embedding_manager = EmbeddingManager()
    embedding_manager.register_provider(FakeEmbeddingProvider(), set_default=True)

    def _fake_create_provider(self, provider_config):
        return FakeLLMProvider()

    original_create_provider = LLMManager._create_provider
    LLMManager._create_provider = _fake_create_provider  # type: ignore[assignment]
    try:
        llm_manager = LLMManager(
            {"provider": "fake", "model": "fake-gpt"},
            {"provider": "fake", "model": "fake-gpt"},
        )
    finally:
        LLMManager._create_provider = original_create_provider  # type: ignore[assignment]

    # In-memory DuckDB provider scoped to tmp_path
    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()

    parser = create_parser_for_language(Language.PYTHON)
    coordinator = IndexingCoordinator(
        db,
        tmp_path,
        embedding_manager.get_default_provider(),
        {Language.PYTHON: parser},
    )
    search_service = SearchService(db, embedding_manager.get_default_provider())
    embedding_service = EmbeddingService(
        db,
        embedding_manager.get_default_provider(),
    )
    services = DatabaseServices(
        provider=db,
        indexing_coordinator=coordinator,
        search_service=search_service,
        embedding_service=embedding_service,
    )

    # Index both files
    await coordinator.process_file(a_file)
    await coordinator.process_file(b_file)

    # Instrument SearchService to capture path_filter usage
    semantic_path_filters: list[str | None] = []
    regex_path_filters: list[str | None] = []

    original_search_semantic = search_service.search_semantic
    original_search_regex_async = search_service.search_regex_async

    async def wrapped_search_semantic(*args, **kwargs):
        semantic_path_filters.append(kwargs.get("path_filter"))
        return await original_search_semantic(*args, **kwargs)

    async def wrapped_search_regex_async(*args, **kwargs):
        regex_path_filters.append(kwargs.get("path_filter"))
        return await original_search_regex_async(*args, **kwargs)

    search_service.search_semantic = wrapped_search_semantic  # type: ignore[assignment]
    search_service.search_regex_async = wrapped_search_regex_async  # type: ignore[assignment]

    # Run deep research with an explicit scope and relaxed threshold
    import chunkhound.services.deep_research_service as dr_mod

    original_threshold = dr_mod.RELEVANCE_THRESHOLD
    dr_mod.RELEVANCE_THRESHOLD = None
    try:
        await deep_research_impl(
            services=services,
            embedding_manager=embedding_manager,
            llm_manager=llm_manager,
            query="alpha or beta functions",
            progress=None,
            path="repo-a",
        )
    finally:
        dr_mod.RELEVANCE_THRESHOLD = original_threshold

    # At least one semantic search must have been performed with the scoped path
    assert semantic_path_filters, "Expected search_semantic to be called at least once"
    assert all(
        pf == "repo-a" for pf in semantic_path_filters
    ), f"All semantic searches should use path_filter='repo-a', got {semantic_path_filters}"

    # Regex stage is optional; if it runs, it must also respect the same scope
    if regex_path_filters:
        assert all(
            pf == "repo-a" for pf in regex_path_filters
        ), f"All regex searches should use path_filter='repo-a', got {regex_path_filters}"
