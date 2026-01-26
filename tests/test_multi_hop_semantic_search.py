"""
Test multi-hop semantic search with reranking functionality.

These tests verify that:
1. Providers with reranking support trigger multi-hop search (because supports_reranking() = True)
2. Path filters are respected at both the search service and database layers
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.search_service import SearchService
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import create_parser_for_language
from tests.fixtures.fake_providers import FakeEmbeddingProvider


@pytest.fixture
async def simple_test_database(tmp_path):
    """Create a simple test database for mock-based tests."""
    # Use file-based database (not :memory:) because semantic search requires ShardManager
    db_path = tmp_path / "test.db"
    db = DuckDBProvider(str(db_path), base_directory=tmp_path)
    db.connect()
    yield db


@pytest.mark.asyncio
async def test_search_strategy_selection_verification(simple_test_database):
    """Verify that SearchService correctly selects search strategy based on provider capabilities."""
    db = simple_test_database

    # Mock providers to test strategy selection
    reranking_provider = Mock()
    reranking_provider.supports_reranking.return_value = True
    reranking_provider.name = "mock_voyage"
    reranking_provider.model = "mock-model"

    non_reranking_provider = Mock()
    non_reranking_provider.supports_reranking.return_value = False
    non_reranking_provider.name = "mock_openai"
    non_reranking_provider.model = "mock-model"

    # Create search services
    voyage_search = SearchService(db, reranking_provider)
    openai_search = SearchService(db, non_reranking_provider)

    query = "user authentication"

    # Test strategy selection by mocking the internal methods
    with patch.object(voyage_search._multi_hop_strategy, 'search', return_value=([], {})) as mock_multi_hop:
        with patch.object(openai_search._single_hop_strategy, 'search', return_value=([], {})) as mock_standard:

            # VoyageAI provider should trigger multi-hop search
            await voyage_search.search_semantic(query, page_size=5)
            mock_multi_hop.assert_called_once_with(
                query=query,
                page_size=5,
                offset=0,
                threshold=None,
                provider="mock_voyage",
                model="mock-model",
                path_filter=None,
                time_limit=None,
                result_limit=None,
            )

            # OpenAI provider should use standard search
            await openai_search.search_semantic(query, page_size=5)
            mock_standard.assert_called_once_with(
                query=query,
                page_size=5,
                offset=0,
                threshold=None,
                provider="mock_openai",
                model="mock-model",
                path_filter=None,
            )


@pytest.mark.asyncio
async def test_multi_hop_respects_path_filter_scope(tmp_path):
    """Semantic search with path_filter should respect scope boundaries.

    Note: FakeEmbeddingProvider creates deterministic but semantically meaningless
    embeddings, so query-based semantic search won't find matches. This test uses
    chunk-based similarity (find_similar_chunks) which works with any embedding
    provider by comparing actual indexed embeddings.
    """
    base_dir = tmp_path
    db_path = tmp_path / "test.db"
    db = DuckDBProvider(str(db_path), base_directory=base_dir)
    db.connect()

    # Use deterministic fake embedding provider with reranking support
    embedding_provider = FakeEmbeddingProvider()

    parser = create_parser_for_language(Language.PYTHON)
    coordinator = IndexingCoordinator(
        db, base_dir, embedding_provider, {Language.PYTHON: parser}
    )

    # Create two synthetic "repos" under the same base directory
    # Each repo has multiple files (not functions in one file) to ensure
    # separate chunks are created - cAST merges small adjacent functions.
    repos = ["repo_a", "repo_b"]
    for repo in repos:
        repo_dir = base_dir / repo
        repo_dir.mkdir(parents=True, exist_ok=True)

        # Create separate files per repo to ensure distinct chunks
        for idx in range(3):
            content = f"""
def shared_function_{repo}_{idx}():
    \"\"\"Shared multi-hop scope test in {repo} module {idx}.\"\"\"
    value = "shared-{repo}-{idx}"
    return value
"""
            file_path = repo_dir / f"module_{idx}.py"
            file_path.write_text(content)
            await coordinator.process_file(file_path)

    # Ensure USearch index is synchronized with DuckDB after all files indexed
    if db.shard_manager:
        db.run_fix_pass(check_quality=False)

    # Verify chunks exist in both repos using regex search
    regex_results, _ = db.search_regex(pattern="shared_function", page_size=50)
    assert regex_results, "Expected chunks from indexing"

    repo_a_chunks = [r for r in regex_results if r.get("file_path", "").startswith("repo_a/")]
    repo_b_chunks = [r for r in regex_results if r.get("file_path", "").startswith("repo_b/")]
    assert repo_a_chunks, "Expected chunks from repo_a"
    assert repo_b_chunks, "Expected chunks from repo_b"

    # Get a chunk from repo_a to use as similarity anchor
    repo_a_chunk = repo_a_chunks[0]
    chunk_id = repo_a_chunk["chunk_id"]

    # Without path_filter, similar chunks should include both repos
    neighbors_unscoped = db.find_similar_chunks(
        chunk_id=chunk_id,
        provider=embedding_provider.name,
        model=embedding_provider.model,
        limit=20,
        threshold=None,
    )
    assert neighbors_unscoped, "Expected unscoped neighbors"
    assert any(
        n.get("file_path", "").startswith("repo_b/") for n in neighbors_unscoped
    ), "Unscoped neighbors should include repo_b results"

    # With path_filter='repo_a', all results must be scoped to repo_a
    neighbors_scoped = db.find_similar_chunks(
        chunk_id=chunk_id,
        provider=embedding_provider.name,
        model=embedding_provider.model,
        limit=20,
        threshold=None,
        path_filter="repo_a",
    )
    assert neighbors_scoped, "Expected scoped neighbors"
    for result in neighbors_scoped:
        file_path = result.get("file_path", "")
        assert file_path.startswith(
            "repo_a/"
        ), f"Result {file_path} should be constrained to repo_a/"


@pytest.mark.asyncio
async def test_find_similar_chunks_enforces_path_filter(tmp_path):
    """find_similar_chunks should enforce path_filter at the database layer."""
    base_dir = tmp_path
    db_path = tmp_path / "test.db"
    db = DuckDBProvider(str(db_path), base_directory=base_dir)
    db.connect()

    embedding_provider = FakeEmbeddingProvider()
    parser = create_parser_for_language(Language.PYTHON)
    coordinator = IndexingCoordinator(
        db, base_dir, embedding_provider, {Language.PYTHON: parser}
    )

    # Create synthetic repos with very similar content and multiple files per repo
    for repo in ["repo_a", "repo_b"]:
        repo_dir = base_dir / repo
        repo_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(2):
            content = f"""
def repo_function_{idx}():
    \"\"\"Repository-specific function {idx} for {repo}.\"\"\"
    return \"{repo}-value-{idx}\"
"""
            file_path = repo_dir / f"module_{idx}.py"
            file_path.write_text(content)
            await coordinator.process_file(file_path)

    # Use regex search to get a chunk from repo_a
    regex_results, _ = db.search_regex(pattern="Repository-specific function", page_size=50)
    assert regex_results, "Expected at least one chunk from regex search"

    repo_a_chunk = next(
        (r for r in regex_results if r.get("file_path", "").startswith("repo_a/")), None
    )
    assert repo_a_chunk is not None, "Expected a chunk from repo_a"

    chunk_id = repo_a_chunk["chunk_id"]

    # Without path_filter, similar chunks should include both repos
    neighbors_unscoped = db.find_similar_chunks(
        chunk_id=chunk_id,
        provider=embedding_provider.name,
        model=embedding_provider.model,
        limit=20,
        threshold=None,
    )
    assert neighbors_unscoped, "Expected unscoped neighbors for similarity search"
    assert any(
        n.get("file_path", "").startswith("repo_b/") for n in neighbors_unscoped
    ), "Unscoped neighbors should include repo_b results"

    # With path_filter='repo_a', all neighbors must stay within repo_a
    neighbors_scoped = db.find_similar_chunks(
        chunk_id=chunk_id,
        provider=embedding_provider.name,
        model=embedding_provider.model,
        limit=20,
        threshold=None,
        path_filter="repo_a",
    )
    assert neighbors_scoped, "Expected scoped neighbors for similarity search"
    assert all(
        n.get("file_path", "").startswith("repo_a/") for n in neighbors_scoped
    ), "Scoped neighbors must all be within repo_a/"
