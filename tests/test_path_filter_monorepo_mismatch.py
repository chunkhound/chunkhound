"""Regression tests for path_filter behavior with monorepo-style roots.

These tests ensure that when the database base_directory is a higher-level
monorepo root, path_filter values that are repo-relative segments (missing
the leading repository name) still successfully scope semantic search.

Scenario:
    base_directory: /tmp/.../workspace
    stored file path: "orion-suite/services/engine/src/EngineModule.py"

We want:
    search_semantic(query, path_filter="services/engine")
to return results from that file, even though the stored path includes the
leading "orion-suite/" segment.
"""

import pytest
from pathlib import Path

from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.services.search_service import SearchService
from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import create_parser_for_language
from tests.fixtures.fake_providers import FakeEmbeddingProvider


@pytest.mark.asyncio
async def test_semantic_search_respects_repo_relative_path_filter(tmp_path: Path) -> None:
    """Semantic search with repo-relative path_filter should still find results.

    Verifies the fix where DuckDB path filtering uses substring matching so that
    a path_filter like "services/engine" matches stored paths of the form
    "orion-suite/services/engine/...".
    """
    # Monorepo-style layout: workspace/orion-suite/services/engine/src/EngineModule.py
    workspace_dir = tmp_path
    repo_dir = workspace_dir / "orion-suite"
    service_dir = repo_dir / "services" / "engine" / "src"
    service_dir.mkdir(parents=True, exist_ok=True)

    engine_file = service_dir / "EngineModule.py"
    engine_file.write_text(
        "def orion_engine_flag():\n"
        "    \"\"\"Unique function used for path_filter regression tests.\"\"\"\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )

    # Database is rooted at the workspace (one level above the repo),
    # so indexed paths will be stored as "orion-suite/services/engine/src/EngineModule.py"
    db = DuckDBProvider(":memory:", base_directory=workspace_dir)
    db.connect()

    embedding_provider = FakeEmbeddingProvider()
    parser = create_parser_for_language(Language.PYTHON)
    coordinator = IndexingCoordinator(
        db,
        workspace_dir,
        embedding_provider,
        {Language.PYTHON: parser},
    )

    # Index the engine file
    await coordinator.process_file(engine_file)

    # Sanity check: unscoped semantic search should find the function
    search_service = SearchService(db, embedding_provider)
    unscoped_results, _ = await search_service.search_semantic(
        query="orion_engine_flag",
        page_size=10,
        offset=0,
        path_filter=None,
        force_strategy="single_hop",
    )
    assert unscoped_results, "Unscoped semantic search should return results"

    # Now use a repo-relative path_filter that omits the leading 'orion-suite/' prefix
    scoped_results, _ = await search_service.search_semantic(
        query="orion_engine_flag",
        page_size=10,
        offset=0,
        path_filter="services/engine",
        force_strategy="single_hop",
    )

    assert scoped_results, "Scoped semantic search with repo-relative path_filter should return results"
    for result in scoped_results:
        file_path = result.get("file_path", "")
        assert "services/engine" in file_path, f"Result {file_path} should be under services/engine"

