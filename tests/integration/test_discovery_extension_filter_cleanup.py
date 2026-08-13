"""Regression test for the discovery-time extension filter's cleanup gap.

Bug found during design review of the discovery extension-support filter
(`IndexingCoordinator._filter_unsupported_extensions`): filtering discovery
alone is not sufficient. The orphan-cleanup reconciliation phase re-checks
inclusion independently via `RealtimePathFilter.should_index()`, which (before
this fix) had no extension-registry check on the custom-include branch. So a
file indexed once with `index_unknown_files=True` under a custom directory-
wildcard include, then re-indexed with `index_unknown_files=False`, would
keep its stale DB row forever — `_discover_files` correctly stops returning
it, but cleanup never recognized it as excluded. This test guards that path.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from chunkhound.core.config.indexing_config import IndexingConfig
from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.directory_indexing_service import DirectoryIndexingService
from chunkhound.services.indexing_coordinator import IndexingCoordinator


def _make_coordinator(
    db: DuckDBProvider, base_dir: Path, cfg: IndexingConfig
) -> IndexingCoordinator:
    parser = create_parser_for_language(Language.PYTHON)
    return IndexingCoordinator(
        db,
        base_dir,
        None,
        {Language.PYTHON: parser},
        None,
        SimpleNamespace(indexing=cfg),
    )


def _db_paths(db: DuckDBProvider) -> set[str]:
    rows = db.execute_query("SELECT path FROM files ORDER BY path", [])
    return {r["path"] for r in rows}


@pytest.mark.asyncio
async def test_disabling_index_unknown_files_cleans_up_stale_unsupported_rows(
    tmp_path: Path,
):
    project = tmp_path / "project"
    pkg_dir = project / "src" / "pkg"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "module.py").write_text("print('ok')\n")
    (pkg_dir / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    db = DuckDBProvider(":memory:", base_directory=project)
    db.connect()

    # --- Step 1: index with a custom include + index_unknown_files=True ---
    cfg1 = IndexingConfig(
        include=["src/**/*"], index_unknown_files=True, parallel_discovery=False
    )
    coordinator1 = _make_coordinator(db, project, cfg1)
    service1 = DirectoryIndexingService(
        indexing_coordinator=coordinator1, config=SimpleNamespace(indexing=cfg1)
    )
    await service1.process_directory(project, no_embeddings=True)

    after_seed = _db_paths(db)
    assert "src/pkg/module.py" in after_seed
    assert "src/pkg/image.png" in after_seed, (
        "png row should be seeded with index_unknown_files=True"
    )

    # --- Step 2: re-index with index_unknown_files back to the default (False) ---
    cfg2 = IndexingConfig(
        include=["src/**/*"], index_unknown_files=False, parallel_discovery=False
    )
    coordinator2 = _make_coordinator(db, project, cfg2)
    service2 = DirectoryIndexingService(
        indexing_coordinator=coordinator2, config=SimpleNamespace(indexing=cfg2)
    )
    await service2.process_directory(project, no_embeddings=True)

    after_cleanup = _db_paths(db)
    assert "src/pkg/module.py" in after_cleanup
    assert "src/pkg/image.png" not in after_cleanup, (
        "stale unsupported-extension row must be cleaned up once "
        "index_unknown_files is disabled again"
    )
