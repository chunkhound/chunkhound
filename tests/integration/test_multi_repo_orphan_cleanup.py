"""Regression test for ALGINF-5452 / GitHub #87.

Bug: re-indexing a sub-directory of a multi-repo workspace deletes all other
repos' data from the database because _cleanup_orphaned_files has no WHERE
clause scoped to the directory being indexed.

Scenario:
  /workspace/
    repo1/src/a.py
    repo2/lib/b.py

  1. Index /workspace  → both files in DB
  2. Re-index /workspace/repo1 → repo2/lib/b.py must NOT be deleted
"""

import os
import subprocess

import pytest
from pathlib import Path

from chunkhound.core.config.indexing_config import IndexingConfig
from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.directory_indexing_service import DirectoryIndexingService
from chunkhound.services.indexing_coordinator import IndexingCoordinator


class _DummyConfig:
    def __init__(self) -> None:
        self.indexing = IndexingConfig()


def _make_coordinator(db: DuckDBProvider, base_dir: Path) -> IndexingCoordinator:
    parser = create_parser_for_language(Language.PYTHON)
    return IndexingCoordinator(
        db,
        base_dir,
        None,
        {Language.PYTHON: parser},
        None,
        None,
    )


def _db_paths(db: DuckDBProvider) -> set[str]:
    rows = db.execute_query("SELECT path FROM files ORDER BY path", [])
    return {r["path"] for r in rows}


@pytest.mark.skipif(
    os.environ.get("CHUNKHOUND_ALLOW_PROCESSPOOL", "0") != "1",
    reason="Requires ProcessPool-friendly environment (SemLock).",
)
@pytest.mark.asyncio
async def test_reindex_subrepo_preserves_other_repos(tmp_path: Path):
    """Re-indexing repo1 must not delete repo2's data from the database."""
    workspace = tmp_path / "workspace"

    repo1 = workspace / "repo1"
    repo2 = workspace / "repo2"
    (repo1 / "src").mkdir(parents=True)
    (repo2 / "lib").mkdir(parents=True)

    file_a = repo1 / "src" / "a.py"
    file_b = repo2 / "lib" / "b.py"
    file_a.write_text("def foo(): pass\n")
    file_b.write_text("def bar(): pass\n")

    db = DuckDBProvider(":memory:", base_directory=workspace)
    db.connect()

    # --- Step 1: index the full workspace ---
    coordinator = _make_coordinator(db, workspace)
    service = DirectoryIndexingService(
        indexing_coordinator=coordinator,
        config=_DummyConfig(),
    )
    await service.process_directory(workspace, no_embeddings=True)

    after_full = _db_paths(db)
    assert "repo1/src/a.py" in after_full, "repo1 file missing after full index"
    assert "repo2/lib/b.py" in after_full, "repo2 file missing after full index"

    # --- Step 2: re-index only repo1 ---
    coordinator2 = _make_coordinator(db, workspace)
    service2 = DirectoryIndexingService(
        indexing_coordinator=coordinator2,
        config=_DummyConfig(),
    )
    await service2.process_directory(repo1, no_embeddings=True)

    after_reindex = _db_paths(db)

    # repo2 data must survive the repo1-only re-index
    assert "repo2/lib/b.py" in after_reindex, (
        "BUG: repo2/lib/b.py was deleted from DB after re-indexing only repo1. "
        "The orphan cleanup is not scoped to the directory being indexed."
    )
    # repo1 data must still be present
    assert "repo1/src/a.py" in after_reindex, "repo1 file missing after re-index"
