"""process_file (the realtime single-file path) must store the file's content hash.

Without it, every file the live watcher re-indexes is written back with
content_hash = NULL, and the Rust differ's hash-confirm cannot skip that file the
next time its mtime moves without its bytes changing (a checkout, a branch switch,
a copied index) - so the whole file is re-parsed and re-embedded.
"""

import asyncio
from pathlib import Path

import pytest

from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.utils.hashing import compute_file_hash


def _stored_hash(provider: DuckDBProvider, rel_path: str) -> str | None:
    rows = provider.execute_query(
        "SELECT content_hash FROM files WHERE path = ?", [rel_path]
    )
    assert len(rows) == 1
    return rows[0]["content_hash"]


def test_process_file_stores_and_refreshes_content_hash(tmp_path: Path):
    pytest.importorskip("duckdb")

    provider = DuckDBProvider(db_path=tmp_path / "db.duckdb", base_directory=tmp_path)
    provider.connect()
    try:
        coordinator = IndexingCoordinator(provider, tmp_path)
        test_file = tmp_path / "live.py"

        test_file.write_text("def first():\n    return 1\n")
        result = asyncio.run(coordinator.process_file(test_file, skip_embeddings=True))
        assert result["status"] == "success"
        assert _stored_hash(provider, "live.py") == compute_file_hash(test_file)

        # A realtime update of an existing row must replace the hash, not null it.
        test_file.write_text("def second():\n    return 2\n")
        result = asyncio.run(coordinator.process_file(test_file, skip_embeddings=True))
        assert result["status"] == "success"
        assert _stored_hash(provider, "live.py") == compute_file_hash(test_file)
    finally:
        provider.disconnect()


def test_process_file_on_a_vanished_path_reports_instead_of_raising(tmp_path: Path):
    pytest.importorskip("duckdb")

    provider = DuckDBProvider(db_path=tmp_path / "db.duckdb", base_directory=tmp_path)
    provider.connect()
    try:
        coordinator = IndexingCoordinator(provider, tmp_path)
        result = asyncio.run(
            coordinator.process_file(tmp_path / "gone.py", skip_embeddings=True)
        )
        assert result["status"] != "success"
    finally:
        provider.disconnect()
