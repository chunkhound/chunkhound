"""Rust pipeline must report parse-time skips in coordinator stats.

Timeouts already split out of rust_result.errors. Binary / unknown-type /
large-config skips used to land only as DB skip_reason rows, with
skipped_filtered always 0 on the Rust path.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import duckdb

from chunkhound.core.config.config import Config
from chunkhound.registry import configure_registry, create_indexing_coordinator
from chunkhound.services.directory_indexing_service import DirectoryIndexingService
from tests.contracts.pipeline_harness import disconnect_registry_db


def _build_config(root: Path, db_dir: Path) -> Config:
    return Config(
        target_dir=root,
        database={"provider": "duckdb", "path": str(db_dir)},
        indexing={"include": ["**/*"], "exclude": []},
        embeddings_disabled=True,
    )


def test_rust_pipeline_reports_binary_file_as_skipped_filtered(tmp_path, monkeypatch):
    monkeypatch.setenv("CHUNKHOUND_USE_RUST", "1")

    root = tmp_path / "repo"
    root.mkdir()
    (root / "real_code.py").write_text("def real_function():\n    return 42\n")
    (root / "blob.bin").write_bytes(b"\x00binary")

    db_dir = tmp_path / "db"
    config = _build_config(root, db_dir)
    configure_registry(config)
    coordinator = create_indexing_coordinator()
    service = DirectoryIndexingService(indexing_coordinator=coordinator, config=config)
    stats = asyncio.run(service.process_directory(root, no_embeddings=True))
    disconnect_registry_db()

    assert stats.skipped_due_to_timeout == []
    assert stats.skipped_filtered >= 1, (
        "binary blob.bin must count as skipped_filtered on the Rust path, "
        f"got skipped_filtered={stats.skipped_filtered}"
    )

    conn = duckdb.connect(str(db_dir / "chunks.db"), read_only=True)
    try:
        row = conn.execute(
            "SELECT skip_reason FROM files WHERE path = 'blob.bin'"
        ).fetchone()
    finally:
        conn.close()

    assert row is not None, "blob.bin must still get a files row"
    assert row[0] == "binary_file", f"expected skip_reason binary_file, got {row[0]!r}"
