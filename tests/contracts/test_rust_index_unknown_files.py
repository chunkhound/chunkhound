"""Rust pipeline splits unrecognized files when index_unknown_files is on.

The Python batch processor already locks this in tests/test_index_unknown_files.py.
This contract checks the same split through the Rust indexing path.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import duckdb

from chunkhound.core.config.config import Config
from chunkhound.registry import configure_registry, create_indexing_coordinator
from chunkhound.services.directory_indexing_service import DirectoryIndexingService
from tests.contracts.pipeline_harness import disconnect_registry_db


def test_rust_pipeline_splits_unknown_files_when_flag_on(tmp_path, monkeypatch):
    monkeypatch.setenv("CHUNKHOUND_USE_RUST", "1")

    root = tmp_path / "repo"
    root.mkdir()
    text_file = root / "notes.xyzunknown"
    text_file.write_text(
        "\n".join(f"line {i}: some content here" for i in range(20)) + "\n"
    )
    (root / "model.xyzunknown").write_bytes(b"\x00\x01\x02" * 1000)

    db_dir = tmp_path / "db"
    config = Config(
        target_dir=root,
        database={"provider": "duckdb", "path": str(db_dir)},
        indexing={
            "include": ["**/*"],
            "exclude": [],
            "index_unknown_files": True,
        },
        embeddings_disabled=True,
    )
    configure_registry(config)
    coordinator = create_indexing_coordinator()
    service = DirectoryIndexingService(indexing_coordinator=coordinator, config=config)
    asyncio.run(service.process_directory(root, no_embeddings=True))
    disconnect_registry_db()

    conn = duckdb.connect(str(db_dir / "chunks.db"), read_only=True)
    try:
        rows = {
            path: (skip_reason, chunks)
            for path, skip_reason, chunks in conn.execute(
                """
                SELECT f.path, f.skip_reason, COUNT(c.id)
                FROM files f
                LEFT JOIN chunks c ON c.file_id = f.id
                GROUP BY f.path, f.skip_reason
                """
            ).fetchall()
        }
    finally:
        conn.close()

    assert rows["notes.xyzunknown"][0] is None
    assert rows["notes.xyzunknown"][1] > 0, (
        "a NUL-free unrecognized file must be stored as chunks when the flag is on"
    )
    assert rows["model.xyzunknown"] == ("binary_file", 0)
