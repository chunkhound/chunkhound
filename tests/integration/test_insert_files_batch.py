"""Contract: insert_files_batch returns ids in input order (Lance cold bulk)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("lancedb")

from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.core.models import File
from chunkhound.core.types.common import Language
from chunkhound.providers.database.lancedb_provider import LanceDBProvider


def test_insert_files_batch_ids_match_input_order(tmp_path: Path) -> None:
    cfg = DatabaseConfig(
        path=tmp_path,
        provider="lancedb",
        lancedb_optimize_fragment_threshold=1000,
    )
    db = LanceDBProvider(
        str(cfg.get_db_path()), base_directory=tmp_path, config=cfg
    )
    db.connect()
    try:
        files = [
            File(
                path=f"batch/f_{i:03d}.py",
                mtime=1_700_000_000.0 + i,
                language=Language.PYTHON,
                size_bytes=10 + i,
            )
            for i in range(25)
        ]
        ids = db.insert_files_batch(files)
        assert len(ids) == 25
        assert len(set(ids)) == 25
        for i, fid in enumerate(ids):
            row = db.get_file_by_path(files[i].path)
            assert row is not None
            assert int(row["id"]) == int(fid)
            assert int(row["size"]) == 10 + i
    finally:
        db.disconnect()
