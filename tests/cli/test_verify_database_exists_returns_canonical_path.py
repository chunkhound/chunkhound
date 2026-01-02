from __future__ import annotations

from pathlib import Path

import pytest

from chunkhound.api.cli.utils.database import verify_database_exists
from chunkhound.core.config.config import Config


def test_verify_database_exists_returns_transformed_duckdb_path(tmp_path: Path) -> None:
    db_dir = tmp_path / ".chunkhound" / "db"
    db_file = db_dir / "chunks.db"
    db_file.parent.mkdir(parents=True, exist_ok=True)
    db_file.write_bytes(b"")

    cfg = Config(target_dir=tmp_path, database={"path": db_dir, "provider": "duckdb"})

    assert verify_database_exists(cfg) == db_file


def test_verify_database_exists_returns_transformed_lancedb_path(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    lancedb_dir = db_dir / "lancedb.lancedb"
    lancedb_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(target_dir=tmp_path, database={"path": db_dir, "provider": "lancedb"})

    assert verify_database_exists(cfg) == lancedb_dir


def test_verify_database_exists_raises_when_missing(tmp_path: Path) -> None:
    cfg = Config(target_dir=tmp_path, database={"path": tmp_path / "missing", "provider": "duckdb"})

    with pytest.raises(FileNotFoundError):
        verify_database_exists(cfg)
