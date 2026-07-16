"""Regression: empty Lance tables are falsy in LanceDB 0.34+.

``bool(empty_table)`` is False when ``count_rows() == 0``. Provider guards must
use ``is not None`` so schema/index setup still runs on a fresh empty DB.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

pytest.importorskip("lancedb")

from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.providers.database.lancedb_provider import LanceDBProvider


def test_empty_lance_table_is_falsy_but_not_none(tmp_path: Path) -> None:
    """Document LanceDB 0.34+ empty-table truthiness (external contract)."""
    import lancedb

    db = lancedb.connect(str(tmp_path / "raw"))
    empty = db.create_table(
        "empty",
        schema=pa.schema([("id", pa.int64())]),
    )
    assert empty is not None
    assert len(empty) == 0
    assert bool(empty) is False


def test_provider_creates_scalar_index_on_empty_connect(tmp_path: Path) -> None:
    """Fresh connect must create id BTree index even when tables have 0 rows."""
    cfg = DatabaseConfig(provider="lancedb", path=tmp_path)
    provider = LanceDBProvider(
        db_path=tmp_path / "db",
        base_directory=tmp_path,
        config=cfg,
    )
    provider.connect()
    try:
        assert provider._chunks_table is not None
        # Empty table remains falsy; our guards must not use bool(table).
        assert bool(provider._chunks_table) is False
        indices = provider._chunks_table.list_indices()
        has_id = any(
            list(getattr(idx, "columns", None) or []) == ["id"]
            or "id" in list(getattr(idx, "columns", None) or [])
            for idx in indices
        )
        assert has_id, f"expected scalar id index on empty connect, got {indices!r}"
        frags = provider.get_fragment_count()
        assert frags.get("chunks") == 0
        assert frags.get("files") == 0
    finally:
        provider.disconnect()
