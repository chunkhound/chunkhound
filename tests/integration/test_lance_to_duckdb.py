"""Contract: LanceDB → DuckDB conversion is usable as a DuckDB index."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("lancedb")
pytest.importorskip("duckdb")

import duckdb

from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.core.models import Chunk, File
from chunkhound.core.types.common import ChunkType, Language
from chunkhound.providers.database.lancedb_provider import LanceDBProvider
from chunkhound.utils.lance_to_duckdb import (
    activate_duckdb_in_config,
    convert_lancedb_to_duckdb,
)


def _seed_lance(tmp: Path) -> Path:
    cfg = DatabaseConfig(
        path=tmp / "lance_cfg",
        provider="lancedb",
        lancedb_optimize_fragment_threshold=10_000,
    )
    lance_path = cfg.get_db_path()
    db = LanceDBProvider(str(lance_path), base_directory=tmp, config=cfg)
    db.connect()
    try:
        fid = int(
            db.insert_file(
                File(
                    path="src/hello.py",
                    mtime=1_700_000_000.0,
                    language=Language.PYTHON,
                    size_bytes=32,
                    content_hash="abc",
                )
            )
        )
        chunks = [
            Chunk(
                file_id=fid,
                code="def hello():\n    return 1\n",
                start_line=1,
                end_line=2,
                chunk_type=ChunkType.FUNCTION,
                language=Language.PYTHON,
                symbol="hello",
            )
        ]
        vec = [0.1] * 32
        ids = db.insert_chunks_with_embeddings_batch(
            chunks, [vec], "fake", "fake-embeddings"
        )
        assert len(ids) == 1
    finally:
        db.disconnect()
    return Path(lance_path)


def test_convert_lancedb_to_duckdb_roundtrip(tmp_path: Path) -> None:
    lance_dir = _seed_lance(tmp_path)
    dest = tmp_path / "duck_out" / "chunks.db"
    stats = convert_lancedb_to_duckdb(
        lance_dir,
        dest,
        overwrite=True,
        batch_size=1,
        compact="never",
        progress=None,
    )
    assert stats.files == 1
    assert stats.chunks == 1
    assert stats.embeddings == 1
    assert 32 in stats.embedding_dims
    assert dest.is_file()
    assert stats.stream_batch_size == 1
    assert stats.compacted is False

    # Verify with raw DuckDB (avoids provider disconnect/HNSW edge cases in CI)
    conn = duckdb.connect(str(dest), read_only=True)
    try:
        n_files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert n_files == 1
        assert n_chunks == 1
        path, code = conn.execute(
            "SELECT f.path, c.code FROM files f JOIN chunks c ON c.file_id = f.id"
        ).fetchone()
        assert path == "src/hello.py"
        assert "def hello" in code

        emb_row = conn.execute(
            "SELECT chunk_id, provider, model, len(embedding), dims FROM embeddings_32"
        ).fetchone()
        assert emb_row is not None
        chunk_id, provider, model, emb_len, dims = emb_row
        assert provider == "fake"
        assert model == "fake-embeddings"
        assert emb_len == 32
        assert dims == 32
        # Remapped dense ids start at 1
        assert int(chunk_id) == 1
        file_id = conn.execute("SELECT id FROM files").fetchone()[0]
        assert int(file_id) == 1
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE file_id = ?", [file_id]
            ).fetchone()[0]
            == 1
        )
    finally:
        conn.close()


def test_convert_progress_callback_emits_phases(tmp_path: Path) -> None:
    """Progress callback receives phase markers and scanned/total lines."""
    lance_dir = _seed_lance(tmp_path)
    dest = tmp_path / "duck_progress" / "chunks.db"
    lines: list[str] = []

    stats = convert_lancedb_to_duckdb(
        lance_dir,
        dest,
        overwrite=True,
        batch_size=1,
        compact="never",
        progress=lines.append,
    )
    assert stats.chunks == 1
    joined = "\n".join(lines)
    assert "=== Phase 0: initialize ===" in joined
    assert "counting Lance rows" in joined
    assert "=== Phase 2: files ===" in joined
    assert "=== Phase 3: chunks" in joined
    assert "scanned " in joined
    assert "duck=" in joined
    assert "=== done ===" in joined
    # Phase 0 must announce before inventory work lines complete.
    assert lines[0].startswith("=== Phase 0:")


def test_convert_progress_none_and_default_silent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Default and progress=None leave stdout/stderr free of phase noise (--json)."""
    lance_dir = _seed_lance(tmp_path)
    dest = tmp_path / "duck_silent" / "chunks.db"

    convert_lancedb_to_duckdb(
        lance_dir,
        dest,
        overwrite=True,
        compact="never",
        progress=None,
    )
    captured = capsys.readouterr()
    assert "Phase" not in captured.out
    assert "Phase" not in captured.err

    convert_lancedb_to_duckdb(
        lance_dir,
        dest,
        overwrite=True,
        compact="never",
    )
    captured = capsys.readouterr()
    assert "Phase" not in captured.out
    assert "Phase" not in captured.err


def test_activate_duckdb_in_config_writes_provider(tmp_path: Path) -> None:
    cfg_path = activate_duckdb_in_config(
        tmp_path, database_path=".chunkhound", require_db_exists=False
    )
    assert cfg_path.is_file()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert data["database"]["provider"] == "duckdb"
    assert data["database"]["path"] == ".chunkhound"


def test_activate_refuses_corrupt_json(tmp_path: Path) -> None:
    bad = tmp_path / ".chunkhound.json"
    bad.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        activate_duckdb_in_config(tmp_path, database_path=".chunkhound")


def test_activate_preserves_other_config_keys(tmp_path: Path) -> None:
    cfg = tmp_path / ".chunkhound.json"
    cfg.write_text(
        json.dumps(
            {
                "embedding": {"provider": "openai", "model": "x"},
                "database": {"provider": "lancedb", "path": ".chunkhound"},
            }
        ),
        encoding="utf-8",
    )
    activate_duckdb_in_config(tmp_path, database_path=".chunkhound")
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data["embedding"]["model"] == "x"
    assert data["database"]["provider"] == "duckdb"
