#!/usr/bin/env python3
"""Contract tests for `chunkhound snapshot` multi-scope selection (v1)."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from chunkhound.core.types.common import ChunkType
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from tests.utils.windows_subprocess import get_safe_subprocess_env


def _write_minimal_project_config_no_llm(root: Path) -> None:
    cfg_path = root / ".chunkhound.json"
    db_dir = root / ".chunkhound" / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        json.dumps({"database": {"path": str(db_dir), "provider": "duckdb"}}),
        encoding="utf-8",
    )


def _index_no_embeddings(root: Path) -> None:
    res = subprocess.run(
        ["uv", "run", "chunkhound", "index", ".", "--no-embeddings"],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=60,
        env=get_safe_subprocess_env(),
    )
    assert res.returncode == 0, res.stderr


def _insert_code_embeddings(root: Path, *, provider: str, model: str, dims: int) -> int:
    db_file = root / ".chunkhound" / "db" / "chunks.db"
    assert db_file.exists()

    db = DuckDBProvider(db_file, base_directory=root)
    db.connect()
    try:
        chunks = db.get_all_chunks_with_metadata()
        code_chunks = [
            c
            for c in chunks
            if ChunkType.from_string(str(c.get("chunk_type", ""))).is_code
        ]
        embeddings_data: list[dict] = []
        for c in code_chunks:
            chunk_id = int(c["chunk_id"])
            vec = [float(((chunk_id * 31) + i) % 97) / 97.0 for i in range(dims)]
            embeddings_data.append(
                {
                    "chunk_id": chunk_id,
                    "provider": provider,
                    "model": model,
                    "dims": dims,
                    "embedding": vec,
                }
            )
        return db.insert_embeddings_batch(embeddings_data)
    finally:
        db.disconnect()


def test_snapshot_multi_scope_union_filters_rows() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_minimal_project_config_no_llm(root)

        api_dir = root / "api"
        tests_dir = root / "tests"
        other_dir = root / "other"
        api_dir.mkdir(parents=True, exist_ok=True)
        tests_dir.mkdir(parents=True, exist_ok=True)
        other_dir.mkdir(parents=True, exist_ok=True)

        (api_dir / "a.py").write_text(
            "class A:\n    def foo(self):\n        return 1\n"
        )
        (tests_dir / "test_a.py").write_text(
            "def test_a() -> None:\n    assert 1 == 1\n"
        )
        (other_dir / "c.py").write_text("def c() -> int:\n    return 3\n")

        _index_no_embeddings(root)
        inserted = _insert_code_embeddings(
            root, provider="testprov", model="testmodel", dims=8
        )
        assert inserted > 0

        # Ensure snapshot does not read/parse files: delete sources after indexing.
        (api_dir / "a.py").unlink()
        (tests_dir / "test_a.py").unlink()
        (other_dir / "c.py").unlink()

        out_dir = root / "out"
        res = subprocess.run(
            [
                "uv",
                "run",
                "chunkhound",
                "snapshot",
                "--scope-root",
                "api",
                "--scope-root",
                "tests",
                "--out-dir",
                str(out_dir),
                "--no-themes",
                "--no-systems",
                "--chunk-systems",
                "--embedding-provider",
                "testprov",
                "--embedding-model",
                "testmodel",
                "--embedding-dims",
                "8",
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=120,
            env=get_safe_subprocess_env(),
        )
        assert res.returncode == 0, res.stderr

        run_payload = json.loads((out_dir / "snapshot.run.json").read_text("utf-8"))
        assert run_payload["schema_version"] == "snapshot.run.v1"
        assert run_payload["schema_revision"] == "2026-02-17"
        assert run_payload.get("scope_root") == ["api", "tests"]
        counts = run_payload.get("counts") or {}
        assert isinstance(counts, dict)

        # Compute expected chunk_ids under union scopes from DB.
        db_file = root / ".chunkhound" / "db" / "chunks.db"
        db = DuckDBProvider(db_file, base_directory=root)
        db.connect()
        try:
            chunks = db.get_all_chunks_with_metadata()
        finally:
            db.disconnect()

        expected_chunk_ids: set[int] = set()
        expected_files: set[str] = set()
        for c in chunks:
            file_path = str(c.get("file_path") or "").replace("\\", "/")
            if not (file_path.startswith("api/") or file_path.startswith("tests/")):
                continue
            if not ChunkType.from_string(str(c.get("chunk_type", ""))).is_code:
                continue
            expected_chunk_ids.add(int(c["chunk_id"]))
            expected_files.add(file_path)

        assert int(counts.get("files") or 0) == len(expected_files)
        assert int(counts.get("chunks_with_embeddings") or 0) == len(expected_chunk_ids)
