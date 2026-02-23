#!/usr/bin/env python3
"""Fail-fast tests for snapshot LLM labeler when LLM config is missing."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from chunkhound.core.types.common import ChunkType
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from tests.utils.windows_subprocess import get_safe_subprocess_env


def _write_config(root: Path, *, with_llm: bool) -> None:
    db_dir = root / ".chunkhound" / "db"
    db_dir.mkdir(parents=True, exist_ok=True)

    cfg: dict = {
        "database": {"path": str(db_dir), "provider": "duckdb"},
        "indexing": {"include": ["*.py"]},
    }
    if with_llm:
        cfg["llm"] = {
            "provider": "codex-cli",
            "utility_model": "gpt-5-nano",
            "synthesis_model": "gpt-5",
        }
    (root / ".chunkhound.json").write_text(json.dumps(cfg), encoding="utf-8")


def _index_and_insert_embeddings(root: Path) -> None:
    res = subprocess.run(
        ["uv", "run", "chunkhound", "index", ".", "--no-embeddings"],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=60,
        env=get_safe_subprocess_env(),
    )
    assert res.returncode == 0, res.stderr

    db_dir = root / ".chunkhound" / "db"
    db_file = db_dir / "chunks.db"
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
            vec = [float(((chunk_id * 17) + i) % 101) / 101.0 for i in range(8)]
            embeddings_data.append(
                {
                    "chunk_id": chunk_id,
                    "provider": "testprov",
                    "model": "testmodel",
                    "dims": 8,
                    "embedding": vec,
                }
            )
        inserted = db.insert_embeddings_batch(embeddings_data)
        assert inserted > 0
    finally:
        db.disconnect()


def test_snapshot_llm_missing_fails_fast() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_config(root, with_llm=False)

        (root / "a.py").write_text("class A:\n    pass\n", encoding="utf-8")

        _index_and_insert_embeddings(root)

        out_dir = root / "out"
        res2 = subprocess.run(
            [
                "uv",
                "run",
                "chunkhound",
                "snapshot",
                ".",
                "--out-dir",
                str(out_dir),
                "--chunk-systems",
                "--no-themes",
                "--no-systems",
                "--labeler",
                "llm",
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
            timeout=60,
            env=get_safe_subprocess_env(),
        )
        assert res2.returncode != 0
        combined = (res2.stderr + res2.stdout).lower()
        assert "llm" in combined


def test_snapshot_llm_dry_run_without_llm_config_succeeds() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_config(root, with_llm=False)

        (root / "a.py").write_text("class A:\n    pass\n", encoding="utf-8")
        _index_and_insert_embeddings(root)

        out_dir = root / "out"
        res = subprocess.run(
            [
                "uv",
                "run",
                "chunkhound",
                "snapshot",
                ".",
                "--out-dir",
                str(out_dir),
                "--chunk-systems",
                "--no-themes",
                "--no-systems",
                "--labeler",
                "llm",
                "--llm-dry-run",
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
            timeout=60,
            env=get_safe_subprocess_env(),
        )
        assert res.returncode == 0, res.stderr
        assert (out_dir / "snapshot.chunk_systems.json").exists()
        assert (out_dir / "snapshot.labels.json").exists()


def test_snapshot_heuristic_labeler_without_llm_config_succeeds() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_config(root, with_llm=False)

        (root / "a.py").write_text("class A:\n    pass\n", encoding="utf-8")
        _index_and_insert_embeddings(root)

        out_dir = root / "out"
        res = subprocess.run(
            [
                "uv",
                "run",
                "chunkhound",
                "snapshot",
                ".",
                "--out-dir",
                str(out_dir),
                "--chunk-systems",
                "--no-themes",
                "--no-systems",
                "--labeler",
                "heuristic",
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
            timeout=60,
            env=get_safe_subprocess_env(),
        )
        assert res.returncode == 0, res.stderr
        assert (out_dir / "snapshot.chunk_systems.json").exists()
        assert not (out_dir / "snapshot.labels.json").exists()
