#!/usr/bin/env python3
"""Fail-fast tests for snapshot when embeddings are missing for the selector."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from tests.utils.windows_subprocess import get_safe_subprocess_env


def test_snapshot_missing_embeddings_fails_fast() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        db_dir = root / ".chunkhound" / "db"
        db_dir.mkdir(parents=True, exist_ok=True)

        (root / ".chunkhound.json").write_text(
            json.dumps(
                {
                    "database": {"path": str(db_dir), "provider": "duckdb"},
                    "indexing": {"include": ["*.py"]},
                    "llm": {"provider": "codex-cli", "utility_model": "gpt-5-nano", "synthesis_model": "gpt-5"},
                }
            ),
            encoding="utf-8",
        )

        (root / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")

        res = subprocess.run(
            ["uv", "run", "chunkhound", "index", ".", "--no-embeddings"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=60,
            env=get_safe_subprocess_env(),
        )
        assert res.returncode == 0, res.stderr

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
        assert res2.returncode != 0
        assert "embeddings" in (res2.stderr + res2.stdout).lower()
