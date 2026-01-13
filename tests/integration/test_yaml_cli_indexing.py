"""Integration test to ensure CLI indexing works on YAML fixtures."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "yaml"


@pytest.mark.skipif(not FIXTURE_DIR.exists(), reason="YAML fixtures missing")
def test_cli_indexes_yaml_repo(tmp_path):
    workdir = tmp_path / "repo"
    shutil.copytree(FIXTURE_DIR, workdir)

    env = os.environ.copy()
    db_path = tmp_path / "chunkhound-yaml.duckdb"
    env["CHUNKHOUND_DATABASE__PATH"] = str(db_path)
    env.setdefault("CHUNKHOUND_DATABASE__PROVIDER", "duckdb")

    cmd = ["uv", "run", "chunkhound", "index", str(workdir), "--no-embeddings"]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )

    assert result.returncode == 0, (
        f"CLI failed: stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert db_path.exists(), "Database file was not created"
