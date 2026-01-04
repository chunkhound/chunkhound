"""Integration test: root-level files are indexed with default patterns.

This test creates a minimal project with only a .chunkhound.json specifying a
DuckDB database path (no include/exclude rules). It writes a root-level Python
file and runs the CLI index command with --no-embeddings. It then verifies the
file exists in the database, confirming default include patterns cover root
files.
"""

import asyncio
import json
import os
from types import SimpleNamespace

import pytest

from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_database_with_dependencies


@pytest.mark.skipif(
    os.environ.get("CHUNKHOUND_ALLOW_PROCESSPOOL", "0") != "1",
    reason="Requires ProcessPool-friendly environment (SemLock).",
)
@pytest.mark.asyncio
async def test_root_level_file_is_indexed_with_defaults(tmp_path):
    """Root-level files should be indexed when using default include patterns."""
    project_dir = tmp_path

    # Write a root-level source file (covered by default patterns)
    root_file = project_dir / "root.py"
    root_file.write_text("def hello():\n    return 'world'\n")

    # Minimal config: only database settings; rely on default include/exclude
    db_path = project_dir / ".chunkhound" / "test.db"
    db_path.parent.mkdir(exist_ok=True)

    config_path = project_dir / ".chunkhound.json"
    config_content = {"database": {"path": str(db_path), "provider": "duckdb"}}
    config_path.write_text(json.dumps(config_content, indent=2))

    # Run indexing via CLI with embeddings disabled
    env = os.environ.copy()
    env.update(
        {
            # Ensure no prompts in CI-like environments
            "CHUNKHOUND_NO_PROMPTS": "1",
            # Keep UV cache within workspace to avoid permission issues
            "UV_CACHE_DIR": str(project_dir / ".uv-cache"),
        }
    )

    proc = await asyncio.create_subprocess_exec(
        "uv",
        "run",
        "chunkhound",
        "index",
        str(project_dir),
        "--no-embeddings",
        cwd=str(project_dir),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await proc.communicate()
    assert proc.returncode == 0, (
        f"Indexing failed (code={proc.returncode})\n"
        f"stdout: {stdout.decode()}\n"
        f"stderr: {stderr.decode()}"
    )

    # Verify the file exists in the database
    fake_args = SimpleNamespace(path=project_dir)
    config_for_db = Config(
        args=fake_args,
        database={"path": str(db_path), "provider": "duckdb"},
    )

    db = create_database_with_dependencies(
        db_path=db_path,
        config=config_for_db,
        embedding_manager=None,
    )
    db.connect()
    try:
        # get_file_by_path accepts absolute paths
        rec = db.get_file_by_path(str(root_file))
        assert rec is not None, "Root-level file should be indexed with defaults"
    finally:
        db.close()
