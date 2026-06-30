import io
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_timeout_prompt_adds_exclusions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Mock coordinator that returns canned values (no actual indexing needed)
    coord = AsyncMock()
    coord._db = SimpleNamespace(
        drop_all_hnsw_indexes=lambda: None,
        ensure_all_hnsw_indexes=lambda: None,
    )
    coord.get_stats.return_value = {"files": 0, "chunks": 0, "embeddings": 0}
    coord.process_directory.return_value = {
        "status": "success",
        "files_processed": 0,
        "total_chunks": 0,
        "skipped": 0,
        "skipped_due_to_timeout": ["big.bin"],
    }
    coord.compact_database_with_metrics.return_value = {
        "status": "skipped",
        "reason": "unsupported",
    }

    # Patch registry hooks used by run_command
    from chunkhound.api.cli.commands import run as run_mod

    monkeypatch.setattr(run_mod, "configure_registry", lambda cfg: None)
    monkeypatch.setattr(run_mod, "create_indexing_coordinator", lambda: coord)

    # Pretend we are in a TTY and accept the prompt
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.delenv("CHUNKHOUND_MCP_MODE", raising=False)
    monkeypatch.delenv("CHUNKHOUND_NO_PROMPTS", raising=False)
    monkeypatch.setattr("builtins.input", lambda *_: "y")

    # Create minimal config file in the target directory
    proj_dir = tmp_path
    config_path = proj_dir / ".chunkhound.json"
    config_path.write_text(json.dumps({}, indent=2))

    # Build args and config
    args = Namespace(
        path=proj_dir,  # pass Path, validation expects Path object
        verbose=False,
        no_embeddings=True,
        include=None,
        exclude=None,
        # Required by validation helpers but not used due to our patches
        db=None,
    )

    # Build a minimal Config object
    from chunkhound.core.config.config import Config as CoreConfig
    from chunkhound.core.config.database_config import DatabaseConfig

    db_cfg = DatabaseConfig(provider="duckdb", path=proj_dir / ".chunkhound" / "db")
    cfg = CoreConfig(target_dir=proj_dir)
    # Assign after creation (pydantic validate_assignment enabled)
    cfg.database = db_cfg

    # Run
    # Capture stdout by temporarily redirecting the Rich console to a string buffer
    buf = io.StringIO()
    real_stdout = sys.stdout
    sys.stdout = buf
    try:
        await run_mod.run_command(args, cfg)
    finally:
        sys.stdout = real_stdout

    out = buf.getvalue()
    assert "Skipped Due to Timeout" in out

    # Verify that the exclusion was appended
    data = json.loads(config_path.read_text())
    assert "indexing" in data and "exclude" in data["indexing"]
    assert "big.bin" in data["indexing"]["exclude"]


@pytest.mark.asyncio
async def test_timeout_prompt_skipped_in_mcp_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """CHUNKHOUND_MCP_MODE=1 must prevent input() call and show info instead."""
    coord = AsyncMock()
    coord._db = SimpleNamespace(
        drop_all_hnsw_indexes=lambda: None,
        ensure_all_hnsw_indexes=lambda: None,
    )
    coord.get_stats.return_value = {"files": 0, "chunks": 0, "embeddings": 0}
    coord.process_directory.return_value = {
        "status": "success",
        "files_processed": 0,
        "total_chunks": 0,
        "skipped": 0,
        "skipped_due_to_timeout": ["big.bin"],
    }
    coord.compact_database_with_metrics.return_value = {
        "status": "skipped",
        "reason": "unsupported",
    }

    from chunkhound.api.cli.commands import run as run_mod

    # Patch registry hooks used by run_command
    monkeypatch.setattr(run_mod, "configure_registry", lambda cfg: None)
    monkeypatch.setattr(run_mod, "create_indexing_coordinator", lambda: coord)

    # Force MCP mode, pretend we are in TTY, but ensure input() would raise if called
    monkeypatch.setenv("CHUNKHOUND_MCP_MODE", "1")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        "builtins.input",
        lambda *_: (_ for _ in ()).throw(RuntimeError("input called")),
    )

    # Minimal config
    proj_dir = tmp_path
    args = Namespace(
        path=proj_dir,
        verbose=False,
        no_embeddings=True,
        include=None,
        exclude=None,
        db=None,
    )

    from chunkhound.core.config.config import Config as CoreConfig
    from chunkhound.core.config.database_config import DatabaseConfig

    db_cfg = DatabaseConfig(provider="duckdb", path=proj_dir / ".chunkhound" / "db")
    cfg = CoreConfig(target_dir=proj_dir)
    cfg.database = db_cfg

    # Capture stdout
    buf = io.StringIO()
    real_stdout = sys.stdout
    sys.stdout = buf
    try:
        await run_mod.run_command(args, cfg)
    finally:
        sys.stdout = real_stdout

    out = buf.getvalue()
    assert "files timed out" in out or "prompts disabled" in out
