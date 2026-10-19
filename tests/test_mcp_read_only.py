"""Contract tests for the --read-only MCP flag.

Covers user-visible invariants:
  (a) An existing DB opens read-only and serves regex search.
  (b) Writes through the provider's public API are rejected.
  (c) Config.validate_for_command rejects read_only=True for non-mcp commands.
  (d) Reopening a DB with a stale WAL under read_only surfaces a recovery hint.
  (e) Read-only deferred-start sets query_ready_at and never starts realtime.
  (f) derive_daemon_status reports "ready" for a pre-indexed read-only DB.
  (g) mcp_command rewrites no_daemon=True when read_only is set.
"""

import argparse
import asyncio
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import duckdb
import pytest
from loguru import logger as loguru_logger

from chunkhound.core.config.config import Config
from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.core.types.common import Language
from chunkhound.mcp_server.base import MCPServerBase
from chunkhound.mcp_server.status import derive_daemon_status
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.indexing_coordinator import IndexingCoordinator


class _ConcreteMCPServer(MCPServerBase):
    """Minimal concrete subclass — abstract methods are no-ops for tests."""

    def _register_tools(self) -> None:
        pass

    async def run(self) -> None:
        pass


@pytest.fixture
def populated_db_path(tmp_path):
    """Build a DuckDB DB on disk, index one Python file, disconnect cleanly."""
    db_path = tmp_path / "chunks.db"
    writer = DuckDBProvider(db_path, base_directory=tmp_path)
    writer.connect()
    try:
        parser = create_parser_for_language(Language.PYTHON)
        coordinator = IndexingCoordinator(
            writer, tmp_path, None, {Language.PYTHON: parser}
        )
        sample = tmp_path / "sample.py"
        sample.write_text(
            "def calculate_tax(income, rate):\n    return income * rate\n"
        )
        asyncio.run(coordinator.process_file(sample))
    finally:
        writer.disconnect()
    return db_path


@pytest.fixture
def read_only_reader(populated_db_path, tmp_path):
    """Open populated_db_path read-only through the production wiring."""
    cfg = DatabaseConfig(read_only=True)
    reader = DuckDBProvider(populated_db_path, base_directory=tmp_path, config=cfg)
    reader.connect()
    try:
        yield reader
    finally:
        reader.disconnect()


def test_read_only_open_runs_regex_search(read_only_reader):
    results, _ = read_only_reader.search_regex("calculate_tax")
    assert any("calculate_tax" in r["content"] for r in results)


def test_read_only_rejects_writes(read_only_reader):
    with pytest.raises(duckdb.Error):
        read_only_reader.execute_query("CREATE TABLE __probe (id INTEGER)")
    with pytest.raises(duckdb.Error):
        read_only_reader.execute_query("INSERT INTO files (path) VALUES ('x')")


def test_validator_rejects_read_only_for_index(tmp_path):
    config = Config(
        target_dir=tmp_path,
        database={"path": tmp_path / "chunks.db", "read_only": True},
    )
    errors = config.validate_for_command("index")
    assert any(
        "database.read_only=True is only valid for the 'mcp' subcommand" in e
        for e in errors
    ), errors
    assert not any(
        "read_only" in e for e in config.validate_for_command("mcp")
    )


@pytest.fixture
def stale_wal_db_path(tmp_path):
    """Build a DB with an unreplayable WAL behind it.

    A subprocess writes via the production stack and exits via os._exit to
    skip the close-time CHECKPOINT (provider.disconnect() unconditionally
    checkpoints). The resulting WAL is structurally valid but chunks-only,
    which DuckDB silently discards in read-only mode. Appending garbage
    bytes forces the replay path to raise — the exact failure mode the
    production guard wraps with the recovery hint.
    """
    db_path = tmp_path / "chunks.db"
    script = textwrap.dedent(
        f"""
        import asyncio, os
        from pathlib import Path
        from chunkhound.core.types.common import Language
        from chunkhound.parsers.parser_factory import create_parser_for_language
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider
        from chunkhound.services.indexing_coordinator import IndexingCoordinator

        base = Path({str(tmp_path)!r})
        writer = DuckDBProvider(Path({str(db_path)!r}), base_directory=base)
        writer.connect()
        parser = create_parser_for_language(Language.PYTHON)
        coord = IndexingCoordinator(writer, base, None, {{Language.PYTHON: parser}})
        sample = base / "stale.py"
        sample.write_text("def f():\\n    return 1\\n")
        asyncio.run(coord.process_file(sample))
        os._exit(0)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    wal_path = db_path.with_suffix(".db.wal")
    assert wal_path.exists() and wal_path.stat().st_size > 0, (
        f"expected stale WAL with bytes; stderr={result.stderr!r}"
    )
    with wal_path.open("ab") as f:
        f.write(b"\xff" * 4096)
    return db_path


def test_read_only_stale_wal_surfaces_recovery_hint(stale_wal_db_path, tmp_path):
    cfg = DatabaseConfig(read_only=True)
    reader = DuckDBProvider(
        stale_wal_db_path, base_directory=tmp_path, config=cfg
    )
    with pytest.raises(RuntimeError, match="Reopen without --read-only to recover"):
        reader.connect()


async def _start_read_only_server(db_path, tmp_path):
    """Initialize a read-only MCPServerBase and wait for deferred start."""
    config = Config(
        target_dir=tmp_path,
        database={"path": db_path, "read_only": True},
    )
    args = SimpleNamespace(path=str(tmp_path))
    with (
        patch("chunkhound.mcp_server.base.EmbeddingManager"),
        patch("chunkhound.mcp_server.base.LLMManager"),
    ):
        server = _ConcreteMCPServer(config=config, args=args)
        await server.initialize()
        assert server._deferred_start_task is not None
        await server._deferred_start_task
        return server


@pytest.mark.asyncio
async def test_read_only_deferred_start_sets_query_ready_and_skips_realtime(
    populated_db_path, tmp_path
):
    server = await _start_read_only_server(populated_db_path, tmp_path)
    try:
        assert server.realtime_indexing is None
        assert server._scan_progress["query_ready_at"] is not None
    finally:
        await server.cleanup()


@pytest.mark.asyncio
async def test_read_only_daemon_status_reports_ready(populated_db_path, tmp_path):
    server = await _start_read_only_server(populated_db_path, tmp_path)
    try:
        status = derive_daemon_status(server._scan_progress)
        assert status["status"] == "ready"
        assert status["query_ready"] is True
    finally:
        await server.cleanup()


@pytest.mark.asyncio
async def test_read_only_forces_stdio_when_daemon_requested(tmp_path, monkeypatch):
    # CHUNKHOUND_DAEMON_MODE feeds into mcp_command's no_daemon derivation,
    # so clear it to stay independent of the developer/CI shell. Also clear
    # CHUNKHOUND_MCP_MODE via monkeypatch so its teardown restores the unset
    # state — otherwise mcp_command's unconditional write would bleed into
    # subsequent tests.
    monkeypatch.delenv("CHUNKHOUND_DAEMON_MODE", raising=False)
    monkeypatch.delenv("CHUNKHOUND_MCP_MODE", raising=False)

    args = argparse.Namespace(
        path=str(tmp_path), no_daemon=False, stdio=False, show_setup=False,
    )
    config = Config(
        target_dir=tmp_path,
        database={"path": tmp_path / "chunks.db", "read_only": True},
    )

    captured: list[str] = []
    sink_id = loguru_logger.add(
        lambda msg: captured.append(msg), level="WARNING", format="{message}"
    )
    try:
        with (
            patch("chunkhound.mcp_server.stdio.main", new=AsyncMock()) as stdio_main,
            patch("chunkhound.daemon.client_proxy.ClientProxy") as client_proxy,
        ):
            from chunkhound.api.cli.commands.mcp import mcp_command

            await mcp_command(args, config)
    finally:
        loguru_logger.remove(sink_id)

    stdio_main.assert_called_once()
    client_proxy.assert_not_called()
    assert any(
        "read-only mode forces single-process stdio" in msg for msg in captured
    ), f"Expected forced-stdio warning, got: {captured}"
