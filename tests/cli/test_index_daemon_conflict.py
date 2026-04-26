"""CLI tests for `chunkhound index` daemon lock conflict recovery.

Covers:
1. Healthy daemon holds DuckDB lock → index exits non-zero with "running" message,
   daemon is NOT killed.
2. Stuck daemon (ping fails) + interactive confirm → daemon is killed, index retries.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.utils import SubprocessJsonRpcClient, create_subprocess_exec_safe, get_safe_subprocess_env


def _chunkhound_exe() -> str:
    exe = shutil.which("chunkhound")
    if exe is None:
        raise RuntimeError("chunkhound not found in PATH — run via 'uv run pytest'")
    return exe


def _make_env(project_dir: Path) -> dict[str, str]:
    env = get_safe_subprocess_env()
    for key in list(env.keys()):
        if key.startswith("CHUNKHOUND_"):
            del env[key]
    env["CHUNKHOUND_MCP_MODE"] = "1"
    return env


async def _start_proxy(project_dir: Path):
    env = _make_env(project_dir)
    proc = await create_subprocess_exec_safe(
        _chunkhound_exe(), "mcp", str(project_dir),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env=env,
        cwd=str(project_dir),
    )
    client = SubprocessJsonRpcClient(proc)
    await client.start()
    return proc, client


async def _do_mcp_handshake(client: SubprocessJsonRpcClient) -> None:
    await client.send_request(
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "clientInfo": {"name": "test", "version": "0.0.1"},
            "capabilities": {},
        },
        timeout=30.0,
    )
    await client.send_notification("notifications/initialized", {})


@pytest.fixture
async def pre_indexed_project_dir(tmp_path: Path) -> AsyncIterator[Path]:
    """Minimal pre-indexed project for daemon conflict tests."""
    (tmp_path / "main.py").write_text("def main():\n    pass\n")

    db_path = tmp_path / ".chunkhound" / "test.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "database": {"path": str(db_path), "provider": "duckdb"},
        "indexing": {"include": ["*.py"]},
    }
    (tmp_path / ".chunkhound.json").write_text(json.dumps(config))

    index_proc = await create_subprocess_exec_safe(
        _chunkhound_exe(), "index", str(tmp_path), "--no-embeddings",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(tmp_path),
        env=get_safe_subprocess_env(),
    )
    _, stderr = await asyncio.wait_for(index_proc.communicate(), timeout=60.0)
    assert index_proc.returncode == 0, f"Pre-indexing failed: {stderr.decode()}"

    yield tmp_path

    # Cleanup: stop any lingering daemon
    from chunkhound.daemon.discovery import DaemonDiscovery
    import psutil

    discovery = DaemonDiscovery(tmp_path)
    lock = discovery.read_lock()
    if lock:
        pid = lock.get("pid")
        if isinstance(pid, int):
            try:
                psutil.Process(pid).terminate()
            except Exception:
                pass
    discovery.remove_lock()


@pytest.mark.asyncio
async def test_index_with_healthy_daemon_shows_running_message(
    pre_indexed_project_dir: Path,
) -> None:
    """When a healthy daemon holds the DuckDB lock, `chunkhound index` should:
    - Exit with non-zero status
    - Print an informational message mentioning the daemon is "running"
    - NOT kill the daemon
    """
    project_dir = pre_indexed_project_dir

    # Start daemon via proxy
    proc, client = await _start_proxy(project_dir)
    try:
        await _do_mcp_handshake(client)

        # Wait for daemon to be up
        from tests.helpers.daemon_test_helpers import wait_for_daemon_start
        started = await wait_for_daemon_start(project_dir, timeout=15.0)
        assert started, "Daemon did not start within 15 seconds"

        # Run chunkhound index with CHUNKHOUND_NO_PROMPTS=1 so no interactive prompt
        env = get_safe_subprocess_env()
        env["CHUNKHOUND_NO_PROMPTS"] = "1"
        result = subprocess.run(
            ["uv", "run", "chunkhound", "index", str(project_dir), "--no-embeddings"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            cwd=str(project_dir),
        )

        combined_output = result.stdout + result.stderr
        assert result.returncode != 0, (
            f"Expected non-zero exit when daemon holds lock, got 0.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "running" in combined_output.lower(), (
            f"Expected 'running' in output when healthy daemon present.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

        # Daemon should still be alive (we did NOT kill it)
        from chunkhound.daemon.discovery import DaemonDiscovery
        discovery = DaemonDiscovery(project_dir)
        assert discovery.is_daemon_alive(), "Daemon should still be alive after healthy-lock detection"

    finally:
        try:
            await client.close()
        except Exception:
            pass
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


@pytest.mark.asyncio
async def test_index_with_stuck_daemon_kills_and_retries(tmp_path: Path) -> None:
    """When a stuck daemon (ping returns False) is present and user confirms:
    - `configure_registry` should be called twice (fail + retry)
    - The daemon process should be killed
    """
    import os
    from chunkhound.daemon.discovery import DaemonDiscovery
    from chunkhound.api.cli.commands.run import run_command
    from chunkhound.core.config.config import Config

    # Start a real subprocess as a stand-in "daemon" (sleep)
    if sys.platform == "win32":
        dummy_proc = subprocess.Popen(["ping", "-n", "9999", "127.0.0.1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        dummy_proc = subprocess.Popen(["sleep", "9999"])

    daemon_pid = dummy_proc.pid

    # Write a lock file pointing to this process
    db_path = tmp_path / ".chunkhound" / "test.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / ".chunkhound.json").write_text(json.dumps({
        "database": {"path": str(db_path), "provider": "duckdb"},
    }))
    discovery = DaemonDiscovery(tmp_path)
    discovery.write_lock(daemon_pid, "tcp:127.0.0.1:1", auth_token="test")

    # Build minimal args / config
    args = argparse.Namespace(
        path=str(tmp_path),
        db=db_path,
        verbose=False,
        no_embeddings=True,
        include=["*.py"],
        exclude=[],
        profile_startup=False,
        simulate=False,
        check_ignores=False,
    )

    lock_error = Exception("Could not set lock on file")
    call_count = 0

    def mock_configure_registry(config):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise lock_error

    mock_formatter = MagicMock()
    mock_formatter.info = MagicMock()
    mock_formatter.warning = MagicMock()
    mock_formatter.error = MagicMock()
    mock_formatter.success = MagicMock()
    mock_formatter.startup_info = MagicMock()
    mock_formatter.initial_stats_panel = MagicMock()
    mock_formatter.completion_summary = MagicMock()
    mock_formatter.create_progress_display = MagicMock()
    mock_formatter.create_progress_display.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_formatter.create_progress_display.return_value.__exit__ = MagicMock(return_value=False)
    mock_formatter.verbose_info = MagicMock()

    try:
        config = MagicMock(spec=Config)
        config.database = MagicMock()
        config.database.path = str(db_path)
        config.embedding = None
        config.indexing = MagicMock()

        env_patch = {"CHUNKHOUND_NO_PROMPTS": ""}

        with (
            patch("chunkhound.api.cli.commands.run.configure_registry", side_effect=mock_configure_registry),
            patch("chunkhound.api.cli.commands.run.RichOutputFormatter", return_value=mock_formatter),
            patch.object(DaemonDiscovery, "ping_daemon", new=AsyncMock(return_value=False)),
            patch("sys.stdin") as mock_stdin,
            patch("builtins.input", return_value="y"),
            patch("chunkhound.api.cli.commands.run.create_indexing_coordinator") as mock_coord_factory,
            patch.dict(os.environ, env_patch),
        ):
            mock_stdin.isatty.return_value = True

            mock_coord = AsyncMock()
            mock_coord.get_stats = AsyncMock(return_value={})
            mock_coord_factory.return_value = mock_coord

            with patch("chunkhound.api.cli.commands.run.DirectoryIndexingService") as mock_svc_cls:
                mock_svc = AsyncMock()
                mock_stats = MagicMock()
                mock_stats.processing_time = 0.0
                mock_stats.skipped_due_to_timeout = []
                mock_svc.process_directory = AsyncMock(return_value=mock_stats)
                mock_svc_cls.return_value = mock_svc

                # run_command calls sys.exit on failure; use pytest.raises to catch it
                try:
                    await run_command(args, config)
                except SystemExit as e:
                    if e.code != 0:
                        raise AssertionError(
                            f"run_command exited with code {e.code}. "
                            f"configure_registry calls: {call_count}\n"
                            f"formatter.error calls: {mock_formatter.error.call_args_list}"
                        )

        # configure_registry should have been called twice (fail + retry)
        assert call_count == 2, (
            f"Expected configure_registry to be called twice (fail + retry), got {call_count}"
        )

        # Dummy process should be dead
        import psutil
        assert not psutil.pid_exists(daemon_pid), (
            f"Expected dummy process (pid={daemon_pid}) to be killed after user confirmed"
        )

    finally:
        try:
            dummy_proc.kill()
            dummy_proc.wait()
        except Exception:
            pass
