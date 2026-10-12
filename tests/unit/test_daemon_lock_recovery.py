"""Unit tests for daemon lock conflict recovery logic."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.api.cli.commands.run import (
    _handle_daemon_lock_conflict,
    _is_db_lock_error,
)

# ---------------------------------------------------------------------------
# TestIsDbLockError
# ---------------------------------------------------------------------------


class TestIsDbLockError:
    def test_could_not_set_lock(self):
        assert _is_db_lock_error(Exception("Could not set lock on file"))

    def test_database_is_locked(self):
        assert _is_db_lock_error(Exception("database is locked"))

    def test_duckdb_locked(self):
        duckdb_error = type("DuckDBError", (Exception,), {"__module__": "duckdb"})
        assert _is_db_lock_error(duckdb_error("locked by another process"))

    def test_windows_being_used_by_another_process(self):
        assert _is_db_lock_error(
            Exception(
                "IO Error: Cannot open file: The process cannot access the file "
                "because it is being used by another process."
            )
        )

    def test_windows_wal_permission_denied(self):
        assert _is_db_lock_error(
            PermissionError(
                "[Errno 13] Permission denied: 'C:\\\\data\\\\chunks.db.wal'"
            )
        )

    def test_wal_error_not_lock(self):
        assert not _is_db_lock_error(Exception("WAL checkpoint failed"))

    def test_table_not_found_not_lock(self):
        assert not _is_db_lock_error(Exception("table not found: chunks"))


@pytest.mark.asyncio
@pytest.mark.parametrize("daemon_state", ["healthy", "stuck", "recover"])
async def test_index_command_recognizes_real_preflight_lock_and_checks_daemon(
    tmp_path, monkeypatch, clean_environment, capsys, daemon_state
):
    import chunkhound.registry as registry_module
    from chunkhound.api.cli.commands.run import run_command
    from chunkhound.core.config.config import Config
    from chunkhound.core.index_format import IndexFormatError
    from chunkhound.daemon.discovery import DaemonDiscovery
    from chunkhound.providers.database.duckdb_provider import DuckDBProvider
    from chunkhound.providers.database.index_format import preflight_duckdb

    registry = registry_module.ProviderRegistry()
    monkeypatch.setattr(registry_module, "_registry", registry)
    recovering = daemon_state == "recover"
    monkeypatch.setenv("CHUNKHOUND_NO_PROMPTS", "0" if recovering else "1")
    config = Config(
        database={"path": tmp_path / "private-index", "provider": "duckdb"},
        target_dir=tmp_path, embedding=None, embeddings_disabled=True,
    )
    path = config.database.get_db_path()
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    provider.disconnect()
    registry = registry_module.ProviderRegistry()
    monkeypatch.setattr(registry_module, "_registry", registry)
    # A real external writer holds the lock; stdin gives deterministic release.
    proc = subprocess.Popen(
        [
            "uv", "run", "python", "-u", "-c",
            "import duckdb, sys; c=duckdb.connect(sys.argv[1]); "
            "print('ready', flush=True); sys.stdin.readline(); c.close()", str(path),
        ],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    try:
        assert proc.stdout.readline().strip() == "ready"
        with pytest.raises(IndexFormatError) as error:
            preflight_duckdb(path)
        assert str(path) not in str(error.value)
        discovery = DaemonDiscovery(tmp_path)
        discovery.write_lock(proc.pid, "tcp:127.0.0.1:1")
        args = argparse.Namespace(
            path=tmp_path, verbose=False, no_embeddings=True,
            include=["*.py"], exclude=[],
        )
        def release_test_holder(*, timeout):
            # Fake only the process-control boundary: release our holder
            # gracefully, never terminate a real daemon.
            proc.communicate("release\n", timeout=timeout)
            return True

        with (
            patch.object(
                DaemonDiscovery, "ping_daemon",
                new=AsyncMock(return_value=daemon_state == "healthy"),
            ) as ping,
            patch.object(
                DaemonDiscovery, "stop_daemon", side_effect=release_test_holder
            ) as stop,
            patch("sys.stdin.isatty", return_value=recovering),
            patch("builtins.input", return_value="y"),
        ):
            if recovering:
                await run_command(args, config)
                stop.assert_called_once()
            else:
                with pytest.raises(SystemExit) as exit_info:
                    await run_command(args, config)
                assert exit_info.value.code == 1
                stop.assert_not_called()
            ping.assert_awaited_once()
        assert _is_db_lock_error(error.value)
        output = capsys.readouterr().out
        if recovering:
            assert "Run command completed successfully" in output
        else:
            assert (
                "is running" if daemon_state == "healthy" else "is unresponsive"
            ) in output
    finally:
        if proc.poll() is None:
            proc.communicate("release\n", timeout=15)
        registered_provider = registry._providers.get("database")
        if registered_provider is not None:
            registered_provider.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["future", "corrupt"])
@pytest.mark.parametrize(
    "filename",
    [
        "locked-secret.duckdb",
        "could not set lock.duckdb",
        "database is locked.duckdb",
        "being used by another process.duckdb",
        "locked by another process.duckdb",
    ],
)
async def test_non_lock_format_refusal_never_consults_daemon(
    tmp_path, monkeypatch, clean_environment, capsys, kind, filename
):
    import os

    import duckdb

    import chunkhound.registry as registry_module
    from chunkhound.api.cli.commands.run import run_command
    from chunkhound.core.config.config import Config
    from chunkhound.core.index_format import IndexFormatError
    from chunkhound.daemon.discovery import DaemonDiscovery
    from chunkhound.providers.database.index_format import preflight_duckdb

    monkeypatch.setattr(
        registry_module, "_registry", registry_module.ProviderRegistry()
    )
    monkeypatch.setenv("CHUNKHOUND_NO_PROMPTS", "0")
    monkeypatch.setenv("CHUNKHOUND_DAEMON_RUNTIME_DIR", str(tmp_path / ".runtime"))
    monkeypatch.setenv("CHUNKHOUND_DAEMON_REGISTRY_DIR", str(tmp_path / ".registry"))
    path = tmp_path / filename
    if kind == "future":
        with duckdb.connect(str(path)) as conn:
            conn.execute("CREATE TABLE schema_version AS SELECT 2 AS version")
    else:
        path.write_bytes(b"not a database")
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    with pytest.raises(IndexFormatError) as error:
        preflight_duckdb(path)
    assert str(path) not in str(error.value)
    config = Config(
        database={"path": path, "provider": "duckdb"},
        target_dir=tmp_path,
        embedding=None,
        embeddings_disabled=True,
    )
    args = argparse.Namespace(
        path=tmp_path,
        verbose=False,
        no_embeddings=True,
        include=["*.py"],
        exclude=[],
    )
    # A live PID makes the unrelated-daemon hazard reachable. Fake only IPC,
    # terminal input and process control; never stop this process or a daemon.
    DaemonDiscovery(tmp_path).write_lock(os.getpid(), "tcp:127.0.0.1:1")
    with (
        patch.object(
            DaemonDiscovery, "ping_daemon", new=AsyncMock(return_value=False)
        ) as ping,
        patch.object(DaemonDiscovery, "stop_daemon", return_value=True) as stop,
        patch("sys.stdin.isatty", return_value=True),
        patch("builtins.input", return_value="y") as prompt,
    ):
        with pytest.raises(SystemExit) as exit_info:
            await run_command(args, config)
        assert exit_info.value.code == 1
        ping.assert_not_awaited()
        prompt.assert_not_called()
        stop.assert_not_called()
    assert not _is_db_lock_error(error.value)
    if kind == "corrupt":
        # Direct engine failures must not classify arbitrary filenames either.
        with pytest.raises(duckdb.IOException) as raw_error:
            duckdb.connect(str(path), read_only=True)
        assert not _is_db_lock_error(raw_error.value)
    output = capsys.readouterr().out
    assert "Run command failed:" in output
    assert "is unresponsive" not in output
    assert "is not a valid DuckDB database" not in output
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


# ---------------------------------------------------------------------------
# TestPingDaemon
# ---------------------------------------------------------------------------


class TestPingDaemon:
    @pytest.mark.asyncio
    async def test_no_lock_file_returns_false(self, tmp_path: Path):
        from chunkhound.daemon.discovery import DaemonDiscovery

        discovery = DaemonDiscovery(tmp_path)
        result = await discovery.ping_daemon(timeout=1.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_socket_unreachable_returns_false(self, tmp_path: Path):
        from chunkhound.daemon import ipc
        from chunkhound.daemon.discovery import DaemonDiscovery

        discovery = DaemonDiscovery(tmp_path)
        discovery.write_lock(99999, "tcp:127.0.0.1:1", auth_token="tok")

        with patch.object(ipc, "authenticated_ping", new=AsyncMock(return_value=False)):
            result = await discovery.ping_daemon(timeout=0.5)
        assert result is False

    @pytest.mark.asyncio
    async def test_mock_ipc_round_trip_returns_true(self, tmp_path: Path):
        from chunkhound.daemon import ipc
        from chunkhound.daemon.discovery import DaemonDiscovery

        discovery = DaemonDiscovery(tmp_path)
        discovery.write_lock(99999, "tcp:127.0.0.1:9999", auth_token="testtoken")

        with patch.object(ipc, "authenticated_ping", new=AsyncMock(return_value=True)):
            result = await discovery.ping_daemon(timeout=2.0)

        assert result is True


# ---------------------------------------------------------------------------
# TestHandleDaemonLockConflict
# ---------------------------------------------------------------------------


class TestHandleDaemonLockConflict:
    def _make_formatter(self):
        fmt = MagicMock()
        fmt.info = MagicMock()
        fmt.warning = MagicMock()
        fmt.error = MagicMock()
        fmt.success = MagicMock()
        return fmt

    @pytest.mark.asyncio
    async def test_no_lock_file_returns_false(self, tmp_path: Path):
        fmt = self._make_formatter()
        result = await _handle_daemon_lock_conflict(tmp_path, fmt)
        assert result is False

    @pytest.mark.asyncio
    async def test_stale_pid_returns_false(self, tmp_path: Path):
        from chunkhound.daemon.discovery import DaemonDiscovery

        discovery = DaemonDiscovery(tmp_path)
        discovery.write_lock(99999999, "tcp:127.0.0.1:1")

        fmt = self._make_formatter()
        with patch("chunkhound.api.cli.commands.run.pid_alive", return_value=False):
            result = await _handle_daemon_lock_conflict(tmp_path, fmt)
        assert result is False

    @pytest.mark.asyncio
    async def test_healthy_daemon_returns_false_no_kill(self, tmp_path: Path):
        from chunkhound.daemon.discovery import DaemonDiscovery

        discovery = DaemonDiscovery(tmp_path)
        discovery.write_lock(12345, "tcp:127.0.0.1:9999")

        fmt = self._make_formatter()
        with (
            patch("chunkhound.api.cli.commands.run.pid_alive", return_value=True),
            patch.object(
                DaemonDiscovery, "ping_daemon", new=AsyncMock(return_value=True)
            ),
            patch.object(DaemonDiscovery, "stop_daemon") as mock_stop,
        ):
            result = await _handle_daemon_lock_conflict(tmp_path, fmt)

        assert result is False
        fmt.warning.assert_called_once()
        assert "running" in fmt.warning.call_args[0][0].lower()
        mock_stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_stuck_mcp_mode_returns_false(self, tmp_path: Path):
        from chunkhound.daemon.discovery import DaemonDiscovery

        discovery = DaemonDiscovery(tmp_path)
        discovery.write_lock(12345, "tcp:127.0.0.1:9999")

        fmt = self._make_formatter()
        with (
            patch("chunkhound.api.cli.commands.run.pid_alive", return_value=True),
            patch.object(
                DaemonDiscovery, "ping_daemon", new=AsyncMock(return_value=False)
            ),
            patch.dict("os.environ", {"CHUNKHOUND_MCP_MODE": "1"}),
        ):
            result = await _handle_daemon_lock_conflict(tmp_path, fmt)

        assert result is False
        fmt.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_stuck_no_tty_returns_false(self, tmp_path: Path):
        from chunkhound.daemon.discovery import DaemonDiscovery

        discovery = DaemonDiscovery(tmp_path)
        discovery.write_lock(12345, "tcp:127.0.0.1:9999")

        fmt = self._make_formatter()
        with (
            patch("chunkhound.api.cli.commands.run.pid_alive", return_value=True),
            patch.object(
                DaemonDiscovery, "ping_daemon", new=AsyncMock(return_value=False)
            ),
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.isatty.return_value = False
            result = await _handle_daemon_lock_conflict(tmp_path, fmt)

        assert result is False
        fmt.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_stuck_interactive_input_n_returns_false(self, tmp_path: Path):
        from chunkhound.daemon.discovery import DaemonDiscovery

        discovery = DaemonDiscovery(tmp_path)
        discovery.write_lock(12345, "tcp:127.0.0.1:9999")

        fmt = self._make_formatter()
        with (
            patch("chunkhound.api.cli.commands.run.pid_alive", return_value=True),
            patch.object(
                DaemonDiscovery, "ping_daemon", new=AsyncMock(return_value=False)
            ),
            patch.dict(
                "os.environ", {"CHUNKHOUND_MCP_MODE": "0", "CHUNKHOUND_NO_PROMPTS": "0"}
            ),
            patch("sys.stdin") as mock_stdin,
            patch("builtins.input", return_value="n"),
        ):
            mock_stdin.isatty.return_value = True
            result = await _handle_daemon_lock_conflict(tmp_path, fmt)

        assert result is False

    @pytest.mark.asyncio
    async def test_stuck_interactive_input_y_kills_and_returns_true(
        self, tmp_path: Path
    ):
        from chunkhound.daemon.discovery import DaemonDiscovery

        discovery = DaemonDiscovery(tmp_path)
        discovery.write_lock(12345, "tcp:127.0.0.1:9999")

        fmt = self._make_formatter()
        with (
            patch("chunkhound.api.cli.commands.run.pid_alive", return_value=True),
            patch.object(
                DaemonDiscovery, "ping_daemon", new=AsyncMock(return_value=False)
            ),
            patch.object(
                DaemonDiscovery, "stop_daemon", return_value=True
            ) as mock_stop,
            patch.dict(
                "os.environ", {"CHUNKHOUND_MCP_MODE": "0", "CHUNKHOUND_NO_PROMPTS": "0"}
            ),
            patch("sys.stdin") as mock_stdin,
            patch("builtins.input", return_value="y"),
        ):
            mock_stdin.isatty.return_value = True
            result = await _handle_daemon_lock_conflict(tmp_path, fmt)

        assert result is True
        mock_stop.assert_called_once()
        fmt.success.assert_called_once()

    @pytest.mark.asyncio
    async def test_stuck_stop_daemon_fails_returns_false(self, tmp_path: Path):
        from chunkhound.daemon.discovery import DaemonDiscovery

        discovery = DaemonDiscovery(tmp_path)
        discovery.write_lock(12345, "tcp:127.0.0.1:9999")

        fmt = self._make_formatter()
        with (
            patch("chunkhound.api.cli.commands.run.pid_alive", return_value=True),
            patch.object(
                DaemonDiscovery, "ping_daemon", new=AsyncMock(return_value=False)
            ),
            patch.object(DaemonDiscovery, "stop_daemon", return_value=False),
            patch.dict(
                "os.environ", {"CHUNKHOUND_MCP_MODE": "0", "CHUNKHOUND_NO_PROMPTS": "0"}
            ),
            patch("sys.stdin") as mock_stdin,
            patch("builtins.input", return_value="y"),
        ):
            mock_stdin.isatty.return_value = True
            result = await _handle_daemon_lock_conflict(tmp_path, fmt)

        assert result is False
        fmt.error.assert_called_once()


# ---------------------------------------------------------------------------
# TestStopPid
# ---------------------------------------------------------------------------


class TestStopPid:
    def test_zero_pid_returns_true(self):
        from chunkhound.daemon.process import stop_pid

        assert stop_pid(0) is True

    def test_already_dead_pid_returns_true(self):
        from chunkhound.daemon.process import stop_pid

        with patch("chunkhound.daemon.process.pid_alive", return_value=False):
            assert stop_pid(12345) is True

    def test_pid_alive_access_denied_treats_process_as_live(self):
        import psutil

        from chunkhound.daemon.process import pid_alive

        mock_process = MagicMock()
        mock_process.status.side_effect = psutil.AccessDenied(pid=12345)
        with patch("psutil.Process", return_value=mock_process):
            assert pid_alive(12345) is True

    def test_real_subprocess_terminated(self):
        import time as _time

        from chunkhound.daemon.process import stop_pid

        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            _time.sleep(0.1)  # give it a moment to start
            result = stop_pid(proc.pid, timeout=5.0)
            assert result is True
            proc.wait(timeout=1.0)
            assert proc.poll() is not None
        finally:
            try:
                proc.kill()
                proc.wait()
            except Exception:
                pass

    def test_escalates_to_force_kill_after_graceful_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from chunkhound.daemon.process import stop_pid

        alive_states = iter([True, True, False])
        monotonic_values = iter([0.0, 10.0, 10.0, 20.0])
        mock_proc = MagicMock()

        monkeypatch.setattr(
            "chunkhound.daemon.process.pid_alive", lambda pid: next(alive_states)
        )
        monkeypatch.setattr(
            "chunkhound.daemon.process.time.monotonic",
            lambda: next(monotonic_values),
        )
        monkeypatch.setattr("chunkhound.daemon.process.time.sleep", lambda _: None)

        with patch("psutil.Process", return_value=mock_proc):
            assert stop_pid(12345, timeout=10.0) is True

        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="signal.SIG_IGN is POSIX-only",
    )
    def test_force_kill_has_separate_deadline_on_exhausted_timeout(self):
        """A process that ignores SIGTERM is killed via SIGKILL even when the
        graceful timeout is very short — Phase 2 gets its own deadline
        independent of what Phase 1 consumed."""
        from chunkhound.daemon.process import stop_pid

        proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-c",
                "import signal, sys, time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "sys.stdout.write('ready\\n'); sys.stdout.flush(); "
                "time.sleep(60)",
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert proc.stdout is not None
            assert proc.stdout.readline().strip() == "ready"
            # timeout=0.5 means Phase 1 has only 0.5s for SIGTERM.
            # SIGTERM is ignored, so Phase 2's 2s deadline must still
            # succeed independently.
            result = stop_pid(proc.pid, timeout=0.5)
            assert result is True
            proc.wait(timeout=1.0)
            assert proc.poll() is not None
        finally:
            try:
                proc.kill()
                proc.wait()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# TestStopDaemon
# ---------------------------------------------------------------------------


class TestStopDaemon:
    def test_no_lock_returns_true(self, tmp_path: Path):
        from chunkhound.daemon.discovery import DaemonDiscovery

        discovery = DaemonDiscovery(tmp_path)
        assert discovery.stop_daemon() is True

    def test_zero_pid_removes_lock_returns_true(self, tmp_path: Path):
        from chunkhound.daemon.discovery import DaemonDiscovery

        discovery = DaemonDiscovery(tmp_path)
        discovery.write_lock(0, "tcp:127.0.0.1:1")

        result = discovery.stop_daemon()
        assert result is True
        assert discovery.read_lock() is None

    def test_legacy_lock_dead_pid_removes_lock(self, tmp_path: Path):
        from chunkhound.daemon.discovery import DaemonDiscovery

        discovery = DaemonDiscovery(tmp_path)
        discovery.get_lock_path().parent.mkdir(parents=True, exist_ok=True)
        discovery.get_lock_path().write_text(
            json.dumps(
                {
                    "pid": 12345,
                    "socket_path": "tcp:127.0.0.1:1",
                    "started_at": 1.0,
                    "project_dir": str(tmp_path.resolve()),
                    "auth_token": "token",
                }
            )
        )

        with patch("chunkhound.daemon.process.pid_alive", return_value=False):
            result = discovery.stop_daemon()

        assert result is True
        assert discovery.read_lock() is None

    def test_legacy_lock_live_pid_refuses_to_stop_without_identity(
        self, tmp_path: Path
    ):
        from chunkhound.daemon.discovery import DaemonDiscovery

        discovery = DaemonDiscovery(tmp_path)
        discovery.get_lock_path().parent.mkdir(parents=True, exist_ok=True)
        discovery.get_lock_path().write_text(
            json.dumps(
                {
                    "pid": 12345,
                    "socket_path": "tcp:127.0.0.1:1",
                    "started_at": 1.0,
                    "project_dir": str(tmp_path.resolve()),
                    "auth_token": "token",
                }
            )
        )

        with (
            patch("chunkhound.daemon.process.pid_alive", return_value=True),
            patch("chunkhound.daemon.process.stop_pid", return_value=True) as stop_pid,
        ):
            result = discovery.stop_daemon()

        assert result is False
        stop_pid.assert_not_called()
        assert discovery.read_lock() is not None

    def test_dead_pid_stops_and_removes_lock(self, tmp_path: Path):
        from chunkhound.daemon.discovery import DaemonDiscovery

        discovery = DaemonDiscovery(tmp_path)
        with patch("chunkhound.daemon.process.process_create_time", return_value=1.0):
            discovery.write_lock(99999999, "tcp:127.0.0.1:1")

        with (
            patch("chunkhound.daemon.process.process_create_time", return_value=None),
            patch("chunkhound.daemon.process.pid_alive", return_value=False),
        ):
            result = discovery.stop_daemon()

        assert result is True
        assert discovery.read_lock() is None

    def test_mismatched_process_identity_does_not_stop_pid(self, tmp_path: Path):
        from chunkhound.daemon.discovery import DaemonDiscovery

        discovery = DaemonDiscovery(tmp_path)
        with patch("chunkhound.daemon.process.process_create_time", return_value=1.0):
            discovery.write_lock(12345, "tcp:127.0.0.1:1")

        with (
            patch("chunkhound.daemon.process.process_create_time", return_value=2.0),
            patch("chunkhound.daemon.process.stop_pid", return_value=True) as stop_pid,
        ):
            result = discovery.stop_daemon()

        assert result is True
        stop_pid.assert_not_called()
        assert discovery.read_lock() is None

    def test_unknown_process_identity_refuses_to_stop_live_pid(self, tmp_path: Path):
        from chunkhound.daemon.discovery import DaemonDiscovery

        discovery = DaemonDiscovery(tmp_path)
        with patch("chunkhound.daemon.process.process_create_time", return_value=1.0):
            discovery.write_lock(12345, "tcp:127.0.0.1:1")

        with (
            patch("chunkhound.daemon.process.process_create_time", return_value=None),
            patch("chunkhound.daemon.process.pid_alive", return_value=True),
            patch("chunkhound.daemon.process.stop_pid", return_value=True) as stop_pid,
        ):
            result = discovery.stop_daemon()

        assert result is False
        stop_pid.assert_not_called()
        assert discovery.read_lock() is not None

    def test_matching_process_identity_stops_pid(self, tmp_path: Path):
        from chunkhound.daemon.discovery import DaemonDiscovery

        discovery = DaemonDiscovery(tmp_path)
        with patch("chunkhound.daemon.process.process_create_time", return_value=1.0):
            discovery.write_lock(12345, "tcp:127.0.0.1:1")

        with (
            patch("chunkhound.daemon.process.process_create_time", return_value=1.0),
            patch("chunkhound.daemon.process.stop_pid", return_value=True) as stop_pid,
        ):
            result = discovery.stop_daemon(timeout=3.0)

        assert result is True
        stop_pid.assert_called_once_with(12345, timeout=3.0)
        assert discovery.read_lock() is None

    def test_matching_process_identity_removes_registry_entry(self, tmp_path: Path):
        from chunkhound.daemon.discovery import DaemonDiscovery

        discovery = DaemonDiscovery(tmp_path)
        with patch("chunkhound.daemon.process.process_create_time", return_value=1.0):
            discovery.write_lock(12345, "tcp:127.0.0.1:1")

        registry_path = discovery.get_registry_entry_path()
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text("{}")

        with (
            patch("chunkhound.daemon.process.process_create_time", return_value=1.0),
            patch("chunkhound.daemon.process.stop_pid", return_value=True),
        ):
            result = discovery.stop_daemon(timeout=3.0)

        assert result is True
        assert discovery.read_lock() is None
        assert not registry_path.exists()

    def test_real_subprocess_stopped(self, tmp_path: Path):
        import time as _time

        from chunkhound.daemon.discovery import DaemonDiscovery

        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        _time.sleep(0.1)

        discovery = DaemonDiscovery(tmp_path)
        discovery.write_lock(proc.pid, "tcp:127.0.0.1:1")

        try:
            result = discovery.stop_daemon(timeout=5.0)
            assert result is True
            proc.wait(timeout=1.0)
            assert proc.poll() is not None
            assert discovery.read_lock() is None
        finally:
            try:
                proc.kill()
                proc.wait()
            except Exception:
                pass
