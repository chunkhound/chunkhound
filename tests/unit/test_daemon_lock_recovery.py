"""Unit tests for daemon lock conflict recovery logic."""

from __future__ import annotations

import json
import signal
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

        signals: list[int] = []
        alive_states = iter([True, True, False])
        monotonic_values = iter([0.0, 10.0, 10.0, 20.0])

        monkeypatch.setattr("chunkhound.daemon.process.sys.platform", "linux")
        monkeypatch.setattr(
            "chunkhound.daemon.process.pid_alive", lambda pid: next(alive_states)
        )
        monkeypatch.setattr(
            "chunkhound.daemon.process.os.kill",
            lambda pid, sig: signals.append(sig),
        )
        monkeypatch.setattr(
            "chunkhound.daemon.process.time.monotonic",
            lambda: next(monotonic_values),
        )
        monkeypatch.setattr("chunkhound.daemon.process.time.sleep", lambda _: None)

        assert stop_pid(12345, timeout=10.0) is True
        assert signals == [signal.SIGTERM, signal.SIGKILL]

    def test_force_kill_uses_remaining_timeout_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from chunkhound.daemon.process import stop_pid

        signals: list[int] = []
        alive_states = iter([True, True, True])
        monotonic_values = iter([0.0, 10.0, 10.0])

        monkeypatch.setattr("chunkhound.daemon.process.sys.platform", "linux")
        monkeypatch.setattr(
            "chunkhound.daemon.process.pid_alive", lambda pid: next(alive_states)
        )
        monkeypatch.setattr(
            "chunkhound.daemon.process.os.kill",
            lambda pid, sig: signals.append(sig),
        )
        monkeypatch.setattr(
            "chunkhound.daemon.process.time.monotonic",
            lambda: next(monotonic_values),
        )
        monkeypatch.setattr("chunkhound.daemon.process.time.sleep", lambda _: None)

        assert stop_pid(12345, timeout=10.0) is False
        assert signals == [signal.SIGTERM, signal.SIGKILL]


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
            assert discovery.read_lock() is None
        finally:
            try:
                proc.kill()
                proc.wait()
            except Exception:
                pass
