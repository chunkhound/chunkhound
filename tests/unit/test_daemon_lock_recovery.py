"""Unit tests for daemon lock conflict recovery logic."""

from __future__ import annotations

import asyncio
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
        DuckDBError = type("DuckDBError", (Exception,), {"__module__": "duckdb"})
        assert _is_db_lock_error(DuckDBError("locked by another process"))

    def test_windows_being_used_by_another_process(self):
        assert _is_db_lock_error(Exception(
            "IO Error: Cannot open file: The process cannot access the file "
            "because it is being used by another process."
        ))

    def test_windows_wal_permission_denied(self):
        assert _is_db_lock_error(PermissionError(
            "[Errno 13] Permission denied: 'C:\\\\data\\\\chunks.db.wal'"
        ))

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
        from chunkhound.daemon.discovery import DaemonDiscovery

        discovery = DaemonDiscovery(tmp_path)
        discovery.write_lock(99999, "tcp:127.0.0.1:1", auth_token="tok")

        result = await discovery.ping_daemon(timeout=0.5)
        assert result is False

    @pytest.mark.asyncio
    async def test_mock_ipc_round_trip_returns_true(self, tmp_path: Path):
        from chunkhound.daemon.discovery import DaemonDiscovery
        from chunkhound.daemon import ipc

        discovery = DaemonDiscovery(tmp_path)
        discovery.write_lock(99999, "tcp:127.0.0.1:9999", auth_token="testtoken")

        mock_reader = AsyncMock()
        mock_writer = MagicMock()
        mock_writer.drain = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()

        # ack frame then ping response
        mock_reader.side_effect = [
            {"type": "registered", "client_id": "abc"},
            {"jsonrpc": "2.0", "id": 1, "result": {}},
        ]

        with (
            patch.object(ipc, "create_client", new=AsyncMock(return_value=(mock_reader, mock_writer))),
            patch.object(ipc, "read_frame", new=AsyncMock(side_effect=[
                {"type": "registered", "client_id": "abc"},
                {"jsonrpc": "2.0", "id": 1, "result": {}},
            ])),
            patch.object(ipc, "write_frame"),
        ):
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
        with patch("chunkhound.daemon.process.pid_alive", return_value=False):
            result = await _handle_daemon_lock_conflict(tmp_path, fmt)
        assert result is False

    @pytest.mark.asyncio
    async def test_healthy_daemon_returns_false_no_kill(self, tmp_path: Path):
        from chunkhound.daemon.discovery import DaemonDiscovery

        discovery = DaemonDiscovery(tmp_path)
        discovery.write_lock(12345, "tcp:127.0.0.1:9999")

        fmt = self._make_formatter()
        with (
            patch("chunkhound.daemon.process.pid_alive", return_value=True),
            patch.object(DaemonDiscovery, "ping_daemon", new=AsyncMock(return_value=True)),
            patch.object(DaemonDiscovery, "stop_daemon") as mock_stop,
        ):
            result = await _handle_daemon_lock_conflict(tmp_path, fmt)

        assert result is False
        fmt.info.assert_called_once()
        assert "running" in fmt.info.call_args[0][0].lower()
        mock_stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_stuck_mcp_mode_returns_false(self, tmp_path: Path):
        from chunkhound.daemon.discovery import DaemonDiscovery

        discovery = DaemonDiscovery(tmp_path)
        discovery.write_lock(12345, "tcp:127.0.0.1:9999")

        fmt = self._make_formatter()
        with (
            patch("chunkhound.daemon.process.pid_alive", return_value=True),
            patch.object(DaemonDiscovery, "ping_daemon", new=AsyncMock(return_value=False)),
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
            patch("chunkhound.daemon.process.pid_alive", return_value=True),
            patch.object(DaemonDiscovery, "ping_daemon", new=AsyncMock(return_value=False)),
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
            patch("chunkhound.daemon.process.pid_alive", return_value=True),
            patch.object(DaemonDiscovery, "ping_daemon", new=AsyncMock(return_value=False)),
            patch.dict("os.environ", {"CHUNKHOUND_MCP_MODE": "0", "CHUNKHOUND_NO_PROMPTS": "0"}),
            patch("sys.stdin") as mock_stdin,
            patch("builtins.input", return_value="n"),
        ):
            mock_stdin.isatty.return_value = True
            result = await _handle_daemon_lock_conflict(tmp_path, fmt)

        assert result is False

    @pytest.mark.asyncio
    async def test_stuck_interactive_input_y_kills_and_returns_true(self, tmp_path: Path):
        from chunkhound.daemon.discovery import DaemonDiscovery

        discovery = DaemonDiscovery(tmp_path)
        discovery.write_lock(12345, "tcp:127.0.0.1:9999")

        fmt = self._make_formatter()
        with (
            patch("chunkhound.daemon.process.pid_alive", return_value=True),
            patch.object(DaemonDiscovery, "ping_daemon", new=AsyncMock(return_value=False)),
            patch.object(DaemonDiscovery, "stop_daemon", return_value=True) as mock_stop,
            patch.dict("os.environ", {"CHUNKHOUND_MCP_MODE": "0", "CHUNKHOUND_NO_PROMPTS": "0"}),
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
            patch("chunkhound.daemon.process.pid_alive", return_value=True),
            patch.object(DaemonDiscovery, "ping_daemon", new=AsyncMock(return_value=False)),
            patch.object(DaemonDiscovery, "stop_daemon", return_value=False),
            patch.dict("os.environ", {"CHUNKHOUND_MCP_MODE": "0", "CHUNKHOUND_NO_PROMPTS": "0"}),
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

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix signal test")
    def test_real_subprocess_terminated(self):
        import subprocess
        import time as _time
        from chunkhound.daemon.process import stop_pid

        proc = subprocess.Popen(["sleep", "60"])
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

    def test_dead_pid_stops_and_removes_lock(self, tmp_path: Path):
        from chunkhound.daemon.discovery import DaemonDiscovery

        discovery = DaemonDiscovery(tmp_path)
        discovery.write_lock(99999999, "tcp:127.0.0.1:1")

        with patch("chunkhound.daemon.process.stop_pid", return_value=True):
            result = discovery.stop_daemon()

        assert result is True
        assert discovery.read_lock() is None

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix signal test")
    def test_real_subprocess_stopped(self, tmp_path: Path):
        import subprocess
        import time as _time
        from chunkhound.daemon.discovery import DaemonDiscovery

        proc = subprocess.Popen(["sleep", "60"])
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
