"""Unit tests for DaemonManager service.

Tests daemon lifecycle management (start, stop, status).
"""

import os
import signal
import socket
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chunkhound.services.daemon_manager import DaemonManager, DaemonStatus


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory for daemon files."""
    data_dir = tmp_path / ".chunkhound"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def manager(temp_data_dir):
    """Create a DaemonManager with temporary data directory."""
    return DaemonManager(data_dir=temp_data_dir, host="127.0.0.1", port=5173)


class TestDaemonStatus:
    """Tests for DaemonStatus dataclass."""

    def test_to_dict_running(self):
        """Test converting running status to dict."""
        status = DaemonStatus(
            running=True,
            pid=12345,
            host="127.0.0.1",
            port=5173,
            url="http://127.0.0.1:5173",
            uptime_seconds=3600.0,
            pid_file=Path("/tmp/daemon.pid"),
        )

        result = status.to_dict()

        assert result["running"] is True
        assert result["pid"] == 12345
        assert result["host"] == "127.0.0.1"
        assert result["port"] == 5173
        assert result["url"] == "http://127.0.0.1:5173"
        assert result["uptime_seconds"] == 3600.0
        assert result["pid_file"] == "/tmp/daemon.pid"

    def test_to_dict_not_running(self):
        """Test converting not running status to dict."""
        status = DaemonStatus(
            running=False,
            pid_file=Path("/tmp/daemon.pid"),
        )

        result = status.to_dict()

        assert result["running"] is False
        assert result["pid"] is None
        assert result["url"] is None


class TestDaemonManagerInit:
    """Tests for DaemonManager initialization."""

    def test_default_values(self):
        """Test default configuration values."""
        manager = DaemonManager()

        assert manager.host == "127.0.0.1"
        assert manager.port == 5173
        assert manager.data_dir == Path.home() / ".chunkhound"

    def test_custom_values(self, temp_data_dir):
        """Test custom configuration values."""
        manager = DaemonManager(
            data_dir=temp_data_dir,
            host="0.0.0.0",
            port=8080,
        )

        assert manager.host == "0.0.0.0"
        assert manager.port == 8080
        assert manager.data_dir == temp_data_dir

    def test_creates_directories(self, temp_data_dir):
        """Test that log directory is created."""
        manager = DaemonManager(data_dir=temp_data_dir)

        assert manager.log_dir.exists()
        assert manager.log_dir == temp_data_dir / "logs"


class TestUrlProperty:
    """Tests for url property."""

    def test_get_url_default(self, manager):
        """Test getting URL with default host/port."""
        assert manager.url == "http://127.0.0.1:5173"

    def test_get_url_custom_port(self, temp_data_dir):
        """Test getting URL with custom port."""
        manager = DaemonManager(data_dir=temp_data_dir, port=8080)

        assert manager.url == "http://127.0.0.1:8080"

    def test_get_url_custom_host(self, temp_data_dir):
        """Test getting URL with custom host."""
        manager = DaemonManager(data_dir=temp_data_dir, host="0.0.0.0")

        assert manager.url == "http://0.0.0.0:5173"


class TestPidFileOperations:
    """Tests for PID file operations."""

    def test_read_pid_file_exists(self, manager):
        """Test reading existing PID file."""
        manager.pid_file.write_text("12345")

        pid = manager._read_pid()

        assert pid == 12345

    def test_read_pid_file_missing(self, manager):
        """Test reading when PID file doesn't exist."""
        pid = manager._read_pid()

        assert pid is None

    def test_read_pid_invalid_content(self, manager):
        """Test reading PID file with invalid content."""
        manager.pid_file.write_text("not-a-number")

        pid = manager._read_pid()

        assert pid is None

    def test_write_pid(self, manager):
        """Test writing PID file."""
        manager._write_pid(54321)

        assert manager.pid_file.read_text() == "54321"

    def test_remove_pid(self, manager):
        """Test removing PID file."""
        manager.pid_file.write_text("12345")
        assert manager.pid_file.exists()

        manager._remove_pid()

        assert not manager.pid_file.exists()

    def test_remove_pid_missing(self, manager):
        """Test removing non-existent PID file doesn't raise."""
        manager._remove_pid()  # Should not raise


class TestIsProcessRunning:
    """Tests for process running detection."""

    def test_process_running(self, manager):
        """Test detecting running process."""
        current_pid = os.getpid()

        result = manager._is_process_running(current_pid)

        assert result is True

    def test_process_not_running(self, manager):
        """Test detecting non-existent process."""
        # Use an unlikely PID
        fake_pid = 999999

        result = manager._is_process_running(fake_pid)

        assert result is False


class TestIsPortInUse:
    """Tests for port usage detection."""

    def test_port_not_in_use(self, temp_data_dir):
        """Test detecting unused port."""
        # Use a random high port unlikely to be in use
        manager = DaemonManager(data_dir=temp_data_dir, port=59999)

        result = manager._is_port_in_use()

        assert result is False

    def test_is_port_in_use_mock(self, manager):
        """Test detecting used port with mock."""
        with patch.object(socket.socket, "bind") as mock_bind:
            mock_bind.side_effect = OSError("Address in use")

            result = manager._is_port_in_use()

            assert result is True


class TestCheckHealth:
    """Tests for health check endpoint."""

    def test_check_health_success(self, manager):
        """Test successful health check."""
        with patch("httpx.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            result = manager._check_health()

            assert result is True
            mock_get.assert_called_once_with(
                "http://127.0.0.1:5173/health", timeout=5.0
            )

    def test_check_health_failure(self, manager):
        """Test failed health check."""
        with patch("httpx.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_get.return_value = mock_response

            result = manager._check_health()

            assert result is False

    def test_check_health_connection_error(self, manager):
        """Test health check with connection error."""
        with patch("httpx.get") as mock_get:
            mock_get.side_effect = Exception("Connection refused")

            result = manager._check_health()

            assert result is False


class TestStatusNotRunning:
    """Tests for status when daemon is not running."""

    def test_status_no_pid_file(self, manager):
        """Test status when no PID file exists."""
        status = manager.status()

        assert status.running is False
        assert status.pid is None
        assert status.url is None

    def test_status_stale_pid_file(self, manager):
        """Test status with stale PID file (process not running)."""
        manager.pid_file.write_text("999999")  # Non-existent PID

        status = manager.status()

        assert status.running is False
        assert status.pid is None
        # Stale PID file should be cleaned up
        assert not manager.pid_file.exists()


class TestStatusRunning:
    """Tests for status when daemon is running."""

    def test_status_running(self, manager):
        """Test status when daemon is running."""
        current_pid = os.getpid()
        manager.pid_file.write_text(str(current_pid))

        with patch.object(manager, "_is_port_in_use", return_value=True):
            with patch("httpx.get") as mock_get:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"uptime_seconds": 3600.0}
                mock_get.return_value = mock_response

                status = manager.status()

        assert status.running is True
        assert status.pid == current_pid
        assert status.url == "http://127.0.0.1:5173"
        assert status.uptime_seconds == 3600.0

    def test_status_running_health_fails(self, manager):
        """Test status when process runs but health check fails."""
        current_pid = os.getpid()
        manager.pid_file.write_text(str(current_pid))

        with patch.object(manager, "_is_port_in_use", return_value=True):
            with patch("httpx.get") as mock_get:
                mock_get.side_effect = Exception("Connection error")

                status = manager.status()

        assert status.running is True
        assert status.pid == current_pid
        assert status.uptime_seconds is None  # Failed to get uptime


class TestIsRunning:
    """Tests for is_running() method."""

    def test_is_running_true(self, manager):
        """Test is_running returns True when daemon is up."""
        current_pid = os.getpid()
        manager.pid_file.write_text(str(current_pid))

        with patch.object(manager, "_is_port_in_use", return_value=True):
            result = manager.is_running()

        assert result is True

    def test_is_running_false_no_pid(self, manager):
        """Test is_running returns False when no PID file."""
        result = manager.is_running()

        assert result is False

    def test_is_running_cleans_stale_pid(self, manager):
        """Test is_running cleans stale PID file."""
        manager.pid_file.write_text("999999")

        result = manager.is_running()

        assert result is False
        assert not manager.pid_file.exists()


class TestGetPid:
    """Tests for get_pid() method."""

    def test_get_pid_running(self, manager):
        """Test getting PID when running."""
        current_pid = os.getpid()
        manager.pid_file.write_text(str(current_pid))

        with patch.object(manager, "_is_port_in_use", return_value=True):
            pid = manager.get_pid()

        assert pid == current_pid

    def test_get_pid_not_running(self, manager):
        """Test getting PID when not running."""
        pid = manager.get_pid()

        assert pid is None


class TestStart:
    """Tests for daemon start."""

    def test_start_already_running(self, manager):
        """Test start when daemon already running."""
        with patch.object(manager, "is_running", return_value=True):
            result = manager.start()

        assert result is True

    def test_start_port_in_use(self, manager):
        """Test start fails when port is already in use."""
        with patch.object(manager, "is_running", return_value=False):
            with patch.object(manager, "_is_port_in_use", return_value=True):
                result = manager.start()

        assert result is False

    def test_start_background(self, manager):
        """Test starting daemon in background."""
        with patch.object(manager, "is_running", return_value=False):
            with patch.object(manager, "_is_port_in_use", return_value=False):
                with patch.object(manager, "_check_health", return_value=True):
                    with patch("subprocess.Popen") as mock_popen:
                        mock_process = MagicMock()
                        mock_process.pid = 12345
                        mock_process.poll.return_value = None
                        mock_popen.return_value = mock_process

                        result = manager.start(background=True, wait=True, timeout=1.0)

        assert result is True
        assert manager.pid_file.read_text() == "12345"

    def test_start_background_health_timeout(self, manager):
        """Test start when health check times out."""
        with patch.object(manager, "is_running", return_value=False):
            with patch.object(manager, "_is_port_in_use", return_value=False):
                with patch.object(manager, "_check_health", return_value=False):
                    with patch("subprocess.Popen") as mock_popen:
                        mock_process = MagicMock()
                        mock_process.pid = 12345
                        mock_process.poll.return_value = None  # Still running
                        mock_popen.return_value = mock_process

                        result = manager.start(background=True, wait=True, timeout=0.5)

        # Should still return True (process started, just health timed out)
        assert result is True

    def test_start_background_process_exits(self, manager):
        """Test start when process exits immediately."""
        with patch.object(manager, "is_running", return_value=False):
            with patch.object(manager, "_is_port_in_use", return_value=False):
                with patch.object(manager, "_check_health", return_value=False):
                    with patch("subprocess.Popen") as mock_popen:
                        mock_process = MagicMock()
                        mock_process.pid = 12345
                        mock_process.poll.return_value = 1  # Process exited
                        mock_process.returncode = 1
                        mock_popen.return_value = mock_process

                        result = manager.start(background=True, wait=True, timeout=0.5)

        assert result is False
        # PID file should be cleaned up
        assert not manager.pid_file.exists()


class TestStop:
    """Tests for daemon stop."""

    def test_stop_no_pid_file(self, manager):
        """Test stop when no PID file exists."""
        result = manager.stop()

        assert result is True

    def test_stop_process_not_running(self, manager):
        """Test stop when process is not running."""
        manager.pid_file.write_text("999999")

        result = manager.stop()

        assert result is True
        assert not manager.pid_file.exists()

    def test_stop_graceful(self, manager):
        """Test graceful stop with SIGTERM."""
        current_pid = os.getpid()

        with patch.object(manager, "_is_process_running") as mock_running:
            # Process running initially, then stops
            mock_running.side_effect = [True, False]
            with patch("os.kill") as mock_kill:
                manager.pid_file.write_text(str(current_pid))

                result = manager.stop(timeout=0.5)

        mock_kill.assert_called_with(current_pid, signal.SIGTERM)
        assert result is True

    def test_stop_force_kill(self, manager):
        """Test force kill when SIGTERM doesn't work."""
        fake_pid = 999998

        with patch.object(manager, "_is_process_running") as mock_running:
            # Process keeps running until SIGKILL
            mock_running.side_effect = [True, True, True, True, False]
            with patch("os.kill") as mock_kill:
                manager.pid_file.write_text(str(fake_pid))

                result = manager.stop(timeout=0.5)

        # Should have sent both SIGTERM and SIGKILL
        calls = mock_kill.call_args_list
        assert any(call[0][1] == signal.SIGTERM for call in calls)
        assert any(call[0][1] == signal.SIGKILL for call in calls)


class TestRestart:
    """Tests for daemon restart."""

    def test_restart(self, manager):
        """Test restart stops and starts daemon."""
        with patch.object(manager, "stop", return_value=True) as mock_stop:
            with patch.object(manager, "start", return_value=True) as mock_start:
                result = manager.restart(background=True)

        mock_stop.assert_called_once()
        mock_start.assert_called_once_with(background=True)
        assert result is True


class TestLogs:
    """Tests for log retrieval."""

    def test_logs_no_file(self, manager):
        """Test logs when no log file exists."""
        result = manager.logs()

        assert result is None

    def test_logs_read_lines(self, manager):
        """Test reading log lines."""
        log_content = "\n".join([f"Line {i}" for i in range(200)])
        manager.log_file.write_text(log_content)

        result = manager.logs(lines=50)

        lines = result.split("\n")
        assert len(lines) == 50
        assert "Line 199" in result

    def test_logs_follow(self, manager):
        """Test following logs calls tail -f."""
        manager.log_file.write_text("test log")

        with patch("subprocess.run") as mock_run:
            manager.logs(follow=True)

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args == ["tail", "-f", str(manager.log_file)]


class TestGetDaemonManager:
    """Tests for get_daemon_manager factory function."""

    def test_get_daemon_manager(self):
        """Test getting default daemon manager."""
        from chunkhound.services.daemon_manager import get_daemon_manager

        manager = get_daemon_manager()

        assert isinstance(manager, DaemonManager)
        assert manager.host == "127.0.0.1"
        assert manager.port == 5173
