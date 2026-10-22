"""Contract tests for the offline runner's isolation and cancellation.

The autouse hermetic fixture changes the process cwd, so the runner path must
be absolute and the assertions must separate a blocked network from a broken
runner.
"""

import os
import selectors
import signal
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_offline.sh"

# Explicit str annotation keeps mypy from folding sys.platform to a literal
# (which would flag the sudo probe in _require_linux as unreachable).
_PLATFORM: str = sys.platform

BLOCKS_EXTERNAL = """
import socket

try:
    socket.create_connection(("8.8.8.8", 53), timeout=2)
except OSError as exc:
    print(type(exc).__name__)
else:
    raise SystemExit("external connection unexpectedly succeeded")
"""

ALLOWS_LOOPBACK = """
import socket

server = socket.socket()
server.bind(("127.0.0.1", 0))
server.listen(1)
port = server.getsockname()[1]

client = socket.create_connection(("127.0.0.1", port), timeout=2)
connection, _ = server.accept()
connection.close()
client.close()
server.close()
print("loopback connection successful")
"""

ALLOWS_FILESYSTEM = """
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as directory:
    path = Path(directory) / "probe"
    path.write_bytes(b"test")
    assert path.read_bytes() == b"test"
    print(path)
"""

ALLOWS_UNIX_SOCKET = """
import os
import socket
from tempfile import TemporaryDirectory

with TemporaryDirectory() as directory:
    path = os.path.join(directory, "unix.sock")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(path)
        server.listen(1)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(path)
            connection, _ = server.accept()
            connection.close()
print("unix socket connection successful")
"""

EXIT_CODE = """
raise SystemExit(7)
"""

STDIN = """
import sys

data = sys.stdin.read()
assert data == "offline-stdin", data
print("stdin ok")
"""

CANCELLATION = """
import os
import signal
import socket
import sys

# TERM-resistant descendants require namespace teardown, not signal forwarding.
signal.signal(signal.SIGTERM, signal.SIG_IGN)
signal.signal(signal.SIGINT, signal.SIG_IGN)
descendant = os.fork()
signal.alarm(60)  # Last-resort cleanup if the test process itself dies.
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
    connection.settimeout(30)
    connection.connect(sys.argv[1])
    descriptor = os.pidfd_open(os.getpid())
    try:
        socket.send_fds(connection, [b"R"], [descriptor])
    finally:
        os.close(descriptor)
    # Keep both processes alive until cancellation; EOF also cleans up failures.
    connection.recv(1)
if descendant:
    os.waitpid(descendant, 0)
"""


def _run_offline(
    source: str, timeout: int = 10, input_text: str | None = None
) -> subprocess.CompletedProcess[str]:
    assert RUNNER.is_file(), f"offline runner missing: {RUNNER}"
    return subprocess.run(
        ["bash", str(RUNNER), sys.executable, "-c", source],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _require_linux() -> None:
    if _PLATFORM != "linux":
        pytest.skip("Offline runner network isolation requires Linux")
    # run_offline.sh needs passwordless sudo (CI has it; local dev may not).
    try:
        probe = subprocess.run(["sudo", "-n", "true"], capture_output=True, timeout=10)
    except FileNotFoundError:
        probe = None
    if probe is None or probe.returncode != 0:
        # A CI runner must never silently drop this contract suite.
        if os.getenv("CI"):
            pytest.fail(
                "run_offline.sh requires passwordless sudo (CI sudoers regression)"
            )
        pytest.skip("run_offline.sh requires passwordless sudo (sudoers config)")


_pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)


def _kill_payload(descriptor: int) -> None:
    if _pidfd_send_signal is None:
        return
    try:
        _pidfd_send_signal(descriptor, signal.SIGKILL)
    except ProcessLookupError:
        pass  # An exited process needs no further cleanup.


def _stop_runner(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _payload_pidfd(listener: socket.socket, cleanup: ExitStack) -> int:
    connection, _ = listener.accept()
    cleanup.enter_context(connection)
    connection.settimeout(10)
    message, descriptors, flags, _ = socket.recv_fds(connection, 1, 1)
    for descriptor in descriptors:
        cleanup.callback(os.close, descriptor)
        cleanup.callback(_kill_payload, descriptor)
    assert message == b"R" and len(descriptors) == 1, message
    assert not flags & socket.MSG_CTRUNC, "payload pidfd was truncated"
    # SCM_RIGHTS transfers the exact process handle across PID namespaces.
    return descriptors[0]


def _assert_payload_exited(selector: selectors.BaseSelector) -> None:
    deadline = time.monotonic() + 10
    while selector.get_map():
        remaining = deadline - time.monotonic()
        assert remaining > 0, "offline runner left a surviving payload process"
        for event, _ in selector.select(remaining):
            selector.unregister(event.fd)


def _cancel_ready_payload(
    process: subprocess.Popen[bytes], descriptors: list[int]
) -> None:
    with selectors.DefaultSelector() as selector:
        for descriptor in descriptors:
            selector.register(descriptor, selectors.EVENT_READ)
        assert not selector.select(0), "payload exited before cancellation"
        process.terminate()
        assert process.wait(timeout=10) == 143
        _assert_payload_exited(selector)


def test_offline_runner_cancellation_terminates_descendants(tmp_path: Path) -> None:
    """Cancellation returns TERM status and leaves neither child nor descendant."""
    _require_linux()
    with ExitStack() as cleanup:
        listener = cleanup.enter_context(
            socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        )
        listener.settimeout(10)
        address = str(tmp_path / "cancellation.sock")
        listener.bind(address)
        listener.listen(2)
        process = subprocess.Popen(
            ["bash", str(RUNNER), sys.executable, "-c", CANCELLATION, address]
        )
        cleanup.callback(_stop_runner, process)
        descriptors = [_payload_pidfd(listener, cleanup) for _ in range(2)]
        _cancel_ready_payload(process, descriptors)


def test_offline_runner_blocks_external_network() -> None:
    """An outbound connection fails inside the runner's network namespace."""
    _require_linux()
    result = _run_offline(BLOCKS_EXTERNAL)
    assert result.returncode == 0, result.stderr
    assert "error" in result.stdout.lower(), result.stdout


def test_offline_runner_allows_loopback() -> None:
    """Loopback stays reachable inside the namespace."""
    _require_linux()
    result = _run_offline(ALLOWS_LOOPBACK)
    assert result.returncode == 0, result.stderr
    assert "loopback connection successful" in result.stdout


def test_offline_runner_preserves_filesystem_access() -> None:
    """The namespace must not break ordinary filesystem operations."""
    _require_linux()
    result = _run_offline(ALLOWS_FILESYSTEM)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


def test_offline_runner_unix_sockets_work() -> None:
    """Unix sockets do not use the network stack and must still work."""
    _require_linux()
    result = _run_offline(ALLOWS_UNIX_SOCKET)
    assert result.returncode == 0, result.stderr
    assert "unix socket connection successful" in result.stdout


def test_offline_runner_propagates_exit_code() -> None:
    """The runner must return the child's status, not sudo's or bash's."""
    _require_linux()
    result = _run_offline(EXIT_CODE)
    assert result.returncode == 7, result.stderr


def test_offline_runner_passes_stdin() -> None:
    """The namespace wrapper must keep the caller's stdin attached."""
    _require_linux()
    result = _run_offline(STDIN, input_text="offline-stdin")
    assert result.returncode == 0, result.stderr
    assert "stdin ok" in result.stdout


@pytest.mark.skipif(sys.platform == "linux", reason="requires non-Linux platform")
def test_offline_runner_refuses_non_linux() -> None:
    """On non-Linux hosts the runner exits immediately with a clear message."""
    result = _run_offline(BLOCKS_EXTERNAL)
    assert result.returncode == 1, result.stderr
    assert "Offline network namespaces require Linux" in result.stderr
