"""Daemon smoke tests — verifies multi-client MCP daemon architecture.

These tests cover the multi-client use case: multiple Claude instances
sharing one DuckDB connection via the ChunkHound daemon.

Run with:
    uv run pytest tests/test_daemon_smoke.py -v

IMPORTANT: These tests require the daemon to start as a subprocess, so they
take longer than unit tests.  They do NOT run with -n auto because daemon
cleanup timing is hard to parallelize safely across test processes.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path

import psutil
import pytest

from tests.helpers.daemon_test_helpers import (
    runtime_dir_env,
    wait_for_daemon_full_cleanup,
    wait_for_daemon_shutdown,
    wait_for_daemon_start,
)
from tests.utils import (
    SubprocessJsonRpcClient,
    create_subprocess_exec_safe,
    get_safe_subprocess_env,
)


def _chunkhound_exe() -> str:
    """Return the absolute path to the chunkhound executable in the active venv."""
    exe = shutil.which("chunkhound")
    if exe is None:
        raise RuntimeError(
            "chunkhound not found in PATH — run tests via 'uv run pytest'"
        )
    return exe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MCP_INIT_PARAMS = {
    "protocolVersion": "2024-11-05",
    "clientInfo": {"name": "test-client", "version": "0.0.1"},
    "capabilities": {},
}

_SEARCH_REGEX_PARAMS = {
    "name": "search",
    "arguments": {"query": "def authenticate", "type": "regex"},
}


def _registry_dir(project_dir: Path, runtime_dir: Path) -> Path:
    """Return the daemon registry directory under a test runtime directory."""
    from chunkhound.daemon.discovery import DaemonDiscovery

    with runtime_dir_env(runtime_dir):
        return DaemonDiscovery(project_dir).get_registry_dir()


def _make_env(
    project_dir: Path,
    *,
    runtime_dir: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return a clean subprocess environment pointing to project_dir."""
    env = get_safe_subprocess_env(os.environ.copy())
    # Clear any stale CHUNKHOUND_* vars that could interfere
    for key in list(env.keys()):
        if key.startswith("CHUNKHOUND_"):
            del env[key]
    env["CHUNKHOUND_MCP_MODE"] = "1"
    if runtime_dir is not None:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        env["CHUNKHOUND_DAEMON_RUNTIME_DIR"] = str(runtime_dir)
    if extra_env is not None:
        env.update(extra_env)
    return env


async def _start_proxy_process(
    project_dir: Path,
    *,
    runtime_dir: Path | None = None,
    stdin: int | None = asyncio.subprocess.PIPE,
    capture_stderr: bool = False,
    extra_env: dict[str, str] | None = None,
) -> asyncio.subprocess.Process:
    """Launch a ``chunkhound mcp`` proxy subprocess and return the process."""
    env = _make_env(project_dir, runtime_dir=runtime_dir, extra_env=extra_env)
    stderr = asyncio.subprocess.PIPE if capture_stderr else asyncio.subprocess.DEVNULL
    return await create_subprocess_exec_safe(
        _chunkhound_exe(),
        "mcp",
        str(project_dir),
        stdin=stdin,
        stdout=asyncio.subprocess.PIPE,
        stderr=stderr,
        env=env,
        cwd=str(project_dir),
    )


async def _start_proxy(
    project_dir: Path,
    *,
    runtime_dir: Path | None = None,
    capture_stderr: bool = False,
    extra_env: dict[str, str] | None = None,
) -> tuple[asyncio.subprocess.Process, SubprocessJsonRpcClient]:
    """Launch a ``chunkhound mcp`` proxy subprocess and return (proc, client)."""
    proc = await _start_proxy_process(
        project_dir,
        runtime_dir=runtime_dir,
        capture_stderr=capture_stderr,
        extra_env=extra_env,
    )
    client = SubprocessJsonRpcClient(proc)
    await client.start()
    return proc, client


async def _run_proxy_to_failure(
    project_dir: Path,
    *,
    runtime_dir: Path | None = None,
    extra_env: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> tuple[int, str, str]:
    """Run ``chunkhound mcp`` until it exits and capture stdio."""
    proc = await _start_proxy_process(
        project_dir,
        runtime_dir=runtime_dir,
        stdin=asyncio.subprocess.DEVNULL,
        capture_stderr=True,
        extra_env=extra_env,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    return proc.returncode or 0, stdout.decode(), stderr.decode()


async def _do_mcp_handshake(
    client: SubprocessJsonRpcClient, timeout: float = 30.0
) -> dict:
    """Send initialize + initialized notification; return server capabilities."""
    result = await client.send_request("initialize", _MCP_INIT_PARAMS, timeout=timeout)
    await client.send_notification("notifications/initialized", {})
    return result


async def _teardown_proxy(
    proc: asyncio.subprocess.Process,
    client: SubprocessJsonRpcClient,
) -> None:
    """Gracefully shut down a proxy subprocess."""
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


def _write_project_files(project_dir: Path) -> None:
    """Create a minimal test project with config and source files."""
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "auth.py").write_text(
        "def authenticate(user, password):\n"
        '    """Authenticate a user."""\n'
        "    return user == 'admin' and password == 'secret'\n"
        "\n"
        "def logout(session):\n"
        "    session.clear()\n"
    )
    (project_dir / "search.py").write_text(
        "def search_items(query, index):\n"
        '    """Search items in the index."""\n'
        "    results = []\n"
        "    for item in index:\n"
        "        if query.lower() in item.lower():\n"
        "            results.append(item)\n"
        "    return results\n"
    )
    (project_dir / "utils.py").write_text(
        "def format_result(result):\n"
        '    """Format a search result."""\n'
        "    return str(result)\n"
        "\n"
        "def validate_input(data):\n"
        "    return data is not None\n"
    )

    db_path = project_dir / ".chunkhound" / "test.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "database": {"path": str(db_path), "provider": "duckdb"},
        "indexing": {"include": ["*.py"]},
    }
    (project_dir / ".chunkhound.json").write_text(json.dumps(config))


async def _prepare_project_dir(project_dir: Path) -> None:
    """Create and index a minimal project for daemon smoke tests."""
    _write_project_files(project_dir)
    index_proc = await create_subprocess_exec_safe(
        _chunkhound_exe(),
        "index",
        str(project_dir),
        "--no-embeddings",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(project_dir),
        env=get_safe_subprocess_env(),
    )
    _, stderr = await asyncio.wait_for(index_proc.communicate(), timeout=60.0)
    assert index_proc.returncode == 0, (
        f"Pre-indexing failed (rc={index_proc.returncode}): {stderr.decode()}"
    )


def _cleanup_project_dir(
    project_dir: Path,
    *,
    runtime_dir: Path | None = None,
) -> None:
    """Stop any lingering daemon for a test project and remove local artifacts."""
    from chunkhound.daemon.discovery import DaemonDiscovery

    with runtime_dir_env(runtime_dir):
        discovery = DaemonDiscovery(project_dir)
    lock = discovery.read_lock()
    if lock:
        pid = lock.get("pid")
        if isinstance(pid, int):
            try:
                proc = psutil.Process(pid)
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except psutil.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            except Exception:
                pass
        discovery.remove_lock()
    socket_path = discovery.get_socket_path()
    if not socket_path.startswith("tcp:"):
        try:
            os.unlink(socket_path)
        except Exception:
            pass


async def _wait_for_registry_entry(
    runtime_dir: Path,
    project_dir: Path,
    timeout: float = 10.0,
) -> Path | None:
    """Wait until a registry entry appears for the canonical project root."""
    deadline = asyncio.get_running_loop().time() + timeout
    expected_root = str(project_dir.resolve())
    registry_dir = _registry_dir(project_dir, runtime_dir)
    while asyncio.get_running_loop().time() < deadline:
        if registry_dir.exists():
            for entry_path in registry_dir.glob("*.json"):
                try:
                    data = json.loads(entry_path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if data.get("project_dir") == expected_root:
                    return entry_path
        await asyncio.sleep(0.1)
    return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def pre_indexed_project_dir(tmp_path: Path) -> AsyncIterator[Path]:
    """Create a minimal pre-indexed project directory.

    Creates three synthetic Python files, writes a .chunkhound.json config,
    and indexes the files without embeddings so that regex search works.
    The fixture does NOT require an embedding API key.
    """
    await _prepare_project_dir(tmp_path)

    yield tmp_path

    _cleanup_project_dir(tmp_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_daemon_single_client_basic(pre_indexed_project_dir: Path) -> None:
    """Single client connects via daemon proxy, performs regex search.

    Verifies the basic daemon lifecycle:
    - Daemon auto-starts on first proxy connection.
    - Lock file is created.
    - MCP tools/call works end-to-end through the proxy.
    - Daemon shuts down after the single client disconnects.
    """
    project_dir = pre_indexed_project_dir
    proc, client = await _start_proxy(project_dir)

    try:
        # --- Handshake ---
        init_result = await _do_mcp_handshake(client, timeout=30.0)
        assert "capabilities" in init_result, (
            f"initialize response missing 'capabilities': {init_result}"
        )

        # --- Wait for daemon to be reachable ---
        daemon_started = await wait_for_daemon_start(project_dir, timeout=15.0)
        assert daemon_started, "Daemon did not start within 15 seconds"

        # --- Regex search (no API key required) ---
        result = await client.send_request(
            "tools/call", _SEARCH_REGEX_PARAMS, timeout=20.0
        )
        assert "content" in result, f"tools/call result missing 'content': {result}"
        content = result["content"]
        assert isinstance(content, list) and len(content) > 0

        text = content[0].get("text", "")
        # Should find 'def authenticate' in auth.py
        assert "auth" in text.lower() or "authenticate" in text.lower(), (
            f"Expected auth-related results in search output: {text[:500]}"
        )

    finally:
        await _teardown_proxy(proc, client)

    # After the proxy disconnects, the daemon should shut down
    # Windows requires significantly more time for cleanup
    # (see test_daemon_lock_file_created comments).
    stopped = await wait_for_daemon_shutdown(project_dir, timeout=30.0)
    assert stopped, "Daemon did not shut down after last client disconnected"


@pytest.mark.asyncio
async def test_daemon_two_clients_concurrent(pre_indexed_project_dir: Path) -> None:
    """Two proxy clients connect to the SAME daemon concurrently.

    This is the critical test that proves the DuckDB lock conflict is resolved.
    Without the daemon, opening two 'chunkhound mcp' processes on the same
    directory would cause one of them to fail with a DuckDB file-lock error.

    With the daemon, both clients share a single DuckDB connection and can
    query concurrently.
    """
    project_dir = pre_indexed_project_dir

    # Start first proxy (this triggers daemon auto-start)
    proc1, client1 = await _start_proxy(project_dir)
    # Start second proxy immediately (should connect to same daemon)
    proc2, client2 = await _start_proxy(project_dir)

    try:
        # Both proxies perform the MCP handshake concurrently
        init1, init2 = await asyncio.gather(
            _do_mcp_handshake(client1, timeout=30.0),
            _do_mcp_handshake(client2, timeout=30.0),
        )
        assert "capabilities" in init1, f"Client 1 init missing capabilities: {init1}"
        assert "capabilities" in init2, f"Client 2 init missing capabilities: {init2}"

        # Verify both are talking to the same daemon (only one should be running)
        daemon_started = await wait_for_daemon_start(project_dir, timeout=15.0)
        assert daemon_started, "Daemon did not start within 15 seconds"

        # Both clients issue regex search concurrently — THIS WOULD FAIL without daemon
        result1, result2 = await asyncio.gather(
            client1.send_request("tools/call", _SEARCH_REGEX_PARAMS, timeout=20.0),
            client2.send_request("tools/call", _SEARCH_REGEX_PARAMS, timeout=20.0),
        )

        # Both should get valid responses
        assert "content" in result1, f"Client 1 result missing 'content': {result1}"
        assert "content" in result2, f"Client 2 result missing 'content': {result2}"
        assert len(result1["content"]) > 0, "Client 1 got empty content"
        assert len(result2["content"]) > 0, "Client 2 got empty content"

        # Both should have found the same data
        text1 = result1["content"][0].get("text", "")
        text2 = result2["content"][0].get("text", "")
        assert text1 == text2, (
            "Both clients should get identical results from the same daemon\n"
            f"Client 1: {text1[:200]}\nClient 2: {text2[:200]}"
        )

    finally:
        await asyncio.gather(
            _teardown_proxy(proc1, client1),
            _teardown_proxy(proc2, client2),
            return_exceptions=True,
        )

    # After both clients disconnect, daemon should shut down
    # Windows requires significantly more time for cleanup
    # (see test_daemon_lock_file_created comments).
    stopped = await wait_for_daemon_shutdown(project_dir, timeout=30.0)
    assert stopped, "Daemon did not shut down after all clients disconnected"


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Daemon shutdown is unreliable on Windows due to process termination timing",
)
async def test_daemon_lock_file_created(pre_indexed_project_dir: Path) -> None:
    """Verify the daemon lock file is created with correct content and cleaned up.

    Checks:
    - Lock file is created at .chunkhound/daemon.lock.
    - Lock file contains valid JSON with 'pid', 'socket_path', 'started_at'.
    - Recorded PID matches a live process.
    - Lock file is removed after the daemon shuts down.

    Note: Skipped on Windows because process termination doesn't reliably trigger
    graceful shutdown, causing the lock file cleanup to be inconsistent.
    """
    project_dir = pre_indexed_project_dir
    lock_path = project_dir / ".chunkhound" / "daemon.lock"

    proc, client = await _start_proxy(project_dir)
    try:
        # Trigger daemon startup via handshake
        await _do_mcp_handshake(client, timeout=30.0)

        # Wait for daemon to be live
        daemon_started = await wait_for_daemon_start(project_dir, timeout=15.0)
        assert daemon_started, "Daemon did not start within 15 seconds"

        # Verify lock file exists
        assert lock_path.exists(), f"Lock file not found at {lock_path}"

        # Parse and validate lock file contents
        lock_data = json.loads(lock_path.read_text())
        assert "pid" in lock_data, f"Lock file missing 'pid': {lock_data}"
        assert "socket_path" in lock_data, (
            f"Lock file missing 'socket_path': {lock_data}"
        )
        assert "started_at" in lock_data, f"Lock file missing 'started_at': {lock_data}"
        assert "project_dir" in lock_data, (
            f"Lock file missing 'project_dir': {lock_data}"
        )
        assert lock_data["project_dir"] == str(project_dir.resolve())

        pid = lock_data["pid"]
        assert isinstance(pid, int) and pid > 0, f"Invalid PID in lock file: {pid}"

        # PID should belong to a live process (cross-platform check)
        try:
            proc = psutil.Process(pid)
            assert proc.is_running(), f"Lock file PID {pid} exists but is not running"
        except psutil.NoSuchProcess:
            pytest.fail(f"Lock file PID {pid} is not a live process")

        # Socket path should exist (Unix) or be a valid TCP address (Windows)
        socket_path = lock_data["socket_path"]
        if socket_path.startswith("tcp:"):
            # On Windows: verify it's a valid TCP address format
            assert socket_path.startswith("tcp:127.0.0.1:"), (
                f"Invalid TCP address format in lock file: {socket_path}"
            )
        else:
            # On Unix: verify the socket file exists
            assert os.path.exists(socket_path), (
                f"Socket file '{socket_path}' listed in lock file does not exist"
            )

    finally:
        await _teardown_proxy(proc, client)

    # After client disconnects, lock file should be removed
    # Windows requires significantly more time for cleanup due to:
    # 1. Background scan task cancellation and cleanup
    # 2. Realtime indexing observer.join() timeout
    # 3. DuckDB connection close (can be slow on Windows)
    # 4. Process termination overhead
    stopped = await wait_for_daemon_shutdown(project_dir, timeout=30.0)
    assert stopped, "Daemon did not shut down in time"
    assert not lock_path.exists(), (
        f"Lock file was not cleaned up after daemon shutdown: {lock_path}"
    )


@pytest.mark.asyncio
async def test_watchman_start_failure_returns_fast_and_skips_lock_publication(
    pre_indexed_project_dir: Path,
) -> None:
    """Watchman startup failures should surface quickly without publishing a lock."""
    project_dir = pre_indexed_project_dir
    config_path = project_dir / ".chunkhound.json"
    config = json.loads(config_path.read_text())
    config.setdefault("indexing", {})["realtime_backend"] = "watchman"
    config_path.write_text(json.dumps(config))

    start_time = time.monotonic()
    returncode, _stdout, stderr_text = await _run_proxy_to_failure(
        project_dir,
        extra_env={"CHUNKHOUND_FAKE_WATCHMAN_FAIL_BEFORE_READY": "1"},
        timeout=20.0,
    )
    elapsed = time.monotonic() - start_time

    assert returncode != 0, "Proxy should fail when watchman startup fails"
    assert elapsed < 20.0, f"Expected fail-fast startup error, got {elapsed:.2f}s"
    assert "Watchman sidecar startup failed" in stderr_text
    assert "Recent daemon log output" in stderr_text

    lock_path = project_dir / ".chunkhound" / "daemon.lock"
    assert not lock_path.exists(), (
        "Daemon lock should not be published on watchman failure"
    )
    assert not await wait_for_daemon_start(project_dir, timeout=1.0)

    daemon_log_path = project_dir / ".chunkhound" / "daemon.log"
    assert daemon_log_path.exists()
    daemon_log_text = daemon_log_path.read_text(encoding="utf-8", errors="replace")
    assert "Watchman sidecar startup failed" in daemon_log_text


@pytest.mark.asyncio
async def test_daemon_sibling_roots_allowed(tmp_path: Path) -> None:
    """Sibling project roots should start separate daemons without conflict."""
    root_a = tmp_path / "repo-a"
    root_b = tmp_path / "repo-b"
    runtime_dir = tmp_path / "runtime"
    await _prepare_project_dir(root_a)
    await _prepare_project_dir(root_b)

    proc_a = client_a = proc_b = client_b = None
    try:
        proc_a, client_a = await _start_proxy(root_a, runtime_dir=runtime_dir)
        await _do_mcp_handshake(client_a, timeout=30.0)
        assert await wait_for_daemon_start(root_a, timeout=15.0)

        proc_b, client_b = await _start_proxy(root_b, runtime_dir=runtime_dir)
        await _do_mcp_handshake(client_b, timeout=30.0)
        assert await wait_for_daemon_start(root_b, timeout=15.0)
    finally:
        await asyncio.gather(
            *[
                _teardown_proxy(proc, client)
                for proc, client in ((proc_a, client_a), (proc_b, client_b))
                if proc is not None and client is not None
            ],
            return_exceptions=True,
        )
        _cleanup_project_dir(root_a, runtime_dir=runtime_dir)
        _cleanup_project_dir(root_b, runtime_dir=runtime_dir)


@pytest.mark.asyncio
async def test_daemon_parent_then_child_blocks(tmp_path: Path) -> None:
    """A nested child root should be rejected while parent daemon is live."""
    parent = tmp_path / "repo"
    child = parent / "subdir"
    runtime_dir = tmp_path / "runtime"
    await _prepare_project_dir(parent)
    await _prepare_project_dir(child)

    proc = client = None
    try:
        proc, client = await _start_proxy(parent, runtime_dir=runtime_dir)
        await _do_mcp_handshake(client, timeout=30.0)
        assert await wait_for_daemon_start(parent, timeout=15.0)

        returncode, _, stderr = await _run_proxy_to_failure(
            child,
            runtime_dir=runtime_dir,
        )
        assert returncode != 0
        assert "Overlapping daemon roots are not supported." in stderr
        assert str(parent.resolve()) in stderr
        assert str(child.resolve()) in stderr
        assert await wait_for_daemon_full_cleanup(
            child,
            runtime_dir=runtime_dir,
            timeout=30.0,
        )
    finally:
        if proc is not None and client is not None:
            await _teardown_proxy(proc, client)
        _cleanup_project_dir(child, runtime_dir=runtime_dir)
        _cleanup_project_dir(parent, runtime_dir=runtime_dir)


@pytest.mark.asyncio
async def test_daemon_child_then_parent_blocks(tmp_path: Path) -> None:
    """A parent root should be rejected while child daemon is live."""
    parent = tmp_path / "repo"
    child = parent / "subdir"
    runtime_dir = tmp_path / "runtime"
    await _prepare_project_dir(parent)
    await _prepare_project_dir(child)

    proc = client = None
    try:
        proc, client = await _start_proxy(child, runtime_dir=runtime_dir)
        await _do_mcp_handshake(client, timeout=30.0)
        assert await wait_for_daemon_start(child, timeout=15.0)

        returncode, _, stderr = await _run_proxy_to_failure(
            parent,
            runtime_dir=runtime_dir,
        )
        assert returncode != 0
        assert "Overlapping daemon roots are not supported." in stderr
        assert str(parent.resolve()) in stderr
        assert str(child.resolve()) in stderr
        assert await wait_for_daemon_full_cleanup(
            parent,
            runtime_dir=runtime_dir,
            timeout=30.0,
        )
    finally:
        if proc is not None and client is not None:
            await _teardown_proxy(proc, client)
        _cleanup_project_dir(child, runtime_dir=runtime_dir)
        _cleanup_project_dir(parent, runtime_dir=runtime_dir)


@pytest.mark.asyncio
async def test_daemon_concurrent_parent_child_first_start_allows_only_one(
    tmp_path: Path,
) -> None:
    """Concurrent overlap starts should allow one daemon and reject the other."""
    parent = tmp_path / "repo"
    child = parent / "subdir"
    runtime_dir = tmp_path / "runtime"
    await _prepare_project_dir(parent)
    await _prepare_project_dir(child)

    parent_proc = parent_client = child_proc = child_client = None
    try:
        (parent_proc, parent_client), (child_proc, child_client) = await asyncio.gather(
            _start_proxy(parent, runtime_dir=runtime_dir, capture_stderr=True),
            _start_proxy(child, runtime_dir=runtime_dir, capture_stderr=True),
        )

        parent_result, child_result = await asyncio.gather(
            _do_mcp_handshake(parent_client, timeout=30.0),
            _do_mcp_handshake(child_client, timeout=30.0),
            return_exceptions=True,
        )
        parent_ok = not isinstance(parent_result, Exception)
        child_ok = not isinstance(child_result, Exception)
        assert parent_ok ^ child_ok, (
            f"Expected exactly one handshake to succeed, got parent={parent_result!r} "
            f"child={child_result!r}"
        )

        winner_root = parent if parent_ok else child
        loser_root = child if parent_ok else parent
        loser_proc = child_proc if parent_ok else parent_proc

        assert await wait_for_daemon_start(winner_root, timeout=15.0)

        assert loser_proc is not None
        loser_returncode = await asyncio.wait_for(loser_proc.wait(), timeout=30.0)
        assert loser_returncode != 0
        assert loser_proc.stderr is not None
        loser_stderr = (await loser_proc.stderr.read()).decode()
        assert "Overlapping daemon roots are not supported." in loser_stderr
        assert str(parent.resolve()) in loser_stderr
        assert str(child.resolve()) in loser_stderr
        assert await wait_for_daemon_full_cleanup(
            loser_root,
            runtime_dir=runtime_dir,
            timeout=30.0,
        )
    finally:
        await asyncio.gather(
            *[
                _teardown_proxy(proc, client)
                for proc, client in (
                    (parent_proc, parent_client),
                    (child_proc, child_client),
                )
                if proc is not None and client is not None
            ],
            return_exceptions=True,
        )
        _cleanup_project_dir(child, runtime_dir=runtime_dir)
        _cleanup_project_dir(parent, runtime_dir=runtime_dir)


@pytest.mark.asyncio
async def test_daemon_symlink_path_reuses_canonical_root(tmp_path: Path) -> None:
    """A symlinked path to the same project should reuse the existing daemon."""
    project_dir = tmp_path / "repo"
    runtime_dir = tmp_path / "runtime"
    await _prepare_project_dir(project_dir)

    alias = tmp_path / "repo-link"
    try:
        alias.symlink_to(project_dir, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symbolic links not supported on this platform")

    proc1 = client1 = proc2 = client2 = None
    try:
        proc1, client1 = await _start_proxy(project_dir, runtime_dir=runtime_dir)
        await _do_mcp_handshake(client1, timeout=30.0)
        assert await wait_for_daemon_start(project_dir, timeout=15.0)
        first_pid = json.loads(
            (project_dir / ".chunkhound" / "daemon.lock").read_text()
        )["pid"]

        proc2, client2 = await _start_proxy(alias, runtime_dir=runtime_dir)
        await _do_mcp_handshake(client2, timeout=30.0)
        second_pid = json.loads(
            (project_dir / ".chunkhound" / "daemon.lock").read_text()
        )["pid"]

        assert first_pid == second_pid
    finally:
        await asyncio.gather(
            *[
                _teardown_proxy(proc, client)
                for proc, client in ((proc1, client1), (proc2, client2))
                if proc is not None and client is not None
            ],
            return_exceptions=True,
        )
        _cleanup_project_dir(project_dir, runtime_dir=runtime_dir)


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Daemon shutdown is unreliable on Windows due to process termination timing",
)
async def test_daemon_registry_entry_removed_on_shutdown(
    pre_indexed_project_dir: Path,
) -> None:
    """Daemon should publish and then remove its registry entry on shutdown."""
    project_dir = pre_indexed_project_dir
    runtime_dir = project_dir / "_runtime"

    proc, client = await _start_proxy(project_dir, runtime_dir=runtime_dir)
    registry_entry = None
    try:
        await _do_mcp_handshake(client, timeout=30.0)
        assert await wait_for_daemon_start(project_dir, timeout=15.0)

        registry_entry = await _wait_for_registry_entry(
            runtime_dir, project_dir, timeout=15.0
        )
        assert registry_entry is not None, "Registry entry was not published"
        assert registry_entry.exists()
    finally:
        await _teardown_proxy(proc, client)

    stopped = await wait_for_daemon_full_cleanup(
        project_dir,
        runtime_dir=runtime_dir,
        timeout=30.0,
    )
    assert stopped, "Daemon did not shut down in time"
    assert registry_entry is not None
    assert not registry_entry.exists(), (
        "Registry entry was not cleaned up after shutdown"
    )
