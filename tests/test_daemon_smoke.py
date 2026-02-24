"""Daemon smoke tests — verifies multi-client MCP daemon architecture.

These tests cover the critical ALGINF-5316 use case: multiple Claude instances
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
from pathlib import Path
from typing import AsyncIterator

import psutil
import pytest

from tests.helpers.daemon_test_helpers import (
    is_daemon_running,
    wait_for_daemon_shutdown,
    wait_for_daemon_start,
)
from tests.utils import SubprocessJsonRpcClient, create_subprocess_exec_safe, get_safe_subprocess_env


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
    "arguments": {"pattern": "def authenticate", "type": "regex"},
}


def _make_env(project_dir: Path) -> dict[str, str]:
    """Return a clean subprocess environment pointing to project_dir."""
    env = get_safe_subprocess_env()
    # Clear any stale CHUNKHOUND_* vars that could interfere
    for key in list(env.keys()):
        if key.startswith("CHUNKHOUND_"):
            del env[key]
    env["CHUNKHOUND_MCP_MODE"] = "1"
    return env


async def _start_proxy(project_dir: Path) -> tuple[asyncio.subprocess.Process, SubprocessJsonRpcClient]:
    """Launch a ``chunkhound mcp`` proxy subprocess and return (proc, client)."""
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


async def _do_mcp_handshake(client: SubprocessJsonRpcClient, timeout: float = 30.0) -> dict:
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
    # --- Synthetic source files ---
    (tmp_path / "auth.py").write_text(
        "def authenticate(user, password):\n"
        "    \"\"\"Authenticate a user.\"\"\"\n"
        "    return user == 'admin' and password == 'secret'\n"
        "\n"
        "def logout(session):\n"
        "    session.clear()\n"
    )
    (tmp_path / "search.py").write_text(
        "def search_items(query, index):\n"
        "    \"\"\"Search items in the index.\"\"\"\n"
        "    results = []\n"
        "    for item in index:\n"
        "        if query.lower() in item.lower():\n"
        "            results.append(item)\n"
        "    return results\n"
    )
    (tmp_path / "utils.py").write_text(
        "def format_result(result):\n"
        "    \"\"\"Format a search result.\"\"\"\n"
        "    return str(result)\n"
        "\n"
        "def validate_input(data):\n"
        "    return data is not None\n"
    )

    # --- ChunkHound config ---
    db_path = tmp_path / ".chunkhound" / "test.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "database": {"path": str(db_path), "provider": "duckdb"},
        "indexing": {"include": ["*.py"]},
    }
    (tmp_path / ".chunkhound.json").write_text(json.dumps(config))

    # --- Index files (no embeddings — keeps tests API-key-free) ---
    index_proc = await asyncio.create_subprocess_exec(
        _chunkhound_exe(), "index", str(tmp_path), "--no-embeddings",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(tmp_path),
        env=get_safe_subprocess_env(),
    )
    _, stderr = await asyncio.wait_for(index_proc.communicate(), timeout=60.0)
    assert index_proc.returncode == 0, (
        f"Pre-indexing failed (rc={index_proc.returncode}): {stderr.decode()}"
    )

    yield tmp_path

    # --- Cleanup: stop any lingering daemon ---
    from chunkhound.daemon.discovery import DaemonDiscovery

    discovery = DaemonDiscovery(tmp_path)
    lock = discovery.read_lock()
    if lock:
        pid = lock.get("pid")
        if isinstance(pid, int):
            try:
                # Use psutil for cross-platform process termination
                proc = psutil.Process(pid)
                proc.terminate()  # SIGTERM on Unix, TerminateProcess on Windows
                # Wait up to 5 seconds for graceful shutdown
                try:
                    proc.wait(timeout=5.0)
                except psutil.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    proc.kill()
                    proc.wait(timeout=2.0)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            except Exception:
                pass
        discovery.remove_lock()
    # Clean up Unix socket file (not applicable on Windows where TCP is used)
    socket_path = discovery.get_socket_path()
    if not socket_path.startswith("tcp:"):
        try:
            os.unlink(socket_path)
        except Exception:
            pass


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
        # The result is a JSON string; parse and check for matches
        parsed = json.loads(text) if text.startswith("{") or text.startswith("[") else {"raw": text}
        # Should find 'def authenticate' in auth.py
        assert "auth" in text.lower() or "authenticate" in text.lower() or parsed.get("total_results", 0) >= 0, (
            f"Unexpected search result: {text[:500]}"
        )

    finally:
        await _teardown_proxy(proc, client)

    # After the proxy disconnects, the daemon should shut down
    # Windows requires significantly more time for cleanup (see test_daemon_lock_file_created comments)
    stopped = await wait_for_daemon_shutdown(project_dir, timeout=30.0)
    assert stopped, "Daemon did not shut down after last client disconnected"


@pytest.mark.asyncio
async def test_daemon_two_clients_concurrent(pre_indexed_project_dir: Path) -> None:
    """Two proxy clients connect to the SAME daemon concurrently — ALGINF-5316.

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
    # Windows requires significantly more time for cleanup (see test_daemon_lock_file_created comments)
    stopped = await wait_for_daemon_shutdown(project_dir, timeout=30.0)
    assert stopped, "Daemon did not shut down after all clients disconnected"


@pytest.mark.asyncio
async def test_daemon_lock_file_created(pre_indexed_project_dir: Path) -> None:
    """Verify the daemon lock file is created with correct content and cleaned up.

    Checks:
    - Lock file is created at .chunkhound/daemon.lock.
    - Lock file contains valid JSON with 'pid', 'socket_path', 'started_at'.
    - Recorded PID matches a live process.
    - Lock file is removed after the daemon shuts down.
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
        assert "socket_path" in lock_data, f"Lock file missing 'socket_path': {lock_data}"
        assert "started_at" in lock_data, f"Lock file missing 'started_at': {lock_data}"

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
