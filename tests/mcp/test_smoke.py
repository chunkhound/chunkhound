#!/usr/bin/env python3
"""
Smoke-level MCP integration tests — subprocess roundtrip startup + protocol checks.

These tests start a real ``chunkhound mcp`` subprocess and exercise JSON-RPC
handshake, tool discovery, and tool calls.  They are inherently slower and
more environment-sensitive than the import/CLI-help tests in
``tests/test_smoke.py``, so they live in their own directory.

Run them when MCP startup or protocol-handling code changes::

    uv run pytest tests/mcp/ -v
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess  # noqa: F401  keep for potential use in subclasses
import tempfile
import time
from pathlib import Path

import httpx
import pytest

from tests.utils import HttpMcpClient, SubprocessJsonRpcClient, SubprocessJsonRpcError
from tests.utils.windows_compat import get_fs_event_timeout, windows_safe_tempdir
from tests.utils.windows_subprocess import (
    create_subprocess_exec_safe,
    get_safe_subprocess_env,
)


def _get_free_port() -> int:
    """Bind an ephemeral loopback port and return it for a spawned server.

    Simpler than the daemon's hash-based-port-with-bindability-check scheme:
    that scheme exists so the daemon can be rediscovered at a stable address
    across restarts, which tests don't need.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def _wait_for_http_ready(
    base_url: str,
    timeout: float,
    proc: asyncio.subprocess.Process | None = None,
) -> None:
    """Poll ``/health`` until the HTTP MCP server accepts connections.

    If ``proc`` is given and exits before becoming ready, fail immediately
    instead of spinning for the full timeout — a crashed server will never
    start answering ``/health``.
    """
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    async with httpx.AsyncClient() as probe:
        while time.monotonic() < deadline:
            if proc is not None and proc.returncode is not None:
                raise AssertionError(
                    f"HTTP MCP server process exited early with code "
                    f"{proc.returncode} before becoming ready at {base_url}"
                )
            try:
                resp = await probe.get(f"{base_url}/health", timeout=2.0)
                if resp.status_code == 200:
                    return
            except httpx.TransportError as exc:
                last_error = exc
            await asyncio.sleep(0.3)
    raise AssertionError(
        f"HTTP MCP server did not become ready at {base_url}: {last_error}"
    )


class TestMCPStdioHelp:
    """Minimal smoke — does the ``--help`` flag work via subprocess?"""

    @pytest.mark.asyncio
    async def test_mcp_stdio_server_help(self) -> None:
        """Test that MCP stdio server responds to help."""
        proc = await create_subprocess_exec_safe(
            "uv",
            "run",
            "chunkhound",
            "mcp",
            "--no-daemon",
            "--help",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=get_safe_subprocess_env(),
        )

        stdout, stderr = await proc.communicate()

        assert proc.returncode == 0, (
            f"MCP stdio help failed with code {proc.returncode}\n"
            f"stderr: {stderr.decode()}"
        )


class TestMCPStartupImports:
    """Does ``chunkhound.mcp_server.stdio`` import without crashing?"""

    @pytest.mark.asyncio
    async def test_mcp_stdio_server_starts(self) -> None:
        """Verify the MCP server module imports and Config() constructs cleanly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            config_path = temp_path / ".chunkhound.json"
            db_path = temp_path / ".chunkhound" / "test.db"
            db_path.parent.mkdir(exist_ok=True)

            config = {
                "database": {"path": str(db_path), "provider": "duckdb"},
                "indexing": {"include": ["*.py"]},
            }
            config_path.write_text(json.dumps(config))

            cwd_repr = repr(os.getcwd())
            proc = await create_subprocess_exec_safe(
                "uv",
                "run",
                "python",
                "-c",
                f"""
import sys
import os

sys.path.insert(0, {cwd_repr})

from chunkhound.mcp_server.stdio import main
import asyncio

async def test():
    os.environ["CHUNKHOUND_EMBEDDING__PROVIDER"] = "openai"
    os.environ["CHUNKHOUND_EMBEDDING__API_KEY"] = "test"

    try:
        from chunkhound.mcp_server.stdio import StdioMCPServer
        from chunkhound.core.config.config import Config

        config = Config()

        print("SUCCESS: MCP server imports and config creation work")
        return 0
    except Exception as e:
        print(f"FAILED: {{e}}")
        return 1

sys.exit(asyncio.run(test()))
                """,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=temp_dir,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=10.0
                )

                if proc.returncode != 0:
                    pytest.fail(
                        f"MCP stdio server initialization failed with code {proc.returncode}\n"
                        f"stdout: {stdout.decode()}\n"
                        f"stderr: {stderr.decode()}"
                    )

                assert "SUCCESS:" in stdout.decode(), (
                    f"Expected success message, got: {stdout.decode()}"
                )

            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                pytest.fail("MCP stdio server test timed out")


class TestMCPProtocolHandshake:
    """Full JSON-RPC handshake + tool discovery via a real subprocess."""

    @pytest.mark.asyncio
    async def test_mcp_stdio_protocol_handshake(self) -> None:
        """MCP stdio server completes initialize → tools/list → daemon_status."""
        with windows_safe_tempdir() as temp_path:
            test_file = temp_path / "test.py"
            test_file.write_text("def hello(): return 'world'")

            config_path = temp_path / ".chunkhound.json"
            db_path = temp_path / ".chunkhound" / "test.db"
            db_path.parent.mkdir(exist_ok=True)

            config = {
                "database": {"path": str(db_path), "provider": "duckdb"},
                "indexing": {"include": ["*.py"]},
            }

            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                config["embedding"] = {
                    "provider": "openai",
                    "model": "text-embedding-3-small",
                }

            config_path.write_text(json.dumps(config))

            mcp_env = get_safe_subprocess_env(os.environ)
            mcp_env["CHUNKHOUND_MCP_MODE"] = "1"
            if api_key:
                mcp_env["CHUNKHOUND_EMBEDDING__API_KEY"] = api_key

            proc = await create_subprocess_exec_safe(
                "uv",
                "run",
                "chunkhound",
                "mcp",
                "--no-daemon",
                str(temp_path),
                cwd=str(temp_path),
                env=mcp_env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Drain stderr in the background so the MCP server does not
            # block on a full stderr buffer during startup.
            stderr_lines: list[str] = []

            async def _drain_stderr() -> None:
                while True:
                    line = await proc.stderr.readline()
                    if not line:
                        break
                    stderr_lines.append(line.decode("utf-8", errors="replace").strip())

            stderr_task = asyncio.create_task(_drain_stderr())

            client = SubprocessJsonRpcClient(proc)
            await client.start()
            await asyncio.sleep(0.5)

            try:
                init_timeout = max(10.0, get_fs_event_timeout())
                init_result = await client.send_request(
                    "initialize",
                    {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1.0"},
                    },
                    timeout=init_timeout,
                )

                assert "serverInfo" in init_result, (
                    f"No serverInfo in result: {init_result}"
                )
                assert init_result["serverInfo"]["name"] == "ChunkHound Code Search"

                await client.send_notification("notifications/initialized")

                tools_result = await client.send_request("tools/list", timeout=5.0)
                tools = tools_result.get("tools", [])
                tool_names = [t["name"] for t in tools]

                assert "search" in tool_names, f"search not in tools: {tool_names}"
                assert "daemon_status" in tool_names, (
                    f"daemon_status not in tools: {tool_names}"
                )

                status_result = await client.send_request(
                    "tools/call",
                    {"name": "daemon_status", "arguments": {}},
                    timeout=5.0,
                )
                status_content = status_result.get("content", [])
                assert isinstance(status_content, list) and len(status_content) > 0

                status_payload = json.loads(status_content[0]["text"])
                assert status_payload["status"] in {
                    "initializing",
                    "ready",
                    "degraded",
                }
                assert "scan_progress" in status_payload
                assert "realtime" in status_payload["scan_progress"]
                startup = status_payload["scan_progress"]["realtime"].get("startup")
                assert isinstance(startup, dict)
                assert isinstance(startup.get("phases"), dict)
                assert "initialize" in startup["phases"]
                assert startup["phases"]["initialize"]["state"] in {
                    "completed",
                    "in_progress",
                    "uninitialized",
                }

            except asyncio.TimeoutError:
                pytest.fail("MCP stdio protocol handshake timed out")
            finally:
                await client.close()
                stderr_task.cancel()
                try:
                    await stderr_task
                except asyncio.CancelledError:
                    pass


class TestMCPWebsearchStdio:
    """Websearch tool roundtrip with external calls stubbed out."""

    @pytest.mark.asyncio
    async def test_mcp_websearch_stdio_mocked(self) -> None:
        """MCP stdio roundtrip for websearch with network calls replaced by stubs."""
        with windows_safe_tempdir() as temp_path:
            (temp_path / "test.py").write_text("def hello(): return 'world'")

            config_path = temp_path / ".chunkhound.json"
            db_path = temp_path / ".chunkhound" / "test.db"
            db_path.parent.mkdir(exist_ok=True)
            config = {
                "database": {"path": str(db_path), "provider": "duckdb"},
                "indexing": {"include": ["*.py"]},
            }
            config_path.write_text(json.dumps(config))

            helpers_dir = Path("tests/helpers").resolve()
            mcp_env = get_safe_subprocess_env(os.environ)
            mcp_env["CHUNKHOUND_MCP_MODE"] = "1"
            mcp_env["CH_TEST_WEBSEARCH_STUB"] = "1"
            mcp_env["PYTHONPATH"] = (
                f"{helpers_dir}{os.pathsep}{mcp_env.get('PYTHONPATH', '')}"
            )

            proc = await create_subprocess_exec_safe(
                "uv",
                "run",
                "chunkhound",
                "mcp",
                "--no-daemon",
                "--no-embeddings",
                str(temp_path),
                cwd=str(temp_path),
                env=mcp_env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            client = SubprocessJsonRpcClient(proc)
            await client.start()

            try:
                init_timeout = max(10.0, get_fs_event_timeout())
                init = await client.send_request(
                    "initialize",
                    {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "smoke", "version": "1.0"},
                    },
                    timeout=init_timeout,
                )
                assert init["serverInfo"]["name"] == "ChunkHound Code Search"

                await client.send_notification("notifications/initialized")

                tools_result = await client.send_request("tools/list", timeout=10.0)
                tool_names = [t["name"] for t in tools_result.get("tools", [])]
                assert "websearch" in tool_names, (
                    f"websearch missing from tools/list: {tool_names}"
                )

                call = await client.send_request(
                    "tools/call",
                    {
                        "name": "websearch",
                        "arguments": {"query": "smoke", "limit": 3},
                    },
                    timeout=30.0,
                )
                contents = call.get("content") or []
                full_text = "\n".join(
                    c.get("text", "") for c in contents if isinstance(c, dict)
                )
                assert "ANSWER" in full_text, (
                    f"missing stubbed ANSWER in response: {full_text!r}"
                )

            except asyncio.TimeoutError:
                pytest.fail("MCP websearch stdio roundtrip timed out")
            finally:
                await client.close()


class TestMCPHttpTransport:
    """HTTP transport startup, protocol handshake, auth, and CORS behavior."""

    @staticmethod
    async def _spawn_http_mcp_server(
        temp_path: Path, port: int, extra_args: list[str] | None = None
    ) -> asyncio.subprocess.Process:
        """Start `chunkhound mcp --transport http` against a minimal project."""
        import json as _json

        test_file = temp_path / "test.py"
        test_file.write_text("def hello(): return 'world'")

        config_path = temp_path / ".chunkhound.json"
        db_path = temp_path / ".chunkhound" / "test.db"
        db_path.parent.mkdir(exist_ok=True)
        config = {
            "database": {"path": str(db_path), "provider": "duckdb"},
            "indexing": {"include": ["*.py"]},
        }
        config_path.write_text(_json.dumps(config))

        mcp_env = get_safe_subprocess_env(os.environ)
        mcp_env["CHUNKHOUND_MCP_MODE"] = "1"

        args = [
            "uv",
            "run",
            "chunkhound",
            "mcp",
            "--transport",
            "http",
            "--port",
            str(port),
            *(extra_args or []),
            str(temp_path),
        ]

        return await create_subprocess_exec_safe(
            *args,
            cwd=str(temp_path),
            env=mcp_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    @staticmethod
    async def _drain(stream: asyncio.StreamReader) -> None:
        while True:
            line = await stream.readline()
            if not line:
                break

    async def _terminate(self, proc: asyncio.subprocess.Process) -> None:
        """Terminate a spawned ``uv run chunkhound mcp ...`` process tree.

        ``uv run`` execs the real server as a child (or grandchild) process;
        terminating only the ``uv``-tracked PID leaves that real process
        running on Windows, where child processes are not part of the same
        process group by default. That orphan keeps listening on its port
        and holding an exclusive lock on its DB file indefinitely. Killing
        the whole descendant tree (via psutil) avoids that leak.
        """
        if proc.returncode is not None:
            return

        import psutil

        try:
            parent = psutil.Process(proc.pid)
            children = parent.children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
            parent = None

        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        if parent is not None:
            try:
                parent.terminate()
            except psutil.NoSuchProcess:
                pass

        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()

        gone, alive = psutil.wait_procs(children, timeout=5.0)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass

    @pytest.mark.asyncio
    async def test_mcp_http_protocol_handshake(self):
        """MCP HTTP server completes the full protocol handshake + tool call."""
        import json

        with windows_safe_tempdir() as temp_path:
            port = _get_free_port()
            proc = await self._spawn_http_mcp_server(temp_path, port)
            drain_tasks = [
                asyncio.create_task(self._drain(proc.stdout)),
                asyncio.create_task(self._drain(proc.stderr)),
            ]

            base_url = f"http://127.0.0.1:{port}"
            client = HttpMcpClient(base_url)
            try:
                ready_timeout = max(30.0, get_fs_event_timeout())
                await _wait_for_http_ready(base_url, ready_timeout, proc=proc)

                init_result = await client.initialize(timeout=15.0)
                assert "serverInfo" in init_result, (
                    f"No serverInfo in result: {init_result}"
                )
                assert init_result["serverInfo"]["name"] == "ChunkHound Code Search"

                await client.send_notification("notifications/initialized")

                tools_result = await client.send_request("tools/list", timeout=10.0)
                tool_names = [t["name"] for t in tools_result.get("tools", [])]
                assert "search" in tool_names, f"search not in tools: {tool_names}"
                assert "daemon_status" in tool_names, (
                    f"daemon_status not in tools: {tool_names}"
                )

                status_result = await client.send_request(
                    "tools/call",
                    {"name": "daemon_status", "arguments": {}},
                    timeout=10.0,
                )
                status_content = status_result.get("content", [])
                assert isinstance(status_content, list) and len(status_content) > 0

                status_payload = json.loads(status_content[0]["text"])
                assert status_payload["status"] in {
                    "initializing",
                    "ready",
                    "degraded",
                }
                assert "scan_progress" in status_payload
            finally:
                await client.close()
                await self._terminate(proc)
                for task in drain_tasks:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    @pytest.mark.asyncio
    async def test_mcp_http_session_teardown(self):
        """DELETE /mcp with Mcp-Session-Id explicitly terminates the session."""
        with windows_safe_tempdir() as temp_path:
            port = _get_free_port()
            proc = await self._spawn_http_mcp_server(temp_path, port)
            drain_tasks = [
                asyncio.create_task(self._drain(proc.stdout)),
                asyncio.create_task(self._drain(proc.stderr)),
            ]

            base_url = f"http://127.0.0.1:{port}"
            client = HttpMcpClient(base_url)
            try:
                ready_timeout = max(30.0, get_fs_event_timeout())
                await _wait_for_http_ready(base_url, ready_timeout, proc=proc)

                await client.initialize(timeout=15.0)
                await client.send_notification("notifications/initialized")

                terminate_resp = await client.terminate_session(timeout=5.0)
                assert terminate_resp.status_code == 200, terminate_resp.text

                # The now-terminated session ID must be rejected, not silently
                # resumed — proves DELETE actually tore down server-side state
                # rather than being a no-op the client can't observe.
                with pytest.raises(SubprocessJsonRpcError) as exc_info:
                    await client.send_request("tools/list", timeout=5.0)
                assert "404" in str(exc_info.value), (
                    f"Expected 404 for terminated session, got: {exc_info.value}"
                )
            finally:
                await client.close()
                await self._terminate(proc)
                for task in drain_tasks:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    @pytest.mark.asyncio
    async def test_mcp_http_health_endpoint(self):
        """A bare GET /health with no prior handshake returns a valid status."""
        with windows_safe_tempdir() as temp_path:
            port = _get_free_port()
            proc = await self._spawn_http_mcp_server(temp_path, port)
            drain_tasks = [
                asyncio.create_task(self._drain(proc.stdout)),
                asyncio.create_task(self._drain(proc.stderr)),
            ]

            base_url = f"http://127.0.0.1:{port}"
            try:
                ready_timeout = max(30.0, get_fs_event_timeout())
                await _wait_for_http_ready(base_url, ready_timeout, proc=proc)

                async with httpx.AsyncClient() as raw_client:
                    resp = await raw_client.get(f"{base_url}/health", timeout=5.0)
                assert resp.status_code == 200
                payload = resp.json()
                assert "status" in payload
                assert "server_version" in payload
                assert "query_ready" in payload
                assert "scan_progress" in payload
            finally:
                await self._terminate(proc)
                for task in drain_tasks:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    @pytest.mark.asyncio
    async def test_mcp_http_bearer_auth(self):
        """Bearer auth gates /mcp but never /health, even when configured."""
        with windows_safe_tempdir() as temp_path:
            port = _get_free_port()
            token = "secret123"
            proc = await self._spawn_http_mcp_server(
                temp_path, port, extra_args=["--auth-token", token]
            )
            drain_tasks = [
                asyncio.create_task(self._drain(proc.stdout)),
                asyncio.create_task(self._drain(proc.stderr)),
            ]

            base_url = f"http://127.0.0.1:{port}"
            init_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0"},
                },
            }
            mcp_headers = {
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            }

            client = None
            try:
                ready_timeout = max(30.0, get_fs_event_timeout())
                await _wait_for_http_ready(base_url, ready_timeout, proc=proc)

                async with httpx.AsyncClient() as raw_client:
                    # No auth header -> 401
                    resp = await raw_client.post(
                        f"{base_url}/mcp",
                        json=init_payload,
                        headers=mcp_headers,
                        timeout=5.0,
                    )
                    assert resp.status_code == 401

                    # Wrong token -> 401
                    resp = await raw_client.post(
                        f"{base_url}/mcp",
                        json=init_payload,
                        headers={**mcp_headers, "Authorization": "Bearer wrong"},
                        timeout=5.0,
                    )
                    assert resp.status_code == 401

                    # /health stays reachable with no auth header at all.
                    resp = await raw_client.get(f"{base_url}/health", timeout=5.0)
                    assert resp.status_code == 200

                # Correct token -> full handshake succeeds.
                client = HttpMcpClient(base_url, auth_token=token)
                init_result = await client.initialize(timeout=15.0)
                assert init_result["serverInfo"]["name"] == "ChunkHound Code Search"
            finally:
                if client is not None:
                    await client.close()
                await self._terminate(proc)
                for task in drain_tasks:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    @pytest.mark.asyncio
    async def test_mcp_http_no_token_nonlocal_host_rejected(self):
        """Non-loopback --host with no --auth-token is refused at startup."""
        with windows_safe_tempdir() as temp_path:
            port = _get_free_port()
            proc = await self._spawn_http_mcp_server(
                temp_path, port, extra_args=["--host", "0.0.0.0"]
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=30.0
                )
            except asyncio.TimeoutError:
                # Use the full process-tree terminate, not a bare kill: if the
                # startup-validation gate regressed and the server actually
                # started serving, `uv run`'s real child/grandchild process
                # would otherwise survive a top-level kill and leak a
                # locked DB/port (see _terminate's docstring).
                await self._terminate(proc)
                pytest.fail("Server did not exit promptly on rejected configuration")

            assert proc.returncode != 0, (
                "Server should refuse to start on a non-loopback host without "
                "an auth token"
            )
            combined = stdout.decode(errors="replace") + stderr.decode(errors="replace")
            assert "auth_token" in combined or "non-loopback" in combined, combined

    @pytest.mark.asyncio
    async def test_mcp_http_cors_without_token_rejected(self):
        """--cors with no --auth-token is refused at startup, even on loopback.

        Without a token, any website open in the same browser could read from
        the HTTP transport via CORS — this must be rejected regardless of
        --host, not just when --host is non-loopback.
        """
        with windows_safe_tempdir() as temp_path:
            port = _get_free_port()
            proc = await self._spawn_http_mcp_server(
                temp_path, port, extra_args=["--cors"]
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=30.0
                )
            except asyncio.TimeoutError:
                # Use the full process-tree terminate, not a bare kill: if the
                # startup-validation gate regressed and the server actually
                # started serving, `uv run`'s real child/grandchild process
                # would otherwise survive a top-level kill and leak a
                # locked DB/port (see _terminate's docstring).
                await self._terminate(proc)
                pytest.fail("Server did not exit promptly on rejected configuration")

            assert proc.returncode != 0, (
                "Server should refuse to start with --cors and no --auth-token"
            )
            combined = stdout.decode(errors="replace") + stderr.decode(errors="replace")
            assert "auth_token" in combined or "cors" in combined.lower(), combined

    @pytest.mark.asyncio
    async def test_mcp_http_cors_with_token_allowed(self):
        """--cors with --auth-token starts up and sends CORS response headers."""
        with windows_safe_tempdir() as temp_path:
            port = _get_free_port()
            token = "secret123"
            proc = await self._spawn_http_mcp_server(
                temp_path, port, extra_args=["--cors", "--auth-token", token]
            )
            drain_tasks = [
                asyncio.create_task(self._drain(proc.stdout)),
                asyncio.create_task(self._drain(proc.stderr)),
            ]

            base_url = f"http://127.0.0.1:{port}"
            try:
                ready_timeout = max(30.0, get_fs_event_timeout())
                await _wait_for_http_ready(base_url, ready_timeout, proc=proc)

                async with httpx.AsyncClient() as raw_client:
                    resp = await raw_client.get(
                        f"{base_url}/health",
                        headers={"Origin": "https://example.com"},
                        timeout=5.0,
                    )
                    assert resp.status_code == 200
                    assert resp.headers.get("access-control-allow-origin") == "*"

                    # The load-bearing case: a browser CORS preflight OPTIONS
                    # request to /mcp carries no Authorization header. CORS
                    # middleware must answer it directly (200, with CORS
                    # headers) rather than let bearer-auth 401 it — see
                    # _BearerAuthMiddleware's docstring for why ordering
                    # matters here.
                    preflight = await raw_client.options(
                        f"{base_url}/mcp",
                        headers={
                            "Origin": "https://example.com",
                            "Access-Control-Request-Method": "POST",
                            "Access-Control-Request-Headers": (
                                "authorization,content-type"
                            ),
                        },
                        timeout=5.0,
                    )
                    assert preflight.status_code == 200, preflight.text
                    assert preflight.headers.get("access-control-allow-origin") == "*"

                    # A real, authenticated request to /mcp with an Origin
                    # header must succeed and still carry CORS response
                    # headers, proving CORS and bearer-auth compose correctly
                    # (not just CORS alone on the auth-exempt /health
                    # endpoint).
                    init_payload = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {"name": "test", "version": "1.0"},
                        },
                    }
                    resp = await raw_client.post(
                        f"{base_url}/mcp",
                        json=init_payload,
                        headers={
                            "Accept": "application/json, text/event-stream",
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {token}",
                            "Origin": "https://example.com",
                        },
                        timeout=15.0,
                    )
                    assert resp.status_code == 200, resp.text
                    assert resp.headers.get("access-control-allow-origin") == "*"
            finally:
                await self._terminate(proc)
                for task in drain_tasks:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
