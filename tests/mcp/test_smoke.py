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
import subprocess  # noqa: F401  keep for potential use in subclasses
import tempfile
from pathlib import Path

import pytest

from tests.utils import SubprocessJsonRpcClient
from tests.utils.windows_compat import get_fs_event_timeout, windows_safe_tempdir
from tests.utils.windows_subprocess import (
    create_subprocess_exec_safe,
    get_safe_subprocess_env,
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