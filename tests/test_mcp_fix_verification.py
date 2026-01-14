"""Test to verify MCP server non-blocking initialization architecture.

Verifies that initialization returns before directory scanning completes
(non-blocking) and that scan progress is available through the stats tool.
"""

import asyncio
import json
import os

import pytest
from tests.utils import SubprocessJsonRpcClient
from tests.utils.windows_compat import windows_safe_tempdir, database_cleanup_context


class TestMCPFixVerification:
    """Test MCP server initialization fix."""

    @pytest.mark.asyncio
    async def test_mcp_initialization_is_non_blocking(self):
        """Test that MCP server initialization is non-blocking.

        Verifies the architectural invariant: init must return BEFORE directory
        scanning completes. Uses observable state (is_scanning == True) rather
        than flaky timing assertions.
        """
        with windows_safe_tempdir() as temp_path:

            # Create enough files that scanning takes meaningful time
            for i in range(200):
                subdir = temp_path / f"module_{i // 20}"
                subdir.mkdir(exist_ok=True)

                test_file = subdir / f"file_{i}.py"
                test_file.write_text(f"""
def function_{i}():
    '''Function {i} for testing.'''
    return "value_{i}"

class Class_{i}:
    '''Class {i} for testing.'''

    def method_{i}(self):
        return "result_{i}"
""")

            # Create minimal config
            config_path = temp_path / ".chunkhound.json"
            db_path = temp_path / ".chunkhound" / "test.db"
            db_path.parent.mkdir(exist_ok=True)

            config = {
                "database": {"path": str(db_path), "provider": "duckdb"},
                "indexing": {"include": ["*.py"]}
            }
            config_path.write_text(json.dumps(config))

            # Use database cleanup context to ensure proper resource management
            with database_cleanup_context():
                # Start MCP server
                mcp_env = os.environ.copy()
                mcp_env["CHUNKHOUND_MCP_MODE"] = "1"

                proc = await asyncio.create_subprocess_exec(
                    "uv", "run", "chunkhound", "mcp", str(temp_path),
                    cwd=temp_path,
                    env=mcp_env,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                client = SubprocessJsonRpcClient(proc)
                await client.start()

                try:
                    # Initialize server
                    init_result = await client.send_request(
                        "initialize",
                        {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {"name": "test", "version": "1.0"}
                        },
                        timeout=30.0
                    )

                    # Verify response structure
                    assert "serverInfo" in init_result, f"No serverInfo in result: {init_result}"
                    assert init_result["serverInfo"]["name"] == "ChunkHound Code Search"

                    # Send initialized notification to enable tool calls
                    await client.send_notification("notifications/initialized")

                    # Query stats IMMEDIATELY after init to check scan state
                    stats_result = await client.send_request(
                        "tools/call",
                        {"name": "get_stats", "arguments": {}},
                        timeout=5.0
                    )

                    stats_data = json.loads(stats_result["content"][0]["text"])
                    scan_info = stats_data["initial_scan"]

                    # THE REAL INVARIANT: init returned before scan completed
                    # Valid states proving non-blocking:
                    # 1. Scan in progress (is_scanning=True)
                    # 2. Scan not yet started (started_at=None) - even more non-blocking!
                    scan_not_completed = (
                        scan_info["is_scanning"] is True or
                        scan_info["started_at"] is None
                    )
                    assert scan_not_completed, (
                        f"Init must return before scan completes (non-blocking architecture). "
                        f"Scan appears to have completed: {scan_info}"
                    )

                finally:
                    await client.close()

    @pytest.mark.asyncio
    async def test_stats_includes_scan_progress(self):
        """Test that get_stats tool now includes scan progress information."""
        with windows_safe_tempdir() as temp_path:
            
            # Create a few test files
            for i in range(5):
                test_file = temp_path / f"test_{i}.py"
                test_file.write_text(f"def test_{i}(): pass")
            
            # Create minimal config
            config_path = temp_path / ".chunkhound.json"
            db_path = temp_path / ".chunkhound" / "test.db"
            db_path.parent.mkdir(exist_ok=True)
            
            config = {
                "database": {"path": str(db_path), "provider": "duckdb"},
                "indexing": {"include": ["*.py"]}
            }
            config_path.write_text(json.dumps(config))
            
            # Use database cleanup context to ensure proper resource management
            with database_cleanup_context():
                # Start MCP server
                mcp_env = os.environ.copy()
                mcp_env["CHUNKHOUND_MCP_MODE"] = "1"

                proc = await asyncio.create_subprocess_exec(
                    "uv", "run", "chunkhound", "mcp", str(temp_path),
                    cwd=temp_path,
                    env=mcp_env,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                client = SubprocessJsonRpcClient(proc)
                await client.start()

                try:
                    # Initialize the server (increase timeout for CI environments)
                    await client.send_request(
                        "initialize",
                        {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {"name": "test", "version": "1.0"}
                        },
                        timeout=10.0
                    )

                    # Send initialized notification
                    await client.send_notification("notifications/initialized")

                    # Call get_stats tool
                    stats_result = await client.send_request(
                        "tools/call",
                        {
                            "name": "get_stats",
                            "arguments": {}
                        },
                        timeout=5.0
                    )

                    # Verify stats structure
                    assert "content" in stats_result, f"No content in stats result: {stats_result}"

                    # Parse the stats content
                    content = stats_result["content"][0]["text"]
                    stats_data = json.loads(content)

                    # Verify scan progress is included
                    assert "initial_scan" in stats_data, f"No initial_scan in stats: {stats_data}"

                    scan_info = stats_data["initial_scan"]
                    assert "is_scanning" in scan_info, f"No is_scanning field: {scan_info}"
                    assert "files_processed" in scan_info, f"No files_processed field: {scan_info}"
                    assert "chunks_created" in scan_info, f"No chunks_created field: {scan_info}"
                    assert "started_at" in scan_info, f"No started_at field: {scan_info}"

                    print(f"✅ Scan progress info: {scan_info}")

                finally:
                    await client.close()