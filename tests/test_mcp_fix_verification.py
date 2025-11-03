"""Test to verify the MCP server initialization fix works correctly.

This test verifies that the server now responds quickly even with large directories,
and that scan progress is available through the stats tool.
"""

import asyncio
import json
import os
import sys
import time

import pytest

from tests.utils import SubprocessJsonRpcClient
from tests.utils.windows_compat import database_cleanup_context, windows_safe_tempdir


def get_init_timeout() -> float:
    """Get appropriate timeout for MCP initialization.

    macOS GitHub Actions runners experience additional overhead from the
    SubprocessJsonRpcClient architecture (background reader task, future-based
    request/response matching). This function provides a more generous timeout
    for CI environments while maintaining strict validation locally.

    Rationale for platform-specific timeout:
    - macOS CI: ~8-10s actual time observed (SubprocessJsonRpcClient overhead)
    - Linux/Windows CI: ~3-5s actual time
    - Local development: ~2-3s actual time
    - VS Code MCP requirement: Must respond within reasonable time (<10s)

    Trade-off: Accept slower CI to avoid flaky tests vs failing legitimately
    slow initializations. The 10s timeout is a pragmatic upper bound.

    Returns:
        10.0 seconds for macOS CI (SubprocessJsonRpcClient overhead + CI variability)
        5.0 seconds otherwise (VS Code timeout requirement)
    """
    if os.environ.get("CI") == "true" and sys.platform == "darwin":
        return 10.0
    return 5.0


class TestMCPFixVerification:
    """Test MCP server initialization fix."""

    @pytest.mark.asyncio
    async def test_mcp_responds_quickly_with_large_directory(self):
        """Test that MCP server now responds quickly even with large directories.
        
        This verifies our fix works - the server should respond to initialize
        within a few seconds even with a large directory.
        """
        with windows_safe_tempdir() as temp_path:

            # Create a moderately large directory to test responsiveness
            for i in range(100):  # Enough files to potentially cause delay
                subdir = temp_path / f"module_{i // 10}"
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
                    start_time = time.time()
                    timeout = get_init_timeout()

                    # Send initialize request
                    init_result = await client.send_request(
                        "initialize",
                        {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {"name": "test", "version": "1.0"}
                        },
                        timeout=timeout
                    )

                    response_time = time.time() - start_time

                    # Verify quick response
                    assert response_time < timeout, (
                        f"Server took {response_time:.2f} seconds to respond "
                        f"(should be < {timeout}s)"
                    )

                    # Verify response structure
                    assert "serverInfo" in init_result, f"No serverInfo in result: {init_result}"
                    assert init_result["serverInfo"]["name"] == "ChunkHound Code Search"

                    print(f"✅ Server responded in {response_time:.2f} seconds")

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
                    timeout = get_init_timeout()

                    # Initialize the server
                    await client.send_request(
                        "initialize",
                        {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {"name": "test", "version": "1.0"}
                        },
                        timeout=timeout
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
