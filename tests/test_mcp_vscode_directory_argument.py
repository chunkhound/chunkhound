"""Test MCP server directory argument handling for VS Code compatibility.

This test reproduces the issue where VS Code invokes the MCP server with a
positional directory argument but from a different working directory.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from subprocess import PIPE

import pytest

from tests.utils import SubprocessJsonRpcClient


@pytest.mark.asyncio
async def test_mcp_server_uses_positional_directory_argument():
    """Test that MCP server correctly uses positional directory argument.

    This reproduces the VS Code issue where the server is invoked as:
    chunkhound mcp /path/to/project
    from a different working directory.
    """
    # Create temporary directories
    home_dir = Path(tempfile.mkdtemp())
    project_dir = Path(tempfile.mkdtemp())

    try:
        # Create project config in project directory (following test patterns)
        db_path = project_dir / ".chunkhound" / "test.db"
        db_path.parent.mkdir(exist_ok=True)

        config = {"database": {"path": str(db_path), "provider": "duckdb"}}
        config_file = project_dir / ".chunkhound.json"
        config_file.write_text(json.dumps(config, indent=2))

        # Set environment for MCP mode
        import os

        mcp_env = os.environ.copy()
        mcp_env["CHUNKHOUND_MCP_MODE"] = "1"

        # Run MCP server from home_dir with project_dir as argument
        # This simulates VS Code's invocation pattern
        proc = await asyncio.create_subprocess_exec(
            "uv",
            "run",
            "chunkhound",
            "mcp",
            "--stdio",
            str(project_dir),
            cwd=str(home_dir),  # Run from different directory
            env=mcp_env,
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
        )

        client = SubprocessJsonRpcClient(proc)
        await client.start()

        try:
            # Send initialize request and get response
            response = await client.send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0.0"},
                },
                timeout=5.0,
            )
            print(f"Parsed response: {response}")

            # Success! The server started and responded
            # This means it correctly used the positional directory argument
            print("✓ MCP server correctly used positional directory argument")
            print("✓ Server started successfully from different working directory")

        finally:
            # Read stderr before closing (subprocess may be blocking on it)
            stderr_text = ""
            if proc.stderr:
                try:
                    stderr_bytes = await asyncio.wait_for(
                        proc.stderr.read(), timeout=0.5
                    )
                    stderr_text = stderr_bytes.decode()
                    print(f"stderr: {stderr_text}")
                except asyncio.TimeoutError:
                    print("No stderr output captured")
                    pass

            await client.close()

            # Check if we got the error we're testing for
            if stderr_text and "No ChunkHound project found" in stderr_text:
                # This would be the bug we're testing for
                pytest.fail(
                    "MCP server failed to use positional directory argument. "
                    f"Error: {stderr_text}"
                )

    finally:
        # Cleanup
        import shutil

        shutil.rmtree(home_dir, ignore_errors=True)
        shutil.rmtree(project_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_mcp_server_handles_empty_directory_gracefully():
    """Test that MCP server handles directories without config files gracefully.

    After the fix, the server should be able to start even when pointing to
    a directory that doesn't have a .chunkhound.json file and properly
    respond to MCP protocol initialization.
    """
    import shutil

    home_dir = Path(tempfile.mkdtemp())
    project_dir = Path(tempfile.mkdtemp())

    try:
        # Don't create any config files - test graceful handling

        # Run MCP server from home_dir with project_dir as argument
        proc = await asyncio.create_subprocess_exec(
            "uv",
            "run",
            "chunkhound",
            "mcp",
            "--stdio",
            str(project_dir),
            cwd=str(home_dir),
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
        )

        client = SubprocessJsonRpcClient(proc)
        await client.start()

        try:
            # Step 1: Send initialize request and receive response
            init_result = await client.send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0"},
                },
                timeout=5.0,
            )

            # Verify we got a valid response with correct structure
            assert "serverInfo" in init_result
            assert "protocolVersion" in init_result

            print(f"✓ Server responded with serverInfo: {init_result['serverInfo']}")

            # Step 2: Send initialized notification (no response expected)
            await client.send_notification("notifications/initialized")

            # Step 3: Test that server is now ready by requesting tools list
            tools_result = await client.send_request("tools/list", timeout=5.0)

            # Verify tools response
            assert "tools" in tools_result

            print(
                f"✓ Server initialized successfully with {len(tools_result['tools'])} tools"
            )
            print("✓ Server handles empty directory gracefully")

        finally:
            await client.close()

    finally:
        shutil.rmtree(home_dir, ignore_errors=True)
        shutil.rmtree(project_dir, ignore_errors=True)
