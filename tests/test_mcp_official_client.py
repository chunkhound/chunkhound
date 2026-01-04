#!/usr/bin/env python3
"""Test MCP server using official MCP client SDK."""

import asyncio
import json
import os
import tempfile
from pathlib import Path

from tests.utils import SubprocessJsonRpcClient

# Try to use the official MCP client SDK
try:
    import mcp.client.stdio
    from mcp.types import Tool

    HAS_MCP_CLIENT = True
except ImportError:
    HAS_MCP_CLIENT = False
    print("Official MCP client not available, trying manual approach...")


async def test_mcp_with_official_client():
    """Test MCP server using the official MCP client SDK."""
    if not HAS_MCP_CLIENT:
        print("Skipping official client test - MCP client SDK not available")
        return False

    # Create a minimal test project
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Test directory: {temp_dir}")

    try:
        # Create test content
        (temp_dir / "test.py").write_text("def hello(): return 'world'")

        # Create config
        config_path = temp_dir / ".chunkhound.json"
        db_path = temp_dir / ".chunkhound" / "test.db"
        db_path.parent.mkdir(exist_ok=True)

        config_content = {
            "database": {"path": str(db_path), "provider": "duckdb"},
            "indexing": {"include": ["*.py"]},
        }
        config_path.write_text(json.dumps(config_content, indent=2))

        # Index the content
        print("Indexing...")
        index_process = await asyncio.create_subprocess_exec(
            "uv",
            "run",
            "chunkhound",
            "index",
            str(temp_dir),
            "--no-embeddings",
            cwd=temp_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await index_process.communicate()

        if index_process.returncode != 0:
            print(f"Index failed: {stderr.decode()}")
            return False

        print("Index completed successfully")

        # Start MCP server
        print("Starting MCP server...")
        mcp_env = os.environ.copy()
        # Clear existing env vars
        for key in list(mcp_env.keys()):
            if key.startswith("CHUNKHOUND_"):
                del mcp_env[key]

        mcp_env.update(
            {
                "CHUNKHOUND_PROJECT_ROOT": str(temp_dir),
                "CHUNKHOUND_DATABASE__PATH": str(db_path),
                "CHUNKHOUND_MCP_MODE": "1",
            }
        )

        # Use the official MCP client
        try:
            async with mcp.client.stdio.stdio_client(
                "uv",
                "run",
                "chunkhound",
                "mcp",
                "--stdio",
                str(temp_dir),
                env=mcp_env,
                cwd=temp_dir,
            ) as client:
                print("MCP client connected successfully!")

                # Initialize the client
                await client.initialize()
                print("MCP client initialized successfully!")

                # List available tools
                tools = await client.list_tools()
                print(f"Available tools: {[tool.name for tool in tools.tools]}")

                # Test a search
                if any(tool.name == "search_regex" for tool in tools.tools):
                    result = await client.call_tool(
                        "search_regex", {"pattern": "hello"}
                    )
                    print(f"Search result: {result}")
                    return True
                else:
                    print("search_regex tool not available")
                    return False

        except Exception as e:
            print(f"Official MCP client failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    finally:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


async def test_mcp_manual_proper_protocol():
    """Test MCP server with manually implemented but proper JSON-RPC protocol."""

    # Create a minimal test project
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Manual test directory: {temp_dir}")

    try:
        # Create test content
        (temp_dir / "test.py").write_text("def hello(): return 'world'")

        # Create config
        config_path = temp_dir / ".chunkhound.json"
        db_path = temp_dir / ".chunkhound" / "test.db"
        db_path.parent.mkdir(exist_ok=True)

        config_content = {
            "database": {"path": str(db_path), "provider": "duckdb"},
            "indexing": {"include": ["*.py"]},
        }
        config_path.write_text(json.dumps(config_content, indent=2))

        # Index the content
        print("Manual: Indexing...")
        index_process = await asyncio.create_subprocess_exec(
            "uv",
            "run",
            "chunkhound",
            "index",
            str(temp_dir),
            "--no-embeddings",
            cwd=temp_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await index_process.communicate()

        if index_process.returncode != 0:
            print(f"Manual: Index failed: {stderr.decode()}")
            return False

        print("Manual: Index completed successfully")

        # Start MCP server
        print("Manual: Starting MCP server...")
        mcp_env = os.environ.copy()
        # Clear existing env vars
        for key in list(mcp_env.keys()):
            if key.startswith("CHUNKHOUND_"):
                del mcp_env[key]

        mcp_env["CHUNKHOUND_MCP_MODE"] = "1"

        mcp_process = await asyncio.create_subprocess_exec(
            "uv",
            "run",
            "chunkhound",
            "mcp",
            "--stdio",
            str(temp_dir),
            cwd=temp_dir,
            env=mcp_env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Wait for startup
        await asyncio.sleep(2)

        if mcp_process.returncode is not None:
            stdout, stderr = await mcp_process.communicate()
            print(f"Manual: MCP server failed: {stderr.decode()}")
            return False

        print("Manual: MCP server started, testing handshake...")

        # Create JSON-RPC client
        client = SubprocessJsonRpcClient(mcp_process)
        await client.start()

        try:
            # 1. Send initialize
            init_params = {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "manual-test", "version": "1.0.0"},
            }

            print(f"Manual: Sending initialize: {json.dumps(init_params)}")

            # 2. Read initialize response
            try:
                init_result = await client.send_request(
                    "initialize", init_params, timeout=10.0
                )
                print(f"Manual: Initialize response: {json.dumps(init_result)}")
                print("Manual: Initialize successful!")

                # 3. Send initialized notification (no response expected)
                print("Manual: Sending initialized notification")
                await client.send_notification("initialized", {})

                # Wait a moment for server to process
                await asyncio.sleep(0.5)

                # 4. Test a tool call
                search_params = {
                    "name": "search_regex",
                    "arguments": {"pattern": "hello"},
                }

                print(f"Manual: Sending search request: {json.dumps(search_params)}")

                # Read search response
                search_result = await client.send_request(
                    "tools/call", search_params, timeout=10.0
                )
                print(f"Manual: Search response: {json.dumps(search_result)}")
                print("Manual: ✓ Search successful!")
                return True

            except Exception as e:
                print(f"Manual: Request failed: {e}")
                import traceback

                traceback.print_exc()
                return False

        finally:
            # Cleanup
            await client.close()

    finally:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


async def main():
    """Run both official and manual MCP client tests."""
    print("=" * 60)
    print("Testing MCP server communication")
    print("=" * 60)

    # Test 1: Official MCP client
    print("\n1. Testing with official MCP client SDK...")
    official_success = await test_mcp_with_official_client()

    # Test 2: Manual protocol implementation
    print("\n2. Testing with manual JSON-RPC protocol...")
    manual_success = await test_mcp_manual_proper_protocol()

    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"Official MCP client: {'✓ SUCCESS' if official_success else '✗ FAILED'}")
    print(f"Manual JSON-RPC:     {'✓ SUCCESS' if manual_success else '✗ FAILED'}")
    print("=" * 60)

    return official_success or manual_success


if __name__ == "__main__":
    asyncio.run(main())
