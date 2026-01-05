"""Test MCP server initialization timeout behavior.

This test reproduces the VS Code MCP integration issue where the server
hangs during initialization due to synchronous directory scanning.

IMPORTANT: This file contains stress tests that create thousands of files
and directories to test performance under load. The large stress tests are
automatically skipped in CI environments due to GitHub Actions I/O
limitations.

To run stress tests locally:
  pytest tests/test_mcp_initialization_timeout.py -v

To run only the stress tests:
  pytest tests/test_mcp_initialization_timeout.py -k "large or deep" -v
"""

import asyncio
import json
import os
import time

import pytest

from tests.utils import JsonRpcTimeoutError, SubprocessJsonRpcClient
from tests.utils.windows_compat import database_cleanup_context, windows_safe_tempdir


class TestMCPInitializationTimeout:
    """Test MCP server initialization timeout scenarios."""

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Stress test with 1000 files too slow for CI",
    )
    @pytest.mark.asyncio
    async def test_mcp_initialization_timeout_on_large_directory(self):
        """Test MCP server times out on initialize with large directories.

        This reproduces the VS Code MCP integration issue where the server
        becomes unresponsive during directory scanning.

        NOTE: This is a stress test that creates 1000 files with
        substantial content. It's skipped in CI due to GitHub Actions I/O
        performance limitations. Run locally to verify performance fixes.
        """
        with windows_safe_tempdir() as temp_path:
            # Create many files to simulate a large codebase
            # that takes time to scan. This reproduces the VS Code issue
            # with large directories.
            for i in range(1000):  # Create enough files to cause significant delay
                subdir = temp_path / f"module_{i // 100}"
                subdir.mkdir(exist_ok=True)

                test_file = subdir / f"file_{i}.py"
                # Add substantial content that will create many chunks during scanning
                # This simulates real-world codebases with large files
                test_file.write_text(f"""
# File {i} - Large content to slow down chunking and indexing

def function_{i}():
    '''Function {i} for testing initialization timeout.
    
    This function has extensive documentation to create more chunks
    during the parsing and indexing process. The goal is to simulate
    a real-world codebase that would take significant time to process.
    '''
    data = {{
        "id": {i},
        "name": "function_{i}",
        "description": "A test function that does various operations",
        "operations": [
            "data_processing",
            "file_handling", 
            "network_requests",
            "database_operations"
        ]
    }}
    
    # Perform some operations
    for operation in data["operations"]:
        process_operation(operation, data["id"])
        
    return data

class Class_{i}:
    '''Class {i} for testing large directory initialization.
    
    This class contains multiple methods and properties to increase
    the parsing and chunking work during initialization.
    '''
    
    def __init__(self):
        self.data = {{
            "class_id": {i},
            "methods": [],
            "properties": [],
            "metadata": {{
                "created": "2025-01-01",
                "version": "1.0.{i}",
                "author": "test_system"
            }}
        }}
    
    def method_{i}(self):
        '''Method {i} implementation.'''
        return self.function_{i}()
        
    def function_{i}(self):
        '''Another function in class {i}.'''
        return f"class_value_{{self.data['class_id']}}"
        
    def process_data(self):
        '''Process data for class {i}.'''
        results = []
        for j in range(10):
            result = {{
                "iteration": j,
                "value": f"{{self.data['class_id']}}_{{j}}",
                "timestamp": "2025-01-01T00:00:00Z"
            }}
            results.append(result)
        return results

def process_operation(operation, operation_id):
    '''Process an operation for function {i}.'''
    print(f"Processing {{operation}} with ID {{operation_id}}")
    return f"{{operation}}_completed_{{operation_id}}"

# Additional module-level code to increase parsing work
CONSTANTS_{i} = {{
    "MAX_RETRIES": 3,
    "TIMEOUT": 30,
    "BUFFER_SIZE": 1024,
    "VERSION": "1.{i}.0"
}}

# More content to make the file substantial
CONFIGURATION_{i} = {{
    "database": {{
        "host": "localhost",
        "port": 5432,
        "name": "test_db_{i}"
    }},
    "cache": {{
        "enabled": True,
        "ttl": 3600,
        "size": 100
    }},
    "logging": {{
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }}
}}
""")

            # Create minimal config
            config_path = temp_path / ".chunkhound.json"
            db_path = temp_path / ".chunkhound" / "test.db"
            db_path.parent.mkdir(exist_ok=True)

            config = {
                "database": {"path": str(db_path), "provider": "duckdb"},
                "indexing": {"include": ["*.py"]},
            }
            config_path.write_text(json.dumps(config))

            # Use database cleanup context to ensure proper resource management
            with database_cleanup_context():
                # Start MCP server (auto-indexes on startup)
                mcp_env = os.environ.copy()
                mcp_env["CHUNKHOUND_MCP_MODE"] = "1"

                proc = await asyncio.create_subprocess_exec(
                    "uv",
                    "run",
                    "chunkhound",
                    "mcp",
                    "stdio",
                    str(temp_path),
                    cwd=temp_path,
                    env=mcp_env,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

            client = SubprocessJsonRpcClient(proc)
            await client.start()

            try:
                # After fix: server should respond quickly with large dirs
                # VS Code waits about 5 seconds, should be under 3 seconds
                start_time = time.time()
                init_result = await client.send_request(
                    "initialize",
                    {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1.0"},
                    },
                    timeout=5.0,
                )
                response_time = time.time() - start_time

                # Verify response within VS Code timeout
                # (should be under 5 seconds)
                assert response_time < 5.0, (
                    f"Server took {response_time:.2f} seconds "
                    f"(should be < 5s to avoid VS Code timeout)"
                )

                print(
                    f"✅ Server responded in {response_time:.2f} seconds "
                    f"(within VS Code timeout)"
                )

                # Verify proper response structure
                assert "serverInfo" in init_result, (
                    f"No serverInfo in result: {init_result}"
                )
                assert init_result["serverInfo"]["name"] == "ChunkHound Code Search"

                # Verify the server process is still running (not crashed)
                assert proc.returncode is None, (
                    "Server process crashed during initialization"
                )

            finally:
                await client.close()

    @pytest.mark.asyncio
    async def test_mcp_initialization_responds_quickly_small_directory(self):
        """Test that MCP server responds quickly with small directories.

        This verifies normal behavior and will help validate the fix.
        """
        with windows_safe_tempdir() as temp_path:
            # Create just a few files - should initialize quickly
            test_file = temp_path / "test.py"
            test_file.write_text("def hello(): return 'world'")

            # Create minimal config
            config_path = temp_path / ".chunkhound.json"
            db_path = temp_path / ".chunkhound" / "test.db"
            db_path.parent.mkdir(exist_ok=True)

            config = {
                "database": {"path": str(db_path), "provider": "duckdb"},
                "indexing": {"include": ["*.py"]},
            }
            config_path.write_text(json.dumps(config))

            # Use database cleanup context to ensure proper resource management
            with database_cleanup_context():
                # Start MCP server
                mcp_env = os.environ.copy()
                mcp_env["CHUNKHOUND_MCP_MODE"] = "1"

                proc = await asyncio.create_subprocess_exec(
                    "uv",
                    "run",
                    "chunkhound",
                    "mcp",
                    "stdio",
                    str(temp_path),
                    cwd=temp_path,
                    env=mcp_env,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

            client = SubprocessJsonRpcClient(proc)
            await client.start()

            try:
                # Should respond quickly with small directory
                init_result = await client.send_request(
                    "initialize",
                    {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1.0"},
                    },
                    timeout=10.0,
                )

                # Verify response structure
                assert "serverInfo" in init_result, (
                    f"No serverInfo in result: {init_result}"
                )
                assert init_result["serverInfo"]["name"] == "ChunkHound Code Search"

            finally:
                await client.close()

    @pytest.mark.asyncio
    async def test_mcp_initialization_eventual_response(self):
        """Test that MCP server eventually responds even with large directories.

        This verifies the server doesn't crash, just takes too long for VS Code.
        """
        with windows_safe_tempdir() as temp_path:
            # Create moderate number of files - enough to delay but not excessive
            for i in range(50):
                test_file = temp_path / f"test_{i}.py"
                test_file.write_text(f"def function_{i}(): return {i}")

            # Create minimal config
            config_path = temp_path / ".chunkhound.json"
            db_path = temp_path / ".chunkhound" / "test.db"
            db_path.parent.mkdir(exist_ok=True)

            config = {
                "database": {"path": str(db_path), "provider": "duckdb"},
                "indexing": {"include": ["*.py"]},
            }
            config_path.write_text(json.dumps(config))

            # Use database cleanup context to ensure proper resource management
            with database_cleanup_context():
                # Start MCP server
                mcp_env = os.environ.copy()
                mcp_env["CHUNKHOUND_MCP_MODE"] = "1"

                proc = await asyncio.create_subprocess_exec(
                    "uv",
                    "run",
                    "chunkhound",
                    "mcp",
                    "stdio",
                    str(temp_path),
                    cwd=temp_path,
                    env=mcp_env,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

            client = SubprocessJsonRpcClient(proc)
            await client.start()

            try:
                # Give it enough time to eventually respond
                # (but VS Code wouldn't wait this long)
                init_result = await client.send_request(
                    "initialize",
                    {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1.0"},
                    },
                    timeout=60.0,
                )

                # Verify it eventually works
                assert "serverInfo" in init_result, (
                    f"No serverInfo in result: {init_result}"
                )

            finally:
                await client.close()

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Stress test with 2500 directories too slow for CI",
    )
    @pytest.mark.asyncio
    async def test_watchdog_recursive_blocking_on_deep_directory_tree(self):
        """Test that watchdog's recursive directory traversal blocks MCP initialization.

        This test creates a deeply nested directory structure that causes
        watchdog's observer.schedule(..., recursive=True) to block for
        several seconds during filesystem watch setup.

        The key difference from other tests is that this creates many directories
        rather than many files, which is what causes watchdog blocking.

        NOTE: This is a stress test that creates 2500 directories in a deep hierarchy.
        It's skipped in CI due to GitHub Actions I/O performance limitations.
        Run locally to verify watchdog performance fixes.
        """
        with windows_safe_tempdir() as temp_path:
            # Create deeply nested directory structure
            # Watchdog blocks on directory traversal, not file count
            for i in range(50):  # 50 top-level dirs
                level1 = temp_path / f"project_{i}"
                level1.mkdir()

                for j in range(10):  # 10 subdirs each
                    level2 = level1 / f"module_{j}"
                    level2.mkdir()

                    for k in range(5):  # 5 more subdirs each
                        level3 = level2 / f"submodule_{k}"
                        level3.mkdir()

                        # Add one small file per leaf directory
                        (level3 / "code.py").write_text("# code")

            print(f"Created {50 * 10 * 5} directories for watchdog traversal test")

            # Setup minimal chunkhound config
            config_path = temp_path / ".chunkhound.json"
            db_path = temp_path / ".chunkhound" / "test.db"
            db_path.parent.mkdir(exist_ok=True)

            config = {
                "database": {"path": str(db_path), "provider": "duckdb"},
                "indexing": {"include": ["*.py"]},
            }
            config_path.write_text(json.dumps(config))

            # Use database cleanup context to ensure proper resource management
            with database_cleanup_context():
                # Start MCP server with deep directory structure
                mcp_env = os.environ.copy()
                mcp_env["CHUNKHOUND_MCP_MODE"] = "1"

                proc = await asyncio.create_subprocess_exec(
                    "uv",
                    "run",
                    "chunkhound",
                    "mcp",
                    "stdio",
                    str(temp_path),
                    cwd=temp_path,
                    env=mcp_env,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

            client = SubprocessJsonRpcClient(proc)
            await client.start()

            try:
                # This SHOULD timeout due to watchdog blocking
                # After fix: should respond quickly
                start_time = time.time()
                try:
                    init_result = await client.send_request(
                        "initialize",
                        {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {"name": "test", "version": "1.0"},
                        },
                        timeout=5.0,  # VS Code timeout
                    )
                    response_time = time.time() - start_time

                    # If we get here, the bug is fixed!
                    print(
                        f"✅ Server responded in {response_time:.2f}s - bug is fixed!"
                    )

                    # Verify proper response structure
                    assert "serverInfo" in init_result, (
                        f"No serverInfo in result: {init_result}"
                    )
                    assert init_result["serverInfo"]["name"] == "ChunkHound Code Search"

                except JsonRpcTimeoutError:
                    # This is expected with the current bug
                    print(
                        "❌ Server timed out during initialization "
                        "(watchdog blocking bug confirmed)"
                    )
                    raise AssertionError(
                        "MCP server timed out during initialization due to "
                        "watchdog recursive directory traversal blocking. "
                        "This reproduces the VS Code integration issue where "
                        "the server becomes unresponsive."
                    )

            finally:
                await client.close()
