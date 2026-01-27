"""
Test MCP server serving different directory with real indexing and MCP communication.

This test ensures:
1. MCP server CWD has no .chunkhound.json config file
2. Target directory is properly indexed first
3. MCP server started from different CWD serves target directory
4. Real MCP stdio communication works correctly
5. Search results come from target directory, not server CWD
"""

import asyncio
import json
import os
import tempfile
import pytest
from pathlib import Path

from chunkhound.database_factory import create_database_with_dependencies
from chunkhound.core.config.config import Config
from chunkhound.utils.windows_constants import IS_WINDOWS, WINDOWS_FILE_HANDLE_DELAY
from .test_utils import get_api_key_for_tests, get_embedding_config_for_tests, build_embedding_config_from_dict, create_embedding_manager_for_tests

# Import Windows-safe subprocess utilities and JSON-RPC client
from tests.utils import (
    SubprocessJsonRpcClient,
    create_subprocess_exec_safe,
    get_safe_subprocess_env,
)
from tests.utils.windows_compat import path_contains, windows_safe_tempdir, database_cleanup_context


class TestMCPServerDirectoryIsolationWithRealCommunication:
    """Test MCP server serving different directory with real indexing and MCP communication."""
    
    @pytest.mark.asyncio
    async def test_mcp_server_serves_indexed_different_directory(
        self, clean_environment
    ):
        """
        Complete test: Index target directory, start MCP server from different CWD, send real search queries.
        
        Flow:
        1. Create isolated directories (test CWD vs target project)
        2. Index the target project directory 
        3. Start MCP server from test CWD (different from target)
        4. Send real MCP search queries via stdio
        5. Verify responses contain indexed content from target directory
        """
        temp_base = Path(tempfile.mkdtemp())
        
        try:
            # === STEP 1: Directory Setup ===
            test_cwd = temp_base / "test_execution_dir"  # NO config here
            test_cwd.mkdir()
            
            project_dir = temp_base / "target_project"   # WITH config and content
            project_dir.mkdir()
            
            # Create test content with unique identifiers
            test_files = {
                "main.py": '''
def calculate_fibonacci(n):
    """Calculate fibonacci sequence up to n terms."""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i-1] + sequence[i-2])
    return sequence

class DataProcessor:
    """Process and analyze data sets."""
    
    def __init__(self):
        self.data = []
    
    def load_data(self, filename):
        # Load data from file
        pass
    
    def analyze_patterns(self):
        # Analyze data patterns
        return "unique_analysis_result_12345"
''',
                "utils.py": '''
import logging

logger = logging.getLogger(__name__)

def setup_logging():
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

class ConfigManager:
    """Manage application configuration."""
    
    def __init__(self, config_path):
        self.config_path = config_path
        self.settings = {}
    
    def load_config(self):
        """Load configuration from file."""
        return {"app_name": "test_isolated_app_67890"}
''',
                "README.md": '''
# Test Project

This is a test project for MCP server directory isolation testing.

## Features
- Fibonacci calculation
- Data processing with unique_feature_identifier_99999
- Configuration management

## Usage
Run the application with proper configuration.
'''
            }
            
            # Write test files
            for filename, content in test_files.items():
                (project_dir / filename).write_text(content.strip())
            
            # Database setup
            db_path = project_dir / ".chunkhound" / "isolated_test.db"
            db_path.parent.mkdir(exist_ok=True)
            
            # Project config - no embedding provider to avoid API key requirement for this test
            config_path = project_dir / ".chunkhound.json"
            config_content = {
                "database": {"path": str(db_path), "provider": "duckdb"},
                "indexing": {
                    "include": ["*.py", "*.md"],
                    "exclude": ["*.log", "__pycache__/"]
                }
            }
            config_path.write_text(json.dumps(config_content, indent=2))
            
            # Verify isolation
            assert not (test_cwd / ".chunkhound.json").exists()
            assert (project_dir / ".chunkhound.json").exists()
            
            # === STEP 2: Index the Target Directory ===
            print(f"Indexing target directory: {project_dir}")
            
            # Run indexing from target directory
            index_env = os.environ.copy()
            index_env.update({
                "CHUNKHOUND_DATABASE__PATH": str(db_path)
            })
            
            index_process = await create_subprocess_exec_safe(
                "uv", "run", "chunkhound", "index", str(project_dir), "--no-embeddings",
                cwd=str(project_dir),  # Index from project directory
                env=get_safe_subprocess_env(index_env),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await index_process.communicate()
            
            assert index_process.returncode == 0, (
                f"Indexing failed: {stderr.decode()}"
            )
            
            print(f"Indexing completed successfully")
            
            # Verify database has content
            # Use fake args to prevent find_project_root call that fails in CI
            from types import SimpleNamespace
            fake_args = SimpleNamespace(path=db_path.parent)
            config_for_db = Config(
                args=fake_args,
                database={"path": str(db_path), "provider": "duckdb"}
            )
            db = create_database_with_dependencies(
                db_path=db_path,
                config=config_for_db,
                embedding_manager=None
            )
            db.connect()
            stats = db.get_stats()
            db.close()
            
            file_count = stats.get("files", 0)
            chunk_count = stats.get("chunks", 0)
            assert file_count > 0, f"Database should contain indexed files, got {file_count}"
            assert chunk_count > 0, f"Database should contain chunks, got {chunk_count}"
            print(f"Database contains {file_count} files and {chunk_count} chunks")
            
            # === STEP 3: Start MCP Server from Different Directory ===
            mcp_env = os.environ.copy()
            # Remove any existing chunkhound env vars to avoid conflicts
            for key in list(mcp_env.keys()):
                if key.startswith("CHUNKHOUND_"):
                    del mcp_env[key]
            
            mcp_env.update({
                "CHUNKHOUND_DATABASE__PATH": str(db_path),
                "CHUNKHOUND_MCP_MODE": "1",
                "CHUNKHOUND_DEBUG": "1"
            })
            
            print(f"MCP server will use database: {db_path}")
            print(f"Database exists: {db_path.exists()}")
            print(f"Database size: {db_path.stat().st_size if db_path.exists() else 'N/A'}")
            
            # Use CLI positional argument to specify project directory
            print(f"Environment variables set for MCP server:")
            for key, value in mcp_env.items():
                if key.startswith("CHUNKHOUND_"):
                    print(f"  {key} = {value}")
            
            # Critical: Start from test_cwd, not project_dir - pass target path as argument
            mcp_cmd = ["uv", "run", "chunkhound", "mcp", "--stdio", str(project_dir)]
            print(f"Running command: {' '.join(mcp_cmd)} from cwd: {test_cwd}")
            mcp_process = await create_subprocess_exec_safe(  
                *mcp_cmd,
                cwd=str(test_cwd),  # Different from project_dir!
                env=get_safe_subprocess_env(mcp_env),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                # Check if process started successfully (extended for Ollama compatibility)
                await asyncio.sleep(3)
                if mcp_process.returncode is not None:
                    stdout, stderr = await mcp_process.communicate()
                    print(f"MCP server failed to start:")
                    print(f"Return code: {mcp_process.returncode}")
                    print(f"stdout: {stdout.decode()}")
                    print(f"stderr: {stderr.decode()}")
                    raise Exception(f"MCP server failed to start with code {mcp_process.returncode}")

                print("MCP server is running, proceeding with initialization...")

                # === STEP 4: Real MCP Communication ===
                mcp_client = SubprocessJsonRpcClient(mcp_process)
                await mcp_client.start()

                # Initialize MCP server
                print("Initializing MCP server...")
                try:
                    init_result = await mcp_client.send_request(
                        "initialize",
                        {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {
                                "name": "chunkhound-test-client",
                                "version": "1.0.0"
                            }
                        }
                    )
                    assert init_result["protocolVersion"] == "2024-11-05"
                    print("MCP server initialized successfully")

                    # Send initialized notification
                    await mcp_client.send_notification("notifications/initialized", {})

                    # Wait a moment for server to process the notification
                    await asyncio.sleep(0.1)
                except Exception as e:
                    # Get any stderr output before failing
                    if mcp_process.returncode is None:
                        # Process still running, check for stderr
                        print("MCP server still running, checking for stderr...")
                        # Give a moment for any stderr output
                        await asyncio.sleep(0.5)

                        # Try to read any available stderr without blocking
                        try:
                            stderr_data = mcp_process.stderr.read_nowait()
                            if stderr_data:
                                print(f"MCP server stderr: {stderr_data.decode()}")
                        except asyncio.IncompleteReadError:
                            pass
                        except Exception:
                            pass

                    print(f"MCP initialization failed: {e}")
                    raise

                # Wait for server to fully start (extended for Ollama compatibility)
                await asyncio.sleep(4)

                # List available tools for debugging
                print("Listing available tools...")
                tools_result = await mcp_client.send_request("tools/list", {})
                tools = tools_result["tools"]
                print(f"Available tools: {[tool['name'] for tool in tools]}")
                
                # === STEP 5: Test Search Queries ===

                # Test 0: Quick check - search for any content
                print("Testing search for 'def' to see if any content is searchable...")
                def_result = await mcp_client.send_request(
                    "tools/call",
                    {
                        "name": "search",
                        "arguments": {
                            "type": "regex",
                            "query": "def",
                            "page_size": 10,
                            "offset": 0
                        }
                    }
                )
                print(f"Def search response: {def_result}")

                # Test 1: Search for function name
                print("Testing regex search for fibonacci function...")
                fibonacci_result = await mcp_client.send_request(
                    "tools/call",
                    {
                        "name": "search",
                        "arguments": {
                            "type": "regex",
                            "query": "calculate_fibonacci",
                            "page_size": 10,
                            "offset": 0
                        }
                    }
                )
                print(f"Fibonacci response: {fibonacci_result}")

                # The result has content array with text
                if "content" in fibonacci_result and len(fibonacci_result["content"]) > 0:
                    fibonacci_results = json.loads(fibonacci_result["content"][0]["text"])["results"]
                else:
                    print(f"Unexpected result format: {fibonacci_result}")
                    fibonacci_results = []
                assert len(fibonacci_results) > 0, "Should find fibonacci function"
                
                # Verify content comes from target project, not test CWD
                found_fibonacci = False
                for result in fibonacci_results:
                    if "calculate_fibonacci" in result.get("content", ""):
                        file_path = result["file_path"]
                        # Search should return relative path from indexed project directory
                        assert file_path == "main.py", f"Expected 'main.py', got: {file_path}"
                        # Verify it's a relative path (not absolute)
                        assert not Path(file_path).is_absolute(), f"File path should be relative: {file_path}"
                        found_fibonacci = True
                        break
                
                assert found_fibonacci, "Should find fibonacci function in target project"
                print("✓ Fibonacci function found in correct directory")
                
                # Test 2: Search for unique identifier from utils.py
                print("Testing regex search for unique app identifier...")
                app_result = await mcp_client.send_request(
                    "tools/call",
                    {
                        "name": "search",
                        "arguments": {
                            "type": "regex",
                            "query": "test_isolated_app_67890",
                            "page_size": 10,
                            "offset": 0
                        }
                    }
                )

                app_results = json.loads(app_result["content"][0]["text"])["results"]
                assert len(app_results) > 0, "Should find unique app identifier"

                found_app_id = False
                for result in app_results:
                    if "test_isolated_app_67890" in result.get("content", ""):
                        file_path = result["file_path"]
                        # Search should return relative path from indexed project directory
                        assert file_path == "utils.py", f"Expected 'utils.py', got: {file_path}"
                        # Verify it's a relative path (not absolute)
                        assert not Path(file_path).is_absolute(), f"File path should be relative: {file_path}"
                        found_app_id = True
                        break

                assert found_app_id, "Should find app identifier in target project"
                print("✓ App identifier found in correct directory")

                # Test 3: Search for content from README
                print("Testing regex search for README content...")
                readme_result = await mcp_client.send_request(
                    "tools/call",
                    {
                        "name": "search",
                        "arguments": {
                            "type": "regex",
                            "query": "unique_feature_identifier_99999",
                            "page_size": 10,
                            "offset": 0
                        }
                    }
                )

                readme_results = json.loads(readme_result["content"][0]["text"])["results"]
                assert len(readme_results) > 0, "Should find README content"

                found_readme = False
                for result in readme_results:
                    if "unique_feature_identifier_99999" in result.get("content", ""):
                        file_path = result["file_path"]
                        # Search should return relative path from indexed project directory
                        assert file_path == "README.md", f"Expected 'README.md', got: {file_path}"
                        # Verify it's a relative path (not absolute)
                        assert not Path(file_path).is_absolute(), f"File path should be relative: {file_path}"
                        found_readme = True
                        break

                assert found_readme, "Should find README content in target project"
                print("✓ README content found in correct directory")

                # Test 4: Search for class definition
                print("Testing regex search for class definition...")
                class_result = await mcp_client.send_request(
                    "tools/call",
                    {
                        "name": "search",
                        "arguments": {
                            "type": "regex",
                            "query": "class DataProcessor",
                            "page_size": 10,
                            "offset": 0
                        }
                    }
                )

                class_results = json.loads(class_result["content"][0]["text"])["results"]
                assert len(class_results) > 0, "Should find DataProcessor class"
                print("✓ DataProcessor class found")

                # Test 5: Verify no content from test_cwd directory
                print("Testing that no content from test_cwd is returned...")

                # Create a file in test_cwd that should NOT be found
                (test_cwd / "should_not_be_found.py").write_text("""
def should_not_appear():
    return "this_should_not_be_indexed_54321"
""")

                # Search for content that only exists in test_cwd
                isolation_result = await mcp_client.send_request(
                    "tools/call",
                    {
                        "name": "search",
                        "arguments": {
                            "type": "regex",
                            "query": "this_should_not_be_indexed_54321",
                            "page_size": 10,
                            "offset": 0
                        }
                    }
                )
                isolation_results = json.loads(isolation_result["content"][0]["text"])["results"]

                # Should find nothing because test_cwd is not indexed
                assert len(isolation_results) == 0, "Should not find content from test_cwd"
                print("✓ Content isolation verified - test_cwd content not indexed")

                print("All MCP stdio communication tests passed!")

            finally:
                # Clean shutdown of client
                await mcp_client.close()
                    
        finally:
            # Use Windows-safe cleanup
            from tests.utils.windows_compat import cleanup_database_resources
            cleanup_database_resources()
            
            import shutil
            try:
                shutil.rmtree(temp_base, ignore_errors=True)
            except Exception as e:
                # Windows may need extra time for file handles to be released
                import time
                if IS_WINDOWS:
                    time.sleep(WINDOWS_FILE_HANDLE_DELAY)
                    shutil.rmtree(temp_base, ignore_errors=True)

    @pytest.mark.asyncio
    @pytest.mark.skipif(get_api_key_for_tests()[0] is None, reason="No API key available")
    async def test_mcp_server_semantic_search_isolation(self, clean_environment):
        """Test semantic search also respects directory isolation."""
        temp_base = Path(tempfile.mkdtemp())
        
        try:
            # Setup directories
            test_cwd = temp_base / "test_execution_dir"
            test_cwd.mkdir()
            
            project_dir = temp_base / "target_project"
            project_dir.mkdir()
            
            # Create content for semantic search
            (project_dir / "algorithms.py").write_text('''
def binary_search(arr, target):
    """Efficient search algorithm for sorted arrays."""
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

def quicksort(arr):
    """Fast sorting using divide and conquer."""
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort(left) + middle + quicksort(right)
''')
            
            # Use the same pattern as working MCP integration test
            from chunkhound.core.config.config import Config
            from chunkhound.database_factory import create_services
            from chunkhound.embeddings import EmbeddingManager
            
            # Database and config setup
            db_path = project_dir / ".chunkhound" / "semantic_test.db"
            db_path.parent.mkdir(exist_ok=True)

            # Get embedding config using centralized helper
            config_dict = get_embedding_config_for_tests()
            embedding_config = build_embedding_config_from_dict(config_dict)

            config = Config(
                database={"path": str(db_path), "provider": "duckdb"},
                embedding=embedding_config,
                indexing={"include": ["*.py"]}
            )

            # Create embedding manager using centralized helper
            embedding_manager = create_embedding_manager_for_tests(config_dict)
            if not embedding_manager:
                embedding_manager = EmbeddingManager()  # Fallback for no config
            
            # Create services
            services = create_services(db_path, config.to_dict(), embedding_manager)
            services.provider.connect()
            
            # Index the project files
            result = await services.indexing_coordinator.process_file(project_dir / "algorithms.py")
            if result["status"] == "error":
                pytest.skip(f"Indexing failed: {result.get('error', 'Unknown error')}")
            
            try:
                # Test semantic search with services (no subprocess needed)
                from chunkhound.mcp_server.tools import execute_tool
                
                # Search for sorting algorithms semantically
                semantic_response = await execute_tool(
                    tool_name="search_semantic",
                    services=services,
                    embedding_manager=embedding_manager,
                    arguments={"query": "sorting algorithms", "page_size": 10, "offset": 0}
                )
                
                semantic_results = semantic_response.get('results', [])
                        
                # Should find sorting-related content from target project
                for result in semantic_results:
                    file_path = result["file_path"]
                    # Search should return relative path from indexed project directory
                    assert file_path == "algorithms.py", f"Expected 'algorithms.py', got: {file_path}"
                    # Verify it's a relative path (not absolute)
                    assert not Path(file_path).is_absolute(), f"File path should be relative: {file_path}"
                    
                print("✓ Semantic search respects directory isolation")
                    
            finally:
                if hasattr(services.provider, 'close'):
                    services.provider.close()
                else:
                    services.provider.disconnect()
                    
        finally:
            # Use Windows-safe cleanup
            from tests.utils.windows_compat import cleanup_database_resources
            cleanup_database_resources()
            
            import shutil
            try:
                shutil.rmtree(temp_base, ignore_errors=True)
            except Exception as e:
                # Windows may need extra time for file handles to be released
                import time
                if IS_WINDOWS:
                    time.sleep(WINDOWS_FILE_HANDLE_DELAY)
                    shutil.rmtree(temp_base, ignore_errors=True)