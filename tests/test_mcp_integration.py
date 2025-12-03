"""Integration tests that use actual MCP server components.

These tests verify the real integration path that users experience:
Filesystem Event → Watchdog → AsyncHandler → IndexingCoordinator → Database → Search Tools
"""

import asyncio
import tempfile
from pathlib import Path
import pytest
import shutil
import time

from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services
from chunkhound.services.realtime_indexing_service import RealtimeIndexingService
from chunkhound.mcp_server.tools import execute_tool
from chunkhound.embeddings import EmbeddingManager
from .test_utils import get_api_key_for_tests


class TestMCPIntegration:
    """Test real MCP server integration with realtime indexing."""
    
    @pytest.fixture
    async def mcp_setup(self):
        """Setup MCP server with real services and temp directory."""
        # Get API key and provider configuration
        api_key, provider = get_api_key_for_tests()
        
        temp_dir = Path(tempfile.mkdtemp())
        db_path = temp_dir / ".chunkhound" / "test.db"
        watch_dir = temp_dir / "project"
        watch_dir.mkdir(parents=True)
        
        # Ensure database directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure embedding based on available API key
        embedding_config = None
        if api_key and provider:
            model = "text-embedding-3-small" if provider == "openai" else "voyage-3.5"
            embedding_config = {
                "provider": provider,
                "api_key": api_key,
                "model": model
            }
        
        # Use fake args to prevent find_project_root call that fails in CI
        from types import SimpleNamespace
        fake_args = SimpleNamespace(path=temp_dir)
        config = Config(
            args=fake_args,
            database={"path": str(db_path), "provider": "duckdb"},
            embedding=embedding_config,
            indexing={"include": ["*.py", "*.js"], "exclude": ["*.log"]}
        )
        
        # Create embedding manager if API key is available
        embedding_manager = None
        if api_key and provider:
            embedding_manager = EmbeddingManager()
            if provider == "openai":
                from chunkhound.providers.embeddings.openai_provider import OpenAIEmbeddingProvider
                embedding_provider = OpenAIEmbeddingProvider(api_key=api_key, model="text-embedding-3-small")
            elif provider == "voyageai":
                from chunkhound.providers.embeddings.voyageai_provider import VoyageAIEmbeddingProvider
                embedding_provider = VoyageAIEmbeddingProvider(api_key=api_key, model="voyage-3.5")
            else:
                embedding_provider = None
            
            if embedding_provider:
                embedding_manager.register_provider(embedding_provider, set_default=True)
        
        # Create services - this is what MCP server uses
        services = create_services(db_path, config, embedding_manager)
        services.provider.connect()


        # Initialize realtime indexing service (what MCP server should do)
        realtime_service = RealtimeIndexingService(services, config)
        await realtime_service.start(watch_dir)
        
        yield services, realtime_service, watch_dir, temp_dir, embedding_manager
        
        # Cleanup
        try:
            await realtime_service.stop()
        except Exception:
            pass
        
        try:
            services.provider.disconnect()
        except Exception:
            pass
            
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.skipif(get_api_key_for_tests()[0] is None, reason="No API key available")
    @pytest.mark.asyncio
    async def test_mcp_semantic_search_finds_new_files(self, mcp_setup):
        """Test that MCP semantic search finds newly created files."""
        services, realtime_service, watch_dir, _, embedding_manager = mcp_setup
        
        # Wait for initial scan
        await asyncio.sleep(1.0)
        
        # Get initial search results using MCP tool execution
        initial_results = await execute_tool(
            tool_name="search_semantic",
            services=services,
            embedding_manager=embedding_manager,
            arguments={
                "query": "unique_mcp_test_function",
                "page_size": 10,
                "offset": 0
            }
        )
        initial_count = len(initial_results.get('results', []))
        
        # Create new file with unique content
        new_file = watch_dir / "mcp_test.py"
        new_file.write_text("""
def unique_mcp_test_function():
    '''This is a unique function for MCP integration testing'''
    return "mcp_realtime_success"
""")
        
        # Wait for debounce + processing
        await asyncio.sleep(2.0)
        
        # Search for new content using MCP tool execution
        new_results = await execute_tool(
            tool_name="search_semantic",
            services=services,
            embedding_manager=embedding_manager,
            arguments={
                "query": "unique_mcp_test_function",
                "page_size": 10,
                "offset": 0
            }
        )
        new_count = len(new_results.get('results', []))
        
        assert new_count > initial_count, \
            f"MCP semantic search should find new file (was {initial_count}, now {new_count})"

    @pytest.mark.asyncio
    async def test_mcp_regex_search_finds_modified_files(self, mcp_setup):
        """Test that MCP regex search finds modified file content."""
        services, realtime_service, watch_dir, _, _ = mcp_setup
        
        # Create initial file
        test_file = watch_dir / "modify_test.py"
        test_file.write_text("def initial_function(): pass")
        
        # Wait for initial processing
        await asyncio.sleep(2.0)
        
        # Verify initial content is found
        initial_results = await execute_tool(
            tool_name="search_regex",
            services=services,
            embedding_manager=None,
            arguments={
                "pattern": "initial_function",
                "page_size": 10,
                "offset": 0
            }
        )
        assert len(initial_results.get('results', [])) > 0, "Initial content should be found"
        
        # Modify file with new unique content
        test_file.write_text("""
def initial_function(): pass

def modified_unique_regex_pattern():
    '''Added by modification - should be found by regex'''
    return "modification_success"
""")
        
        # Wait for debounce + processing
        await asyncio.sleep(2.0)
        
        # Search for modified content using MCP tool execution
        modified_results = await execute_tool(
            tool_name="search_regex",
            services=services,
            embedding_manager=None,
            arguments={
                "pattern": "modified_unique_regex_pattern",
                "page_size": 10,
                "offset": 0
            }
        )
        
        assert len(modified_results.get('results', [])) > 0, \
            "MCP regex search should find modified content"

    @pytest.mark.asyncio
    async def test_mcp_database_stats_change_with_realtime(self, mcp_setup):
        """Test that database stats reflect real-time indexing changes."""
        services, realtime_service, watch_dir, _, _ = mcp_setup
        
        # Wait for initial scan
        await asyncio.sleep(1.0)
        
        # Get initial stats
        initial_stats = await execute_tool(
            tool_name="get_stats",
            services=services,
            embedding_manager=None,
            arguments={}
        )
        initial_files = initial_stats.get('total_files', 0)
        initial_chunks = initial_stats.get('total_chunks', 0)
        
        # Create multiple new files
        for i in range(3):
            new_file = watch_dir / f"stats_test_{i}.py"
            new_file.write_text(f"""
def stats_test_function_{i}():
    '''File {i} for testing database stats updates'''
    return "stats_test_{i}"

class StatsTestClass_{i}:
    def method_{i}(self):
        pass
""")
        
        # Wait for all files to be processed
        await asyncio.sleep(3.0)
        
        # Get updated stats
        updated_stats = await execute_tool(
            tool_name="get_stats",
            services=services,
            embedding_manager=None,
            arguments={}
        )
        updated_files = updated_stats.get('total_files', 0)
        updated_chunks = updated_stats.get('total_chunks', 0)
        
        assert updated_files > initial_files, \
            f"File count should increase (was {initial_files}, now {updated_files})"
        assert updated_chunks > initial_chunks, \
            f"Chunk count should increase (was {initial_chunks}, now {updated_chunks})"

    @pytest.mark.asyncio
    async def test_mcp_search_after_file_deletion(self, mcp_setup):
        """Test that MCP search handles file deletions correctly."""
        services, realtime_service, watch_dir, _, _ = mcp_setup
        
        # Create file with unique content
        delete_file = watch_dir / "delete_test.py"
        delete_file.write_text("""
def delete_test_unique_function():
    '''This function will be deleted'''
    return "to_be_deleted"
""")
        
        # Wait for processing
        await asyncio.sleep(2.0)
        
        # Verify content is searchable
        before_delete = await execute_tool(
            tool_name="search_regex",
            services=services,
            embedding_manager=None,
            arguments={
                "pattern": "delete_test_unique_function",
                "page_size": 10,
                "offset": 0
            }
        )
        assert len(before_delete.get('results', [])) > 0, "Content should be found before deletion"
        
        # Delete the file
        delete_file.unlink()

        # Wait for deletion processing
        await asyncio.sleep(4.0)  # Deletion + 3s consistency interval
        
        # Verify content is no longer searchable
        after_delete = await execute_tool(
            tool_name="search_regex",
            services=services,
            embedding_manager=None,
            arguments={
                "pattern": "delete_test_unique_function",
                "page_size": 10,
                "offset": 0
            }
        )
        assert len(after_delete.get('results', [])) == 0, "Content should not be found after deletion"

    @pytest.mark.asyncio
    async def test_mcp_realtime_service_actually_starts(self, mcp_setup):
        """Test that realtime indexing service is actually running."""
        services, realtime_service, watch_dir, _, _ = mcp_setup
        
        # Check service state
        stats = await realtime_service.get_stats()
        
        assert stats.get('observer_alive', False), "Filesystem observer should be running"
        assert stats.get('watching_directory') == str(watch_dir), \
            f"Should be watching {watch_dir}, but watching {stats.get('watching_directory')}"
        
        # Verify service responds to filesystem events
        test_file = watch_dir / "service_test.py"
        test_file.write_text("def service_running_test(): pass")
        
        # Wait for processing
        await asyncio.sleep(2.0)
        
        # Check that file was actually processed
        # Use resolve() to get the real path (handles /private/var symlink on macOS)
        file_record = services.provider.get_file_by_path(str(test_file.resolve()))
        assert file_record is not None, "Realtime service should process new files"
    
    @pytest.mark.asyncio
    async def test_file_modification_detection_comprehensive(self, mcp_setup):
        """Comprehensive test to reproduce file modification detection issues."""
        services, realtime_service, watch_dir, _, _ = mcp_setup
        
        # Create initial file
        test_file = watch_dir / "comprehensive_modify_test.py"
        initial_content = """def original_function():
    return "version_1"
"""
        test_file.write_text(initial_content)
        
        # Wait for initial indexing
        await asyncio.sleep(2.5)
        
        # Verify initial content is indexed (use multiline-compatible regex)
        initial_results = services.provider.search_chunks_regex("original_function")
        assert len(initial_results) > 0, "Initial content should be indexed"
        
        # Get initial file record
        initial_record = services.provider.get_file_by_path(str(test_file.resolve()))
        assert initial_record is not None, "Initial file should exist"
        # Get chunk count for initial state
        initial_chunks = services.provider.search_chunks_regex(".*", file_path=str(test_file.resolve()))
        initial_chunk_count = len(initial_chunks)
        
        print(f"Initial state: chunks={initial_chunk_count}")
        
        # Modify the file - change existing and add new content
        modified_content = """def original_function():
    return "version_2"  # CHANGED

def newly_added_function():
    '''This function was added during modification'''
    return "modification_detected"

class NewlyAddedClass:
    '''This class was added to test modification detection'''
    def new_method(self):
        return "class_method_added"
"""
        test_file.write_text(modified_content)
        
        # Touch file to ensure modification time changes
        import time
        time.sleep(0.1)
        test_file.touch()
        
        # Wait for modification to be processed
        await asyncio.sleep(3.5)
        
        # Check if modification was detected
        modified_record = services.provider.get_file_by_path(str(test_file.resolve()))
        assert modified_record is not None, "Modified file should still exist"
        # Get chunk count for modified state
        modified_chunks = services.provider.search_chunks_regex(".*", file_path=str(test_file.resolve()))
        modified_chunk_count = len(modified_chunks)
        
        print(f"Modified state: chunks={modified_chunk_count}")
        
        # Key assertions for content-based change detection
        
        assert modified_chunk_count >= initial_chunk_count, \
            f"Chunk count should not decrease (was {initial_chunk_count}, now {modified_chunk_count})"
        
        # Check if new content is searchable
        new_func_results = services.provider.search_chunks_regex("newly_added_function")
        assert len(new_func_results) > 0, "New function should be searchable after modification"
        
        new_class_results = services.provider.search_chunks_regex("NewlyAddedClass")
        assert len(new_class_results) > 0, "New class should be indexed after modification"
        
        # Check that content-based deduplication works - old version replaced by new
        v1_results = services.provider.search_chunks_regex("version_1")
        v2_results = services.provider.search_chunks_regex("version_2")
        
        assert len(v1_results) == 0, "Old version_1 should be replaced via content-based chunk deduplication"
        assert len(v2_results) > 0, "New version_2 should be indexed"
    
    @pytest.mark.asyncio
    async def test_file_modification_with_filesystem_ops(self, mcp_setup):
        """Test modification using different filesystem operations to ensure OS detection."""
        services, realtime_service, watch_dir, _, _ = mcp_setup
        
        test_file = watch_dir / "fs_ops_test.py"
        
        # Create with explicit file operations
        with open(test_file, 'w') as f:
            f.write("def func(): return 'initial'")
            f.flush()
            import os
            os.fsync(f.fileno())
        
        # Wait for initial indexing
        await asyncio.sleep(2.5)
        
        initial_results = services.provider.search_chunks_regex("func.*initial")
        assert len(initial_results) > 0, "Initial content should be indexed"
        
        # Modify with explicit operations and different content
        with open(test_file, 'w') as f:
            f.write("def func(): return 'modified'\ndef new_func(): return 'added'")
            f.flush()
            os.fsync(f.fileno())
        
        # Also change mtime explicitly
        import time
        current_time = time.time()
        os.utime(test_file, (current_time, current_time))
        
        # Wait for processing
        await asyncio.sleep(3.5)
        
        # Verify modification was detected
        modified_results = services.provider.search_chunks_regex("func.*modified")
        new_results = services.provider.search_chunks_regex("new_func.*added")
        
        assert len(modified_results) > 0, "Modified content should be indexed"
        assert len(new_results) > 0, "Added content should be indexed"
        
        # Original should be gone
        old_results = services.provider.search_chunks_regex("func.*initial")
        assert len(old_results) == 0, "Original content should be replaced"