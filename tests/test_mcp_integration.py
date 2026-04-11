"""Integration tests that use actual MCP server components.

These tests verify the real integration path that users experience:
Filesystem Event → Watchdog → AsyncHandler → IndexingCoordinator → Database → Search Tools
"""

import asyncio
import shutil
import tempfile
import time
from pathlib import Path

import pytest

from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services
from chunkhound.mcp_server.tools import execute_tool
from chunkhound.services.realtime_indexing_service import RealtimeIndexingService
from tests.utils.windows_compat import (
    get_fs_event_timeout,
    is_ci,
    is_windows,
    should_use_polling,
)

from .test_utils import get_api_key_for_tests, get_embedding_config_for_tests, build_embedding_config_from_dict, create_embedding_manager_for_tests


class TestMCPIntegration:
    """Test real MCP server integration with realtime indexing."""

    @pytest.fixture
    async def mcp_setup(self):
        """Setup MCP server with real services and temp directory."""
        # Get embedding config using centralized helper
        config_dict = get_embedding_config_for_tests()
        embedding_config = build_embedding_config_from_dict(config_dict)

        temp_dir = Path(tempfile.mkdtemp())
        db_path = temp_dir / ".chunkhound" / "test.db"
        watch_dir = temp_dir / "project"
        watch_dir.mkdir(parents=True)

        # Ensure database directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

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
        # create_services() handles None manager gracefully
        embedding_manager = create_embedding_manager_for_tests(config_dict)

        # Create services - this is what MCP server uses
        services = create_services(db_path, config, embedding_manager)
        services.provider.connect()


        # Initialize realtime indexing service (what MCP server should do)
        # Use polling mode on Windows CI where watchdog is unreliable
        force_polling = should_use_polling()
        realtime_service = RealtimeIndexingService(services, config, force_polling=force_polling)
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

    @pytest.mark.asyncio
    async def test_mcp_rejects_during_compaction(self, mcp_setup):
        """MCP tool calls return CompactionError JSON when compaction gate is closed."""
        import json
        from chunkhound.mcp_server.common import handle_tool_call

        services, _, _, _, _ = mcp_setup

        # Simulate active compaction: drop cached connection then close gate.
        # Access _connection_allowed directly — no public API exists to
        # close the compaction gate without running a real compaction cycle.
        services.provider.soft_disconnect()
        services.provider._connection_allowed.clear()
        try:
            init_event = asyncio.Event()
            init_event.set()

            result = await handle_tool_call(
                tool_name="search",
                arguments={"type": "regex", "query": "test", "page_size": 10, "offset": 0},
                services=services,
                embedding_manager=None,
                initialization_complete=init_event,
            )

            assert len(result) == 1
            body = json.loads(result[0].text)
            assert body["error"]["type"] == "CompactionError"
            assert "compaction in progress" in body["error"]["message"]
            assert "retry_hint" in body["error"]
        finally:
            services.provider._connection_allowed.set()
            services.provider.connect()

    @pytest.mark.skipif(get_api_key_for_tests()[0] is None, reason="No API key available")
    @pytest.mark.asyncio
    async def test_mcp_semantic_search_finds_new_files(self, mcp_setup):
        """Test that MCP semantic search finds newly created files."""
        services, realtime_service, watch_dir, _, embedding_manager = mcp_setup

        # Wait for initial scan
        await asyncio.sleep(1.0)

        # Get initial search results using MCP tool execution
        initial_results = await execute_tool(
            tool_name="search",
            services=services,
            embedding_manager=embedding_manager,
            arguments={
                "type": "semantic",
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
            tool_name="search",
            services=services,
            embedding_manager=embedding_manager,
            arguments={
                "type": "semantic",
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

        # Wait for file to be indexed
        found = await realtime_service.wait_for_file_indexed(test_file, timeout=get_fs_event_timeout())
        assert found, "Initial content should be found"

        # Modify file with new unique content
        realtime_service.reset_file_tracking(test_file)
        test_file.write_text("""
def initial_function(): pass

def modified_unique_regex_pattern():
    '''Added by modification - should be found by regex'''
    return "modification_success"
""")

        # Wait for modified file to be re-indexed
        found = await realtime_service.wait_for_file_indexed(test_file, timeout=get_fs_event_timeout())

        assert found, "MCP regex search should find modified content"

    @pytest.mark.asyncio
    async def test_mcp_database_stats_change_with_realtime(self, mcp_setup):
        """Test that database stats reflect real-time indexing changes."""
        services, realtime_service, watch_dir, _, _ = mcp_setup

        # Wait for initial scan
        await asyncio.sleep(1.0)

        # Get initial stats directly from database provider
        initial_stats = services.provider.get_stats()
        initial_files = initial_stats.get('files', 0)
        initial_chunks = initial_stats.get('chunks', 0)

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

        # Wait for files to be processed with polling
        timeout = get_fs_event_timeout() * 1.5  # Extra margin for multiple files
        deadline = time.monotonic() + timeout
        updated_stats = None

        while time.monotonic() < deadline:
            updated_stats = services.provider.get_stats()
            if updated_stats.get('files', 0) > initial_files:
                break
            await asyncio.sleep(0.3)

        updated_files = updated_stats.get('files', 0) if updated_stats else 0
        updated_chunks = updated_stats.get('chunks', 0) if updated_stats else 0

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
        realtime_service.reset_file_tracking(delete_file)
        delete_file.write_text("""
def delete_test_unique_function():
    '''This function will be deleted'''
    return "to_be_deleted"
""")

        # Wait for processing
        found = await realtime_service.wait_for_file_indexed(delete_file, timeout=get_fs_event_timeout())
        assert found, "File should be indexed"

        # Verify content is searchable
        before_delete = await execute_tool(
            tool_name="search",
            services=services,
            embedding_manager=None,
            arguments={
                "type": "regex",
                "query": "delete_test_unique_function",
                "page_size": 10,
                "offset": 0
            }
        )
        assert len(before_delete.get('results', [])) > 0, "Content should be found before deletion"

        # Delete the file
        realtime_service.reset_file_tracking(delete_file)
        delete_file.unlink()

        # Wait for deletion processing
        removed = await realtime_service.wait_for_file_removed(delete_file, timeout=get_fs_event_timeout())
        assert removed, "File should be removed"

        # Verify content is no longer searchable
        after_delete = await execute_tool(
            tool_name="search",
            services=services,
            embedding_manager=None,
            arguments={
                "type": "regex",
                "query": "delete_test_unique_function",
                "page_size": 10,
                "offset": 0
            }
        )
        assert len(after_delete.get('results', [])) == 0, "Content should not be found after deletion"

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

        # Wait for file to be indexed
        found = await realtime_service.wait_for_file_indexed(test_file, timeout=get_fs_event_timeout())
        assert found, "Initial content should be indexed"

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
        realtime_service.reset_file_tracking(test_file)
        test_file.write_text(modified_content)

        # Touch file to ensure modification time changes
        import time
        time.sleep(0.1)
        test_file.touch()

        # Wait for modified file to be re-indexed
        found = await realtime_service.wait_for_file_indexed(test_file, timeout=get_fs_event_timeout())
        assert found, "Modified content should be searchable"

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

    @pytest.mark.xfail(
        condition=is_windows() and is_ci(),
        reason="Polling mtime detection unreliable on NTFS (fixed in PR #220)",
        strict=False,
    )
    @pytest.mark.asyncio
    async def test_file_modification_with_filesystem_ops(self, mcp_setup):
        """Test modification using different filesystem operations to ensure OS detection."""
        services, realtime_service, watch_dir, _, _ = mcp_setup
        import os

        test_file = watch_dir / "fs_ops_test.py"

        # Create with explicit file operations
        with open(test_file, 'w') as f:
            f.write("def func(): return 'initial'")
            f.flush()
            os.fsync(f.fileno())

        # Wait for file to be indexed
        found = await realtime_service.wait_for_file_indexed(test_file, timeout=get_fs_event_timeout())
        assert found, "Initial content should be indexed"

        # Modify with explicit operations and different content
        realtime_service.reset_file_tracking(test_file)
        with open(test_file, 'w') as f:
            f.write("def func(): return 'modified'\ndef new_func(): return 'added'")
            f.flush()
            os.fsync(f.fileno())

        # Also change mtime explicitly
        import time
        current_time = time.time()
        os.utime(test_file, (current_time, current_time))

        # Wait for modified file to be re-indexed
        found = await realtime_service.wait_for_file_indexed(test_file, timeout=get_fs_event_timeout())
        assert found, "Added content should be indexed"

        # Verify modification was detected
        modified_results = services.provider.search_chunks_regex("func.*modified")
        assert len(modified_results) > 0, "Modified content should be indexed"

        # Original should be gone
        old_results = services.provider.search_chunks_regex("func.*initial")
        assert len(old_results) == 0, "Original content should be replaced"

    @pytest.mark.asyncio
    async def test_modified_file_replaces_old_content(self, mcp_setup):
        """Test that process_file replaces old content on modification.

        Deterministic test that calls process_file() directly, bypassing
        filesystem monitoring. Tests the reaction layer independently of
        platform-dependent change detection.
        """
        services, _, watch_dir, _, _ = mcp_setup

        test_file = watch_dir / "direct_modify_test.py"

        # Write initial content and process directly
        test_file.write_text("def func(): return 'initial'")
        await services.indexing_coordinator.process_file(test_file)

        initial_results = services.provider.search_chunks_regex("func.*initial")
        assert len(initial_results) > 0, "Initial content should be searchable"

        # Overwrite with new content and process again
        test_file.write_text("def func(): return 'replaced'\ndef added(): pass")
        await services.indexing_coordinator.process_file(test_file)

        # New content should be searchable
        new_results = services.provider.search_chunks_regex("func.*replaced")
        assert len(new_results) > 0, "New content should be searchable"

        added_results = services.provider.search_chunks_regex("added")
        assert len(added_results) > 0, "Added function should be searchable"

        # Old content should be gone
        old_results = services.provider.search_chunks_regex("func.*initial")
        assert len(old_results) == 0, "Old content should be replaced"

    @pytest.mark.asyncio
    async def test_search_returns_modified_content_after_compaction(self, mcp_setup):
        """A file mutated between compaction's on-disk export and the atomic swap
        must be reindexed by the on_complete callback and be searchable
        afterwards.

        The pausing stub sits inside `_export_database_for_compaction`, which
        runs after `soft_disconnect(skip_checkpoint=False)` has flushed the DB
        file and opens its own read-only connection against that flushed copy
        (duckdb_provider.py:3352). A mutation arriving during this pause is
        guaranteed to miss this compaction cycle's data and must be picked up
        by the post-swap reindex.

        This test exercises the production reindex path via
        `DirectoryIndexingService.process_directory()`, covering the
        mtime/size/content_hash change detection in `indexing_coordinator`
        (indexing_coordinator.py:1256-1298) — the same path used by
        `chunkhound/mcp_server/base.py:345-370`.

        Scope: this covers the pre-swap portion of the race window and uses
        `no_embeddings=True` to keep the test scoped (production uses
        `no_embeddings=False`). Mutations landing mid-swap or during the
        callback itself, and embedding-pipeline interactions, are not
        exercised here.
        """
        import threading
        from types import SimpleNamespace
        from unittest.mock import patch

        from chunkhound.services.compaction_service import CompactionService
        from chunkhound.services.directory_indexing_service import (
            DirectoryIndexingService,
        )

        services, _, watch_dir, temp_dir, _ = mcp_setup

        test_file = watch_dir / "modify_during_compact.py"
        test_file.write_text("def original_marker():\n    return 'OLD'\n")
        await services.indexing_coordinator.process_file(test_file)

        # Build a dedicated Config that bypasses the auto-compaction thresholds
        # (real tiny test DBs never cross the default 100MB reclaimable gate).
        # threshold=0.0 + min_size=0 guarantees compact_background() starts.
        fake_args = SimpleNamespace(path=temp_dir)
        compaction_config = Config(
            args=fake_args,
            database={
                "path": str(Path(services.provider.db_path).parent),
                "provider": "duckdb",
                "compaction_enabled": True,
                "compaction_threshold": 0.0,
                "compaction_min_size_mb": 0,
            },
            indexing={"include": ["*.py"], "exclude": []},
        )

        export_started = threading.Event()
        export_proceed = threading.Event()
        real_export = services.provider._export_database_for_compaction

        def pausing_export(db_p, export_dir):
            # Signal that export phase has started (compaction gate is closed)
            export_started.set()
            # Wait for the test to mutate the file before finishing export
            assert export_proceed.wait(timeout=10.0), "export_proceed never set"
            return real_export(db_p, export_dir)

        async def reindex_modified() -> None:
            # Mirror chunkhound/mcp_server/base.py:361-370 — drive the
            # production reindex path so change detection (mtime/size/
            # content_hash) discovers the mutated file, rather than
            # hand-feeding process_file() a known path.
            indexing_service = DirectoryIndexingService(
                indexing_coordinator=services.indexing_coordinator,
                config=compaction_config,
            )
            await indexing_service.process_directory(watch_dir, no_embeddings=True)

        with patch.object(
            services.provider,
            "_export_database_for_compaction",
            side_effect=pausing_export,
        ):
            compaction_service = CompactionService(
                db_path=Path(services.provider.db_path),
                config=compaction_config,
            )
            started = await compaction_service.compact_background(
                provider=services.provider,
                on_complete=reindex_modified,
            )
            assert started, "Compaction should start with zero thresholds"

            # Capture the background task so we can await its completion
            # without going through shutdown() (which cancels instead).
            compaction_task = compaction_service._compaction_task
            assert compaction_task is not None

            # Wait for the thread to enter the paused export phase, then mutate
            # the file. asyncio.to_thread so the event loop is not blocked.
            await asyncio.to_thread(export_started.wait, 10.0)
            assert export_started.is_set(), "Export phase never started"

            test_file.write_text("def modified_marker():\n    return 'NEW'\n")

            # Unblock export and wait for the full compaction + on_complete
            # callback to finish (on_complete runs inline inside the task).
            export_proceed.set()
            await asyncio.wait_for(compaction_task, timeout=30.0)

        # Post-compaction reindex callback should have picked up the new content.
        after = await execute_tool(
            tool_name="search",
            services=services,
            embedding_manager=None,
            arguments={
                "type": "regex",
                "query": "modified_marker",
                "page_size": 10,
                "offset": 0,
            },
        )
        assert len(after.get("results", [])) > 0, (
            "Modified content must be searchable after post-compaction reindex"
        )

        old = await execute_tool(
            tool_name="search",
            services=services,
            embedding_manager=None,
            arguments={
                "type": "regex",
                "query": "original_marker",
                "page_size": 10,
                "offset": 0,
            },
        )
        assert len(old.get("results", [])) == 0, (
            "Old content must be replaced by reindex"
        )

    @pytest.mark.asyncio
    async def test_mcp_search_works_after_compaction(self, mcp_setup):
        """Search results are preserved after database compaction."""
        services, _, watch_dir, _, _ = mcp_setup

        # Index a file with unique content
        test_file = watch_dir / "compaction_search_test.py"
        test_file.write_text(
            "def compaction_survivor_func():\n"
            "    return 'data_that_must_survive_compaction'\n"
        )
        await services.indexing_coordinator.process_file(test_file)

        # Verify content is searchable before compaction
        before = await execute_tool(
            tool_name="search",
            services=services,
            embedding_manager=None,
            arguments={
                "type": "regex",
                "query": "compaction_survivor_func",
                "page_size": 10,
                "offset": 0,
            },
        )
        assert len(before.get("results", [])) > 0, "Content should exist before compaction"

        # Compact the database
        services.provider.optimize()

        # Regex search must still return the indexed content
        after = await execute_tool(
            tool_name="search",
            services=services,
            embedding_manager=None,
            arguments={
                "type": "regex",
                "query": "compaction_survivor_func",
                "page_size": 10,
                "offset": 0,
            },
        )
        assert len(after.get("results", [])) > 0, (
            "Search results must be preserved after compaction"
        )
