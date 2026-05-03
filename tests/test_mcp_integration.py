"""Integration tests that use actual MCP server components.

These tests keep MCP tool assertions on the user-facing search path while using
deterministic helpers for indexing-only contracts that do not need filesystem
event coverage.
"""

import asyncio
import shutil
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from chunkhound.core.config.config import Config
from chunkhound.core.exceptions import CompactionError
from chunkhound.database_factory import create_services
from chunkhound.mcp_server.base import MCPServerBase
from chunkhound.mcp_server.tools import execute_tool
from chunkhound.services.compaction_service import CompactionService
from chunkhound.services.realtime_indexing_service import (
    RealtimeIndexingService,
    normalize_file_path,
)
from tests.test_utils import (
    build_embedding_config_from_dict,
    create_embedding_manager_for_tests,
    get_api_key_for_tests,
    get_embedding_config_for_tests,
)
from tests.utils.realtime_test_helpers import (
    remove_file_from_index,
    write_and_index_file,
)
from tests.utils.windows_compat import (
    get_fs_event_timeout,
    is_ci,
    is_windows,
)


class _TestMCPServer(MCPServerBase):
    """Minimal concrete MCP server for integration callback tests."""

    def _register_tools(self) -> None:
        pass

    async def run(self) -> None:
        pass


async def _wait_for_scan_completed(
    server: MCPServerBase, timeout: float = 15.0
) -> None:
    """Wait until MCP background scan reports completion."""

    async def _wait() -> None:
        while server._scan_progress["scan_completed_at"] is None:
            await asyncio.sleep(0.05)

    await asyncio.wait_for(_wait(), timeout=timeout)


async def _wait_for_realtime_started(
    server: MCPServerBase, timeout: float = 10.0
) -> RealtimeIndexingService:
    """Wait until MCP deferred startup creates the realtime service."""

    async def _wait() -> RealtimeIndexingService:
        while server.realtime_indexing is None:
            await asyncio.sleep(0.05)
        return server.realtime_indexing

    return await asyncio.wait_for(_wait(), timeout=timeout)


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
        fake_args = SimpleNamespace(path=temp_dir)
        config = Config(
            args=fake_args,
            database={"path": str(db_path), "provider": "duckdb"},
            embedding=embedding_config,
            indexing={"include": ["*.py", "*.js"], "exclude": ["*.log"]},
        )

        # Create embedding manager if API key is available
        # create_services() handles None manager gracefully
        embedding_manager = create_embedding_manager_for_tests(config_dict)

        # Create services - this is what MCP server uses
        services = create_services(db_path, config, embedding_manager)
        services.provider.connect()

        # Create the realtime service, but let tests opt into watcher startup explicitly.
        realtime_service = RealtimeIndexingService(services, config)

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
                arguments={
                    "type": "regex",
                    "query": "test",
                    "page_size": 10,
                    "offset": 0,
                },
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

    @pytest.mark.skipif(
        get_api_key_for_tests()[0] is None, reason="No API key available"
    )
    @pytest.mark.asyncio
    async def test_mcp_semantic_search_finds_new_files(self, mcp_setup):
        """Test that MCP semantic search finds newly created files."""
        services, _, watch_dir, _, embedding_manager = mcp_setup

        # Get initial search results using MCP tool execution
        initial_results = await execute_tool(
            tool_name="search",
            services=services,
            embedding_manager=embedding_manager,
            arguments={
                "type": "semantic",
                "query": "unique_mcp_test_function",
                "page_size": 10,
                "offset": 0,
            },
        )
        initial_count = len(initial_results.get("results", []))

        # Create new file with unique content
        new_file = watch_dir / "mcp_test.py"
        await write_and_index_file(
            services,
            new_file,
            """
def unique_mcp_test_function():
    '''This is a unique function for MCP integration testing'''
    return "mcp_realtime_success"
""",
        )

        # Search for new content using MCP tool execution
        new_results = await execute_tool(
            tool_name="search",
            services=services,
            embedding_manager=embedding_manager,
            arguments={
                "type": "semantic",
                "query": "unique_mcp_test_function",
                "page_size": 10,
                "offset": 0,
            },
        )
        new_count = len(new_results.get("results", []))

        assert new_count > initial_count, (
            f"MCP semantic search should find new file (was {initial_count}, now {new_count})"
        )

    @pytest.mark.asyncio
    async def test_mcp_regex_search_finds_modified_files(self, mcp_setup):
        """Test that MCP regex search returns modified file content."""
        services, _, watch_dir, _, _ = mcp_setup

        # Create initial file
        test_file = watch_dir / "modify_test.py"
        await write_and_index_file(services, test_file, "def initial_function(): pass")

        # Modify file with new unique content
        await write_and_index_file(
            services,
            test_file,
            """
def initial_function(): pass

def modified_unique_regex_pattern():
    '''Added by modification - should be found by regex'''
    return "modification_success"
""",
        )
        modified_results = await execute_tool(
            tool_name="search",
            services=services,
            embedding_manager=None,
            arguments={
                "type": "regex",
                "query": "modified_unique_regex_pattern",
                "page_size": 10,
                "offset": 0,
            },
        )
        assert len(modified_results.get("results", [])) > 0, (
            "MCP regex search should find modified content"
        )

    @pytest.mark.asyncio
    async def test_mcp_database_stats_change_with_realtime(self, mcp_setup):
        """Test that database stats reflect direct indexing updates."""
        services, _, watch_dir, _, _ = mcp_setup

        # Get initial stats directly from database provider
        initial_stats = services.provider.get_stats()
        initial_files = initial_stats.get("files", 0)
        initial_chunks = initial_stats.get("chunks", 0)

        # Create multiple new files
        for i in range(3):
            new_file = watch_dir / f"stats_test_{i}.py"
            await write_and_index_file(
                services,
                new_file,
                f"""
def stats_test_function_{i}():
    '''File {i} for testing database stats updates'''
    return "stats_test_{i}"

class StatsTestClass_{i}:
    def method_{i}(self):
        pass
""",
            )

        updated_stats = services.provider.get_stats()
        updated_files = updated_stats.get("files", 0)
        updated_chunks = updated_stats.get("chunks", 0)

        assert updated_files > initial_files, (
            f"File count should increase (was {initial_files}, now {updated_files})"
        )
        assert updated_chunks > initial_chunks, (
            f"Chunk count should increase (was {initial_chunks}, now {updated_chunks})"
        )

    @pytest.mark.asyncio
    async def test_mcp_search_after_file_deletion(self, mcp_setup):
        """Test that MCP search handles file deletions correctly."""
        services, realtime_service, watch_dir, _, _ = mcp_setup

        # Create file with unique content
        delete_file = watch_dir / "delete_test.py"
        await write_and_index_file(
            services,
            delete_file,
            """
def delete_test_unique_function():
    '''This function will be deleted'''
    return "to_be_deleted"
""",
        )

        # Verify content is searchable
        before_delete = await execute_tool(
            tool_name="search",
            services=services,
            embedding_manager=None,
            arguments={
                "type": "regex",
                "query": "delete_test_unique_function",
                "page_size": 10,
                "offset": 0,
            },
        )
        assert len(before_delete.get("results", [])) > 0, (
            "Content should be found before deletion"
        )

        # Delete the file
        await remove_file_from_index(realtime_service, delete_file)

        # Verify content is no longer searchable
        after_delete = await execute_tool(
            tool_name="search",
            services=services,
            embedding_manager=None,
            arguments={
                "type": "regex",
                "query": "delete_test_unique_function",
                "page_size": 10,
                "offset": 0,
            },
        )
        assert len(after_delete.get("results", [])) == 0, (
            "Content should not be found after deletion"
        )

    @pytest.mark.asyncio
    async def test_file_modification_detection_comprehensive(self, mcp_setup):
        """Comprehensive test to reproduce file modification detection issues."""
        services, _, watch_dir, _, _ = mcp_setup

        # Create initial file
        test_file = watch_dir / "comprehensive_modify_test.py"
        initial_content = """def original_function():
    return "version_1"
"""
        await write_and_index_file(services, test_file, initial_content)

        # Verify initial content is indexed (use multiline-compatible regex)
        initial_results = services.provider.search_chunks_regex("original_function")
        assert len(initial_results) > 0, "Initial content should be indexed"

        # Get initial file record
        initial_record = services.provider.get_file_by_path(str(test_file.resolve()))
        assert initial_record is not None, "Initial file should exist"
        # Get chunk count for initial state
        initial_chunks = services.provider.search_chunks_regex(
            ".*", file_path=str(test_file.resolve())
        )
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
        await write_and_index_file(services, test_file, modified_content)

        # Check if modification was detected
        modified_record = services.provider.get_file_by_path(str(test_file.resolve()))
        assert modified_record is not None, "Modified file should still exist"
        # Get chunk count for modified state
        modified_chunks = services.provider.search_chunks_regex(
            ".*", file_path=str(test_file.resolve())
        )
        modified_chunk_count = len(modified_chunks)

        print(f"Modified state: chunks={modified_chunk_count}")

        # Key assertions for content-based change detection

        assert modified_chunk_count >= initial_chunk_count, (
            f"Chunk count should not decrease (was {initial_chunk_count}, now {modified_chunk_count})"
        )

        # Check if new content is searchable
        new_func_results = services.provider.search_chunks_regex("newly_added_function")
        assert len(new_func_results) > 0, (
            "New function should be searchable after modification"
        )

        new_class_results = services.provider.search_chunks_regex("NewlyAddedClass")
        assert len(new_class_results) > 0, (
            "New class should be indexed after modification"
        )

        # Check that content-based deduplication works - old version replaced by new
        v1_results = services.provider.search_chunks_regex("version_1")
        v2_results = services.provider.search_chunks_regex("version_2")

        assert len(v1_results) == 0, (
            "Old version_1 should be replaced via content-based chunk deduplication"
        )
        assert len(v2_results) > 0, "New version_2 should be indexed"

    @pytest.mark.xfail(
        condition=is_windows() and is_ci(),
        reason="Polling mtime detection unreliable on NTFS (fixed in PR #220)",
        strict=False,
    )
    @pytest.mark.native_watcher
    @pytest.mark.asyncio
    async def test_file_modification_with_filesystem_ops(self, mcp_setup):
        """Test modification using different filesystem operations to ensure OS detection."""
        services, realtime_service, watch_dir, _, _ = mcp_setup
        import os

        await realtime_service.start(watch_dir)
        assert await realtime_service.wait_for_monitoring_ready(
            timeout=get_fs_event_timeout()
        ), "Realtime monitoring did not become ready"

        test_file = watch_dir / "fs_ops_test.py"

        # Create with explicit file operations
        with open(test_file, "w") as f:
            f.write("def func(): return 'initial'")
            f.flush()
            os.fsync(f.fileno())

        # Wait for file to be indexed
        found = await realtime_service.wait_for_file_indexed(
            test_file, timeout=get_fs_event_timeout()
        )
        assert found, "Initial content should be indexed"

        # Modify with explicit operations and different content
        realtime_service.reset_file_tracking(test_file)
        with open(test_file, "w") as f:
            f.write("def func(): return 'modified'\ndef new_func(): return 'added'")
            f.flush()
            os.fsync(f.fileno())

        # Also change mtime explicitly
        import time

        current_time = time.time()
        os.utime(test_file, (current_time, current_time))

        # Wait for modified file to be re-indexed
        found = await realtime_service.wait_for_file_indexed(
            test_file, timeout=get_fs_event_timeout()
        )
        assert found, "Added content should be indexed"

        # Verify modification was detected
        modified_results = services.provider.search_chunks_regex("func.*modified")
        assert len(modified_results) > 0, "Modified content should be indexed"

        # Original should be gone
        old_results = services.provider.search_chunks_regex("func.*initial")
        assert len(old_results) == 0, "Original content should be replaced"

    @pytest.mark.native_watcher
    @pytest.mark.asyncio
    async def test_mcp_startup_scans_preexisting_files_and_realtime_indexes_new_ones(
        self,
    ):
        """MCP owns startup scan; realtime service owns files created after monitoring starts."""
        temp_dir = Path(tempfile.mkdtemp())
        db_path = temp_dir / ".chunkhound" / "test.db"
        watch_dir = temp_dir / "project"
        watch_dir.mkdir(parents=True)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        fake_args = SimpleNamespace(path=watch_dir)
        config = Config(
            args=fake_args,
            database={"path": str(db_path), "provider": "duckdb"},
            indexing={"include": ["*.py", "*.js"], "exclude": ["*.log"]},
        )
        server = _TestMCPServer(config=config, args=fake_args)

        preexisting_file = watch_dir / "before_start.py"
        preexisting_file.write_text("def before_start(): return 'initial_scan'")

        try:
            await server.initialize()
            realtime_service = await _wait_for_realtime_started(server)
            assert await realtime_service.wait_for_monitoring_ready(
                timeout=get_fs_event_timeout()
            ), "Realtime monitoring did not become ready during MCP startup"

            realtime_file = watch_dir / "after_start.py"
            realtime_service.reset_file_tracking(realtime_file)
            realtime_file.write_text("def after_start(): return 'realtime'")
            found = await realtime_service.wait_for_file_indexed(
                realtime_file, timeout=get_fs_event_timeout()
            )
            assert found, (
                "Files created after MCP startup should be indexed by realtime monitoring"
            )

            await _wait_for_scan_completed(server)

            assert server.services is not None
            preexisting_record = server.services.provider.get_file_by_path(
                str(preexisting_file.resolve())
            )
            realtime_record = server.services.provider.get_file_by_path(
                str(realtime_file.resolve())
            )
            assert preexisting_record is not None, (
                "MCP startup scan must index files that already exist before initialize()"
            )
            assert realtime_record is not None, (
                "Realtime monitoring must index files created after initialize()"
            )
        finally:
            await server.cleanup()
            shutil.rmtree(temp_dir, ignore_errors=True)

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
        await write_and_index_file(services, test_file, "def func(): return 'initial'")

        initial_results = services.provider.search_chunks_regex("func.*initial")
        assert len(initial_results) > 0, "Initial content should be searchable"

        # Overwrite with new content and process again
        await write_and_index_file(
            services, test_file, "def func(): return 'replaced'\ndef added(): pass"
        )

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
        services, realtime_service, watch_dir, temp_dir, _ = mcp_setup

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
            indexing={"include": ["*.py"], "exclude": [], "force_reindex": True},
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

        server = _TestMCPServer(config=compaction_config)
        server.services = services
        server.realtime_indexing = realtime_service
        server._target_path = watch_dir

        with patch.object(
            services.provider,
            "_export_database_for_compaction",
            side_effect=pausing_export,
        ):
            compaction_service = CompactionService(
                db_path=Path(services.provider.db_path),
                config=compaction_config,
            )
            server._compaction_service = compaction_service
            started = await compaction_service.compact_background(
                provider=services.provider,
                on_complete=server._post_compaction_reindex,
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

        assert compaction_service.last_error is None, (
            "Post-compaction callback must complete without recorded error"
        )

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
    async def test_deferred_delete_is_replayed_after_compaction_with_cleanup_disabled(
        self, mcp_setup
    ):
        """Deferred deletes must be replayed even when orphan cleanup is disabled."""
        services, realtime_service, watch_dir, temp_dir, _ = mcp_setup

        test_file = watch_dir / "delete_during_compact.py"
        test_file.write_text("def delete_me_after_compaction():\n    return 'OLD'\n")
        await services.indexing_coordinator.process_file(test_file)
        before = await execute_tool(
            tool_name="search",
            services=services,
            embedding_manager=None,
            arguments={
                "type": "regex",
                "query": "delete_me_after_compaction",
                "page_size": 10,
                "offset": 0,
            },
        )
        assert len(before.get("results", [])) > 0, (
            "Test precondition: deleted symbol must be searchable before "
            "compaction replay"
        )

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
            indexing={
                "include": ["*.py"],
                "exclude": [],
                "force_reindex": True,
                "cleanup": False,
            },
        )

        export_started = threading.Event()
        export_proceed = threading.Event()
        real_export = services.provider._export_database_for_compaction

        def pausing_export(db_p, export_dir):
            export_started.set()
            assert export_proceed.wait(timeout=10.0), "export_proceed never set"
            return real_export(db_p, export_dir)

        server = _TestMCPServer(config=compaction_config)
        server.services = services
        server.realtime_indexing = realtime_service
        server._target_path = watch_dir

        with patch.object(
            services.provider,
            "_export_database_for_compaction",
            side_effect=pausing_export,
        ):
            compaction_service = CompactionService(
                db_path=Path(services.provider.db_path),
                config=compaction_config,
            )
            server._compaction_service = compaction_service
            started = await compaction_service.compact_background(
                provider=services.provider,
                on_complete=server._post_compaction_reindex,
            )
            assert started, "Compaction should start with zero thresholds"

            compaction_task = compaction_service._compaction_task
            assert compaction_task is not None

            try:
                await asyncio.to_thread(export_started.wait, 10.0)
                assert export_started.is_set(), "Export phase never started"

                realtime_service.reset_file_tracking(test_file)
                test_file.unlink()
                await realtime_service.remove_file(test_file)

                normalized = normalize_file_path(test_file)
                assert normalized in realtime_service._compaction_deferred_files
                assert normalized in realtime_service._compaction_deferred_removals
            finally:
                export_proceed.set()
                if not compaction_task.done():
                    await asyncio.wait_for(compaction_task, timeout=30.0)

        after = await execute_tool(
            tool_name="search",
            services=services,
            embedding_manager=None,
            arguments={
                "type": "regex",
                "query": "delete_me_after_compaction",
                "page_size": 10,
                "offset": 0,
            },
        )
        assert len(after.get("results", [])) == 0, (
            "Deferred delete replay must remove stale searchable content"
        )
        assert normalized not in realtime_service._compaction_deferred_files
        assert normalized not in realtime_service._compaction_deferred_removals
        assert normalized not in realtime_service.failed_files

    @pytest.mark.asyncio
    async def test_deferred_directory_delete_is_replayed_after_compaction(
        self, mcp_setup
    ):
        """Deleted-directory events during compaction must remove stale search rows."""
        services, realtime_service, watch_dir, temp_dir, _ = mcp_setup

        deleted_dir = watch_dir / "deleted_during_compaction"
        kept_dir = watch_dir / "kept_during_compaction"
        deleted_dir.mkdir()
        kept_dir.mkdir()
        deleted_file = deleted_dir / "gone.py"
        kept_file = kept_dir / "kept.py"
        deleted_symbol = "directory_replay_deleted_marker"
        kept_symbol = "directory_replay_kept_marker"
        deleted_file.write_text(f"def {deleted_symbol}():\n    return 'gone'\n")
        kept_file.write_text(f"def {kept_symbol}():\n    return 'kept'\n")
        await services.indexing_coordinator.process_file(deleted_file)
        await services.indexing_coordinator.process_file(kept_file)

        async def assert_search_count(symbol: str, expected: int, reason: str) -> None:
            results = await execute_tool(
                tool_name="search",
                services=services,
                embedding_manager=None,
                arguments={
                    "type": "regex",
                    "query": symbol,
                    "page_size": 10,
                    "offset": 0,
                },
            )
            assert len(results.get("results", [])) == expected, reason

        await assert_search_count(
            deleted_symbol,
            1,
            "Test precondition: deleted directory symbol must be searchable",
        )
        await assert_search_count(
            kept_symbol,
            1,
            "Test precondition: sibling symbol must be searchable",
        )

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
            indexing={
                "include": ["*.py"],
                "exclude": [],
                "force_reindex": True,
                "cleanup": False,
            },
        )

        export_started = threading.Event()
        export_proceed = threading.Event()
        real_export = services.provider._export_database_for_compaction

        def pausing_export(db_p, export_dir):
            export_started.set()
            assert export_proceed.wait(timeout=10.0), "export_proceed never set"
            return real_export(db_p, export_dir)

        server = _TestMCPServer(config=compaction_config)
        server.services = services
        server.realtime_indexing = realtime_service
        server._target_path = watch_dir

        with patch.object(
            services.provider,
            "_export_database_for_compaction",
            side_effect=pausing_export,
        ):
            compaction_service = CompactionService(
                db_path=Path(services.provider.db_path),
                config=compaction_config,
            )
            server._compaction_service = compaction_service
            started = await compaction_service.compact_background(
                provider=services.provider,
                on_complete=server._post_compaction_reindex,
            )
            assert started, "Compaction should start with zero thresholds"

            compaction_task = compaction_service._compaction_task
            assert compaction_task is not None
            consumer = asyncio.create_task(realtime_service._consume_events())

            try:
                await asyncio.to_thread(export_started.wait, 10.0)
                assert export_started.is_set(), "Export phase never started"

                shutil.rmtree(deleted_dir)
                await realtime_service.event_queue.put(("dir_deleted", deleted_dir))
                await asyncio.wait_for(realtime_service.event_queue.join(), timeout=5.0)

                normalized_deleted_dir = normalize_file_path(deleted_dir)
                assert normalized_deleted_dir in (
                    realtime_service._compaction_deferred_directories
                )
            finally:
                realtime_service._stopping = True
                consumer.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await consumer
                export_proceed.set()
                if not compaction_task.done():
                    await asyncio.wait_for(compaction_task, timeout=30.0)

        assert compaction_service.last_error is None
        await assert_search_count(
            deleted_symbol,
            0,
            "Deferred directory replay must remove stale deleted-directory content",
        )
        await assert_search_count(
            kept_symbol,
            1,
            "Deferred directory replay must not remove sibling content",
        )
        assert server.get_background_compaction_status()["pending_recovery"] is False

    @pytest.mark.asyncio
    async def test_ensure_services_recovers_failed_post_compaction_retry_end_to_end(
        self, mcp_setup
    ):
        """A failed post-compaction callback is retried once via ensure_services()."""
        services, realtime_service, watch_dir, temp_dir, _ = mcp_setup

        test_file = watch_dir / "retry_after_compaction.py"
        test_file.write_text("def retry_old_marker():\n    return 'OLD'\n")
        await services.indexing_coordinator.process_file(test_file)

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
            indexing={"include": ["*.py"], "exclude": [], "force_reindex": True},
        )

        export_started = threading.Event()
        export_proceed = threading.Event()
        real_export = services.provider._export_database_for_compaction

        def pausing_export(db_p, export_dir):
            export_started.set()
            assert export_proceed.wait(timeout=10.0), "export_proceed never set"
            return real_export(db_p, export_dir)

        server = _TestMCPServer(config=compaction_config)
        server.services = services
        server.realtime_indexing = realtime_service
        server._target_path = watch_dir

        real_process_directory = services.indexing_coordinator.process_directory
        callback_attempts = 0

        async def fail_once_process_directory(*args, **kwargs):
            nonlocal callback_attempts
            callback_attempts += 1
            if callback_attempts == 1:
                raise RuntimeError("simulated first callback failure")
            return await real_process_directory(*args, **kwargs)

        services.indexing_coordinator.process_directory = fail_once_process_directory

        with patch.object(
            services.provider,
            "_export_database_for_compaction",
            side_effect=pausing_export,
        ):
            compaction_service = CompactionService(
                db_path=Path(services.provider.db_path),
                config=compaction_config,
            )
            server._compaction_service = compaction_service
            started = await compaction_service.compact_background(
                provider=services.provider,
                on_complete=server._post_compaction_reindex,
            )
            assert started, "Compaction should start with zero thresholds"

            compaction_task = compaction_service._compaction_task
            assert compaction_task is not None

            await asyncio.to_thread(export_started.wait, 10.0)
            assert export_started.is_set(), "Export phase never started"

            test_file.write_text("def retry_new_marker():\n    return 'NEW'\n")

            export_proceed.set()
            await asyncio.wait_for(compaction_task, timeout=30.0)

        status_after_failure = server.get_background_compaction_status()
        assert compaction_service.last_error is not None
        assert status_after_failure["pending_recovery"] is True
        assert "simulated first callback failure" in status_after_failure["last_error"]

        stale_new = await execute_tool(
            tool_name="search",
            services=services,
            embedding_manager=None,
            arguments={
                "type": "regex",
                "query": "retry_new_marker",
                "page_size": 10,
                "offset": 0,
            },
        )
        assert len(stale_new.get("results", [])) == 0, (
            "Search must remain stale before recovery retry runs"
        )

        stale_old = await execute_tool(
            tool_name="search",
            services=services,
            embedding_manager=None,
            arguments={
                "type": "regex",
                "query": "retry_old_marker",
                "page_size": 10,
                "offset": 0,
            },
        )
        assert len(stale_old.get("results", [])) > 0, (
            "Old content must remain searchable before recovery retry runs"
        )

        await server.ensure_services()

        assert callback_attempts == 2, "ensure_services() must retry exactly once"
        assert compaction_service.last_error is None

        recovered_status = server.get_background_compaction_status()
        assert recovered_status["pending_recovery"] is False
        assert recovered_status["last_error"] is None

        recovered_new = await execute_tool(
            tool_name="search",
            services=services,
            embedding_manager=None,
            arguments={
                "type": "regex",
                "query": "retry_new_marker",
                "page_size": 10,
                "offset": 0,
            },
        )
        assert len(recovered_new.get("results", [])) > 0, (
            "Recovery retry must make new content searchable"
        )

        recovered_old = await execute_tool(
            tool_name="search",
            services=services,
            embedding_manager=None,
            arguments={
                "type": "regex",
                "query": "retry_old_marker",
                "page_size": 10,
                "offset": 0,
            },
        )
        assert len(recovered_old.get("results", [])) == 0, (
            "Recovery retry must remove stale old content from search results"
        )

    @pytest.mark.asyncio
    async def test_ensure_services_retries_partial_deferred_delete_replay_end_to_end(
        self, mcp_setup
    ):
        """Recovery must replay only unresolved deferred deletes against real search."""
        services, realtime_service, watch_dir, temp_dir, _ = mcp_setup

        files = {
            "a": (watch_dir / "partial_delete_a.py", "partial_delete_marker_a"),
            "b": (watch_dir / "partial_delete_b.py", "partial_delete_marker_b"),
            "c": (watch_dir / "partial_delete_c.py", "partial_delete_marker_c"),
        }
        for path, symbol in files.values():
            path.write_text(f"def {symbol}():\n    return '{symbol}'\n")
            await services.indexing_coordinator.process_file(path)
            before = await execute_tool(
                tool_name="search",
                services=services,
                embedding_manager=None,
                arguments={
                    "type": "regex",
                    "query": symbol,
                    "page_size": 10,
                    "offset": 0,
                },
            )
            assert len(before.get("results", [])) > 0

        normalized_paths = {
            key: normalize_file_path(path) for key, (path, _) in files.items()
        }
        for path, _ in files.values():
            path.unlink()

        await realtime_service.restore_compaction_deferred_removals(
            set(normalized_paths.values())
        )

        fake_args = SimpleNamespace(path=temp_dir)
        recovery_config = Config(
            args=fake_args,
            database={
                "path": str(Path(services.provider.db_path).parent),
                "provider": "duckdb",
            },
            indexing={
                "include": ["*.py"],
                "exclude": [],
                "force_reindex": True,
                "cleanup": False,
            },
        )

        server = _TestMCPServer(config=recovery_config)
        server.services = services
        server.realtime_indexing = realtime_service
        server._target_path = watch_dir
        server._compaction_service = MagicMock()
        server._compaction_service.is_compacting = False
        server._compaction_service.last_error = None
        server._compaction_service.clear_last_error = MagicMock()

        real_delete = services.provider.delete_file_completely_async
        fail_b_once = True

        async def fail_one_delete(path: str) -> None:
            nonlocal fail_b_once
            if path == normalized_paths["b"] and fail_b_once:
                fail_b_once = False
                raise CompactionError("busy deleting b", operation="delete")
            await real_delete(path)

        with patch.object(
            services.provider,
            "delete_file_completely_async",
            side_effect=fail_one_delete,
        ):
            with pytest.raises(CompactionError, match="busy deleting b"):
                await server._post_compaction_reindex()

            a_after_failure = await execute_tool(
                tool_name="search",
                services=services,
                embedding_manager=None,
                arguments={
                    "type": "regex",
                    "query": files["a"][1],
                    "page_size": 10,
                    "offset": 0,
                },
            )
            assert len(a_after_failure.get("results", [])) == 0, (
                "Successful replay before failure must remove file a from search"
            )

            for key in ("b", "c"):
                still_searchable = await execute_tool(
                    tool_name="search",
                    services=services,
                    embedding_manager=None,
                    arguments={
                        "type": "regex",
                        "query": files[key][1],
                        "page_size": 10,
                        "offset": 0,
                    },
                )
                assert len(still_searchable.get("results", [])) > 0, (
                    f"File {key} must remain searchable before recovery"
                )

            status_after_failure = server.get_background_compaction_status()
            assert status_after_failure["pending_recovery"] is True

            await server.ensure_services()

        for _, symbol in files.values():
            after = await execute_tool(
                tool_name="search",
                services=services,
                embedding_manager=None,
                arguments={
                    "type": "regex",
                    "query": symbol,
                    "page_size": 10,
                    "offset": 0,
                },
            )
            assert len(after.get("results", [])) == 0

        assert realtime_service._compaction_deferred_files == set()
        assert realtime_service._compaction_deferred_removals == set()
        recovered_status = server.get_background_compaction_status()
        assert recovered_status["pending_recovery"] is False
        assert recovered_status["last_error"] is None

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
        assert len(before.get("results", [])) > 0, (
            "Content should exist before compaction"
        )

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
