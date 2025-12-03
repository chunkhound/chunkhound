"""Tests that expose real failures in the real-time indexing implementation.

These tests are designed to fail and show what's actually broken.
"""

import asyncio
import tempfile
from pathlib import Path
import pytest
import shutil

from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services
from chunkhound.services.realtime_indexing_service import RealtimeIndexingService


class TestRealtimeFailures:
    """Tests that expose actual implementation failures."""
    
    @pytest.fixture
    async def realtime_setup(self):
        """Setup real service with temp database and project directory."""
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
            indexing={"include": ["*.py", "*.js"], "exclude": ["*.log"]}
        )
        
        services = create_services(db_path, config)
        services.provider.connect()


        realtime_service = RealtimeIndexingService(services, config)
        
        yield realtime_service, watch_dir, temp_dir, services
        
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
    async def test_threading_integration_is_broken(self, realtime_setup):
        """Test that threading integration between watchdog and asyncio works."""
        service, watch_dir, _, services = realtime_setup
        await service.start(watch_dir)
        
        # Wait for initial scan to complete
        await asyncio.sleep(1.0)
        
        # Now create file AFTER initial scan - should only be caught by real-time monitoring
        test_file = watch_dir / "realtime_test.py"
        test_file.write_text("def realtime_test(): pass")
        
        # Wait for debounce delay (0.5s) + processing + database commit
        await asyncio.sleep(2.0)
        
        # If threading integration works, file should be processed by real-time monitoring
        # Use resolved path to match what the real-time service stores
        file_record = services.provider.get_file_by_path(str(test_file.resolve()))
        assert file_record is not None, "Real-time file should be processed by filesystem monitoring"
        
        await service.stop()

    @pytest.mark.asyncio 
    async def test_indexing_coordinator_skip_embeddings_not_implemented(self, realtime_setup):
        """Test that IndexingCoordinator.process_file doesn't support skip_embeddings parameter."""
        service, watch_dir, _, services = realtime_setup
        
        # Try to call process_file with skip_embeddings directly
        test_file = watch_dir / "skip_test.py"
        test_file.write_text("def skip_embeddings_test(): pass")
        
        # This should fail because process_file signature doesn't match usage
        try:
            result = await services.indexing_coordinator.process_file(
                test_file, 
                skip_embeddings=True  # This parameter might not exist
            )
            # If we get here, the parameter exists but might not work correctly
            assert result.get('embeddings_skipped') == True, \
                "skip_embeddings parameter should actually skip embeddings"
        except TypeError as e:
            pytest.fail(f"IndexingCoordinator.process_file doesn't support skip_embeddings: {e}")

    @pytest.mark.asyncio
    async def test_file_debouncing_creates_memory_leaks(self, realtime_setup):
        """Test that file debouncing properly cleans up timers."""
        service, watch_dir, _, _ = realtime_setup
        await service.start(watch_dir)
        
        # Wait for initial scan to complete
        await asyncio.sleep(1.0)
        
        # Create many rapid file changes to the SAME file - should reuse timer slot
        test_file = watch_dir / "reused_file.py"
        for i in range(20):
            test_file.write_text(f"def func_{i}(): pass # iteration {i}")
            await asyncio.sleep(0.01)  # Very rapid changes to same file
        
        # Wait for debounce delay to let timers execute and cleanup
        await asyncio.sleep(1.0)
        
        # Get reference to debouncer timers after cleanup should occur
        if service.event_handler and hasattr(service.event_handler, 'debouncer'):
            active_timers = len(service.event_handler.debouncer.timers)
            # Should only have 1 timer max for the single file, or 0 if cleaned up
            assert active_timers <= 1, f"Too many active timers ({active_timers}) - should cleanup after execution"
        
        await service.stop()

    @pytest.mark.asyncio
    async def test_background_scan_conflicts_with_realtime(self, realtime_setup):
        """Test that background scan and real-time processing conflict."""
        service, watch_dir, _, services = realtime_setup
        
        # Create file before starting (will be in initial scan)
        initial_file = watch_dir / "conflict_test.py"
        initial_file.write_text("def initial(): pass")
        
        await service.start(watch_dir)
        
        # Immediately modify the same file (real-time processing)
        initial_file.write_text("def initial_modified(): pass")
        
        # Wait for both to potentially process
        await asyncio.sleep(1.5)
        
        # Check if file was processed multiple times (race condition)
        file_record = services.provider.get_file_by_path(str(initial_file))
        if file_record:
            chunks = services.provider.get_chunks_by_file_id(file_record['id'])
            
            # If there are duplicate chunks or processing conflicts, this will show
            chunk_contents = [chunk.get('content', '') for chunk in chunks]
            unique_contents = set(chunk_contents)
            
            assert len(chunk_contents) == len(unique_contents), \
                f"Duplicate processing detected: {len(chunk_contents)} chunks, {len(unique_contents)} unique"
        
        await service.stop()

    @pytest.mark.asyncio
    async def test_observer_not_properly_recursive(self, realtime_setup):
        """Test that filesystem observer doesn't properly watch subdirectories."""
        service, watch_dir, _, services = realtime_setup
        await service.start(watch_dir)
        
        # Create subdirectory and file
        subdir = watch_dir / "subdir"
        subdir.mkdir()
        subdir_file = subdir / "nested.py"
        subdir_file.write_text("def nested(): pass")
        
        await asyncio.sleep(2.0)
        
        # Check if nested file was detected
        file_record = services.provider.get_file_by_path(str(subdir_file.resolve()))
        assert file_record is not None, "Nested files should be detected by recursive monitoring"
        
        await service.stop()

    @pytest.mark.asyncio
    async def test_service_doesnt_handle_file_deletions(self, realtime_setup):
        """Test that service doesn't handle file deletions properly."""
        service, watch_dir, _, services = realtime_setup
        await service.start(watch_dir)
        
        # Create and process a file
        test_file = watch_dir / "delete_test.py"
        test_file.write_text("def to_be_deleted(): pass")
        
        # Wait for debounce + processing to complete
        await asyncio.sleep(1.5)
        
        # Verify file was processed - use resolved path
        file_record = services.provider.get_file_by_path(str(test_file.resolve()))
        assert file_record is not None, "File should be processed initially"
        
        # Delete the file
        test_file.unlink()

        # Wait for deletion processing
        await asyncio.sleep(4.0)  # Deletion + 3s consistency interval
        
        # Check if file was removed from database - use resolved path
        file_record_after = services.provider.get_file_by_path(str(test_file.resolve()))
        assert file_record_after is None, "Deleted files should be removed from database"
        
        await service.stop()

    @pytest.mark.asyncio
    async def test_error_in_processing_loop_kills_service(self, realtime_setup):
        """Test that an error in the processing loop kills the entire service."""
        service, watch_dir, _, _ = realtime_setup
        await service.start(watch_dir)
        
        # Force an error by creating a file and then deleting it before processing
        test_file = watch_dir / "error_test.py"
        test_file.write_text("def error_test(): pass")
        
        # Wait just enough for file to be queued but not processed
        await asyncio.sleep(0.3)
        
        # Delete file while it's queued for processing
        test_file.unlink()
        
        # Wait for processing to attempt and fail
        await asyncio.sleep(1.0)
        
        # Check if service is still alive after error
        stats = await service.get_stats()
        assert stats.get('observer_alive', False), "Service should survive processing errors"
        
        await service.stop()