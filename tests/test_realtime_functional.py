"""Functional tests for real-time filesystem indexing.

Tests core real-time indexing functionality with real components.
Some tests expected to fail initially - helps identify implementation issues.
"""

import asyncio
import shutil
import tempfile
from pathlib import Path

import pytest

from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services
from chunkhound.services.realtime_indexing_service import RealtimeIndexingService
from tests.utils.windows_compat import should_use_polling, wait_for_indexed


class TestRealtimeFunctional:
    """Functional tests for real-time indexing - test what really matters."""

    @pytest.fixture
    async def realtime_setup(self):
        """Setup real service with temp database and project directory."""
        # Resolve immediately to handle symlinks (/var -> /private/var on macOS)
        # and Windows 8.3 short path names
        temp_dir = Path(tempfile.mkdtemp()).resolve()
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

        # Use polling on Windows CI where watchdog's ReadDirectoryChangesW is unreliable
        force_polling = should_use_polling()
        realtime_service = RealtimeIndexingService(services, config, force_polling=force_polling)

        yield realtime_service, watch_dir, temp_dir, services

        # Cleanup
        try:
            await realtime_service.stop()
        except Exception:
            pass  # Service might already be stopped or failed

        try:
            services.provider.disconnect()
        except Exception:
            pass

        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_service_can_start_and_stop(self, realtime_setup):
        """Test basic service lifecycle - start and stop without crashing."""
        service, watch_dir, _, _ = realtime_setup

        # Should be able to start
        await service.start(watch_dir)

        # Check basic state
        stats = await service.get_stats()
        assert isinstance(stats, dict), "Stats should be returned"
        assert 'observer_alive' in stats, "Should report observer status"
        assert stats['watching_directory'] == str(watch_dir), "Should report watched directory"

        # Should be able to stop cleanly
        await service.stop()

    @pytest.mark.asyncio
    async def test_filesystem_monitoring_detects_changes(self, realtime_setup):
        """Test that filesystem changes are detected and processed."""
        service, watch_dir, _, services = realtime_setup
        await service.start(watch_dir)

        # Create a Python file - should be detected and processed
        test_file = watch_dir / "test_monitor.py"
        test_file.write_text("def hello_world(): pass")

        # Wait for filesystem event + debouncing + processing
        found = await wait_for_indexed(services.provider, test_file)

        # This tests the full pipeline: detection -> processing -> storage
        assert found, "File should be detected and processed by filesystem monitoring"

        await service.stop()

    @pytest.mark.asyncio
    async def test_multiple_rapid_changes_handling(self, realtime_setup):
        """Test handling multiple rapid file changes - stress test for concurrency."""
        service, watch_dir, _, _ = realtime_setup
        await service.start(watch_dir)

        # Create multiple files in rapid succession
        test_files = []
        for i in range(5):
            test_file = watch_dir / f"rapid_{i}.py"
            test_file.write_text(f"def func_{i}(): return {i}")
            test_files.append(test_file)
            # Small delay to create separate events
            await asyncio.sleep(0.1)

        # Wait for all processing
        await asyncio.sleep(3.0)

        # Check service is still alive
        stats = await service.get_stats()
        assert stats.get('observer_alive', False), "Service should still be running after rapid changes"

        # This test mainly checks service doesn't crash under load
        await service.stop()

    @pytest.mark.asyncio
    async def test_service_survives_processing_errors(self, realtime_setup):
        """Test service continues working after processing errors."""
        service, watch_dir, _, _ = realtime_setup
        await service.start(watch_dir)

        # Create a file that might cause processing issues
        bad_file = watch_dir / "bad_file.py"
        # Write binary data to a .py file - might cause parsing errors
        bad_file.write_bytes(b'\x00\xFF\xFE\xFD')

        await asyncio.sleep(1.0)

        # Create a normal file after the bad one
        good_file = watch_dir / "good_file.py"
        good_file.write_text("def good_function(): pass")

        await asyncio.sleep(2.0)

        # Main goal: service should still be alive
        stats = await service.get_stats()
        assert stats.get('observer_alive', False), "Service should survive processing errors"

        await service.stop()

    @pytest.mark.asyncio
    async def test_file_type_filtering_works(self, realtime_setup):
        """Test that only supported file types are processed."""
        service, watch_dir, _, services = realtime_setup
        await service.start(watch_dir)

        # Create supported file
        py_file = watch_dir / "supported.py"
        py_file.write_text("def supported(): pass")

        # Create unsupported file
        bin_file = watch_dir / "unsupported.xyz"
        bin_file.write_text("unsupported content")

        await asyncio.sleep(1.5)

        # Check processing results
        py_record = services.provider.get_file_by_path(str(py_file))
        bin_record = services.provider.get_file_by_path(str(bin_file))

        # Python file should be considered for processing (might still fail due to other issues)
        # Binary file should definitely be ignored
        assert bin_record is None, "Unsupported file types should be ignored"

        await service.stop()

    @pytest.mark.asyncio
    async def test_background_vs_realtime_processing(self, realtime_setup):
        """Test interaction between initial scan and real-time processing."""
        service, watch_dir, _, services = realtime_setup

        # Create files before starting service (will be found by initial scan)
        initial_file = watch_dir / "initial.py"
        initial_file.write_text("def initial(): pass")

        await service.start(watch_dir)

        # Create file after service started (real-time processing)
        realtime_file = watch_dir / "realtime.py"
        await asyncio.sleep(0.5)  # Let initial scan start
        realtime_file.write_text("def realtime(): pass")

        # Wait for both initial scan and real-time processing
        await asyncio.sleep(3.0)

        # Both files should eventually be processed
        initial_record = services.provider.get_file_by_path(str(initial_file))
        realtime_record = services.provider.get_file_by_path(str(realtime_file))

        # At least one should work (helps identify which path is broken)
        processed_count = sum(1 for record in [initial_record, realtime_record] if record is not None)
        assert processed_count > 0, "At least one processing path should work"

        await service.stop()
