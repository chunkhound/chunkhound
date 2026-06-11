"""Test the warm-ready timing summary logic.

Tests ``_emit_warm_ready_summary_if_ready()`` which emits a one-shot
timing summary after deferred initial scan and/or fresh-instance resync
complete during startup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from chunkhound.mcp_server.base import MCPServerBase


class _TestableMCPServer(MCPServerBase):
    """Subclass that stubs abstract methods and runs in stdio mode."""

    def _register_tools(self) -> None:
        pass

    async def run(self) -> None:
        pass


class TestWarmReadyTiming:
    """Verify _emit_warm_ready_summary_if_ready behavior."""

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    @pytest.fixture
    def config(self, tmp_path: Path) -> Any:
        from chunkhound.core.config.config import Config

        db_path = tmp_path / ".chunkhound" / "test.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return Config(
            target_dir=tmp_path,
            database={"path": str(db_path), "provider": "duckdb"},
            indexing={"index_unknown_files": False},
        )

    @pytest.fixture
    def server(self, config: Any) -> _TestableMCPServer:
        """Create a server with warm-ready window set up and tracker seeded."""
        srv = _TestableMCPServer(config)
        # Simulate being inside a warm-ready window
        srv._warm_ready_started_monotonic = 1000.0
        srv._warm_ready_summary_emitted = False
        # Seed the startup tracker so snapshot() returns a float total
        srv._startup_tracker.reset("stdio")
        srv._startup_tracker.complete_phase("initialize")
        # Replace _timing_log with a mock for assertion
        srv._timing_log = MagicMock()
        return srv

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _call_and_assert_emitted(self, server: _TestableMCPServer) -> str:
        """Call the method and return the captured summary message."""
        server._emit_warm_ready_summary_if_ready()
        assert server._warm_ready_summary_emitted, (
            "Summary should have been emitted"
        )
        assert server._timing_log.call_count == 1
        msg: str = server._timing_log.call_args[0][0]
        return msg

    def _call_and_assert_not_emitted(self, server: _TestableMCPServer) -> None:
        """Call the method and verify no emission occurred."""
        server._emit_warm_ready_summary_if_ready()
        assert not server._warm_ready_summary_emitted, (
            "Summary should NOT have been emitted"
        )
        server._timing_log.assert_not_called()

    # ------------------------------------------------------------------
    # Test cases — each verifies a specific code path in the method
    # ------------------------------------------------------------------

    def test_initial_scan_complete_triggers_summary(
        self, server: _TestableMCPServer
    ) -> None:
        """Summary emitted when initial scan completes and no resync is pending.

        Covers the simplest path: only ``_warm_ready_initial_scan_completed_monotonic``
        is set, no fresh-instance resync state exists.
        """
        server._warm_ready_initial_scan_completed_monotonic = 1015.0

        msg = self._call_and_assert_emitted(server)

        assert "blocking_startup=" in msg
        assert "warm_ready=15.000s" in msg
        # No fresh_instance_resync or initial_scan parts expected
        assert "fresh_instance_resync=" not in msg
        assert "initial_scan=" not in msg
        assert msg.startswith("summary blocking_startup=")

    def test_skip_and_resync_triggers_summary(
        self, server: _TestableMCPServer
    ) -> None:
        """Summary emitted when initial scan is skipped AND resync completed.

        Covers the path where ``_warm_ready_initial_scan_skipped_monotonic``
        satisfies the scan-work guard and a completed fresh-instance resync
        satisfies the resync guard.  ``initial_scan=`` is absent because
        ``_warm_ready_initial_scan_total_seconds`` is ``None``.
        """
        server._warm_ready_initial_scan_skipped_monotonic = 1012.0
        server._warm_ready_fresh_instance_resync_requested = True
        server._warm_ready_fresh_instance_resync_completed_monotonic = 1018.0
        server._warm_ready_fresh_instance_resync_total_seconds = 6.0

        msg = self._call_and_assert_emitted(server)

        assert "blocking_startup=" in msg
        # warm_ready = max(1012.0, 1018.0) - 1000.0 = 18.000s
        assert "warm_ready=18.000s" in msg
        assert "fresh_instance_resync=6.000s" in msg
        assert "initial_scan=" not in msg

    def test_double_work_includes_both_timing_parts(
        self, server: _TestableMCPServer
    ) -> None:
        """Both initial scan and resync completed back-to-back.

        Covers the path where both ``initial_scan_total_seconds`` and
        ``fresh_instance_resync_total_seconds`` are set, producing a summary
        with both timing parts.
        """
        server._warm_ready_initial_scan_completed_monotonic = 1015.0
        server._warm_ready_initial_scan_total_seconds = 15.0
        server._warm_ready_fresh_instance_resync_requested = True
        server._warm_ready_fresh_instance_resync_completed_monotonic = 1020.0
        server._warm_ready_fresh_instance_resync_total_seconds = 3.5

        msg = self._call_and_assert_emitted(server)

        assert "blocking_startup=" in msg
        # warm_ready = max(1015.0, 1020.0) - 1000.0 = 20.000s
        assert "warm_ready=20.000s" in msg
        assert "fresh_instance_resync=3.500s" in msg
        assert "initial_scan=15.000s" in msg

    def test_not_ready_when_scan_not_complete(
        self, server: _TestableMCPServer
    ) -> None:
        """Summary NOT emitted when no initial scan completed/skipped.

        Covers the guard that returns early when both
        ``_warm_ready_initial_scan_completed_monotonic`` and
        ``_warm_ready_initial_scan_skipped_monotonic`` are ``None``.
        """
        server._warm_ready_fresh_instance_resync_requested = True

        self._call_and_assert_not_emitted(server)

    def test_already_emitted_is_noop(self, server: _TestableMCPServer) -> None:
        """Guard prevents double emission.

        Covers the ``_warm_ready_summary_emitted`` guard at the top of the
        method.  The flag stays ``True`` and no log call is made.
        """
        server._warm_ready_summary_emitted = True

        server._emit_warm_ready_summary_if_ready()

        # Flag must remain True and _timing_log must not be called again
        assert server._warm_ready_summary_emitted
        server._timing_log.assert_not_called()

    def test_no_started_time_is_noop(self, server: _TestableMCPServer) -> None:
        """Summary NOT emitted when _warm_ready_started_monotonic is None.

        Covers the ``_warm_ready_started_monotonic is None`` guard.
        """
        server._warm_ready_started_monotonic = None

        self._call_and_assert_not_emitted(server)

    def test_summary_format(self, server: _TestableMCPServer) -> None:
        """Verify the summary string structure matches expected format.

        Asserts the individual key=value tokens appear in the expected order
        and that the ``warm_ready=``, ``fresh_instance_resync=``, and
        ``initial_scan=`` values are correctly formatted to three decimal
        places.
        """
        server._warm_ready_initial_scan_completed_monotonic = 1015.0
        server._warm_ready_initial_scan_total_seconds = 15.0
        server._warm_ready_fresh_instance_resync_requested = True
        server._warm_ready_fresh_instance_resync_completed_monotonic = 1020.0
        server._warm_ready_fresh_instance_resync_total_seconds = 3.5

        msg = self._call_and_assert_emitted(server)

        # Expected: "summary blocking_startup=X.XXXs warm_ready=20.000s
        #           fresh_instance_resync=3.500s initial_scan=15.000s"
        assert msg.startswith("summary ")
        parts = msg.split(" ")
        assert parts[0] == "summary"
        # blocking_startup is dynamic but must be a valid key=value
        assert parts[1].startswith("blocking_startup=")
        assert parts[1].endswith("s")
        assert "." in parts[1]  # Has decimal
        assert parts[2] == "warm_ready=20.000s"
        assert parts[3] == "fresh_instance_resync=3.500s"
        assert parts[4] == "initial_scan=15.000s"
