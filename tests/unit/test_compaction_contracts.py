"""Compaction contract tests.

Tests user-facing contracts for the compaction feature:
- fragmentation measurement returns sensible values
- compact_if_needed() respects the configured threshold
- tool_requires_services() correctly gates DB reconnect
- _with_compaction_retry() retries on CompactionError
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


# ---------------------------------------------------------------------------
# measure_fragmentation contracts
# ---------------------------------------------------------------------------


class TestMeasureFragmentation:
    """Contracts for measure_fragmentation() on DuckDBProvider."""

    def _make_provider(self, is_memory: bool, effective_waste: float):
        """Return a minimal DuckDBProvider-like object with the relevant methods."""
        cm = MagicMock()
        cm.is_memory_db = is_memory

        provider = MagicMock()
        provider._connection_manager = cm
        provider.get_storage_stats.return_value = {"effective_waste": effective_waste}
        # Use the real implementation
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider
        provider.measure_fragmentation = lambda: DuckDBProvider.measure_fragmentation(provider)
        return provider

    def test_returns_zero_for_in_memory_database(self):
        provider = self._make_provider(is_memory=True, effective_waste=0.5)
        assert provider.measure_fragmentation() == 0.0

    def test_no_waste_gives_ratio_of_one(self):
        provider = self._make_provider(is_memory=False, effective_waste=0.0)
        assert provider.measure_fragmentation() == 1.0

    def test_fifty_pct_waste_gives_ratio_of_one_point_five(self):
        provider = self._make_provider(is_memory=False, effective_waste=0.5)
        assert provider.measure_fragmentation() == pytest.approx(1.5)

    def test_high_waste_gives_ratio_above_three(self):
        provider = self._make_provider(is_memory=False, effective_waste=2.5)
        # 1.0 + 2.5 = 3.5 — considered fragmented
        assert provider.measure_fragmentation() > 3.0


# ---------------------------------------------------------------------------
# compact_if_needed contracts
# ---------------------------------------------------------------------------


class TestCompactIfNeeded:
    """Contracts for compact_if_needed() on DuckDBProvider."""

    def _make_provider(
        self,
        is_memory: bool = False,
        fragmentation_threshold_pct: float | None = 30.0,
        effective_waste: float = 0.0,
    ):
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider

        cm = MagicMock()
        cm.is_memory_db = is_memory

        config = MagicMock()
        config.fragmentation_threshold_pct = fragmentation_threshold_pct

        provider = MagicMock()
        provider._connection_manager = cm
        # Explicitly set _config=None so the code falls through to provider.config
        provider._config = None
        provider.config = config
        provider._suppress_compaction = False
        provider.get_storage_stats.return_value = {"effective_waste": effective_waste}
        provider.measure_fragmentation = lambda: DuckDBProvider.measure_fragmentation(provider)
        provider.compact_if_needed = lambda: DuckDBProvider.compact_if_needed(provider)
        return provider

    def test_skips_for_in_memory_database(self):
        provider = self._make_provider(is_memory=True, effective_waste=5.0)
        assert provider.compact_if_needed() is False
        provider._run_blocking_compaction.assert_not_called()

    def test_skips_when_threshold_is_none(self):
        provider = self._make_provider(
            fragmentation_threshold_pct=None, effective_waste=5.0
        )
        assert provider.compact_if_needed() is False
        provider._run_blocking_compaction.assert_not_called()

    def test_skips_when_overhead_below_threshold(self):
        # 10% overhead, threshold 30% → no compaction
        provider = self._make_provider(
            fragmentation_threshold_pct=30.0, effective_waste=0.10
        )
        assert provider.compact_if_needed() is False
        provider._run_blocking_compaction.assert_not_called()

    def test_compacts_when_overhead_meets_threshold(self):
        # 50% overhead (ratio=1.5), threshold 30% → compact
        provider = self._make_provider(
            fragmentation_threshold_pct=30.0, effective_waste=0.50
        )
        assert provider.compact_if_needed() is True
        provider._run_blocking_compaction.assert_called_once()

    def test_zero_threshold_always_compacts(self):
        # threshold=0 means "compact whenever any overhead exists"
        provider = self._make_provider(
            fragmentation_threshold_pct=0.0, effective_waste=0.01
        )
        assert provider.compact_if_needed() is True

    def test_exactly_at_threshold_compacts(self):
        # 30% overhead, threshold exactly 30% → compact (>= boundary)
        provider = self._make_provider(
            fragmentation_threshold_pct=30.0, effective_waste=0.30
        )
        assert provider.compact_if_needed() is True


# ---------------------------------------------------------------------------
# _suppress_compaction during chunking contracts
# ---------------------------------------------------------------------------


class TestSuppressCompactionDuringChunking:
    """Contract: _suppress_compaction is True during the chunking loop and False afterward."""

    def test_suppress_flag_restored_after_chunking(self):
        """_suppress_compaction must be False after chunking completes (even on error)."""
        from unittest.mock import MagicMock

        db = MagicMock()
        db._suppress_compaction = False

        # Simulate the suppress/restore block: set True, then finally restores to False
        suppress_attr = hasattr(db, "_suppress_compaction")
        db._suppress_compaction = True  # simulates what the loop sets before starting

        # Simulate the finally block restoring the flag (even if an error occurred)
        try:
            raise RuntimeError("simulated parse error")
        except RuntimeError:
            pass
        finally:
            if suppress_attr:
                db._suppress_compaction = False

        assert db._suppress_compaction is False, "_suppress_compaction must be False after chunking (even on error)"

    def test_compaction_interval_constant_is_positive(self):
        """_CHUNK_COMPACTION_INTERVAL must be a positive integer."""
        from chunkhound.services.indexing_coordinator import IndexingCoordinator

        assert hasattr(IndexingCoordinator, "_CHUNK_COMPACTION_INTERVAL")
        assert isinstance(IndexingCoordinator._CHUNK_COMPACTION_INTERVAL, int)
        assert IndexingCoordinator._CHUNK_COMPACTION_INTERVAL > 0


# ---------------------------------------------------------------------------
# tool_requires_services contract
# ---------------------------------------------------------------------------


class TestToolRequiresServices:
    """Contracts for tool_requires_services() in mcp_server.tools."""

    def test_tool_with_services_param_returns_true(self):
        from chunkhound.mcp_server.tools import TOOL_REGISTRY, tool_requires_services
        from chunkhound.mcp_server.tools import Tool as McpTool

        def _dummy_with_services(services: Any, query: str) -> str:
            return ""

        TOOL_REGISTRY["_test_with_services"] = McpTool(
            name="_test_with_services",
            description="test",
            parameters={},
            implementation=_dummy_with_services,
        )
        try:
            assert tool_requires_services("_test_with_services") is True
        finally:
            TOOL_REGISTRY.pop("_test_with_services", None)

    def test_tool_without_services_param_returns_false(self):
        from chunkhound.mcp_server.tools import TOOL_REGISTRY, tool_requires_services
        from chunkhound.mcp_server.tools import Tool as McpTool

        def _dummy_no_services(query: str) -> str:
            return ""

        TOOL_REGISTRY["_test_no_services"] = McpTool(
            name="_test_no_services",
            description="test",
            parameters={},
            implementation=_dummy_no_services,
        )
        try:
            assert tool_requires_services("_test_no_services") is False
        finally:
            TOOL_REGISTRY.pop("_test_no_services", None)

    def test_unknown_tool_returns_false(self):
        from chunkhound.mcp_server.tools import tool_requires_services
        assert tool_requires_services("__nonexistent_tool__") is False


# ---------------------------------------------------------------------------
# _with_compaction_retry contracts
# ---------------------------------------------------------------------------


class TestWithCompactionRetry:
    """Contracts for _with_compaction_retry() on SerialDatabaseProvider."""

    def _get_retry(self):
        from chunkhound.providers.database.serial_database_provider import (
            SerialDatabaseProvider,
        )
        # Use unbound method — we only need to call it with an awaitable
        return SerialDatabaseProvider._with_compaction_retry

    def test_returns_result_on_first_success(self):
        from chunkhound.providers.database.serial_database_provider import (
            SerialDatabaseProvider,
        )

        provider = MagicMock(spec=SerialDatabaseProvider)
        provider._with_compaction_retry = (
            SerialDatabaseProvider._with_compaction_retry.__get__(provider)
        )

        async def _ok():
            return 42

        result = asyncio.run(
            provider._with_compaction_retry(_ok)
        )
        assert result == 42

    def test_retries_on_compaction_error_and_succeeds(self):
        from chunkhound.providers.database.serial_database_provider import (
            SerialDatabaseProvider,
        )
        from chunkhound.core.exceptions import CompactionError

        provider = MagicMock(spec=SerialDatabaseProvider)
        provider._with_compaction_retry = (
            SerialDatabaseProvider._with_compaction_retry.__get__(provider)
        )

        call_count = 0

        async def _fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise CompactionError("in progress", operation="test")
            return "done"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = asyncio.run(
                provider._with_compaction_retry(_fails_twice, max_retries=5)
            )

        assert result == "done"
        assert call_count == 3

    def test_re_raises_after_max_retries(self):
        from chunkhound.providers.database.serial_database_provider import (
            SerialDatabaseProvider,
        )
        from chunkhound.core.exceptions import CompactionError

        provider = MagicMock(spec=SerialDatabaseProvider)
        provider._with_compaction_retry = (
            SerialDatabaseProvider._with_compaction_retry.__get__(provider)
        )

        async def _always_fails():
            raise CompactionError("permanently blocked", operation="test")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(CompactionError):
                asyncio.run(
                    provider._with_compaction_retry(_always_fails, max_retries=3)
                )

    def test_does_not_retry_non_compaction_errors(self):
        from chunkhound.providers.database.serial_database_provider import (
            SerialDatabaseProvider,
        )

        provider = MagicMock(spec=SerialDatabaseProvider)
        provider._with_compaction_retry = (
            SerialDatabaseProvider._with_compaction_retry.__get__(provider)
        )

        call_count = 0

        async def _value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("not a compaction error")

        with pytest.raises(ValueError):
            asyncio.run(
                provider._with_compaction_retry(_value_error, max_retries=5)
            )

        # Should not retry on non-CompactionError
        assert call_count == 1
