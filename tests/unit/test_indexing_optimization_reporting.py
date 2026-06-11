"""Contract: process_directory reports whether a CHECKPOINT ran via db_optimized key."""

import asyncio
import pytest
from pathlib import Path

from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.indexing_coordinator import IndexingCoordinator


def test_stats_include_db_optimized_true_when_checkpoint_ran(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stats dict must contain db_optimized=True when has_reclaimable_space() is True."""
    provider = DuckDBProvider(db_path=tmp_path / "db.duckdb", base_directory=tmp_path)
    provider.connect()

    # At least one file on disk so discovery doesn't short-circuit to no_files
    (tmp_path / "sample.py").write_text("x = 1\n")

    optimize_called = []

    monkeypatch.setattr(provider, "has_reclaimable_space", lambda operation="": True)
    monkeypatch.setattr(
        provider, "optimize_tables", lambda: optimize_called.append(True)
    )

    coordinator = IndexingCoordinator(provider, tmp_path)

    result = asyncio.run(
        coordinator.process_directory(
            tmp_path, patterns=["**/*.py"], exclude_patterns=[]
        )
    )

    assert result.get("db_optimized") is True
    assert optimize_called, "optimize_tables must be called when has_reclaimable_space is True"


def test_stats_include_db_optimized_false_when_no_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stats dict must contain db_optimized=False when has_reclaimable_space() is False."""
    provider = DuckDBProvider(db_path=tmp_path / "db.duckdb", base_directory=tmp_path)
    provider.connect()

    # At least one file on disk so discovery doesn't short-circuit to no_files
    (tmp_path / "sample.py").write_text("x = 1\n")

    monkeypatch.setattr(provider, "has_reclaimable_space", lambda operation="": False)

    optimize_called = []
    monkeypatch.setattr(
        provider, "optimize_tables", lambda: optimize_called.append(True)
    )

    coordinator = IndexingCoordinator(provider, tmp_path)

    result = asyncio.run(
        coordinator.process_directory(
            tmp_path, patterns=["**/*.py"], exclude_patterns=[]
        )
    )

    assert result.get("db_optimized") is False
    assert not optimize_called, "optimize_tables must NOT be called when no space to reclaim"
