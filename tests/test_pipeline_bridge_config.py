"""Tests for run_rust_pipeline's config-dict mapping.

Covers the contract that user-facing settings (set via CLI, .chunkhound.json,
or env var) are actually forwarded to the Rust pipeline, not overridden by
internal defaults.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _fake_report(**overrides: object) -> SimpleNamespace:
    defaults = dict(
        files_processed=0,
        chunks_written=0,
        embeddings_generated=0,
        elapsed_secs=0.0,
        files_skipped=0,
        errors=[],
        # Option<(f64, f64)> on the Rust side: None or (current_mb, limit_mb)
        disk_limit=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@dataclass
class _FakeNativePipeline:
    """Fake chunkhound_native.IndexingPipeline, installed via sys.modules.

    `captured_config` is populated with whatever config_dict
    IndexingPipeline.__init__ was called with; tests that only care about
    run_rust_pipeline's return value can ignore it. `instance.run.return_value`
    defaults to `_fake_report()` -- override it before calling
    run_rust_pipeline for disk-limit-style tests.
    """

    instance: MagicMock
    captured_config: dict


@pytest.fixture
def fake_native_pipeline(monkeypatch: pytest.MonkeyPatch) -> _FakeNativePipeline:
    captured_config: dict = {}
    instance = MagicMock()
    instance.run.return_value = _fake_report()

    def _init_side_effect(config_dict: dict) -> MagicMock:
        captured_config.update(config_dict)
        return instance

    fake_module = types.SimpleNamespace(
        IndexingPipeline=MagicMock(side_effect=_init_side_effect)
    )
    monkeypatch.setitem(sys.modules, "chunkhound_native", fake_module)
    return _FakeNativePipeline(instance=instance, captured_config=captured_config)


@pytest.mark.asyncio
async def test_fragmentation_threshold_pct_forwarded_as_compaction_ratio(
    tmp_path: Path, fake_native_pipeline: _FakeNativePipeline
) -> None:
    """A user-set --fragmentation-threshold-pct must reach Rust as a ratio.

    DatabaseConfig.fragmentation_threshold_pct is a percentage (e.g. 60.0)
    and is already honored by the Python indexing path (duckdb_provider.py).
    The Rust pipeline bridge must forward the same setting, converted to the
    ratio Rust's PipelineConfig.compaction_threshold expects (0.60) — not a
    hardcoded default.
    """
    from chunkhound import pipeline_bridge

    config = SimpleNamespace(
        database=SimpleNamespace(fragmentation_threshold_pct=60.0),
        indexing=SimpleNamespace(),
        embedding=None,
    )

    await pipeline_bridge.run_rust_pipeline(
        files_to_process=[],
        db_path=tmp_path,
        project_root=tmp_path,
        skip_embeddings=True,
        config=config,
    )

    assert fake_native_pipeline.captured_config["compaction_threshold"] == pytest.approx(0.60)


@pytest.mark.asyncio
async def test_missing_fragmentation_threshold_pct_defaults_to_30_pct(
    tmp_path: Path, fake_native_pipeline: _FakeNativePipeline
) -> None:
    """No database config at all falls back to the documented 30% default."""
    from chunkhound import pipeline_bridge

    await pipeline_bridge.run_rust_pipeline(
        files_to_process=[],
        db_path=tmp_path,
        project_root=tmp_path,
        skip_embeddings=True,
        config=None,
    )

    assert fake_native_pipeline.captured_config["compaction_threshold"] == pytest.approx(0.30)


@pytest.mark.asyncio
async def test_explicit_none_fragmentation_threshold_pct_disables_compaction(
    tmp_path: Path, fake_native_pipeline: _FakeNativePipeline
) -> None:
    """fragmentation_threshold_pct=None means "never auto-compact" (see
    DatabaseConfig's docstring), not "unset — use the 30% default". Rust's
    compaction_threshold must receive that None verbatim so it can disable
    the auto-compaction check entirely, matching what the Python indexing
    path already does via duckdb_provider.py's
    _fragmentation_exceeds_threshold(threshold=None) -> False.
    """
    from chunkhound import pipeline_bridge

    config = SimpleNamespace(
        database=SimpleNamespace(fragmentation_threshold_pct=None),
        indexing=SimpleNamespace(),
        embedding=None,
    )

    await pipeline_bridge.run_rust_pipeline(
        files_to_process=[],
        db_path=tmp_path,
        project_root=tmp_path,
        skip_embeddings=True,
        config=config,
    )

    assert fake_native_pipeline.captured_config["compaction_threshold"] is None


@pytest.mark.asyncio
async def test_explicit_zero_config_file_threshold_disables_gate(
    tmp_path: Path, fake_native_pipeline: _FakeNativePipeline
) -> None:
    """config_file_size_threshold_kb=0 (explicitly disabling the gate) must
    reach Rust as 0, not silently fall back to the default 20 -- matching
    the Python path's documented "<= 0 disables the gate" contract.
    """
    from chunkhound import pipeline_bridge

    config = SimpleNamespace(
        database=SimpleNamespace(),
        indexing=SimpleNamespace(config_file_size_threshold_kb=0),
        embedding=None,
    )

    await pipeline_bridge.run_rust_pipeline(
        files_to_process=[],
        db_path=tmp_path,
        project_root=tmp_path,
        skip_embeddings=True,
        config=config,
    )

    assert fake_native_pipeline.captured_config["config_file_size_threshold_kb"] == 0


@pytest.mark.asyncio
async def test_explicit_zero_mtime_epsilon_forwarded(
    tmp_path: Path, fake_native_pipeline: _FakeNativePipeline
) -> None:
    """mtime_epsilon_seconds=0.0 (exact mtime match) must reach Rust as 0.0,
    not silently fall back to the default 0.01.
    """
    from chunkhound import pipeline_bridge

    config = SimpleNamespace(
        database=SimpleNamespace(),
        indexing=SimpleNamespace(mtime_epsilon_seconds=0.0),
        embedding=None,
    )

    await pipeline_bridge.run_rust_pipeline(
        files_to_process=[],
        db_path=tmp_path,
        project_root=tmp_path,
        skip_embeddings=True,
        config=config,
    )

    assert fake_native_pipeline.captured_config["mtime_epsilon_seconds"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_disk_limit_exceeded_report_becomes_structured_error_dict(
    tmp_path: Path, fake_native_pipeline: _FakeNativePipeline
) -> None:
    """A tripped PipelineReport.disk_limit must surface as the same
    error-dict shape DiskUsageLimitExceededError.to_error_dict() builds --
    also used by IndexingCoordinator._store_parsed_results for the Python
    path (indexing_coordinator.py:1039-1042) -- {"file": None, "error": ...,
    "disk_limit_exceeded": True, "current_size_mb": ..., "limit_mb": ...} --
    so the coordinator's existing generic disk-limit scan
    (indexing_coordinator.py:2175-2183) picks it up with zero coordinator
    changes, regardless of which pipeline ran.
    """
    from chunkhound import pipeline_bridge

    fake_native_pipeline.instance.run.return_value = _fake_report(disk_limit=(12.0, 10.0))

    result = await pipeline_bridge.run_rust_pipeline(
        files_to_process=[],
        db_path=tmp_path,
        project_root=tmp_path,
        skip_embeddings=True,
        config=None,
    )

    assert result["errors"] == [
        {
            "file": None,
            "error": "Database disk usage limit exceeded: 12.0 MB >= 10.0 MB",
            "disk_limit_exceeded": True,
            "current_size_mb": 12.0,
            "limit_mb": 10.0,
        }
    ]


@pytest.mark.asyncio
async def test_disk_limit_not_exceeded_report_has_no_extra_error(
    tmp_path: Path, fake_native_pipeline: _FakeNativePipeline
) -> None:
    """A normal (non-tripped) report must not append any disk-limit error."""
    from chunkhound import pipeline_bridge

    result = await pipeline_bridge.run_rust_pipeline(
        files_to_process=[],
        db_path=tmp_path,
        project_root=tmp_path,
        skip_embeddings=True,
        config=None,
    )

    assert result["errors"] == []
