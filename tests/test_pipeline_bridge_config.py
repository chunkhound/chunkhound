"""Tests for run_rust_pipeline's config-dict mapping.

Covers the contract that user-facing settings (set via CLI, .chunkhound.json,
or env var) are actually forwarded to the Rust pipeline, not overridden by
internal defaults.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _fake_report() -> SimpleNamespace:
    return SimpleNamespace(
        files_processed=0,
        chunks_written=0,
        embeddings_generated=0,
        elapsed_secs=0.0,
        files_skipped=0,
        errors=[],
    )


@pytest.mark.asyncio
async def test_fragmentation_threshold_pct_forwarded_as_compaction_ratio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A user-set --fragmentation-threshold-pct must reach Rust as a ratio.

    DatabaseConfig.fragmentation_threshold_pct is a percentage (e.g. 60.0)
    and is already honored by the Python indexing path (duckdb_provider.py).
    The Rust pipeline bridge must forward the same setting, converted to the
    ratio Rust's PipelineConfig.compaction_threshold expects (0.60) — not a
    hardcoded default.
    """
    from chunkhound import pipeline_bridge

    captured_config_dict: dict = {}

    def _capture_init(self: object, config_dict: dict) -> None:
        captured_config_dict.update(config_dict)

    fake_pipeline_instance = MagicMock()
    fake_pipeline_instance.run.return_value = _fake_report()

    fake_indexing_pipeline_cls = MagicMock(
        side_effect=lambda config_dict: (
            _capture_init(fake_pipeline_instance, config_dict),
            fake_pipeline_instance,
        )[1]
    )

    fake_native_module = types.SimpleNamespace(
        IndexingPipeline=fake_indexing_pipeline_cls
    )
    monkeypatch.setitem(sys.modules, "chunkhound_native", fake_native_module)

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

    assert captured_config_dict["compaction_threshold"] == pytest.approx(0.60)


@pytest.mark.asyncio
async def test_missing_fragmentation_threshold_pct_defaults_to_30_pct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No database config at all falls back to the documented 30% default."""
    from chunkhound import pipeline_bridge

    captured_config_dict: dict = {}

    def _capture_init(self: object, config_dict: dict) -> None:
        captured_config_dict.update(config_dict)

    fake_pipeline_instance = MagicMock()
    fake_pipeline_instance.run.return_value = _fake_report()

    fake_indexing_pipeline_cls = MagicMock(
        side_effect=lambda config_dict: (
            _capture_init(fake_pipeline_instance, config_dict),
            fake_pipeline_instance,
        )[1]
    )

    fake_native_module = types.SimpleNamespace(
        IndexingPipeline=fake_indexing_pipeline_cls
    )
    monkeypatch.setitem(sys.modules, "chunkhound_native", fake_native_module)

    await pipeline_bridge.run_rust_pipeline(
        files_to_process=[],
        db_path=tmp_path,
        project_root=tmp_path,
        skip_embeddings=True,
        config=None,
    )

    assert captured_config_dict["compaction_threshold"] == pytest.approx(0.30)
