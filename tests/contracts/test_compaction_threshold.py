"""Threshold zero requires real waste, not FLOAT array storage segments."""

from pathlib import Path

import pytest

from tests.contracts.compaction_seed import seed_compaction_waste
from tests.contracts.pipeline_harness import IndexResult, index_with_rust

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"


def _run(db_dir: Path, enabled: bool) -> tuple[IndexResult, set[str]]:
    phases: set[str] = set()
    result = index_with_rust(
        _FIXTURES,
        db_dir,
        incremental=True,
        duckdb_hnsw_enabled=enabled,
        compaction_threshold=0.0,
        compaction_min_size_mb=0,
        progress_callback=lambda phase, *counts: phases.add(phase),
    )
    assert not result.errors, result.errors
    return result, phases


@pytest.mark.parametrize(
    "enabled",
    [
        pytest.param(True, marks=pytest.mark.hnsw, id="enabled"),
        pytest.param(False, id="disabled"),
    ],
)
def test_zero_threshold_skips_clean_store_but_compacts_real_waste(tmp_path, enabled):
    db_dir = tmp_path / "db"
    clean, clean_phases = _run(db_dir, enabled)
    suffix = "" if enabled else "-disabled"
    assert clean.embeddings_generated > 0
    assert f"write-index{suffix}" in clean_phases
    assert not any(phase.startswith("write-compact") for phase in clean_phases)
    seed_compaction_waste(db_dir)
    compacted, compact_phases = _run(db_dir, enabled)
    assert f"write-compact{suffix}" in compact_phases
    assert not any(phase.startswith("write-index") for phase in compact_phases)
    assert compacted.files_processed == compacted.chunks_written == 0
    assert compacted.chunk_tuples == clean.chunk_tuples
    assert compacted.embedding_tuples == clean.embedding_tuples
