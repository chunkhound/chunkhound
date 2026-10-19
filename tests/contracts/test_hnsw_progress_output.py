"""Native indexing must render only the HNSW work that actually runs."""

from pathlib import Path

import pytest

from tests.contracts import test_hnsw_unavailable_progress as unavailable_progress
from tests.contracts.hnsw_progress_output import ProgressOutput, assert_completed_rows
from tests.contracts.pipeline_harness import hnsw_index_names, index_with_rust

unavailable_native_runs = unavailable_progress.unavailable_native_runs

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline"


def _run(db_dir: Path, output: ProgressOutput, *, compact: bool = False,
         enabled: bool = True):
    result = index_with_rust(
        _FIXTURES, db_dir, incremental=True, skip_embeddings=False,
        duckdb_hnsw_enabled=enabled,
        compaction_threshold=0.0 if compact else 1.0,
        compaction_min_size_mb=0 if compact else 10**9,
        progress_callback=output,
    )
    assert not result.errors, result.errors
    assert_completed_rows(output.render())
    assert bool(hnsw_index_names(db_dir)) is enabled
    return result


@pytest.mark.hnsw
def test_enabled_build_and_unchanged_run_render_distinct_work(tmp_path, monkeypatch):
    first = ProgressOutput(monkeypatch)
    assert _run(tmp_path / "db", first).chunks_written > 0
    assert "building..." in first.during("write-index")
    assert "done" in first.render()
    second = ProgressOutput(monkeypatch)
    assert _run(tmp_path / "db", second).chunks_written == 0
    assert "up to date (no changes)" in second.render()
    assert all("building..." not in frame for _, frame in second.frames)
    assert "not needed" in second.render()


@pytest.mark.parametrize(
    "enabled",
    [
        pytest.param(True, marks=pytest.mark.hnsw, id="enabled"),
        pytest.param(False, id="disabled"),
    ],
)
@pytest.mark.parametrize("compact", [True, False], ids=["compact", "normal"])
def test_native_write_output_matches_index_capability(
    tmp_path, monkeypatch, enabled, compact,
):
    output = ProgressOutput(monkeypatch, enabled=enabled)
    result = _run(tmp_path / "db", output, compact=compact, enabled=enabled)
    assert result.chunks_written
    phase = "write-compact" if compact else "write-index"
    if not enabled:
        phase += "-disabled"
        assert all("Vector index" not in frame for _, frame in output.frames)
    if compact:
        label = "with index" if enabled else "no index"
        assert f"compacting ({label})..." in output.during(phase)
        if enabled:
            assert "maintained during compaction" in output.render()
    else:
        assert "not needed" in output.render()


@pytest.mark.parametrize("mode", ["normal", "compact"])
def test_unavailable_native_phases_hide_index_work(
    monkeypatch, unavailable_native_runs, mode,
):
    output = ProgressOutput(monkeypatch)
    # Replay real isolated-process phases; the fixture also verifies persistence.
    for phase in unavailable_native_runs[mode]:
        output(phase, 1 if phase in ("write-done", "done") else 0, 1)
    rendered = output.render()
    assert "Vector index" not in rendered
    assert all("building..." not in frame for _, frame in output.frames)
    if mode == "compact":
        assert "compacting (no index)..." in output.during("write-compact-unavailable")
    else:
        assert "not needed" in rendered
    assert_completed_rows(rendered)
