"""Contract test: progress_callback fires the documented phase sequence.

`IndexingPipeline.run()`'s doc comment (`src/pipeline/pipeline.rs`) specifies
``progress_callback(phase: str, current: int, total: int)`` firing at phase
transitions for ``"diff"``, ``"parse"``, ``"embed"``, ``"write-prepare"``,
``"write-data"``, one of ``"write-index"``/``"write-compact"``,
``"write-done"``, and ``"done"``.

Note: the design doc's original contract (`docs/rust-pipeline-architecture.html`
§5.3) specified a 4-arg callback returning `bool` to support abort-on-`False`.
That was never implemented — `emit_progress`/`emit_progress_gil` call the
Python callback with exactly 3 positional args and discard the return value
via `let _ = cb.bind(py).call1(...)`, so a callback's return value is ignored
and, crucially, an exception it raises is swallowed rather than propagated.
These tests pin down the contract as actually implemented, not the stale
pseudocode.
"""

from pathlib import Path

import pytest


def _rust_config(project_root: Path, db_dir: Path) -> dict:
    return {
        "project_root": str(project_root.resolve()),
        "db_path": str(db_dir.resolve()),
        "db_batch_size": 100,
        "compaction_threshold": 0.60,
        "compaction_min_size_mb": 10,
        "parse_batch_size": 200,
        "parse_thread_pool_size": 4,
        "embed_batch_size": 200,
        "force_reindex": False,
        "mtime_epsilon_seconds": 0.01,
        "do_cleanup": True,
        "skip_embeddings": True,
        "per_file_timeout_secs": 3.0,
        "per_file_timeout_min_size_kb": 128,
        "detect_embedded_sql": True,
        "config_file_size_threshold_kb": 20,
        "embedding_provider": "",
        "embedding_model": "",
    }


def _write_fixture_files(tmp_path: Path) -> list[str]:
    files = []
    for i in range(3):
        f = tmp_path / f"mod_{i}.py"
        f.write_text(f"def fn_{i}():\n    return {i}\n")
        files.append(str(f))
    return files


class TestProgressCallback:
    """progress_callback(phase, current, total) fires the documented phase sequence."""

    def test_fires_expected_phases_in_order(self, tmp_path):
        try:
            from chunkhound_native import IndexingPipeline  # type: ignore[import-untyped]
        except ImportError:
            pytest.fail(
                "Rust IndexingPipeline is not yet available in chunkhound_native."
            )
        from chunkhound.pipeline_bridge import parse_batch_callback

        file_paths = _write_fixture_files(tmp_path)
        db_dir = tmp_path / "db"
        db_dir.mkdir(parents=True, exist_ok=True)

        calls: list[tuple[str, int, int]] = []

        def progress_callback(phase: str, current: int, total: int, chunks: int = 0) -> None:
            calls.append((phase, current, total))

        pipeline = IndexingPipeline(_rust_config(tmp_path, db_dir))
        report = pipeline.run(
            files=file_paths,
            parse_batch_callback=parse_batch_callback,
            embed_batch_callback=None,
            progress_callback=progress_callback,
            incremental=True,
        )

        assert not report.errors, f"Unexpected errors: {report.errors}"
        assert report.chunks_written > 0
        assert calls, "progress_callback should have fired at least once"

        phases_seen = [c[0] for c in calls]

        # "diff" fires first — this is a first-ever run (incremental=True,
        # no DB yet), so compute_diff_blocking short-circuits immediately.
        assert phases_seen[0] == "diff"
        assert "parse" in phases_seen
        assert "write-prepare" in phases_seen
        assert "write-data" in phases_seen
        assert "write-done" in phases_seen
        # write-index and write-compact are mutually exclusive — compaction
        # rebuilds the HNSW index as part of its own rewrite (see
        # test_compaction_before_index.py), so exactly one of the two fires.
        has_index = "write-index" in phases_seen
        has_compact = "write-compact" in phases_seen
        assert has_index != has_compact, (
            f"Expected exactly one of write-index/write-compact, got phases: {phases_seen}"
        )
        # "done" is always the final call.
        assert phases_seen[-1] == "done"

        # Ordering: diff < parse < write-prepare < write-done < done.
        assert phases_seen.index("diff") < phases_seen.index("parse")
        assert phases_seen.index("parse") < phases_seen.index("write-prepare")
        assert phases_seen.index("write-prepare") < phases_seen.index("write-done")
        assert phases_seen.index("write-done") < phases_seen.index("done")

        # "parse" progress is monotonically non-decreasing and its final
        # call reaches (total_files, total_files).
        parse_calls = [c for c in calls if c[0] == "parse"]
        currents = [c[1] for c in parse_calls]
        assert currents == sorted(currents), "parse phase current should never regress"
        assert parse_calls[-1][1] == parse_calls[-1][2] == len(file_paths)

        # The final "done" call reports (file_count, file_count).
        done_call = calls[-1]
        assert done_call[1] == done_call[2] == len(file_paths)

    def test_exception_in_callback_does_not_abort_pipeline(self, tmp_path):
        """A callback that raises must not crash or abort the run — Rust
        discards the callback's PyResult (`let _ = cb.bind(py).call1(...)`),
        so an exception is silently swallowed rather than propagated.
        """
        try:
            from chunkhound_native import IndexingPipeline  # type: ignore[import-untyped]
        except ImportError:
            pytest.fail(
                "Rust IndexingPipeline is not yet available in chunkhound_native."
            )
        from chunkhound.pipeline_bridge import parse_batch_callback

        file_paths = _write_fixture_files(tmp_path)
        db_dir = tmp_path / "db"
        db_dir.mkdir(parents=True, exist_ok=True)

        def raising_progress_callback(phase: str, current: int, total: int) -> None:
            raise RuntimeError("simulated progress callback failure")

        pipeline = IndexingPipeline(_rust_config(tmp_path, db_dir))

        # Must not raise.
        report = pipeline.run(
            files=file_paths,
            parse_batch_callback=parse_batch_callback,
            embed_batch_callback=None,
            progress_callback=raising_progress_callback,
            incremental=False,
        )

        assert not report.errors, f"Unexpected errors: {report.errors}"
        assert report.chunks_written > 0
        assert report.files_processed == len(file_paths)
