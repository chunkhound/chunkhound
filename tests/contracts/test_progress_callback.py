"""Contract test: progress_callback fires the documented phase sequence.

`IndexingPipeline.run()`'s doc comment (`src/pipeline/pipeline.rs`) specifies
``progress_callback(phase: str, current: int, total: int)`` firing at phase
transitions for ``"diff"``, ``"parse"``, ``"embed"``, ``"write-prepare"``,
``"write-data"``, one of ``"write-index"``/``"write-compact"``,
``"write-done"``, and ``"done"``.

Note: the original design specified a 4-arg callback returning `bool` to
support abort-on-`False`.
That was never implemented — `emit_progress`/`emit_progress_gil` call the
Python callback with exactly 3 positional args and discard the return value
via `let _ = cb.bind(py).call1(...)`, so a callback's return value is ignored
and, crucially, an exception it raises is swallowed rather than propagated.
These tests pin down the contract as actually implemented, not the stale
pseudocode.
"""

from pathlib import Path

from tests.contracts.pipeline_harness import default_rust_config


def _write_fixture_files(tmp_path: Path) -> list[tuple[str, str]]:
    """Returns (absolute_path, relative_key) pairs, per IndexingPipeline.run()'s
    files contract — these fixtures live directly under tmp_path, so the
    relative key is just the filename."""
    files = []
    for i in range(3):
        name = f"mod_{i}.py"
        f = tmp_path / name
        f.write_text(f"def fn_{i}():\n    return {i}\n")
        files.append((str(f), name))
    return files


class TestProgressCallback:
    """progress_callback(phase, current, total) fires the documented phase sequence."""

    def test_fires_expected_phases_in_order(self, tmp_path):
        from chunkhound.pipeline_bridge import parse_batch_callback
        from chunkhound_native import IndexingPipeline  # type: ignore[import-untyped]

        file_paths = _write_fixture_files(tmp_path)
        db_dir = tmp_path / "db"
        db_dir.mkdir(parents=True, exist_ok=True)

        calls: list[tuple[str, int, int]] = []

        def progress_callback(phase: str, current: int, total: int, chunks: int = 0) -> None:
            calls.append((phase, current, total))

        pipeline = IndexingPipeline(default_rust_config(db_dir))
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
        from chunkhound.pipeline_bridge import parse_batch_callback
        from chunkhound_native import IndexingPipeline  # type: ignore[import-untyped]

        file_paths = _write_fixture_files(tmp_path)
        db_dir = tmp_path / "db"
        db_dir.mkdir(parents=True, exist_ok=True)

        def raising_progress_callback(phase: str, current: int, total: int) -> None:
            raise RuntimeError("simulated progress callback failure")

        pipeline = IndexingPipeline(default_rust_config(db_dir))

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
