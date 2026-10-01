"""Contract test: a symlinked file's DB row must survive incremental re-indexing.

Regression test for the symlink path-key divergence bug: `run_rust_pipeline()`
(`chunkhound/pipeline_bridge.py`) used to call `Path.resolve()` unconditionally
on every discovered file before crossing into Rust, discarding the symlink's
*logical* path — the one Python's own DB-write path preserves via
`get_relative_path_safe()` (git-worktree support: a symlink's logical path is
kept even when its target resolves outside project_root). The Rust differ
(`src/pipeline/differ.rs`) then re-derived the relative DB key from that
already-resolved absolute path via a naive `strip_prefix`, producing a
different (or missing) key than the one the file's DB row was actually
written under — so an incremental run misclassified the symlink's row as
`removed` and deleted it, even though the file was simultaneously being
processed as `changed`.

The fix moved relative-key computation to Python (the single already-tested
source of truth), passed alongside each absolute path — see
`IndexingPipeline.run()`'s `files: Vec<(String, String)>` parameter. This test
proves the fix end-to-end: index a symlinked file, then re-index incrementally
with zero filesystem changes, and confirm its row is not deleted.
"""

import pytest

from chunkhound.pipeline_bridge import run_rust_pipeline
from tests.contracts.pipeline_harness import collect_table_counts, files_table_paths


class TestSymlinkPreservesLogicalPath:
    """A symlinked file's DB row must survive a no-op incremental re-index."""

    @pytest.mark.asyncio
    async def test_symlinked_file_survives_incremental_reindex(self, tmp_path):
        project_root = tmp_path / "project"
        project_root.mkdir()
        target = project_root / "real_module.py"
        target.write_text("def real():\n    return 1\n")
        link = project_root / "linked_module.py"
        link.symlink_to(target)

        db_dir = tmp_path / "db"
        db_dir.mkdir()

        files_to_process = [(target, None), (link, None)]

        first = await run_rust_pipeline(
            files_to_process,
            db_path=db_dir,
            project_root=project_root,
            force_reindex=True,
            skip_embeddings=True,
        )
        assert not first["errors"], f"Unexpected errors on first index: {first['errors']}"
        assert first["total_files"] == 2

        before_counts = collect_table_counts(db_dir)
        assert before_counts["files"] == 2
        before_paths = files_table_paths(db_dir)
        assert before_paths == {"real_module.py", "linked_module.py"}, (
            "the symlink's DB row must be keyed by its own logical path, "
            f"not its resolved target's path — got {before_paths}"
        )

        # Second, incremental pass with no filesystem changes at all — nothing
        # should be classified as changed or removed.
        second = await run_rust_pipeline(
            files_to_process,
            db_path=db_dir,
            project_root=project_root,
            force_reindex=False,
            skip_embeddings=True,
        )
        assert not second["errors"], f"Unexpected errors on incremental re-index: {second['errors']}"
        assert second["total_files"] == 0, (
            "nothing changed on disk — an incremental run must not reprocess "
            "either file"
        )

        after_counts = collect_table_counts(db_dir)
        assert after_counts["files"] == 2, (
            "the symlinked file's DB row must survive an incremental "
            f"re-index with no filesystem changes, got {after_counts}"
        )
        after_paths = files_table_paths(db_dir)
        assert after_paths == before_paths, (
            "file paths must be stable across a no-op incremental run — "
            f"before={before_paths} after={after_paths}"
        )
