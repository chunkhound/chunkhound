import pytest

import chunkhound_native as native


@pytest.mark.parametrize("files", [[], [("source.py", "source.py")]])
def test_direct_pipeline_refuses_marker_before_even_empty_shortcut(tmp_path, files):
    marker = tmp_path / "chunks.db.chunkhound-install.json"
    marker.write_bytes(b"{")
    pipeline = native.IndexingPipeline(
        {"db_path": str(tmp_path), "skip_embeddings": True}
    )
    with pytest.raises(native.DatabaseInterruptedError):
        pipeline.run(files, lambda paths, config: [("python", []) for _ in paths])
    assert not (tmp_path / "chunks.db").exists()


def test_pipeline_owns_diff_to_store_and_after_backend_close(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("def answer(): return 42\n")
    db_dir = tmp_path / "index"
    target = db_dir / "chunks.db"
    phases = []
    failures = []

    def progress(phase, current, total, *extra):
        # Before writer open / after writer close but before joins.
        if (phase == "parse" and current == 0) or phase == "write-done":
            try:
                lease = native.acquire_install_lease(str(target), "duckdb")
            except native.DatabaseBusyError:
                phases.append(phase)
            else:
                lease.close()
                failures.append(phase)

    pipeline = native.IndexingPipeline(
        {"db_path": str(db_dir), "skip_embeddings": True}
    )
    pipeline.run(
        [(str(source), "source.py")],
        lambda paths, config: [("python", []) for _ in paths],
        progress_callback=progress,
    )
    assert failures == []  # native intentionally ignores callback exceptions
    assert "parse" in phases and "write-done" in phases
    native.acquire_install_lease(str(target), "duckdb").close()
