import errno
import os

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


@pytest.mark.skipif(os.name != "posix", reason="requires Unix filename bytes")
@pytest.mark.parametrize("empty", [True, False], ids=["empty", "nonempty"])
@pytest.mark.parametrize("db_present", [False, True], ids=["absent", "existing"])
def test_pipeline_rejects_non_utf8_canonical_target_without_mutation(
    tmp_path, empty, db_present
):
    raw_dir = os.fsencode(tmp_path) + b"/index-\xff"
    try:
        os.mkdir(raw_dir)
    except OSError as error:
        if error.errno == errno.EILSEQ:
            pytest.skip("filesystem does not support non-UTF8 directory names")
        raise
    alias = tmp_path / "alias"
    os.symlink(raw_dir, os.fsencode(alias))
    neighbor = tmp_path / "index-\ufffd"
    neighbor.mkdir()
    # Invalid DB/recovery inputs make premature preflight fail with a different
    # error. The absent case also exercises the empty-run shortcut.
    for directory in (alias, neighbor):
        suffixes = [".old", ".compact", ".wal", ".swap_intent"]
        if db_present:
            suffixes.append("")
        for suffix in suffixes:
            (directory / f"chunks.db{suffix}").write_bytes(b"untouched input")
        # Stable coordination creation is allowed; include it in the snapshot.
        native.acquire_database_lease(str(directory / "chunks.db")).close()

    def snapshot():
        return {
            (directory.name, path.name): (path.read_bytes(), path.stat().st_mtime_ns)
            for directory in (alias, neighbor)
            for path in directory.iterdir()
        }

    before = snapshot()
    source = tmp_path / "source.py"
    source.write_text("def answer(): return 42\n")
    files = [] if empty else [(str(source), "source.py")]
    pipeline = native.IndexingPipeline(
        {"db_path": str(alias), "skip_embeddings": True}
    )
    with pytest.raises(native.DatabaseLayoutError, match="UTF-8"):
        pipeline.run(files, lambda paths, config: [("python", []) for _ in paths])
    assert snapshot() == before
    # Error release must not depend on destroying the pipeline object.
    for directory in (alias, neighbor):
        native.acquire_install_lease(str(directory / "chunks.db"), "duckdb").close()
    assert snapshot() == before


@pytest.mark.parametrize("location", ["unicode", "alias", "memory", "empty-memory"])
def test_pipeline_supports_utf8_targets_and_memory(tmp_path, location):
    target_dir = tmp_path / "索引-é"
    if location == "alias":
        target_dir.mkdir()
        alias = tmp_path / "alias"
        try:
            alias.symlink_to(target_dir, target_is_directory=True)
        except OSError:
            pytest.skip("directory symlink creation is unavailable")
        db_path = str(alias)
    elif location == "memory":
        db_path = ":memory:"
    elif location == "empty-memory":
        db_path = ""
    else:
        db_path = str(target_dir)
    source = tmp_path / "source.py"
    source.write_text("def answer(): return 42\n")
    pipeline = native.IndexingPipeline({"db_path": db_path, "skip_embeddings": True})
    for incremental in (False, True):
        report = pipeline.run(
            [(str(source), "source.py")],
            lambda paths, config: [("python", []) for _ in paths],
            incremental=incremental,
        )
        assert report.errors == []
    if location in ("unicode", "alias"):
        assert (target_dir / "chunks.db").is_file()
        native.acquire_install_lease(str(target_dir / "chunks.db"), "duckdb").close()
