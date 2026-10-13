"""Real-backend contracts for owned and borrowed inspection lifetimes."""

import traceback
import weakref
from pathlib import Path

import lance
import pytest

import chunkhound_native as native
from chunkhound.core.index_format import IndexFormatError
from chunkhound.hub.inspect import inspect_database
from chunkhound.providers.database.index_format import (
    preflight_duckdb,
    read_format,
    read_only_duckdb,
)
from chunkhound.providers.database.ownership import DatabaseOwnership
from tests.hub.artifacts import build_database
from tests.hub.test_index_format import run_python


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
def test_inventory_can_borrow_exclusive_ownership_without_nested_lock(
    tmp_path, backend
):
    path = build_database(tmp_path / "published", backend)
    with DatabaseOwnership(native.acquire_install_lease(str(path), backend)) as owner:
        inventory = inspect_database(path, backend, ownership=owner)
        assert inventory.format_version == 1
        assert sum(group.count for group in inventory.groups) == 100
        owner.assert_covers(path)
        with pytest.raises(native.DatabaseLayoutError):
            inspect_database(tmp_path / "other.db", backend, ownership=owner)
    with pytest.raises(native.DatabaseLayoutError):
        inspect_database(path, backend, ownership=owner)
    replacement = native.acquire_install_lease(str(path), backend)
    replacement.close()


def test_ordinary_inspection_owns_database_until_context_exit(tmp_path):
    path = build_database(tmp_path / "published")
    with read_only_duckdb(path) as conn:
        assert conn.execute("SELECT count(*) FROM inspected.main.files").fetchone()[0]
        run_python(
            """
            import sys
            import chunkhound_native as native
            try:
                native.acquire_install_lease(sys.argv[1], 'duckdb')
            except native.DatabaseBusyError:
                pass
            else:
                raise AssertionError('inspection released ownership early')
            """,
            path,
        )
    exclusive = native.acquire_install_lease(str(path), "duckdb")
    exclusive.close()


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
def test_marker_refuses_ordinary_inspection_but_not_exclusive_identity_validation(
    tmp_path, backend
):
    path = build_database(tmp_path / "published", backend)
    marker = Path(str(path) + ".chunkhound-install.json")
    marker.write_bytes(b"{")
    with pytest.raises(native.DatabaseInterruptedError):
        inspect_database(path, backend)
    lease = native.acquire_install_lease(str(path), backend)
    with DatabaseOwnership(lease) as owner:
        assert lease.interrupted
        # Explicit recovery may inspect a still-readable partial target. The
        # parent skips inspection if its recovery policy needs no old inventory.
        assert inspect_database(path, backend, ownership=owner).format_version == 1
    assert marker.read_bytes() == b"{"


@pytest.mark.parametrize("operation", ["format", "inventory"])
def test_retained_chained_inspection_error_releases_all_datasets(
    tmp_path, monkeypatch, operation
):
    path = build_database(tmp_path / "published", "lancedb")
    real_dataset = lance.dataset
    datasets = []
    target = "schema_version.lance" if operation == "format" else "chunks.lance"

    def failed_dataset(uri, *args, **kwargs):
        if Path(uri).name != target:
            return real_dataset(uri, *args, **kwargs)

        def failed_read():
            dataset = real_dataset(uri, *args, **kwargs)
            datasets.append(weakref.ref(dataset))
            raise OSError("backend read failed")

        try:
            failed_read()
        except OSError as context:
            # Distinct cause/context branches plus a cycle model a third-party
            # wrapper. Both branches retain real handles in completed frames.
            try:
                failed_read()
            except OSError as cause:
                cause.__context__ = None
                error = RuntimeError("dataset inspection failed")
                error.__cause__ = cause
                error.__context__ = context
                context.__cause__ = error
        raise error

    monkeypatch.setattr(lance, "dataset", failed_dataset)
    with pytest.raises(RuntimeError, match="dataset inspection failed") as raised:
        inspect_path(path, operation, "lancedb")
    native.acquire_install_lease(str(path), "lancedb").close()
    # Diagnostic chaining remains intact, including cycles and hidden context.
    assert raised.value.__cause__ is not raised.value.__context__
    assert raised.value.__context__.__cause__ is raised.value
    assert traceback.extract_tb(raised.value.__cause__.__traceback__)
    assert traceback.extract_tb(raised.value.__context__.__traceback__)
    assert len(datasets) == 2
    assert all(dataset() is None for dataset in datasets)


INSPECTIONS = [
    ("format", "duckdb"),
    ("format", "lancedb"),
    ("inventory", "duckdb"),
    ("inventory", "lancedb"),
    ("reader", "duckdb"),
    ("preflight", "duckdb"),
]


def inspect_path(path, operation, backend, *, ownership=None):
    if operation == "inventory":
        return inspect_database(path, backend, ownership=ownership)
    if operation == "format":
        return read_format(path, backend, ownership=ownership)
    if operation == "preflight":
        return preflight_duckdb(path, ownership=ownership)
    with read_only_duckdb(path, ownership=ownership):
        raise AssertionError("absent inspection opened a DuckDB connection")


def assert_absent_inspection(path, operation, backend):
    if operation == "inventory":
        with pytest.raises(ValueError, match="Database is absent"):
            inspect_path(path, operation, backend)
    elif backend == "duckdb" and operation != "preflight":
        with pytest.raises(IndexFormatError, match="format compatibility is unknown"):
            inspect_path(path, operation, backend)
    else:
        expected = None if operation == "preflight" else 1
        assert inspect_path(path, operation, backend) == expected


@pytest.mark.parametrize("operation,backend", INSPECTIONS)
def test_absent_parent_alias_inspection_does_not_create_paths(
    tmp_path, operation, backend
):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    path = alias / "missing" / "database"
    assert_absent_inspection(path, operation, backend)
    assert list(real.iterdir()) == []


@pytest.mark.parametrize("operation,backend", INSPECTIONS)
def test_missing_parent_does_not_bypass_borrowed_ownership(
    tmp_path, operation, backend
):
    filename = "chunks.db" if backend == "duckdb" else "lancedb.lancedb"
    owner = DatabaseOwnership(
        native.acquire_install_lease(str(tmp_path / filename), backend)
    )
    path = tmp_path / "missing" / filename
    try:
        with pytest.raises(native.DatabaseLayoutError):
            inspect_path(path, operation, backend, ownership=owner)
    finally:
        owner.close()
    with pytest.raises(native.DatabaseLayoutError):
        inspect_path(path, operation, backend, ownership=owner)
    assert not path.parent.exists()


@pytest.mark.parametrize("operation,backend", INSPECTIONS)
def test_missing_parent_spelling_cannot_bypass_existing_parent_marker(
    tmp_path, operation, backend
):
    real = tmp_path / "real"
    (real / "nested").mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(real / "nested", target_is_directory=True)
    filename = "chunks.db" if backend == "duckdb" else "lancedb.lancedb"
    path = alias / ".." / "missing" / ".." / filename
    marker = real / (filename + ".chunkhound-install.json")
    marker.write_bytes(b"{")
    with pytest.raises(native.DatabaseInterruptedError):
        inspect_path(path, operation, backend)
    assert marker.read_bytes() == b"{"
    assert not (real / "missing").exists()


@pytest.mark.parametrize("operation,backend", INSPECTIONS)
def test_parent_creation_after_absence_observation_never_opens_backend(
    tmp_path, monkeypatch, operation, backend
):
    import duckdb
    import lance

    staged = build_database(tmp_path / "staged", backend)
    path = tmp_path / "missing" / staged.name
    original_stat = Path.stat
    observed_absence = False

    def stat_with_concurrent_creation(candidate, *args, **kwargs):
        nonlocal observed_absence
        if candidate == path.parent and not observed_absence:
            with pytest.raises(FileNotFoundError):
                original_stat(candidate, *args, **kwargs)
            observed_absence = True
            # Publish real storage after the filesystem has observed absence,
            # before that observation reaches the inspection caller.
            path.parent.mkdir()
            staged.rename(path)
            raise FileNotFoundError(str(candidate))
        return original_stat(candidate, *args, **kwargs)

    def forbid_backend_open(*args, **kwargs):
        raise AssertionError("absence inspection opened a backend")

    monkeypatch.setattr(Path, "stat", stat_with_concurrent_creation)
    monkeypatch.setattr(duckdb, "connect", forbid_backend_open)
    monkeypatch.setattr(lance, "dataset", forbid_backend_open)
    assert_absent_inspection(path, operation, backend)
    assert observed_absence
    assert path.exists()
    assert not Path(str(path) + ".chunkhound.lock").exists()
