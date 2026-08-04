"""Contract test — a failed chunkhound_native import must explain itself.

Historically the failure mode when the bundled DuckDB runtime library can't be
found or loaded is a raw, unhelpful OS-level error (e.g. Windows' "DLL load
failed... The specified module could not be found" never says which module).
`chunkhound_native/__init__.py` wraps the load and must raise a specific,
actionable message instead, while still chaining the original exception --
but only for genuine load failures, not for unrelated problems like a stale
build missing an expected symbol.
"""

import importlib
import sys
import tempfile
import types
from pathlib import Path

import pytest


def _import_with_broken_extension(monkeypatch):
    """Simulate the compiled extension failing to load, and (re)trigger
    `chunkhound_native/__init__.py`'s import wrapper."""
    monkeypatch.setitem(sys.modules, "chunkhound_native.chunkhound_native", None)
    with pytest.raises(ImportError) as exc_info:
        if "chunkhound_native" in sys.modules:
            importlib.reload(sys.modules["chunkhound_native"])
        else:
            import chunkhound_native  # noqa: F401
    return exc_info.value


@pytest.fixture(autouse=True)
def _restore_real_native_module():
    """Undo the simulated failure after each test so later tests in the same
    process import the real, working extension again."""
    yield
    sys.modules.pop("chunkhound_native.chunkhound_native", None)
    sys.modules.pop("chunkhound_native", None)
    import chunkhound_native  # noqa: F401


def test_missing_duckdb_library_raises_clear_error(monkeypatch):
    """Simulates an install where the bundled runtime library never made it
    onto disk, regardless of whatever the dev tree happens to have copied
    there already. Moved fully outside pkg_dir, not just renamed in place --
    the diagnostic globs for "*duckdb*", which a same-directory rename would
    still match. Moved into a sibling of pkg_dir (not pytest's tmp_path,
    which lives under the OS temp dir) so the move stays on the same drive
    -- on Windows, a plain rename works even on a DLL already loaded into
    this process, but a cross-drive move falls back to copy+delete, and
    deleting a loaded DLL is denied."""
    import chunkhound_native as native_pkg

    pkg_dir = Path(native_pkg.__file__).resolve().parent
    bundled = list(pkg_dir.glob("*duckdb*"))
    with tempfile.TemporaryDirectory(dir=pkg_dir.parent) as tmp_dir:
        tmp_path = Path(tmp_dir)
        moved = [(p, tmp_path / p.name) for p in bundled]
        for src, dst in moved:
            src.rename(dst)
        try:
            err = _import_with_broken_extension(monkeypatch)
        finally:
            for src, dst in moved:
                dst.rename(src)

    message = str(err)
    assert "bundled DuckDB" in message
    assert "could not be found" in message
    assert "force-reinstall" in message
    assert err.__cause__ is not None


def test_present_but_unloadable_duckdb_library_raises_clear_error(monkeypatch):
    import chunkhound_native as native_pkg

    pkg_dir = Path(native_pkg.__file__).resolve().parent
    fake_lib = pkg_dir / "libduckdb.so.fake"
    fake_lib.write_bytes(b"not a real library")
    try:
        err = _import_with_broken_extension(monkeypatch)
    finally:
        fake_lib.unlink()

    message = str(err)
    assert "libduckdb.so.fake" in message
    assert "architecture mismatch" in message
    assert err.__cause__ is not None


def test_bundled_duckdb_lookup_checks_sibling_libs_directory(monkeypatch, tmp_path):
    """A real `maturin build --auditwheel repair` wheel bundles the DuckDB
    runtime library into a *sibling* directory (e.g. chunkhound_native.libs/),
    not inside the chunkhound_native/ package itself -- confirmed by building
    a real wheel and inspecting its layout. The lookup must check there too,
    or every real installed wheel would be misreported as missing the
    library regardless of the actual problem."""
    import chunkhound_native as native_pkg

    fake_pkg_dir = tmp_path / "chunkhound_native"
    fake_pkg_dir.mkdir()
    sibling_libs_dir = tmp_path / "chunkhound_native.libs"
    sibling_libs_dir.mkdir()
    expected = sibling_libs_dir / "libduckdb-deadbeef.so"
    expected.write_bytes(b"not a real library")

    monkeypatch.setattr(native_pkg, "_PKG_DIR", fake_pkg_dir)

    assert native_pkg._find_bundled_duckdb_files() == [expected]


def test_present_but_unloadable_error_reports_actual_sibling_location(monkeypatch, tmp_path):
    """The error message must name where the file actually is -- a real
    installed wheel's bundled library lives in the sibling .libs directory,
    not inside chunkhound_native/ itself, and the message must say so rather
    than always claiming _PKG_DIR regardless of the real location."""
    import chunkhound_native as native_pkg

    sibling_libs_dir = Path(native_pkg.__file__).resolve().parent.parent / "chunkhound_native.libs"
    sibling_libs_dir.mkdir(exist_ok=True)
    fake_lib = sibling_libs_dir / "libduckdb-sibling-fake.so"
    fake_lib.write_bytes(b"not a real library")
    try:
        err = _import_with_broken_extension(monkeypatch)
    finally:
        fake_lib.unlink()
        sibling_libs_dir.rmdir()

    message = str(err)
    assert str(sibling_libs_dir) in message
    assert "libduckdb-sibling-fake.so" in message
    assert err.__cause__ is not None


def test_stale_build_missing_symbol_does_not_blame_duckdb(monkeypatch):
    """The compiled submodule loading successfully but lacking an expected
    symbol (e.g. a stale build) is a different problem than a DuckDB load
    failure and must not be misreported as one."""
    fake_module = types.ModuleType("chunkhound_native.chunkhound_native")
    monkeypatch.setitem(sys.modules, "chunkhound_native.chunkhound_native", fake_module)

    with pytest.raises(AttributeError) as exc_info:
        importlib.reload(sys.modules["chunkhound_native"])

    message = str(exc_info.value)
    assert "scan_files" in message
    assert "duckdb" not in message.lower()
