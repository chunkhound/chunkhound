"""Contract test — the compiled extension must resolve its DuckDB dependency
relative to itself, not via a build-machine-absolute path or an external
environment variable.

This is what makes the extension portable from a build machine to an
arbitrary install location: the RPATH is `$ORIGIN` (Linux) / `@loader_path`
(macOS), so the dynamic loader looks next to wherever the extension actually
ends up, rather than at wherever it happened to be linked.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import chunkhound_native as _native_pkg

    _NATIVE_AVAILABLE = True
except ImportError:
    _native_pkg = None
    _NATIVE_AVAILABLE = False

requires_native = pytest.mark.skipif(not _NATIVE_AVAILABLE, reason="native extension not built")
requires_elf = pytest.mark.skipif(sys.platform not in ("linux",), reason="RPATH inspection is Linux-specific here")


@requires_native
@requires_elf
def test_extension_rpath_is_self_relative():
    so_path = Path(_native_pkg.__file__).resolve().parent / "chunkhound_native.abi3.so"
    assert so_path.exists(), f"expected compiled extension at {so_path}"

    out = subprocess.run(
        ["readelf", "-d", str(so_path)], capture_output=True, text=True, check=True
    ).stdout

    rpath_lines = [line for line in out.splitlines() if "RPATH" in line or "RUNPATH" in line]
    assert rpath_lines, f"no RPATH/RUNPATH entry found in {so_path}"
    assert any("$ORIGIN" in line for line in rpath_lines), (
        f"expected a $ORIGIN-relative RPATH entry, got: {rpath_lines}"
    )


@requires_native
@requires_elf
def test_extension_imports_without_absolute_fallback_path():
    """The real acceptance criterion: importing must work even when the
    build-machine-absolute path baked in alongside $ORIGIN is unavailable --
    proving $ORIGIN plus the co-located library is actually sufficient.

    `libduckdb-sys`'s own build.rs unconditionally adds an absolute-path
    RPATH entry for the download-cache directory alongside our $ORIGIN entry
    (confirmed via `readelf`) -- so unsetting LD_LIBRARY_PATH alone doesn't
    prove anything, since that absolute entry still resolves the dependency
    on the same build machine regardless of $ORIGIN. This test must actually
    remove that fallback to be a real regression guard for the $ORIGIN
    mechanism."""
    pkg_dir = Path(_native_pkg.__file__).resolve().parent
    lib_present = any(pkg_dir.glob("libduckdb.so*"))
    assert lib_present, (
        "expected a co-located libduckdb.so next to the extension -- run "
        "scripts/copy_duckdb_runtime.py first"
    )

    repo_root = pkg_dir.parent
    download_dir = repo_root / "target" / "duckdb-download"
    hidden_dir = repo_root / "target" / "duckdb-download.hidden-for-test"
    if hidden_dir.is_dir() and not download_dir.is_dir():
        # Self-heal: a prior run of this test was killed/interrupted before
        # its own finally-block restore ran, leaving the rename half-done.
        # Restoring it here (rather than asserting) avoids sending someone
        # off to rebuild/re-download when the real fix is just this rename.
        hidden_dir.rename(download_dir)
    assert download_dir.is_dir(), (
        f"expected the DUCKDB_DOWNLOAD_LIB absolute-path fallback at {download_dir} "
        "to hide for this test to prove anything -- build with DUCKDB_DOWNLOAD_LIB=1 first"
    )
    download_dir.rename(hidden_dir)

    try:
        env = dict(os.environ)
        env.pop("LD_LIBRARY_PATH", None)
        result = subprocess.run(
            [sys.executable, "-c", "from chunkhound_native import scan_files; scan_files('.', ['py'])"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            env=env,
        )
    finally:
        hidden_dir.rename(download_dir)

    assert result.returncode == 0, result.stderr
