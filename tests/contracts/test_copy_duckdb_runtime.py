"""Regression test for scripts/copy_duckdb_runtime.py's version selection.

A GitHub Actions cache restore can bring back a stale
target/duckdb-download/<triple>/<old-version>/ directory from before a
Cargo.toml duckdb version pin, sitting alongside a freshly downloaded
directory for the version Cargo.lock now actually resolves to. Selecting
by mtime alone can pick the stale one if its cached timestamp happens to
be newer -- which is exactly what silently shipped a reverted DuckDB
version regression back into CI. See
scripts/copy_duckdb_runtime.py::_locked_duckdb_core_version().
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "copy_duckdb_runtime.py"

_RUNTIME_LIB_NAME = {"win32": "duckdb.dll", "darwin": "libduckdb.dylib"}.get(
    sys.platform, "libduckdb.so"
)


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    (tmp_path / "scripts").mkdir()
    shutil.copy(SCRIPT, tmp_path / "scripts" / "copy_duckdb_runtime.py")
    (tmp_path / "chunkhound_native").mkdir()
    return tmp_path


def _write_lockfile(repo: Path, libduckdb_sys_version: str) -> None:
    (repo / "Cargo.lock").write_text(
        "# auto-generated, do not edit\n"
        "[[package]]\n"
        'name = "libduckdb-sys"\n'
        f'version = "{libduckdb_sys_version}"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
    )


def _run_script(repo: Path) -> None:
    subprocess.run(
        [sys.executable, str(repo / "scripts" / "copy_duckdb_runtime.py")],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def test_prefers_the_cargo_lock_locked_version_over_a_newer_stale_cache_entry(
    fake_repo: Path,
) -> None:
    _write_lockfile(fake_repo, "1.10504.0")  # DuckDB core 1.5.4

    download = fake_repo / "target" / "duckdb-download" / "some-triple"
    locked_dir = download / "1.5.4"
    stale_dir = download / "1.5.5"
    locked_dir.mkdir(parents=True)
    stale_dir.mkdir(parents=True)
    (locked_dir / _RUNTIME_LIB_NAME).write_text("correct-locked-build")
    (stale_dir / _RUNTIME_LIB_NAME).write_text("stale-cached-from-before-the-pin")

    # The stale entry has a newer mtime than the freshly downloaded, locked
    # one -- e.g. a cache-restore step preserves a recent cache-save
    # timestamp while the correct download's archive carries an older
    # embedded release date. Exactly the case naive newest-mtime-wins
    # selection gets wrong.
    os.utime(locked_dir / _RUNTIME_LIB_NAME, (1_700_000_000, 1_700_000_000))
    os.utime(stale_dir / _RUNTIME_LIB_NAME, (1_800_000_000, 1_800_000_000))

    _run_script(fake_repo)

    installed = (fake_repo / "chunkhound_native" / _RUNTIME_LIB_NAME).read_text()
    assert installed == "correct-locked-build"


def test_falls_back_to_newest_mtime_when_cargo_lock_is_absent(
    fake_repo: Path,
) -> None:
    """No Cargo.lock (e.g. a stripped sdist build context) must not crash --
    degrade to the old newest-mtime-wins behavior rather than erroring."""
    download = fake_repo / "target" / "duckdb-download" / "some-triple" / "1.5.4"
    download.mkdir(parents=True)
    (download / _RUNTIME_LIB_NAME).write_text("only-candidate")

    _run_script(fake_repo)

    installed = (fake_repo / "chunkhound_native" / _RUNTIME_LIB_NAME).read_text()
    assert installed == "only-candidate"
