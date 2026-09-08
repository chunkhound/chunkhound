"""Copy the downloaded DuckDB runtime library next to the compiled
chunkhound_native extension, for local maturin build + install_native.py
setups.

Wheel-repair tools (auditwheel/delocate/delvewheel) bundle the runtime into
release wheels, but local maturin-build wheels don't reliably carry it (see
_select_files in scripts/install_native.py for what gets extracted). This
script closes that gap for the local dev loop, on every platform, so the
self-relative $ORIGIN/@loader_path RPATH baked in at link time has something
to find.
"""

import pathlib
import shutil
import sys

RUNTIME_LIB_NAMES = {
    "win32": "duckdb.dll",
    "darwin": "libduckdb.dylib",
}
RUNTIME_LIB_NAME = RUNTIME_LIB_NAMES.get(sys.platform, "libduckdb.so")

repo_root = pathlib.Path(__file__).resolve().parent.parent

# Cargo copies a dependency's shared library into its own deps/ directory
# after every successful build, regardless of how that library was sourced
# (DUCKDB_DOWNLOAD_LIB, DUCKDB_LIB_DIR, a system install, ...) -- prefer that
# canonical location over reaching into the download cache directly. Compare
# mtimes across *all* candidate locations together (not tier-by-tier) so a
# leftover release-profile artifact can't shadow a fresher debug build (or
# vice versa) just because it happens to be checked first.
candidates = sorted(
    [
        *repo_root.glob(f"target/release/deps/{RUNTIME_LIB_NAME}"),
        *repo_root.glob(f"target/debug/deps/{RUNTIME_LIB_NAME}"),
        *repo_root.glob(f"target/duckdb-download/*/*/{RUNTIME_LIB_NAME}"),
    ],
    key=lambda p: p.stat().st_mtime,
)
if not candidates:
    sys.exit(
        f"No built/downloaded {RUNTIME_LIB_NAME} found under target/ -- run "
        "the build with DUCKDB_DOWNLOAD_LIB=1 first."
    )
source = candidates[-1]

destinations = [repo_root / "chunkhound_native"]
destinations += [
    p.parent
    for p in repo_root.glob(".venv/**/site-packages/chunkhound_native/__init__.py")
]

copied = []
for dest_dir in destinations:
    if not dest_dir.is_dir():
        continue
    dest = dest_dir / source.name
    # Copy to a temp file and atomically rename it into place: `dest` may be
    # hardlinked into uv's shared package cache (uv links from cache whenever
    # cache and target share a filesystem), and shutil.copy2 writing directly
    # into `dest` would overwrite content in place -- silently corrupting
    # that shared inode for every other venv sharing the cache. Renaming a
    # fresh temp file over `dest` replaces the directory entry instead of
    # writing through the old inode, and is atomic -- no window where `dest`
    # is missing if the process is interrupted mid-copy.
    tmp = dest.parent / (dest.name + ".tmp")
    shutil.copy2(source, tmp)
    tmp.replace(dest)
    copied.append(str(dest))

if not copied:
    sys.exit(f"No chunkhound_native/ destination found to copy {source} into.")
print(f"Copied {source} -> " + ", ".join(copied))
