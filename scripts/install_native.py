"""Extract chunkhound_native from the maturin wheel into the active venv.

Also writes the same files into the source tree's chunkhound_native/ (and
any wheel-repair *.libs/ sibling directory) -- the source tree is a real
package too (installed editable), and Python resolves it ahead of
site-packages on sys.path, so leaving it untouched would make this script
silently install a build that never actually gets imported.
"""
import pathlib
import subprocess
import sys
import sysconfig
import zipfile

wheels = sorted(pathlib.Path("target/wheels").glob("chunkhound*.whl"))
if not wheels:
    sys.exit("No wheel found in target/wheels/ — run 'maturin build --out target/wheels/' first")
if len(wheels) > 1:
    names = "\n  ".join(str(w) for w in wheels)
    sys.exit(f"Multiple wheels found in target/wheels/ — clear the directory before rebuilding:\n  {names}")

repo_root = pathlib.Path(__file__).resolve().parent.parent
site = pathlib.Path(sysconfig.get_path("platlib"))
if not site.exists():
    sys.exit(f"site-packages not found at {site}")

destinations = [repo_root, site]


def _is_dirty(relative_path: str) -> bool:
    """True if `relative_path` has uncommitted changes in the repo's git tree."""
    result = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", relative_path],
        cwd=repo_root,
    )
    return result.returncode == 1


with zipfile.ZipFile(wheels[-1]) as z:
    installed = []
    for name in z.namelist():
        is_extension = name.startswith("chunkhound_native") and (
            name.endswith(".so") or name.endswith(".pyd")
        )
        is_init = name == "chunkhound_native/__init__.py"
        # Wheel-repair tools (auditwheel/delocate/delvewheel) bundle shared
        # libraries the extension depends on (e.g. DuckDB) into a sibling
        # "<pkg>.libs/" directory alongside the package, not inside it.
        is_libs_file = name.split("/", 1)[0].endswith(".libs")
        if not (is_extension or is_init or is_libs_file):
            continue
        data = z.read(name)
        for dest_root in destinations:
            dest = dest_root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            # The source tree's copy is a real, hand-editable file (unlike
            # site-packages' copy) -- refuse to clobber an uncommitted local
            # edit with whatever happened to be baked into this wheel.
            if (
                dest_root is repo_root
                and dest.exists()
                and dest.read_bytes() != data
                and _is_dirty(name)
            ):
                sys.exit(
                    f"{dest} has uncommitted changes that differ from this "
                    "build -- commit or stash it first, or you'll lose those "
                    "edits."
                )
            # Write to a temp file and atomically rename it into place:
            # `dest` may be hardlinked into uv's shared package cache (uv
            # links from cache whenever cache and target share a filesystem),
            # so writing content in place would corrupt that shared inode for
            # every other venv sharing the cache. Renaming a fresh temp file
            # over `dest` replaces the directory entry instead of writing
            # through the old inode, and is atomic -- no window where `dest`
            # is missing if the process is interrupted mid-write.
            tmp = dest.parent / (dest.name + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(dest)
            installed.append(str(dest))
    print(f"Installed {len(installed)} file(s) from {wheels[-1].name}")
