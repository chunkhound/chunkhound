"""Extract chunkhound_native from the maturin wheel into the active venv.

Also writes the same files into the source tree's chunkhound_native/ (and
any wheel-repair *.libs/ sibling directory) -- the source tree is a real
package too (installed editable), and Python resolves it ahead of
site-packages on sys.path, so leaving it untouched would make this script
silently install a build that never actually gets imported.
"""
import sys
import sysconfig
import zipfile
import pathlib

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
            # unlink before writing: if `dest` already exists it may be
            # hardlinked into uv's shared package cache (uv links from cache
            # whenever cache and target share a filesystem), and writing the
            # content in place would silently corrupt that shared inode for
            # every other venv sharing the cache instead of just this one.
            dest.unlink(missing_ok=True)
            dest.write_bytes(data)
            installed.append(str(dest))
    print(f"Installed {len(installed)} file(s) from {wheels[-1].name}")
