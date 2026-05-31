"""Extract chunkhound_native from the maturin wheel into the active venv."""
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

site = pathlib.Path(sysconfig.get_path("platlib"))
if not site.exists():
    sys.exit(f"site-packages not found at {site}")

with zipfile.ZipFile(wheels[-1]) as z:
    installed = []
    for name in z.namelist():
        is_extension = name.startswith("chunkhound_native") and (
            name.endswith(".so") or name.endswith(".pyd")
        )
        is_init = name == "chunkhound_native/__init__.py"
        if is_extension or is_init:
            dest = site / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(name))
            installed.append(str(dest))
    print(f"Installed {len(installed)} file(s) from {wheels[-1].name}")
