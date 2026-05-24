"""Extract chunkhound_native from the maturin wheel into the active venv."""
import sys
import zipfile
import pathlib

wheels = sorted(pathlib.Path("target/wheels").glob("chunkhound*.whl"))
if not wheels:
    sys.exit("No wheel found in target/wheels/ — run 'maturin build --out target/wheels/' first")

site = pathlib.Path(".venv/lib") / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if not site.exists():
    sys.exit(f"site-packages not found at {site}")

with zipfile.ZipFile(wheels[-1]) as z:
    installed = []
    for name in z.namelist():
        if name.startswith("chunkhound_native/"):
            dest = site / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(name))
            installed.append(str(dest))
    print(f"Installed {len(installed)} file(s) from {wheels[-1].name}")
