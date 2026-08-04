import importlib
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent


def _find_bundled_duckdb_files() -> list[Path]:
    """Look for a bundled DuckDB runtime library both directly inside this
    package and in a sibling directory (e.g. `chunkhound_native.libs/`).
    Wheel-repair tools (maturin's built-in auditwheel/delocate/delvewheel
    re-implementation) bundle the library into such a sibling directory, not
    inside the package itself -- checking only _PKG_DIR would report every
    real installed wheel as "missing" regardless of the actual problem.
    Returns full paths (not just names) so a caller can report where each
    file actually lives -- for a real installed wheel that's the sibling
    directory, not _PKG_DIR."""
    candidates = list(_PKG_DIR.glob("*duckdb*"))
    for sibling in _PKG_DIR.parent.glob(f"{_PKG_DIR.name}.*"):
        if sibling.is_dir():
            candidates.extend(sibling.glob("*duckdb*"))
    return sorted(candidates)


if sys.platform == "win32":
    import os

    os.add_dll_directory(str(_PKG_DIR))

try:
    # Import the compiled submodule itself first, separately from any name
    # inside it: a failure here is a genuine load failure (dlopen/DLL-load,
    # e.g. the bundled DuckDB runtime library missing or unloadable) and gets
    # the diagnostic below. A missing attribute (e.g. a stale build lacking
    # scan_files) is a different problem and should raise its own plain,
    # undisguised AttributeError instead of being blamed on DuckDB.
    # importlib.import_module (not `from . import chunkhound_native`) is
    # required here: the bare `from package import name` form resolves via
    # getattr(package, name) if that attribute already exists from a prior
    # successful import, silently skipping the sys.modules check entirely --
    # which would mask a genuinely broken reimport.
    _native_module = importlib.import_module(".chunkhound_native", __name__)
except ImportError as e:
    _bundled = _find_bundled_duckdb_files()
    if not _bundled:
        raise ImportError(
            "chunkhound_native failed to load because its bundled DuckDB "
            f"runtime library could not be found in {_PKG_DIR} or its "
            "sibling wheel-repair directories. This usually means the "
            "chunkhound-native install is incomplete or corrupted. "
            "Try: pip install --force-reinstall chunkhound-native"
        ) from e
    raise ImportError(
        "chunkhound_native failed to load even though a bundled DuckDB "
        f"runtime library was found ({[str(p) for p in _bundled]}). This "
        "usually means an architecture mismatch, a corrupted file, or the "
        "file being blocked/quarantined by antivirus or security software."
    ) from e

scan_files = _native_module.scan_files

try:
    from .chunkhound_native import RustDbWriter
except ImportError:
    RustDbWriter = None  # native extension not built; callers should guard with `if RustDbWriter`

try:
    from .chunkhound_native import (  # type: ignore[import-untyped]
        IndexingPipeline,
        ParseCallConfig,
        PipelineReport,
    )
except ImportError:
    IndexingPipeline = None  # compiled without rust-pipeline feature
    PipelineReport = None
    ParseCallConfig = None
