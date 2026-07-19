"""Contract test conftest — sets up LD_LIBRARY_PATH for DuckDB shared lib."""

import os
import sys
from pathlib import Path


def _setup_duckdb_lib_path() -> None:
    """Ensure libduckdb.so is findable at runtime."""
    # Search for libduckdb.so in common build output locations
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidates = [
        repo_root / "target" / "release" / "deps",
        repo_root / "target" / "debug" / "deps",
    ]
    for candidate in candidates:
        if (candidate / "libduckdb.so").exists():
            existing = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = (
                f"{candidate}{os.pathsep}{existing}" if existing else str(candidate)
            )
            return


_setup_duckdb_lib_path()