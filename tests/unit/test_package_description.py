"""Guard: `chunkhound.__description__` matches pyproject's description.

Sibling copy-pin guards: tests/site/test_configurator_defaults_contract.py,
tests/unit/test_matryoshka_embeddings.py (voyage batch-limit pin).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import chunkhound

try:  # tomllib ships with 3.11+; tomli is the pinned fallback on 3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - runs only on 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]


def test_package_description_matches_pyproject() -> None:
    """`chunkhound.__description__` and pyproject's `project.description` are
    two copies of one user-facing string; this guard fails loudly on drift."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)

    assert chunkhound.__description__ == pyproject["project"]["description"]


def test_package_imports_under_python_optimize() -> None:
    """External contract: importing chunkhound under -OO must succeed and
    still expose the pyproject description (docstring is stripped by -OO)."""
    result = subprocess.run(
        [sys.executable, "-c", "import chunkhound; print(chunkhound.__description__)"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**os.environ, "PYTHONOPTIMIZE": "2"},
        check=True,
    )
    with (ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)

    assert result.stdout.strip() == pyproject["project"]["description"]
