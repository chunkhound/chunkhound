"""Guard: the package docstring matches pyproject's description.

pyproject.toml `project.description` is the single source of truth; the
`chunkhound` module docstring is a user-facing copy (help(chunkhound)) with no
runtime consumer. This guard is the static pin the project's SSOT rule
requires for the pair. It replaces the deleted
tests/unit/test_package_description.py, which guarded the removed
`__description__` runtime machinery rather than the docstring itself; the
import-under-`-OO` contract survives below, only the attribute's value is gone.
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


def test_package_docstring_matches_pyproject_description() -> None:
    """The module docstring and pyproject's `project.description` are two
    copies of one user-facing string; this guard fails loudly on drift."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)

    assert chunkhound.__doc__ is not None
    # Docstrings end with a period; pyproject's description does not.
    docstring = chunkhound.__doc__.strip().removesuffix(".")
    assert docstring == pyproject["project"]["description"]


def test_package_imports_under_python_optimize() -> None:
    """-OO strips docstrings; the package must still import. This pins the
    import contract the removed `__description__` fallback used to carry."""
    result = subprocess.run(
        [sys.executable, "-c", "import chunkhound"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**os.environ, "PYTHONOPTIMIZE": "2"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
