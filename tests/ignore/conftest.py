"""Ignore-engine test prerequisites.

This conftest is scoped to the tests/ignore/ package to enforce that
the development environment has a working Git executable available.
We do NOT skip these tests silently: if git is missing, fail fast.

The ``backend`` fixture lives in ``tests/ignore/_backends.py`` so that
tests/contracts/ can reuse the same parametrization and libgit2 skip rule.
"""

import shutil

import pytest

from tests.ignore._backends import backend  # noqa: F401  (re-exported fixture)


@pytest.fixture(scope="session", autouse=True)
def _require_git_available() -> None:
    git = shutil.which("git")
    assert git is not None, (
        "git is required for ignore-engine tests; please install git"
    )
