"""Ignore-engine test prerequisites.

This conftest is scoped to the tests/ignore/ package to enforce that
the development environment has a working Git executable available.
We do NOT skip these tests silently: if git is missing, fail fast.
"""

import shutil

import pytest


@pytest.fixture(scope="session", autouse=True)
def _require_git_available() -> None:
    git = shutil.which("git")
    assert git is not None, (
        "git is required for ignore-engine tests; please install git"
    )
