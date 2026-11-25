"""Ignore-engine test prerequisites and the shared ``backend`` fixture.

This conftest is scoped to the tests/ignore/ package to enforce that
the development environment has a working Git executable available.
We do NOT skip these tests silently: if git is missing, fail fast.

``backend`` parametrizes every parity case over the gitignore backends.
Phase 2 adds "rust" to ``IGNORE_BACKENDS``; nothing else should change.
"""

import shutil

import pytest

# Phase 2: append "rust" here once the Rust matcher exists.
IGNORE_BACKENDS = ("python", "libgit2")


@pytest.fixture(scope="session", autouse=True)
def _require_git_available() -> None:
    git = shutil.which("git")
    assert git is not None, (
        "git is required for ignore-engine tests; please install git"
    )


@pytest.fixture(params=IGNORE_BACKENDS)
def backend(request: pytest.FixtureRequest) -> str:
    name: str = request.param
    if name == "libgit2":
        try:
            import pygit2  # noqa: F401
        except Exception:
            pytest.skip("pygit2 not available (libgit2 backend)")
    return name
