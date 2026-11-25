"""Shared ``backend`` fixture for ignore-engine tests.

Imported by tests/ignore/conftest.py and by tests/contracts/ so every
suite that parametrizes over gitignore backends uses the same list and the
same libgit2 skip rule.

``backend`` parametrizes over the gitignore backends. Phase 2 adds "rust"
to ``IGNORE_BACKENDS``; nothing else should change.
"""

from __future__ import annotations

import pytest

# Phase 2: append "rust" here once the Rust matcher exists.
IGNORE_BACKENDS = ("python", "libgit2")


def skip_if_backend_unavailable(name: str) -> None:
    if name == "libgit2":
        try:
            import pygit2  # noqa: F401
        except Exception:
            pytest.skip("pygit2 not available (libgit2 backend)")


@pytest.fixture(params=IGNORE_BACKENDS)
def backend(request: pytest.FixtureRequest) -> str:
    name: str = request.param
    skip_if_backend_unavailable(name)
    return name
