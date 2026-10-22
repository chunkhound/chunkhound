"""Shared fixtures for remote-config tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def _isolate(hermetic_environment: Path) -> Path:
    """Persistence targets use the shared sandbox HOME, not the project cwd."""
    return hermetic_environment
