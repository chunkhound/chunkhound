"""Shared fixtures for remote-config tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect HOME so `_get_global_config_candidates` targets tmp_path
    and strip CHUNKHOUND_* env leakage between tests.

    ``tmp_path`` is pre-resolved so comparisons against the pipeline's
    resolved target paths hold on platforms where the temp dir is behind
    a symlink (macOS ``/tmp`` → ``/private/tmp``).
    """
    for key in list(os.environ):
        if key.startswith("CHUNKHOUND_") or key in {
            "VOYAGE_API_KEY",
            "OPENAI_API_KEY",
        }:
            monkeypatch.delenv(key, raising=False)
    resolved = tmp_path.resolve()
    monkeypatch.setenv("HOME", str(resolved))
    monkeypatch.setenv("USERPROFILE", str(resolved))
    return resolved
