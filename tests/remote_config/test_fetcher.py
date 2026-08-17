"""Unit tests for the remote-config HTTP fetcher.

Covers the private `_interpolate_env` helper directly. The pipeline e2e
tests replace `fetcher.fetch` with a fake, so this module is the sole
guardian of `_interpolate_env`'s two branches: unset-var → drop header,
set-var → substitute.
"""

from __future__ import annotations

import pytest

from chunkhound.core.config.remote.fetcher import _interpolate_env


def test_var_interpolation_dropped_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REMOTE_TOKEN", raising=False)
    assert _interpolate_env("Bearer ${REMOTE_TOKEN}") is None


def test_var_interpolation_substituted_when_env_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REMOTE_TOKEN", "sekret")
    assert _interpolate_env("Bearer ${REMOTE_TOKEN}") == "Bearer sekret"


def test_var_interpolation_dropped_when_env_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `export REMOTE_TOKEN=` must drop the header just like an unset var —
    # otherwise the wire sees `Authorization: Bearer ` (empty credential).
    monkeypatch.setenv("REMOTE_TOKEN", "")
    assert _interpolate_env("Bearer ${REMOTE_TOKEN}") is None
