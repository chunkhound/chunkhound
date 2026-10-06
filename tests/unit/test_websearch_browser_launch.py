"""Contract tests for the hardened zendriver launch path.

zendriver's default connect budget (~2.5s) flakes on cold CI browser starts.
``_launch_chrome`` must use a wider budget and retry once so production and
the integration test do not drop to urllib on a cold first launch.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import zendriver

from chunkhound.utils import websearch_core


@pytest.fixture(autouse=True)
def _fast_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep retry tests instant and out of zendriver's global CDP patches."""
    monkeypatch.setattr(websearch_core, "_BROWSER_LAUNCH_BACKOFF_S", 0.0)
    monkeypatch.setattr(websearch_core, "_install_late_completion_guard", lambda: None)


@pytest.mark.asyncio
async def test_launch_chrome_retries_after_transient_connect_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    sentinel = SimpleNamespace(name="browser")

    async def fake_start(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("Failed to connect to browser")
        return sentinel

    monkeypatch.setattr(zendriver, "start", fake_start)

    browser = await websearch_core._launch_chrome("/usr/bin/google-chrome")

    assert browser is sentinel
    assert calls == 2


@pytest.mark.asyncio
async def test_launch_chrome_uses_budget_above_zendriver_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_start(**kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(name="browser")

    monkeypatch.setattr(zendriver, "start", fake_start)

    await websearch_core._launch_chrome("/usr/bin/google-chrome")

    # zendriver 0.15.3 (pinned in pyproject.toml) defaults:
    # browser_connection_timeout=0.25, browser_connection_max_tries=10.
    # The cold-start budget must exceed them so a slow CI Chrome does not
    # trip the generic "Failed to connect" and force the urllib fallback.
    assert captured["browser_executable_path"] == "/usr/bin/google-chrome"
    assert captured["browser_connection_timeout"] > 0.25
    assert captured["browser_connection_max_tries"] > 10


@pytest.mark.asyncio
async def test_launch_chrome_raises_after_all_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def fake_start(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(zendriver, "start", fake_start)

    with pytest.raises(RuntimeError, match="boom"):
        await websearch_core._launch_chrome("/usr/bin/google-chrome")

    assert calls == websearch_core._BROWSER_LAUNCH_ATTEMPTS


@pytest.mark.asyncio
async def test_managed_browser_warns_and_yields_none_when_launch_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_start(**kwargs: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(zendriver, "start", fake_start)
    monkeypatch.setattr(
        websearch_core, "_resolve_chrome_path", lambda cb=None: "/usr/bin/google-chrome"
    )
    warnings: list[str] = []

    async with websearch_core._managed_browser(warnings.append) as browser:
        assert browser is None

    assert warnings
    assert "Falling back to urllib" in warnings[0]
