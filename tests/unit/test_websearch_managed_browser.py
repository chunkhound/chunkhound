"""Managed browser contracts with real configuration and teardown logic."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import zendriver
import zendriver.core.browser as browser_mod
from zendriver.core.util import get_registered_instances

from chunkhound.utils import websearch_core

_CHROME = sys.executable  # Portable executable; browser startup is injected.


class FakeCdpConnection:
    """Stand-in for the CDP WebSocket Connection (the browser's real I/O boundary)."""

    def __init__(self) -> None:
        self.closed = False
        self.sent: list[Any] = []
        self.aclose_calls = 0
        self.aclose_error: RuntimeError | None = None

    async def send(self, message: Any) -> None:
        self.sent.append(message)

    async def aclose(self) -> None:
        self.aclose_calls += 1
        self.closed = True
        if self.aclose_error is not None:
            raise self.aclose_error


def _stub_resolve_chrome_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make _managed_browser find a portable executable."""
    monkeypatch.setattr(
        websearch_core, "_resolve_chrome_path", lambda _callback=None: _CHROME
    )


def _stub_launch(
    monkeypatch: pytest.MonkeyPatch,
    connection: FakeCdpConnection | None = None,
    *,
    start_error: Exception | None = None,
) -> None:
    """Bypass Chrome I/O while retaining real configuration and stop logic."""

    async def fake_start(self: Any) -> Any:
        if start_error is not None:
            raise start_error
        if connection is not None:
            self.connection = connection
        return self

    monkeypatch.setattr(browser_mod.Browser, "start", fake_start)
    _stub_resolve_chrome_path(monkeypatch)


def _attach_fake_process(browser: zendriver.Browser) -> subprocess.Popen[bytes]:
    """Attach a long-lived fake child process to a browser instance."""
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    browser._process = process
    browser._process_pid = process.pid
    return process


def _make_browser_with_profile() -> zendriver.Browser:
    """Create a Browser with a real temporary profile directory."""
    browser = websearch_core._new_chrome(_CHROME)
    Path(browser.config.user_data_dir).mkdir(parents=True, exist_ok=True)
    return browser


@pytest.fixture
def fast_start_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the start timeout so tests do not wait 20 s."""
    monkeypatch.setattr(websearch_core, "_BROWSER_START_TIMEOUT_S", 0.1)


@pytest.mark.asyncio
async def test_managed_browser_warns_and_yields_none_when_launch_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_launch(monkeypatch, start_error=RuntimeError("launch failed"))
    warnings: list[str] = []

    async with websearch_core._managed_browser(warnings.append) as browser:
        assert browser is None

    assert len(warnings) == 1
    assert "Falling back to urllib" in warnings[0]
    assert "launch failed" in warnings[0]


@pytest.mark.asyncio
async def test_managed_browser_yields_real_browser_with_required_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import zendriver

    connection = FakeCdpConnection()
    _stub_launch(monkeypatch, connection)

    async with websearch_core._managed_browser() as browser:
        assert isinstance(browser, zendriver.Browser)
        assert browser.config.headless is True
        assert browser.config.browser_executable_path == _CHROME
        # Config merges zendriver defaults; assert our flags survive the merge.
        for flag in ("--headless=new", "--disable-dev-shm-usage", "--disable-gpu"):
            assert flag in browser.config.browser_args
        get_registered_instances().add(browser)

    # Stop attempted and completed: CDP close sent, connection closed.
    assert len(connection.sent) == 1
    assert connection.aclose_calls == 1
    assert connection.closed is True
    assert browser not in get_registered_instances()


@pytest.mark.asyncio
async def test_managed_browser_stop_failure_does_not_mask_body_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import zendriver

    connection = FakeCdpConnection()
    # Wedged stop: CDP close send succeeds, closing the connection fails hard.
    connection.aclose_error = RuntimeError("stop wedged")
    _stub_launch(monkeypatch, connection)

    # stop() raising inside finally must not replace the body's exception.
    with pytest.raises(ValueError, match="body failed"):
        async with websearch_core._managed_browser() as browser:
            assert isinstance(browser, zendriver.Browser)
            raise ValueError("body failed")

    # Stop was attempted and the wedged stop observed.
    assert len(connection.sent) == 1
    assert connection.aclose_calls >= 1
    assert connection.closed is True


@pytest.mark.asyncio
async def test_failed_start_reaps_owned_process_and_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed Browser.start() leaves no running child or temp profile."""
    owned_processes: list[subprocess.Popen[bytes]] = []
    profiles: list[Path] = []

    async def fake_start(self: Any) -> Any:
        profiles.append(Path(self.config.user_data_dir))
        owned_processes.append(_attach_fake_process(self))
        raise RuntimeError("failed to connect")

    monkeypatch.setattr(browser_mod.Browser, "start", fake_start)
    _stub_resolve_chrome_path(monkeypatch)
    warnings: list[str] = []

    async with websearch_core._managed_browser(warnings.append) as browser:
        assert browser is None

    assert len(warnings) == 1
    assert "Falling back to urllib" in warnings[0]
    assert len(owned_processes) == 1
    assert owned_processes[0].poll() is not None, "Owned process was not reaped"
    assert not profiles[0].exists(), "Owned profile was not removed"


@pytest.mark.asyncio
async def test_cancelled_start_reaps_owned_child_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation during start kills only the owned browser child."""
    owned_processes: list[subprocess.Popen[bytes]] = []
    owned_browsers: list[zendriver.Browser] = []

    async def fake_start(self: Any) -> Any:
        owned_processes.append(_attach_fake_process(self))
        owned_browsers.append(self)
        get_registered_instances().add(self)
        await asyncio.Event().wait()

    monkeypatch.setattr(browser_mod.Browser, "start", fake_start)
    _stub_resolve_chrome_path(monkeypatch)

    unrelated = _make_browser_with_profile()
    unrelated_process = _attach_fake_process(unrelated)
    get_registered_instances().add(unrelated)

    async def run_managed() -> None:
        async with websearch_core._managed_browser() as browser:
            assert browser is not None  # pragma: no cover

    try:
        task = asyncio.create_task(run_managed())
        await asyncio.sleep(0.05)  # let the task reach fake_start
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert owned_processes[0].poll() is not None
        assert owned_browsers[0] not in get_registered_instances()
        assert not Path(owned_browsers[0].config.user_data_dir).exists()
        assert unrelated_process.poll() is None
        assert Path(unrelated.config.user_data_dir).exists()
        assert unrelated in get_registered_instances()
    finally:
        unrelated_process.kill()
        unrelated_process.wait(timeout=5)
        get_registered_instances().discard(unrelated)
        shutil.rmtree(unrelated.config.user_data_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_start_timeout_raises_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    fast_start_timeout: None,
) -> None:
    """A start that exceeds the timeout is killed and reaped."""
    owned_processes: list[subprocess.Popen[bytes]] = []

    async def fake_start(self: Any) -> Any:
        owned_processes.append(_attach_fake_process(self))
        await asyncio.sleep(1.0)
        return self

    monkeypatch.setattr(browser_mod.Browser, "start", fake_start)
    _stub_resolve_chrome_path(monkeypatch)
    warnings: list[str] = []

    async with websearch_core._managed_browser(warnings.append) as browser:
        assert browser is None

    assert len(warnings) == 1
    assert "Falling back to urllib" in warnings[0]
    assert "exceeded" in warnings[0]
    assert "0.1s" in warnings[0]
    assert "Verify Google Chrome" in warnings[0]
    assert len(owned_processes) == 1
    assert owned_processes[0].poll() is not None, "Owned process was not reaped"


@pytest.mark.asyncio
async def test_custom_profile_preserved_on_failed_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A user-supplied profile directory is not removed on launch failure."""
    custom_profile = tmp_path / "user-profile"
    custom_profile.mkdir(parents=True, exist_ok=True)
    (custom_profile / "marker").write_text("keep", encoding="utf-8")
    default_profiles: list[Path] = []

    async def fake_start(self: Any) -> Any:
        # _new_chrome already created a default profile; switching to a custom
        # one makes it unreachable, so the test must remove it.
        default_profiles.append(Path(self.config.user_data_dir))
        self.config.user_data_dir = str(custom_profile)
        raise RuntimeError("failed to connect")

    monkeypatch.setattr(browser_mod.Browser, "start", fake_start)
    _stub_resolve_chrome_path(monkeypatch)

    try:
        async with websearch_core._managed_browser() as browser:
            assert browser is None

        assert (custom_profile / "marker").read_text() == "keep"
    finally:
        for profile in default_profiles:
            shutil.rmtree(profile, ignore_errors=True)


@pytest.mark.asyncio
async def test_stop_failure_reaps_owned_process_and_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed stop must not leave the child, profile, or registration behind."""

    class FailingClose(FakeCdpConnection):
        async def aclose(self) -> None:
            self.aclose_calls += 1
            self.closed = True
            raise RuntimeError("aclose failed")

    connection = FailingClose()
    owned_processes: list[subprocess.Popen[bytes]] = []

    async def fake_start(self: Any) -> Any:
        owned_processes.append(_attach_fake_process(self))
        get_registered_instances().add(self)
        self.connection = connection
        return self

    monkeypatch.setattr(browser_mod.Browser, "start", fake_start)
    _stub_resolve_chrome_path(monkeypatch)

    async with websearch_core._managed_browser() as browser:
        assert browser is not None
        profile = Path(browser.config.user_data_dir)

    assert owned_processes[0].poll() is not None
    assert not profile.exists()
    assert browser not in get_registered_instances()


@pytest.mark.asyncio
async def test_repeated_cancellation_finishes_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second cancel must not interrupt the owned-browser teardown."""

    class BlockingClose(FakeCdpConnection):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def aclose(self) -> None:
            self.aclose_calls += 1
            self.entered.set()
            await self.release.wait()
            self.closed = True

    connection = BlockingClose()
    owned_processes: list[subprocess.Popen[bytes]] = []
    owned_browsers: list[zendriver.Browser] = []

    async def fake_start(self: Any) -> Any:
        owned_processes.append(_attach_fake_process(self))
        owned_browsers.append(self)
        get_registered_instances().add(self)
        self.connection = connection
        await asyncio.Event().wait()

    monkeypatch.setattr(browser_mod.Browser, "start", fake_start)
    _stub_resolve_chrome_path(monkeypatch)

    async def run_managed() -> None:
        async with websearch_core._managed_browser() as browser:
            assert browser is not None  # pragma: no cover

    task = asyncio.create_task(run_managed())
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.wait_for(connection.entered.wait(), timeout=5)
    task.cancel()
    await asyncio.sleep(0)
    connection.release.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)

    assert owned_processes[0].poll() is not None
    assert owned_browsers[0] not in get_registered_instances()
    assert not Path(owned_browsers[0].config.user_data_dir).exists()
    assert connection.closed is True


@pytest.mark.asyncio
async def test_start_cleanup_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CDP close that never returns cannot block the urllib fallback."""

    class HangingClose(FakeCdpConnection):
        async def aclose(self) -> None:
            self.aclose_calls += 1
            await asyncio.Event().wait()

    connection = HangingClose()
    owned_processes: list[subprocess.Popen[bytes]] = []

    async def fake_start(self: Any) -> Any:
        owned_processes.append(_attach_fake_process(self))
        self.connection = connection
        raise RuntimeError("failed to connect")

    monkeypatch.setattr(browser_mod.Browser, "start", fake_start)
    _stub_resolve_chrome_path(monkeypatch)
    monkeypatch.setattr(websearch_core, "_BROWSER_CLEANUP_CLOSE_TIMEOUT_S", 0.1)
    warnings: list[str] = []

    started = time.monotonic()
    async with websearch_core._managed_browser(warnings.append) as browser:
        assert browser is None
    assert time.monotonic() - started < 2.0

    assert len(warnings) == 1
    assert "failed to connect" in warnings[0]
    assert owned_processes[0].poll() is not None


@pytest.mark.asyncio
async def test_reap_process_clears_state_for_exited_child() -> None:
    """An already-exited child must not keep its pipes or process handle."""
    browser = _make_browser_with_profile()
    process = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    process.wait()
    browser._process = process
    browser._process_pid = process.pid

    websearch_core._reap_chrome_process(browser)

    assert browser._process is None
    assert browser._process_pid is None
    assert process.stdin is not None and process.stdin.closed
    assert process.stdout is not None and process.stdout.closed
    assert process.stderr is not None and process.stderr.closed
    websearch_core._cleanup_browser_sync(browser)
