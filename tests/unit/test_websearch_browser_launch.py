"""Launch contracts at the external browser-process boundary."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import zendriver
from loguru import logger
from zendriver.core.util import get_registered_instances

from chunkhound.utils import websearch_core


@dataclass
class Attempt:
    browser: zendriver.Browser
    process: subprocess.Popen[bytes]
    profile: Path

    def assert_removed(self) -> None:
        assert self.process.poll() is not None, "Launch left its child running"
        assert not self.profile.exists(), "Launch left its temporary profile"
        assert self.browser not in get_registered_instances()

    def dispose(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=5)
        shutil.rmtree(self.profile, ignore_errors=True)
        get_registered_instances().discard(self.browser)


@dataclass
class StartBoundary:
    outcomes: list[str] = field(default_factory=lambda: ["fail"])
    attempts: list[Attempt] = field(default_factory=list)
    started: asyncio.Event = field(default_factory=asyncio.Event)
    custom_profile: Path | None = None
    delay: float = 0.0

    def spawn(self, browser: zendriver.Browser) -> Attempt:
        if self.custom_profile is not None:
            browser.config.user_data_dir = self.custom_profile
        profile = Path(browser.config.user_data_dir)
        profile.mkdir(parents=True, exist_ok=True)
        (profile / "profile-data").write_text("owned profile", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"]
        )
        browser._process, browser._process_pid = process, process.pid
        get_registered_instances().add(browser)
        attempt = Attempt(browser, process, profile)
        self.attempts.append(attempt)
        self.started.set()
        return attempt

    async def start(self, browser: zendriver.Browser) -> zendriver.Browser:
        self.spawn(browser)
        await asyncio.sleep(self.delay)
        outcome = self.outcomes[min(len(self.attempts) - 1, len(self.outcomes) - 1)]
        if outcome == "fail":
            raise RuntimeError("Failed to connect to browser")
        if outcome == "stall":
            await asyncio.Event().wait()
        return browser


@pytest.fixture
def boundary(monkeypatch: pytest.MonkeyPatch) -> Iterator[StartBoundary]:
    boundary = StartBoundary()

    async def start(browser: zendriver.Browser) -> zendriver.Browser:
        return await boundary.start(browser)

    monkeypatch.setattr(zendriver.Browser, "start", start)
    monkeypatch.setattr(websearch_core, "_BROWSER_LAUNCH_BACKOFF_S", 0.0)
    monkeypatch.setattr(websearch_core, "_install_late_completion_guard", lambda: None)
    try:
        yield boundary
    finally:
        for attempt in boundary.attempts:
            attempt.dispose()


@pytest.mark.asyncio
async def test_launch_chrome_retries_after_transient_connect_failure(
    boundary: StartBoundary,
) -> None:
    boundary.outcomes = ["fail", "ok"]

    browser = await websearch_core._launch_chrome(sys.executable)

    assert len(boundary.attempts) == 2
    boundary.attempts[0].assert_removed()
    assert browser is boundary.attempts[1].browser
    assert boundary.attempts[1].process.poll() is None
    assert boundary.attempts[1].profile.exists()


@pytest.fixture
def cleanup_errors() -> Iterator[list[str]]:
    errors: list[str] = []
    handler = logger.add(errors.append, level="ERROR")
    try:
        yield errors
    finally:
        logger.remove(handler)


def test_asyncio_run_stopped_browser_exits_without_cleanup_errors(
    boundary: StartBoundary,
    cleanup_errors: list[str],
) -> None:
    boundary.outcomes = ["ok"]

    async def run() -> None:
        browser = await websearch_core._launch_chrome(sys.executable)
        await browser.stop()

    asyncio.run(run())

    attempt = boundary.attempts[0]
    assert attempt.process.poll() is not None
    assert not attempt.profile.exists()
    assert not cleanup_errors


@pytest.mark.parametrize("custom_profile", [False, True])
def test_asyncio_run_reaps_owned_browser_and_preserves_unrelated_profiles(
    boundary: StartBoundary,
    cleanup_errors: list[str],
    tmp_path: Path,
    custom_profile: bool,
) -> None:
    profile = tmp_path / "user-profile" if custom_profile else None
    unrelated = asyncio.run(launch_beside_unrelated(boundary, profile))

    assert_exited_browser(boundary.attempts[1], custom_profile)
    assert unrelated.process.poll() is None
    assert unrelated.profile.exists()
    assert unrelated.browser in get_registered_instances()
    assert not cleanup_errors


async def launch_beside_unrelated(
    boundary: StartBoundary, custom_profile: Path | None
) -> Attempt:
    unrelated = boundary.spawn(
        zendriver.Browser(zendriver.Config(browser_executable_path=sys.executable))
    )
    boundary.outcomes = ["ok"]
    boundary.custom_profile = custom_profile
    await websearch_core._launch_chrome(sys.executable)
    assert boundary.attempts[1].process.poll() is None
    return unrelated


def assert_exited_browser(attempt: Attempt, custom_profile: bool) -> None:
    if custom_profile:
        assert attempt.process.poll() is not None
        assert attempt.browser not in get_registered_instances()
        assert (attempt.profile / "profile-data").read_text() == "owned profile"
    else:
        attempt.assert_removed()


@pytest.mark.asyncio
async def test_launch_chrome_allows_at_least_15_seconds_for_cold_connect(
    boundary: StartBoundary,
) -> None:
    boundary.outcomes = ["ok"]

    browser = await websearch_core._launch_chrome(sys.executable)

    config = browser.config
    assert config.browser_executable_path == sys.executable
    assert config.browser_connection_timeout * config.browser_connection_max_tries >= 15
    assert config.headless
    assert "--headless=new" in config.browser_args


@pytest.mark.asyncio
async def test_launch_chrome_raises_after_all_attempts(boundary: StartBoundary) -> None:
    with pytest.raises(RuntimeError, match="Failed to connect"):
        await websearch_core._launch_chrome(sys.executable)

    assert len(boundary.attempts) == websearch_core._BROWSER_LAUNCH_ATTEMPTS
    for attempt in boundary.attempts:
        attempt.assert_removed()


@pytest.mark.asyncio
async def test_managed_browser_warns_and_yields_none_when_launch_fails(
    boundary: StartBoundary,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        websearch_core, "_resolve_chrome_path", lambda cb=None: sys.executable
    )
    warnings: list[str] = []

    async with websearch_core._managed_browser(warnings.append) as browser:
        assert browser is None

    assert len(warnings) == 1
    assert "Falling back to urllib" in warnings[0]
    assert "Verify Google Chrome" in warnings[0]
    assert ">=124" in warnings[0]
    for attempt in boundary.attempts:
        attempt.assert_removed()


def set_budget(
    monkeypatch: pytest.MonkeyPatch,
    backoff: float,
    total: float = 1.0,
) -> None:
    monkeypatch.setattr(websearch_core, "_BROWSER_LAUNCH_TOTAL_BUDGET_S", total)
    monkeypatch.setattr(websearch_core, "_BROWSER_LAUNCH_BACKOFF_S", backoff)


@contextmanager
def launch_budget_bound() -> Iterator[None]:
    started = time.monotonic()
    try:
        yield
    finally:
        # Allow 150 ms for CI scheduling, not extra launch/cleanup work.
        assert (
            time.monotonic() - started
            < websearch_core._BROWSER_LAUNCH_TOTAL_BUDGET_S + 0.15
        )


async def launch_with_watchdog() -> zendriver.Browser:
    # The watchdog detects hangs; launch_budget_bound enforces the contract.
    return await asyncio.wait_for(
        websearch_core._launch_chrome(sys.executable), timeout=5
    )


@pytest.mark.asyncio
async def test_delayed_retry_succeeds_before_shared_deadline(
    boundary: StartBoundary,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary.outcomes = ["fail", "ok"]
    boundary.delay = 0.06
    set_budget(monkeypatch, backoff=0.05, total=0.5)

    with launch_budget_bound():
        browser = await launch_with_watchdog()

    assert len(boundary.attempts) == 2
    boundary.attempts[0].assert_removed()
    assert browser is boundary.attempts[1].browser
    assert boundary.attempts[1].process.poll() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("outcomes", [["stall"], ["fail"]])
async def test_launch_deadline_includes_backoff_and_cleanup(
    boundary: StartBoundary,
    monkeypatch: pytest.MonkeyPatch,
    outcomes: list[str],
) -> None:
    boundary.outcomes = outcomes
    set_budget(monkeypatch, backoff=0.75)

    with launch_budget_bound(), pytest.raises(TimeoutError, match="budget"):
        await launch_with_watchdog()

    assert len(boundary.attempts) == 1
    boundary.attempts[0].assert_removed()


def test_launch_budget_defaults_cap_total_time() -> None:
    # Pin production budget so silent drift cannot reintroduce unbounded hangs.
    assert websearch_core._BROWSER_LAUNCH_TOTAL_BUDGET_S == 20.0
    assert websearch_core._BROWSER_LAUNCH_CLEANUP_RESERVE_S == 2.0
    assert (
        websearch_core._BROWSER_CONNECT_TIMEOUT_S
        * websearch_core._BROWSER_CONNECT_MAX_TRIES
        >= 15
    )


@pytest.mark.asyncio
async def test_retry_shares_remaining_budget_with_nonzero_backoff(
    boundary: StartBoundary,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary.outcomes = ["fail", "stall"]
    set_budget(monkeypatch, backoff=0.3, total=1.0)

    with launch_budget_bound(), pytest.raises(TimeoutError, match="budget"):
        await launch_with_watchdog()

    assert len(boundary.attempts) == 2
    for attempt in boundary.attempts:
        attempt.assert_removed()


@pytest.mark.asyncio
async def test_cancelled_launch_reaps_only_its_owned_child(
    boundary: StartBoundary,
) -> None:
    unrelated = boundary.spawn(
        zendriver.Browser(
            zendriver.Config(
                browser_executable_path=sys.executable,
            )
        )
    )
    boundary.started.clear()
    boundary.outcomes = ["stall"]
    task = asyncio.create_task(websearch_core._launch_chrome(sys.executable))
    await asyncio.wait_for(boundary.started.wait(), timeout=5)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)

    boundary.attempts[1].assert_removed()
    assert unrelated.process.poll() is None
    assert unrelated.profile.exists()
    assert unrelated.browser in get_registered_instances()


@dataclass
class ConnectionClose:
    """Hold the external connection close so cancellation can overlap cleanup."""

    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    closed: bool = False

    async def aclose(self) -> None:
        self.entered.set()
        await self.release.wait()
        self.closed = True


@pytest.mark.asyncio
async def test_repeated_cancellation_does_not_interrupt_failed_start_cleanup(
    boundary: StartBoundary,
) -> None:
    boundary.outcomes = ["stall"]
    connection = ConnectionClose()
    task = asyncio.create_task(websearch_core._launch_chrome(sys.executable))
    await asyncio.wait_for(boundary.started.wait(), timeout=5)
    boundary.attempts[0].browser.connection = connection
    task.cancel()
    await asyncio.wait_for(connection.entered.wait(), timeout=5)
    task.cancel()
    await asyncio.sleep(0)
    connection.release.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)

    boundary.attempts[0].assert_removed()
    assert connection.closed


@pytest.mark.asyncio
async def test_unresponsive_connection_close_cannot_extend_launch_budget(
    boundary: StartBoundary,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary.outcomes = ["stall"]
    set_budget(monkeypatch, backoff=0.7)
    with launch_budget_bound():
        task = asyncio.create_task(launch_with_watchdog())
        await asyncio.wait_for(boundary.started.wait(), timeout=5)
        boundary.attempts[0].browser.connection = ConnectionClose()

        with pytest.raises(TimeoutError, match="budget"):
            await task

    boundary.attempts[0].assert_removed()


@pytest.mark.asyncio
async def test_reap_at_exhausted_deadline_reaps_killed_child(
    boundary: StartBoundary,
) -> None:
    attempt = boundary.spawn(
        zendriver.Browser(zendriver.Config(browser_executable_path=sys.executable))
    )
    attempt.process.kill()

    websearch_core._reap_chrome(attempt.browser, 0.0)

    assert attempt.process.poll() is not None
    assert not attempt.profile.exists()


@pytest.mark.asyncio
async def test_abort_chrome_reaps_when_deadline_passed(
    boundary: StartBoundary,
) -> None:
    attempt = boundary.spawn(
        zendriver.Browser(zendriver.Config(browser_executable_path=sys.executable))
    )

    await websearch_core._abort_chrome(
        attempt.browser, asyncio.get_running_loop().time() - 1
    )

    attempt.assert_removed()


@pytest.mark.asyncio
async def test_failed_launch_preserves_custom_profile(
    boundary: StartBoundary,
    tmp_path: Path,
) -> None:
    boundary.custom_profile = tmp_path / "user-profile"

    with pytest.raises(RuntimeError, match="Failed to connect"):
        await websearch_core._launch_chrome(sys.executable)

    assert (boundary.custom_profile / "profile-data").read_text() == "owned profile"
    assert all(attempt.process.poll() is not None for attempt in boundary.attempts)
