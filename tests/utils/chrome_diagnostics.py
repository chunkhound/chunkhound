"""Observe zendriver's existing launch/probes without adding readiness requests.

The namespace hooks are scoped to startup, not global subprocess/asyncio hooks.
zendriver owns Chrome stderr: retain the tail of its existing failure read rather
than race that reader or change its 0.25s deadline. A failed/timed-out library read
is explicitly reported as unavailable, not mistaken for empty Chrome stderr.

Fixture-specific and coupled to zendriver 0.15.3 private Browser/HTTPApi/util
entry points. Overlapping diagnosed startups in the same event loop are rejected
without waiting. Unrelated concurrent zendriver activity (including plain starts)
and use from multiple threads are unsupported while these module hooks are active.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from tests.utils.startup_diagnostics import TAIL_BYTES, StartupDiagnostics

if TYPE_CHECKING:
    import zendriver as zd

_startup_active = False


def resolve_chrome_for_test() -> tuple[str | None, str]:
    from chunkhound.utils import websearch_core

    original_subprocess = websearch_core.subprocess
    version = "unavailable"

    class ObservedSubprocess:
        def __getattr__(self, name: str) -> Any:
            return getattr(original_subprocess, name)

        def check_output(self, *args: Any, **kwargs: Any) -> str:
            nonlocal version
            output = original_subprocess.check_output(*args, **kwargs)
            version = output[-TAIL_BYTES:].strip()
            return output

    # Capture the already-issued --version command, retaining its timeout and
    # exception behavior; never issue an extra version subprocess.
    with patch.object(websearch_core, "subprocess", ObservedSubprocess()):
        path = websearch_core._resolve_chrome_path()
    return path, version


@asynccontextmanager
async def diagnosed_chrome(
    chrome_path: str, *, version: str = "unavailable"
) -> AsyncIterator[zd.Browser]:
    import zendriver as zd
    from zendriver.core import browser as browser_module

    global _startup_active
    if _startup_active:
        raise RuntimeError("Overlapping diagnosed Chrome starts are unsupported")

    evidence = StartupDiagnostics(
        "Chrome", stderr_label="existing library-read tail, potentially incomplete"
    )
    evidence.mark(f"executable={chrome_path}; version={version}")
    original_util = browser_module.util
    instance = None
    failed = False

    class ObservedHTTPApi(browser_module.HTTPApi):
        async def get(self, endpoint: str) -> Any:
            evidence.mark(f"probe.begin {self.api}/json/{endpoint}")
            try:
                result = await super().get(endpoint)
            except BaseException as exc:
                evidence.mark(f"probe.error {type(exc).__name__}: {exc}")
                raise
            evidence.mark(f"probe.end version={result.get('Browser', 'unavailable')}")
            return result

    class ObservedUtil:
        def __getattr__(self, name: str) -> Any:
            return getattr(original_util, name)

        def _start_process(self, *args: Any) -> Any:
            evidence.mark("launch.begin")
            process = original_util._start_process(*args)
            evidence.process = process
            evidence.mark("launch.end")
            return process

        async def _read_process_stderr(self, process: Any, n: int = 2**16) -> str:
            evidence.mark(f"stderr.library-read.begin limit={n}")
            stderr = await original_util._read_process_stderr(process, n)
            evidence.capture_stderr(stderr.encode("utf-8", errors="replace"))
            evidence.mark(
                "stderr.library-read.end"
                if stderr
                else "stderr unavailable: existing library read empty or timed out"
            )
            return stderr

    class ObservedBrowser(browser_module.Browser):
        def __init__(self, *args: Any, **kwargs: Any):
            nonlocal instance
            super().__init__(*args, **kwargs)
            instance = self

        async def stop(self) -> None:
            if not self.info:
                evidence.report(
                    "startup before-stop (before temporary profile cleanup)"
                )
            await super().stop()

    # No await between the overlap check and acquisition: fail fast rather than
    # serializing starts or extending a caller's startup deadline.
    _startup_active = True
    try:
        try:
            # zd.start still constructs the same Config and Browser.create still
            # owns registration and its original launch/probe/stop control flow.
            with (
                patch.object(browser_module, "Browser", ObservedBrowser),
                patch.object(browser_module, "HTTPApi", ObservedHTTPApi),
                patch.object(browser_module, "util", ObservedUtil()),
            ):
                evidence.mark("browser.start.begin")
                browser = await zd.start(
                    headless=True,
                    browser_args=[
                        "--headless=new",
                        "--disable-dev-shm-usage",
                        "--disable-gpu",
                    ],
                    browser_executable_path=chrome_path,
                )
        finally:
            # Restore hooks first; reset on success, failure, or cancellation.
            _startup_active = False
        evidence.mark("browser.start.end")
        yield browser
    except BaseException as exc:
        failed = True
        evidence.report(f"failure {type(exc).__name__}: {exc}")
        raise
    finally:
        if instance is not None:
            try:
                # Keep the fixture's existing shutdown cap. Failed create() may
                # not register atexit, so also clean up its partial instance.
                evidence.mark("browser.stop.begin")
                await asyncio.wait_for(instance.stop(), timeout=10)
                evidence.mark("browser.stop.end")
            except asyncio.TimeoutError:
                pass
            except BaseException as exc:
                evidence.report(f"cleanup.error {type(exc).__name__}: {exc}")
                if not failed:
                    raise
