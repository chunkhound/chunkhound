"""Integration proof that cancelled Antigravity CLI calls do not leave child orphans."""

import asyncio
import os
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import patch

import psutil
import pytest

from chunkhound.providers.llm.antigravity_cli_provider import AntigravityCLIProvider

pytestmark = pytest.mark.integration


async def _wait_for_pid_file(path: Path, timeout: float = 5.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return int(text)
        await asyncio.sleep(0.05)
    raise AssertionError(f"child pid file was not written: {path}")


async def _wait_until_pid_gone(pid: int, timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not psutil.pid_exists(pid):
            return
        await asyncio.sleep(0.1)
    raise AssertionError(f"child process {pid} survived cancellation cleanup")


@pytest.mark.skipif(sys.platform == "win32", reason="Unix process-group proof")
@pytest.mark.asyncio
async def test_antigravity_orphan_child_terminated_after_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Fake agy spawns a long-running child; cancellation kills the group."""
    child_pid_file = tmp_path / "child.pid"
    fake_agy = tmp_path / "agy"
    fake_agy.write_text(
        textwrap.dedent(
            f"""
            #!{sys.executable}
            import os
            import subprocess
            import sys
            import time

            child = subprocess.Popen([
                sys.executable,
                "-c",
                "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
            ])
            pid_file = "{child_pid_file.as_posix()}"
            with open(pid_file, "w", encoding="utf-8") as handle:
                handle.write(str(child.pid))
                handle.flush()
            time.sleep(60)
            """
        ).lstrip(),
        encoding="utf-8",
    )
    fake_agy.chmod(0o755)

    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")

    provider = AntigravityCLIProvider(
        model="gemini-3.5-flash",
        timeout=30,
    )

    task = asyncio.create_task(
        provider.complete("prompt")
    )
    child_pid = await _wait_for_pid_file(child_pid_file)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    try:
        await _wait_until_pid_gone(child_pid)
    finally:
        if psutil.pid_exists(child_pid):
            try:
                psutil.Process(child_pid).kill()
            except psutil.NoSuchProcess:
                pass


@pytest.mark.skipif(sys.platform == "win32", reason="Unix process-group proof")
@pytest.mark.asyncio
async def test_antigravity_orphan_child_terminated_after_clean_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Fake agy backgrounds a child, writes a valid response, and exits 0.

    Cleanup must still reap the lingering process-group member even though the
    wrapper terminated cleanly (``returncode == 0``) — the scenario the previous
    ``returncode is None`` guard skipped.
    """
    child_pid_file = tmp_path / "child_clean.pid"
    fake_agy = tmp_path / "agy"
    fake_agy.write_text(
        textwrap.dedent(
            f"""
            #!{sys.executable}
            import subprocess
            import sys

            # Detach the child's stdio so it does not hold agy's stdout/stderr
            # pipes open; otherwise communicate() would block for EOF until the
            # child exits (60s). A real daemonized orphan closes its fds too.
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            pid_file = "{child_pid_file.as_posix()}"
            with open(pid_file, "w", encoding="utf-8") as handle:
                handle.write(str(child.pid))
                handle.flush()
            sys.stdout.write("Antigravity OK response")
            sys.stdout.flush()
            sys.exit(0)
            """
        ).lstrip(),
        encoding="utf-8",
    )
    fake_agy.chmod(0o755)

    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")

    provider = AntigravityCLIProvider(
        model="gemini-3.5-flash",
        timeout=30,
    )

    # Clean exit 0: the call succeeds and returns the CLI stdout.
    response = await provider.complete("prompt")
    assert response.content

    child_pid = await _wait_for_pid_file(child_pid_file)
    try:
        await _wait_until_pid_gone(child_pid)
    finally:
        if psutil.pid_exists(child_pid):
            try:
                psutil.Process(child_pid).kill()
            except psutil.NoSuchProcess:
                pass
