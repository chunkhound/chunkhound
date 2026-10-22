"""Bounded, failure-only test diagnostics; never consume protocol stdout."""

import asyncio
import sys
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from tests.utils import SubprocessJsonRpcClient, create_subprocess_exec_safe

TAIL_BYTES = 16 * 1024


class StartupDiagnostics:
    """Keep bounded evidence in memory so temporary-file teardown cannot erase it."""

    def __init__(
        self,
        label: str,
        debug_file: Path | None = None,
        *,
        stderr_label: str = "stderr tail",
    ):
        self.label = label
        self.stderr_label = stderr_label
        self.debug_file = debug_file
        self.started = time.monotonic()
        self.events = bytearray()
        self.stderr = bytearray()
        self.process: Any = None
        self.reader: asyncio.Task[None] | None = None
        self.output_remaining = 56 * 1024

    def mark(self, event: str) -> None:
        self.events.extend(
            f"+{time.monotonic() - self.started:.3f}s {event}\n".encode(
                "utf-8", errors="replace"
            )[-TAIL_BYTES:]
        )
        del self.events[:-TAIL_BYTES]

    def capture_stderr(self, data: bytes) -> None:
        self.stderr.extend(data[-TAIL_BYTES:])
        del self.stderr[:-TAIL_BYTES]

    def attach(self, process: asyncio.subprocess.Process) -> None:
        self.process = process
        self.mark("launch.end")
        self.reader = asyncio.create_task(self._drain_stderr(process))

    async def _drain_stderr(self, process: asyncio.subprocess.Process) -> None:
        assert process.stderr is not None
        try:
            while data := await process.stderr.read(4096):
                self.capture_stderr(data)
            self.mark("stderr.eof")
        except Exception as exc:
            self.mark(f"stderr.error {type(exc).__name__}: {exc}")

    async def close(self) -> None:
        if self.reader is not None:
            self.reader.cancel()
            try:
                await self.reader
            except asyncio.CancelledError:
                pass

    def report(self, reason: str) -> None:
        """Snapshot synchronously before cleanup; reporting must not replace errors."""
        try:
            debug = b""
            if self.debug_file is not None:
                try:
                    with self.debug_file.open("rb") as stream:
                        stream.seek(0, 2)
                        stream.seek(max(0, stream.tell() - TAIL_BYTES))
                        debug = stream.read(TAIL_BYTES)
                except OSError as exc:
                    debug = f"debug-file unavailable: {exc}".encode()[-TAIL_BYTES:]
            process = self.process
            returncode = getattr(process, "returncode", None)
            if process is not None and hasattr(process, "poll"):
                returncode = process.poll()
            self.mark(reason)
            header = (
                f"[{self.label} diagnostics] "
                f"elapsed={time.monotonic() - self.started:.3f}s "
                f"pid={getattr(process, 'pid', None)} returncode={returncode}\n"
            )

            # Replacement decoding can expand invalid bytes; cap the final UTF-8
            # representation too, keeping each section <=16 KiB on CI.
            def text(data: bytes | bytearray) -> str:
                return (
                    bytes(data)
                    .decode("utf-8", errors="replace")
                    .encode()[-TAIL_BYTES:]
                    .decode("utf-8", errors="ignore")
                )

            output = (
                header
                + text(self.events)
                + f"{self.stderr_label}:\n"
                + text(self.stderr)
                + "\ndebug-file tail:\n"
                + text(debug)
                + "\n"
            ).encode("utf-8")
            output = output[: self.output_remaining]
            self.output_remaining -= len(output)
            sys.stderr.write(output.decode("utf-8", errors="ignore"))
        except Exception:
            # Capturing evidence is best effort, even when the output sink fails.
            pass


class _DiagnosedClient(SubprocessJsonRpcClient):
    def __init__(
        self, process: asyncio.subprocess.Process, evidence: StartupDiagnostics
    ):
        super().__init__(process)
        self.evidence = evidence

    async def send_request(
        self, method: str, params: dict[str, Any] | None = None, timeout: float = 5.0
    ) -> dict[str, Any]:
        self.evidence.mark(f"{method}.begin timeout={timeout}")
        result = await super().send_request(method, params, timeout)
        self.evidence.mark(f"{method}.end")
        return result

    async def send_notification(
        self, method: str, params: dict[str, Any] | None = None
    ) -> None:
        self.evidence.mark(f"{method}.begin")
        await super().send_notification(method, params)
        self.evidence.mark(f"{method}.end")


@asynccontextmanager
async def diagnosed_mcp_client(
    command: Sequence[str], *, debug_file: Path | None = None, **kwargs: Any
) -> AsyncIterator[SubprocessJsonRpcClient]:
    evidence = StartupDiagnostics("MCP", debug_file)
    client = None
    failed = False
    try:
        evidence.mark(f"launch.begin executable={command[0]}")
        process = await create_subprocess_exec_safe(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **kwargs,
        )
        evidence.attach(process)
        client = _DiagnosedClient(process, evidence)
        await client.start()
        evidence.mark("client.started")
        yield client
    except BaseException as exc:
        failed = True
        evidence.report(f"failure {type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            if client is not None:
                evidence.mark("client.close.begin")
                await client.close()
                evidence.mark("client.close.end")
        except BaseException as exc:
            evidence.report(f"cleanup.error {type(exc).__name__}: {exc}")
            if not failed:
                raise
        finally:
            await evidence.close()
