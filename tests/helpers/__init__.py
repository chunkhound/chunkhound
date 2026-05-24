"""Shared test helpers for mocking subprocess interactions."""


class DummyPipe:
    """Mock stdin pipe for subprocess."""

    def __init__(self) -> None:
        self._buf = b""

    def write(self, data: bytes) -> None:
        self._buf += data

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        return None


class DummyProc:
    """Mock asyncio subprocess process."""

    def __init__(
        self,
        rc: int = 0,
        out: bytes = b"OK",
        err: bytes = b"",
        stdin: DummyPipe | None = None,
    ) -> None:
        self.returncode = rc
        self._out = out
        self._err = err
        self.stdin = stdin

    async def communicate(self):
        return self._out, self._err

    def kill(self) -> None:
        return None

    async def wait(self) -> None:
        return None
