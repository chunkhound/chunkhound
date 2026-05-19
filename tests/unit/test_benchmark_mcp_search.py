from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from typing import Any

import pytest

benchmark_mcp_search = importlib.import_module("scripts.benchmark_mcp_search")


class _FakeProcess:
    stdin = object()
    stdout = object()
    stderr = object()


def test_launch_args_use_stdio_transport() -> None:
    assert benchmark_mcp_search._launch_args() == (
        "uv",
        "run",
        "chunkhound",
        "mcp",
        "stdio",
    )


def test_require_search_tool_raises_when_missing() -> None:
    with pytest.raises(
        benchmark_mcp_search.BenchmarkStartupError, match="'search' tool not found"
    ):
        benchmark_mcp_search._require_search_tool(
            {"tools": [{"name": "code_research"}]}
        )


@pytest.mark.asyncio
async def test_launch_mcp_server_uses_safe_subprocess_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    async def fake_create_subprocess_exec_safe(
        *args: str, **kwargs: Any
    ) -> _FakeProcess:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return _FakeProcess()

    def fake_get_safe_subprocess_env(base_env: dict[str, str]) -> dict[str, str]:
        observed["base_env"] = base_env
        return {"SAFE_ENV": "1"}

    monkeypatch.setattr(
        benchmark_mcp_search,
        "create_subprocess_exec_safe",
        fake_create_subprocess_exec_safe,
    )
    monkeypatch.setattr(
        benchmark_mcp_search,
        "get_safe_subprocess_env",
        fake_get_safe_subprocess_env,
    )

    cwd = Path("/tmp/chunkhound-benchmark")
    proc = await benchmark_mcp_search._launch_mcp_server(cwd)

    assert proc is not None
    assert observed["args"] == benchmark_mcp_search._launch_args()
    assert observed["kwargs"]["cwd"] == str(cwd)
    assert observed["kwargs"]["env"] == {"SAFE_ENV": "1"}
    assert observed["kwargs"]["stdin"] is asyncio.subprocess.PIPE
    assert observed["kwargs"]["stdout"] is asyncio.subprocess.PIPE
    assert observed["kwargs"]["stderr"] is asyncio.subprocess.PIPE
    assert isinstance(observed["base_env"], dict)


@pytest.mark.asyncio
async def test_settle_before_measurement_waits_fixed_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[float] = []

    async def fake_sleep(delay: float) -> None:
        observed.append(delay)

    monkeypatch.setattr(benchmark_mcp_search.asyncio, "sleep", fake_sleep)

    await benchmark_mcp_search._settle_before_measurement()

    assert observed == [benchmark_mcp_search.SETTLE_DELAY]


def test_run_cli_exits_nonzero_when_search_tool_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    events: list[str] = []

    async def fake_launch_mcp_server(_cwd: Path) -> _FakeProcess:
        events.append("launch")
        return _FakeProcess()

    class FakeClient:
        def __init__(self, _process: _FakeProcess) -> None:
            pass

        async def start(self) -> None:
            events.append("start")

        async def send_request(
            self, method: str, *_args: Any, **_kwargs: Any
        ) -> dict[str, Any]:
            if method == "initialize":
                return {}
            if method == "tools/list":
                return {"tools": [{"name": "code_research"}]}
            raise AssertionError(f"unexpected method {method}")

        async def send_notification(self, *_args: Any, **_kwargs: Any) -> None:
            events.append("initialized")

        async def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(
        benchmark_mcp_search, "_launch_mcp_server", fake_launch_mcp_server
    )
    monkeypatch.setattr(benchmark_mcp_search, "SubprocessJsonRpcClient", FakeClient)

    with pytest.raises(SystemExit) as excinfo:
        benchmark_mcp_search._run_cli()

    assert excinfo.value.code == 1
    assert events == ["launch", "start", "initialized", "close"]
    assert "'search' tool not found" in capsys.readouterr().err


def test_main_settles_before_running_queries(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    events: list[str] = []

    async def fake_launch_mcp_server(_cwd: Path) -> _FakeProcess:
        return _FakeProcess()

    class FakeClient:
        def __init__(self, _process: _FakeProcess) -> None:
            pass

        async def start(self) -> None:
            events.append("start")

        async def close(self) -> None:
            events.append("close")

    async def fake_handshake(_client: Any) -> list[str]:
        events.append("handshake")
        return ["regex"]

    async def fake_settle_before_measurement(
        *_args: Any, **_kwargs: Any
    ) -> None:
        events.append("settle")

    async def fake_run_queries(
        _client: Any, _queries: list[str], search_type: str
    ) -> dict[str, list[float]]:
        events.append(f"run:{search_type}")
        return {"query": [1.0, 2.0, 3.0]}

    monkeypatch.setattr(
        benchmark_mcp_search, "_launch_mcp_server", fake_launch_mcp_server
    )
    monkeypatch.setattr(benchmark_mcp_search, "SubprocessJsonRpcClient", FakeClient)
    monkeypatch.setattr(benchmark_mcp_search, "_handshake", fake_handshake)
    monkeypatch.setattr(
        benchmark_mcp_search,
        "_settle_before_measurement",
        fake_settle_before_measurement,
    )
    monkeypatch.setattr(benchmark_mcp_search, "_run_queries", fake_run_queries)

    asyncio.run(benchmark_mcp_search.main())

    assert events == ["start", "handshake", "settle", "run:regex", "close"]
    assert "manual input only" in capsys.readouterr().out
