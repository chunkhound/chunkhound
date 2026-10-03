from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from chunkhound.interfaces.llm_provider import PROVIDER_MANAGED_OUTPUT
from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider
from tests.helpers import DummyPipe, DummyProc


@pytest.mark.asyncio
async def test_codex_cli_provider_passes_model_max_output_tokens_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def _fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> DummyProc:
        # args: (binary, "exec", "-", *extra_args, ...)
        captured["args"] = list(args)
        captured["kwargs"] = kwargs
        return DummyProc(out=b"OK", stdin=DummyPipe())

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)
    monkeypatch.setattr(
        "chunkhound.providers.llm.codex_cli_provider.resolve_cli_binary",
        lambda name, env_var=None: "codex",
    )

    provider = CodexCLIProvider(model="test-explicit-model", reasoning_effort="high")

    resp = await provider.complete("hi", max_completion_tokens=123)
    assert resp.content == "OK"

    argv = captured.get("args") or []
    argv_str = " ".join(str(a) for a in argv)
    assert "model_max_output_tokens=123" in argv_str
    assert "--sandbox read-only" in argv_str
    assert 'approval_policy="on-request"' in argv_str
    assert 'model_reasoning_effort="high"' in argv_str


@pytest.mark.asyncio
async def test_codex_provider_managed_output_uses_configured_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def _fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> DummyProc:
        captured["args"] = list(args)
        return DummyProc(out=b"OK", stdin=DummyPipe())

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)
    monkeypatch.setattr(
        "chunkhound.providers.llm.codex_cli_provider.resolve_cli_binary",
        lambda name, env_var=None: "codex",
    )

    provider = CodexCLIProvider(model="test-explicit-model")
    provider.configure_synthesis_output_limit_policy(
        output_limits_enabled=False, fallback_tokens=8192
    )
    response = await provider.complete(
        "hi", max_completion_tokens=PROVIDER_MANAGED_OUTPUT
    )

    assert response.content == "OK"
    argv_str = " ".join(str(arg) for arg in captured["args"])
    assert "model_max_output_tokens=8192" in argv_str
    assert "provider_managed" not in argv_str


@pytest.mark.asyncio
async def test_codex_cli_omission_path_does_not_restore_default_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def _fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> DummyProc:
        captured["args"] = list(args)
        return DummyProc(out=b"OK", stdin=DummyPipe())

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)
    monkeypatch.setattr(
        "chunkhound.providers.llm.codex_cli_provider.resolve_cli_binary",
        lambda name, env_var=None: "codex",
    )

    provider = CodexCLIProvider(model="test-explicit-model")
    content = await provider._run_cli_command("hi", max_completion_tokens=None)

    assert content == "OK"
    argv_str = " ".join(str(arg) for arg in captured["args"])
    assert "model_max_output_tokens" not in argv_str
    assert "model_max_output_tokens=4096" not in argv_str


@pytest.mark.asyncio
async def test_codex_cli_provider_default_output_cap_remains_4096(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def _fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> DummyProc:
        captured["args"] = list(args)
        return DummyProc(out=b"OK", stdin=DummyPipe())

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)
    monkeypatch.setattr(
        "chunkhound.providers.llm.codex_cli_provider.resolve_cli_binary",
        lambda name, env_var=None: "codex",
    )

    provider = CodexCLIProvider(model="test-explicit-model")
    await provider.complete("hi")

    argv_str = " ".join(str(arg) for arg in captured["args"])
    assert "model_max_output_tokens=4096" in argv_str


@pytest.mark.asyncio
async def test_codex_cli_provider_parses_agent_message_from_jsonl_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHUNKHOUND_CODEX_JSON", "1")

    fixture = (
        Path(__file__).resolve().parent / "fixtures" / "codex_exec_reply_ok.jsonl"
    ).read_bytes()

    captured: dict[str, Any] = {}

    async def _fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> DummyProc:
        captured["args"] = list(args)
        captured["kwargs"] = kwargs
        return DummyProc(out=fixture, stdin=DummyPipe())

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)
    monkeypatch.setattr(
        "chunkhound.providers.llm.codex_cli_provider.resolve_cli_binary",
        lambda name, env_var=None: "codex",
    )

    provider = CodexCLIProvider(model="test-explicit-model", reasoning_effort="high")
    resp = await provider.complete("hi", max_completion_tokens=123)

    assert resp.content == "OK"

    argv = captured.get("args") or []
    argv_str = " ".join(str(a) for a in argv)
    assert "--json" in argv_str


@pytest.mark.asyncio
async def test_codex_batch_shim_keeps_quoted_config_and_prompt_off_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A .cmd Codex shim cannot carry quoted -c values or the prompt."""
    captured: dict[str, Any] = {}

    async def _fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> DummyProc:
        captured["args"] = list(args)
        captured["stdin"] = kwargs.get("stdin")
        env = kwargs.get("env") or {}
        home = env.get("CODEX_HOME")
        if home:
            captured["config"] = (Path(home) / "config.toml").read_text(
                encoding="utf-8"
            )
        return DummyProc(out=b"OK", stdin=DummyPipe())

    monkeypatch.setenv("CHUNKHOUND_CODEX_STDIN_FIRST", "0")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)
    monkeypatch.setattr(
        "chunkhound.providers.llm.codex_cli_provider.resolve_cli_binary",
        lambda name, env_var=None: r"C:\npm\codex.cmd",
    )
    monkeypatch.setattr(
        "chunkhound.providers.llm.base_cli_provider.sys.platform",
        "win32",
    )

    provider = CodexCLIProvider(model="test-explicit-model", reasoning_effort="high")
    prompt = 'say "hi" & echo %PATH%\nnext'
    resp = await provider.complete(prompt, max_completion_tokens=123)

    assert resp.content == "OK"
    argv = [str(arg) for arg in captured["args"]]
    assert "-c" not in argv
    assert prompt not in argv
    assert "\n" not in " ".join(argv)
    assert "%" not in " ".join(argv)
    assert captured["stdin"] is asyncio.subprocess.PIPE
    config = captured["config"]
    assert 'approval_policy = "on-request"' in config
    assert "model_max_output_tokens = 123" in config
    assert 'model_reasoning_effort = "high"' in config


@pytest.mark.asyncio
async def test_codex_batch_shim_does_not_retry_prompt_on_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    async def _fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> DummyProc:
        calls["n"] += 1
        return DummyProc(
            rc=1,
            out=b"",
            err=b"stdin is not supported",
            stdin=DummyPipe(),
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)
    monkeypatch.setattr(
        "chunkhound.providers.llm.codex_cli_provider.resolve_cli_binary",
        lambda name, env_var=None: r"C:\npm\codex.cmd",
    )
    monkeypatch.setattr(
        "chunkhound.providers.llm.base_cli_provider.sys.platform",
        "win32",
    )

    provider = CodexCLIProvider(model="test-explicit-model", max_retries=3)
    with pytest.raises(RuntimeError, match="batch shim"):
        await provider.complete("hi")
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_codex_timeout_terminates_the_process_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminated: list[DummyProc] = []

    class _TimeoutProc(DummyProc):
        def __init__(self) -> None:
            super().__init__(stdin=DummyPipe())
            self.returncode = None
            self.killed = False

        async def communicate(self):  # type: ignore[override]
            raise asyncio.TimeoutError

        def kill(self) -> None:
            self.killed = True

    proc = _TimeoutProc()

    async def _fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> DummyProc:
        return proc

    async def _terminate(process: DummyProc) -> None:
        terminated.append(process)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)
    monkeypatch.setattr(
        "chunkhound.providers.llm.codex_cli_provider.resolve_cli_binary",
        lambda name, env_var=None: "codex",
    )
    monkeypatch.setattr(
        "chunkhound.providers.llm.codex_cli_provider.terminate_cli_process",
        _terminate,
    )

    provider = CodexCLIProvider(model="test-explicit-model", max_retries=1)
    with pytest.raises(RuntimeError, match="timed out"):
        await provider.complete("hi")
    assert terminated == [proc]
    assert proc.killed is False
