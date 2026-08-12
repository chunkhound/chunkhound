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

    provider = CodexCLIProvider(model="test-explicit-model", reasoning_effort="high")
    resp = await provider.complete("hi", max_completion_tokens=123)

    assert resp.content == "OK"

    argv = captured.get("args") or []
    argv_str = " ".join(str(a) for a in argv)
    assert "--json" in argv_str
