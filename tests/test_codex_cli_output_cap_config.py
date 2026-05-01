from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

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
