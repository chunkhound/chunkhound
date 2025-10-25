import asyncio
import os
from pathlib import Path

import pytest


class _DummyProc:
    def __init__(self, rc: int = 1, out: bytes = b"", err: bytes = b"") -> None:
        self.returncode = rc
        self._out = out
        self._err = err
        self.stdin = None

    async def communicate(self):  # pragma: no cover - exercised indirectly
        return self._out, self._err

    def kill(self) -> None:  # pragma: no cover - trivial
        return None

    async def wait(self) -> None:  # pragma: no cover - trivial
        return None


@pytest.mark.asyncio
async def test_codex_error_redaction(monkeypatch, tmp_path: Path):
    """Provider should redact sensitive tokens from stderr and truncate output."""
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider
    # Make redaction limit small for test
    monkeypatch.setenv("CHUNKHOUND_CODEX_LOG_MAX_ERR", "120")
    # Force argv path to avoid needing a real stdin pipe in the dummy proc
    monkeypatch.setenv("CHUNKHOUND_CODEX_STDIN_FIRST", "0")

    # Force availability
    monkeypatch.setattr(CodexCLIProvider, "_codex_available", lambda self: True, raising=True)

    # Deterministic overlay path
    overlay_dir = tmp_path / "overlay-home"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    requested_model = {}

    def _fake_overlay_home(self, model_override=None):
        requested_model["value"] = model_override
        return str(overlay_dir)

    monkeypatch.setattr(CodexCLIProvider, "_build_overlay_home", _fake_overlay_home, raising=True)

    # Prepare stderr with secrets
    secret = (
        b"Authorization: Bearer abc.def.ghi\n"
        b"api_key=sk-1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ\n"
        b"Set-Cookie: SESSIONID=verysecretcookievalue\n"
        b"Some regular message."
    )

    async def _fake_create_subprocess_exec(*args, **kwargs):  # noqa: ANN001
        return _DummyProc(rc=2, out=b"", err=secret)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec, raising=True)

    prov = CodexCLIProvider(model="codex", max_retries=1)

    with pytest.raises(RuntimeError) as ei:
        await prov._run_exec("ping", cwd=None, max_tokens=32, timeout=10, model="codex")  # type: ignore[attr-defined]

    msg = str(ei.value)
    # Secret substrings should be redacted
    assert "Bearer abc.def.ghi" not in msg
    assert "sk-1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ" not in msg
    assert "SESSIONID=verysecretcookievalue" not in msg
    assert "[REDACTED]" in msg
    # Overlay should be cleaned
    assert not overlay_dir.exists(), "overlay CODEX_HOME should be removed after error"
    assert requested_model.get("value") == "gpt-5-codex"
