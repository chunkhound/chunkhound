import asyncio
from pathlib import Path

import pytest


class _DummyPipe:
    def __init__(self) -> None:  # pragma: no cover - trivial
        self._buf = b""

    def write(self, data: bytes) -> None:  # pragma: no cover - trivial
        self._buf += data

    async def drain(self) -> None:  # pragma: no cover - trivial
        return None

    def close(self) -> None:  # pragma: no cover - trivial
        return None


class _DummyProc:
    def __init__(self, rc: int = 0, out: bytes = b"OK", err: bytes = b"") -> None:
        self.returncode = rc
        self._out = out
        self._err = err
        self.stdin = _DummyPipe()

    async def communicate(self):  # pragma: no cover - exercised indirectly
        return self._out, self._err

    def kill(self) -> None:  # pragma: no cover - trivial
        return None

    async def wait(self) -> None:  # pragma: no cover - trivial
        return None


@pytest.mark.asyncio
async def test_codex_stdin_is_default(monkeypatch, tmp_path: Path):
    """By default, provider should use stdin (argument '-' after 'exec')."""
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider
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

    captured_args = {}

    async def _fake_create_subprocess_exec(*args, **kwargs):  # noqa: ANN001
        # Save invocation for assertions
        captured_args["args"] = list(args)
        captured_args["kwargs"] = dict(kwargs)
        return _DummyProc(rc=0, out=b"OK", err=b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec, raising=True)

    prov = CodexCLIProvider(model="codex")
    out = await prov._run_exec("ping", cwd=None, max_tokens=32, timeout=10, model="codex")  # type: ignore[attr-defined]
    assert out.strip() == "OK"

    # Validate that the provider invoked: codex exec -
    args = captured_args.get("args") or []
    # Expected shape: [binary, "exec", "-", ...]
    assert len(args) >= 3, f"unexpected argv: {args!r}"
    assert args[1] == "exec"
    assert args[2] == "-", f"provider did not use stdin-first; argv was: {args!r}"

    # Ensure overlay is cleaned
    assert not overlay_dir.exists(), "overlay CODEX_HOME should be removed after run"
    assert requested_model.get("value") == "gpt-5-codex"
