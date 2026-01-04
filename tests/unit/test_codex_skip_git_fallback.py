import asyncio
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
async def test_codex_skip_git_unknown_flag_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    """Fallback to retry without skip-git flag when CLI reports it as unsupported."""
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider

    monkeypatch.setenv("CHUNKHOUND_CODEX_STDIN_FIRST", "0")
    monkeypatch.setattr(
        CodexCLIProvider, "_codex_available", lambda self: True, raising=True
    )

    overlay_dir = tmp_path / "overlay-home"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    def _fake_overlay_home(self, model_override=None):
        return str(overlay_dir)

    monkeypatch.setattr(
        CodexCLIProvider, "_build_overlay_home", _fake_overlay_home, raising=True
    )

    calls: list[list[str]] = []
    procs = [
        _DummyProc(
            rc=2,
            err=b"error: unexpected argument '--skip-git-repo-check' found\n",
        ),
        _DummyProc(rc=0, out=b"OK", err=b""),
    ]

    async def _fake_create_subprocess_exec(*args, **kwargs):  # noqa: ANN001
        calls.append(list(args))
        return procs.pop(0)

    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _fake_create_subprocess_exec, raising=True
    )

    prov = CodexCLIProvider(model="codex", max_retries=2)
    out = await prov._run_exec(
        "ping", cwd=None, max_tokens=8, timeout=10, model="codex"
    )  # type: ignore[attr-defined]

    assert out == "OK"
    assert len(calls) >= 2
    assert "--skip-git-repo-check" in calls[0]
    assert "--skip-git-repo-check" not in calls[1]
    assert not overlay_dir.exists(), "overlay CODEX_HOME should be removed after run"
