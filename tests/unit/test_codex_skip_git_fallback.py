import asyncio
import subprocess
from pathlib import Path

import pytest

from tests.helpers import DummyProc


@pytest.mark.asyncio
async def test_codex_skip_git_required_flag_retry(monkeypatch, tmp_path: Path) -> None:
    """Retry with skip-git flag when CLI reports it as required."""
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider

    monkeypatch.setenv("CHUNKHOUND_CODEX_STDIN_FIRST", "0")
    monkeypatch.setenv("CHUNKHOUND_CODEX_SKIP_GIT_CHECK", "0")
    monkeypatch.setattr(
        CodexCLIProvider, "_codex_available", lambda self: True, raising=True
    )

    overlay_dir = tmp_path / "overlay-home-required"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    def _fake_overlay_home(self, model_override=None, **kwargs):
        return str(overlay_dir)

    monkeypatch.setattr(
        CodexCLIProvider, "_build_overlay_home", _fake_overlay_home, raising=True
    )

    CodexCLIProvider.get_highest_priority_available_model.cache_clear()

    model_discovery_output = (
        b'{"models":['
        b'{"slug":"low","visibility":"list","priority":1},'
        b'{"slug":"high","visibility":"list","priority":20}'
        b"]}\n"
    )

    def _fake_subprocess_run(*args, **kwargs):  # noqa: ANN001, ANN202, ARG001
        return type("Result", (), {"returncode": 0, "stdout": model_discovery_output})()

    monkeypatch.setattr(subprocess, "run", _fake_subprocess_run)

    calls: list[list[str]] = []
    procs = [
        DummyProc(
            rc=2,
            err=b"error: exec requires --skip-git-repo-check for this workspace\n",
        ),
        DummyProc(rc=0, out=b"OK", err=b""),
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
    assert "--skip-git-repo-check" not in calls[0]
    assert "--skip-git-repo-check" in calls[1]
    assert not overlay_dir.exists(), "overlay CODEX_HOME should be removed after run"


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

    def _fake_overlay_home(self, model_override=None, **kwargs):
        return str(overlay_dir)

    monkeypatch.setattr(
        CodexCLIProvider, "_build_overlay_home", _fake_overlay_home, raising=True
    )

    CodexCLIProvider.get_highest_priority_available_model.cache_clear()

    model_discovery_output = (
        b'{"models":['
        b'{"slug":"low","visibility":"list","priority":1},'
        b'{"slug":"high","visibility":"list","priority":20}'
        b"]}\n"
    )

    def _fake_subprocess_run(*args, **kwargs):  # noqa: ANN001, ANN202, ARG001
        return type("Result", (), {"returncode": 0, "stdout": model_discovery_output})()

    monkeypatch.setattr(subprocess, "run", _fake_subprocess_run)

    calls: list[list[str]] = []
    procs = [
        DummyProc(
            rc=2,
            err=b"error: unexpected argument '--skip-git-repo-check' found\n",
        ),
        DummyProc(rc=0, out=b"OK", err=b""),
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
