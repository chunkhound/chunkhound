from unittest.mock import patch

import pytest

from chunkhound.core.config.llm_config import DEFAULT_LLM_TIMEOUT


@pytest.fixture(autouse=True)
def clear_codex_model_discovery_cache():
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider

    CodexCLIProvider.get_highest_priority_available_model.cache_clear()
    yield
    CodexCLIProvider.get_highest_priority_available_model.cache_clear()


def test_codex_cli_provider_import_and_name():
    # Red test: module does not exist yet
    from chunkhound.providers.llm.codex_cli_provider import (
        CodexCLIProvider,  # type: ignore[attr-defined]
    )

    with patch.object(CodexCLIProvider, "_codex_available", return_value=True):
        provider = CodexCLIProvider(model="codex")
    assert provider.name == "codex-cli"


def test_codex_cli_model_resolution_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    from chunkhound.providers.llm.codex_cli_provider import (
        CodexCLIProvider,  # type: ignore[attr-defined]
    )

    monkeypatch.delenv("CHUNKHOUND_CODEX_DEFAULT_MODEL", raising=False)
    with patch.object(
        CodexCLIProvider,
        "get_highest_priority_available_model",
        return_value="test-discovered-model",
    ):
        resolved, source = CodexCLIProvider.describe_model_resolution("codex")
    assert resolved == "test-discovered-model"
    assert source == "discovered"


def test_codex_cli_model_resolution_discovery_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chunkhound.providers.llm.codex_cli_provider import (
        CODEX_DEFAULT_SYNTHESIS_MODEL,
        CodexCLIProvider,  # type: ignore[attr-defined]
    )

    monkeypatch.delenv("CHUNKHOUND_CODEX_DEFAULT_MODEL", raising=False)
    with patch.object(
        CodexCLIProvider,
        "get_highest_priority_available_model",
        return_value=None,
    ):
        resolved, source = CodexCLIProvider.describe_model_resolution("codex")
        assert resolved == CODEX_DEFAULT_SYNTHESIS_MODEL
        assert source == "fallback"


def test_codex_cli_model_resolution_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chunkhound.providers.llm.codex_cli_provider import (
        CodexCLIProvider,  # type: ignore[attr-defined]
    )

    monkeypatch.setenv("CHUNKHOUND_CODEX_DEFAULT_MODEL", "test-env-override-model")
    resolved, source = CodexCLIProvider.describe_model_resolution("codex")
    assert resolved == "test-env-override-model"
    assert source == "env:CHUNKHOUND_CODEX_DEFAULT_MODEL"


def test_codex_cli_model_resolution_env_override_to_gpt52(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chunkhound.providers.llm.codex_cli_provider import (
        CodexCLIProvider,  # type: ignore[attr-defined]
    )

    monkeypatch.setenv("CHUNKHOUND_CODEX_DEFAULT_MODEL", "gpt-5.2-codex")
    resolved, source = CodexCLIProvider.describe_model_resolution("codex")
    assert resolved == "gpt-5.2-codex"
    assert source == "env:CHUNKHOUND_CODEX_DEFAULT_MODEL"


def test_codex_cli_effort_resolution_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from chunkhound.providers.llm.codex_cli_provider import (
        CodexCLIProvider,  # type: ignore[attr-defined]
    )

    monkeypatch.delenv("CHUNKHOUND_CODEX_REASONING_EFFORT", raising=False)
    resolved, source = CodexCLIProvider.describe_reasoning_effort_resolution(None)
    assert resolved == "low"
    assert source == "default"


def _stub_codex_bin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "chunkhound.providers.llm.codex_cli_provider.resolve_cli_binary",
        lambda name, env_var=None: "codex",
    )
    # Static discovery is lru_cached; clear between cases that change fake_run.
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider

    CodexCLIProvider.get_highest_priority_available_model.cache_clear()


def _patch_discovery_proc(
    monkeypatch: pytest.MonkeyPatch, *, returncode: int, stdout: bytes
) -> None:
    class _Proc:
        def __init__(self) -> None:
            self.returncode = returncode
            self.pid = 1

        def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
            return stdout, b""

        def kill(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            return returncode

    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: _Proc())


def test_codex_cli_model_discovery_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider

    _stub_codex_bin(monkeypatch)
    _patch_discovery_proc(monkeypatch, returncode=1, stdout=b"")

    assert CodexCLIProvider.get_highest_priority_available_model() is None


def test_codex_cli_model_discovery_no_visible_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider

    _stub_codex_bin(monkeypatch)
    output = b'{"models":[{"slug":"hidden","visibility":"hidden","priority":10}]}\n'
    _patch_discovery_proc(monkeypatch, returncode=0, stdout=output)

    assert CodexCLIProvider.get_highest_priority_available_model() is None


def test_codex_cli_model_discovery_malformed_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider

    _stub_codex_bin(monkeypatch)
    _patch_discovery_proc(monkeypatch, returncode=0, stdout=b"not json\n")

    assert CodexCLIProvider.get_highest_priority_available_model() is None


def test_codex_cli_model_discovery_priority_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider

    _stub_codex_bin(monkeypatch)
    output = (
        b'{"models":['
        b'{"slug":"low","visibility":"list","priority":1},'
        b'{"slug":"high","visibility":"list","priority":20},'
        b'{"slug":"hidden","visibility":"hidden","priority":100}'
        b"]}\n"
    )

    _patch_discovery_proc(monkeypatch, returncode=0, stdout=output)

    assert CodexCLIProvider.get_highest_priority_available_model() == "high"


def test_default_timeout():
    """Default timeout resolves to DEFAULT_LLM_TIMEOUT."""
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider
    p = CodexCLIProvider()
    assert p.timeout == DEFAULT_LLM_TIMEOUT


def test_unshortenable_shim_reports_broken(monkeypatch: pytest.MonkeyPatch) -> None:
    """A spaced .cmd that cannot be shortened does not crash provider setup."""
    import sys

    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        "chunkhound.providers.llm.base_cli_provider.sys.platform", "win32"
    )
    monkeypatch.setattr(
        "chunkhound.providers.llm.codex_cli_provider.resolve_cli_binary",
        lambda name, env_var=None: r"C:\Program Files\npm\codex.cmd",
    )
    provider = CodexCLIProvider(model="gpt-explicit")
    assert provider._codex_available_status() == "broken"


def test_model_discovery_unshortenable_shim_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        "chunkhound.providers.llm.base_cli_provider.sys.platform", "win32"
    )
    monkeypatch.setattr(
        "chunkhound.providers.llm.codex_cli_provider.resolve_cli_binary",
        lambda name, env_var=None: r"C:\Program Files\npm\codex.cmd",
    )
    assert CodexCLIProvider.get_highest_priority_available_model() is None


def test_model_discovery_timeout_kills_shim_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess
    import sys

    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        "chunkhound.providers.llm.base_cli_provider.sys.platform", "win32"
    )
    monkeypatch.setattr(
        "chunkhound.providers.llm.codex_cli_provider.resolve_cli_binary",
        lambda name, env_var=None: r"C:\npm\codex.cmd",
    )
    recorded: dict[str, object] = {}

    class _TimeoutProc:
        pid = 99

        def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
            raise subprocess.TimeoutExpired("codex", timeout or 10)

        def kill(self) -> None:
            recorded["killed"] = True

        def wait(self, timeout: float | None = None) -> int:
            return 1

    def fake_run(args, **kwargs):  # noqa: ANN001
        recorded["args"] = args
        recorded["stdout"] = kwargs.get("stdout")
        recorded["stderr"] = kwargs.get("stderr")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: _TimeoutProc())
    monkeypatch.setattr(
        "chunkhound.providers.llm.base_cli_provider.subprocess.run", fake_run
    )

    assert CodexCLIProvider.get_highest_priority_available_model() is None
    assert recorded["args"] == ["taskkill", "/T", "/F", "/PID", "99"]
    assert recorded["stdout"] is subprocess.DEVNULL
    assert recorded["stderr"] is subprocess.DEVNULL
    assert recorded["killed"] is True


@pytest.mark.asyncio
async def test_overlay_write_failure_does_not_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A failed overlay config write fails the call instead of dropping settings."""
    import asyncio
    from pathlib import Path

    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider

    monkeypatch.setattr(
        CodexCLIProvider, "_codex_available", lambda self: True, raising=True
    )
    monkeypatch.setattr(
        "chunkhound.providers.llm.codex_cli_provider.resolve_cli_binary",
        lambda name, env_var=None: "codex",
    )
    monkeypatch.setattr(
        CodexCLIProvider, "_get_base_codex_home", lambda self: None, raising=True
    )
    overlay = tmp_path / "overlay"
    monkeypatch.setattr(
        "chunkhound.providers.llm.codex_cli_provider.tempfile.mkdtemp",
        lambda prefix=None: overlay.mkdir() or str(overlay),
    )
    original = Path.write_text

    def fail_config(self, *args, **kwargs):  # noqa: ANN001
        if self.name == "config.toml":
            raise OSError("disk full")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_config)
    launched = False

    async def fail_if_launched(*args, **kwargs):  # noqa: ANN001
        nonlocal launched
        launched = True
        raise AssertionError("codex launched with a partial overlay")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_if_launched)
    provider = CodexCLIProvider(model="gpt-explicit")
    with pytest.raises(RuntimeError, match="Codex overlay config"):
        await provider._run_exec(
            "ping", cwd=None, max_tokens=16, timeout=10, model="gpt-explicit"
        )
    assert launched is False
    assert not overlay.exists()
