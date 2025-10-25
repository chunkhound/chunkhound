import asyncio
import os
from pathlib import Path

import pytest


class _DummyProc:
    def __init__(self, rc: int = 0, out: bytes = b"OK", err: bytes = b"") -> None:
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
async def test_codex_config_only_mode_uses_config_env_and_no_codex_home(monkeypatch, tmp_path: Path):
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider

    # Force argv path to simplify dummy proc (no stdin pipe)
    monkeypatch.setenv("CHUNKHOUND_CODEX_STDIN_FIRST", "0")
    # Use env-based config override by default
    monkeypatch.setenv("CHUNKHOUND_CODEX_CONFIG_OVERRIDE", "env")
    # Pass through specific auth/passthrough envs
    monkeypatch.setenv("CHUNKHOUND_CODEX_AUTH_ENV", "OPENAI_API_KEY,CODEX_API_KEY")
    monkeypatch.setenv("CHUNKHOUND_CODEX_PASSTHROUGH_ENV", "EXTRA_VAR")

    # Provide some env values to be forwarded
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("CODEX_API_KEY", "sk-test-codex")
    monkeypatch.setenv("EXTRA_VAR", "123")

    # Avoid copying any real Codex home during the test
    monkeypatch.setattr(CodexCLIProvider, "_get_base_codex_home", lambda self: None, raising=True)
    # Force availability
    monkeypatch.setattr(CodexCLIProvider, "_codex_available", lambda self: True, raising=True)

    captured = {"args": None, "env": None, "config_text": None}

    async def _fake_create_subprocess_exec(*args, **kwargs):  # noqa: ANN001
        captured["args"] = args
        env = kwargs.get("env", {})
        captured["env"] = env
        cfg_key = os.getenv("CHUNKHOUND_CODEX_CONFIG_ENV", "CODEX_CONFIG")
        cfg_path = env.get(cfg_key)
        if cfg_path:
            cfg = Path(cfg_path)
            if cfg.exists():
                captured["config_text"] = cfg.read_text()
        return _DummyProc(rc=0, out=b"OK", err=b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec, raising=True)

    prov = CodexCLIProvider(model="gpt-5-codex-pro")
    out = await prov._run_exec("ping", cwd=None, max_tokens=16, timeout=10, model="gpt-5-codex-pro")  # type: ignore[attr-defined]

    assert out.strip() == "OK"
    assert captured["env"] is not None
    child_env = captured["env"]
    cfg_key = os.getenv("CHUNKHOUND_CODEX_CONFIG_ENV", "CODEX_CONFIG")
    # CODEX_HOME should point at the sandboxed directory we created
    assert child_env.get("CODEX_HOME") == str(Path(child_env[cfg_key]).parent)
    # Config override via env should be present and point to a file
    assert cfg_key in child_env
    cfg_path = Path(child_env[cfg_key])
    # Provider may already clean up the temp file by the time we assert; only validate shape
    assert cfg_path.name == "config.toml"
    assert "chunkhound-codex-overlay-" in str(cfg_path.parent)
    # CODEX_HOME should be rewritten to the config directory
    assert child_env.get("CODEX_HOME") == str(cfg_path.parent)
    # Auth and passthrough envs forwarded
    assert child_env.get("OPENAI_API_KEY") == "sk-test-openai"
    assert child_env.get("CODEX_API_KEY") == "sk-test-codex"
    assert child_env.get("EXTRA_VAR") == "123"
    # Config toml should contain resolved model and reasoning effort default
    assert captured["config_text"] is not None
    assert 'model = "gpt-5-codex-pro"' in captured["config_text"]
    assert 'model_reasoning_effort = "low"' in captured["config_text"]


@pytest.mark.asyncio
async def test_codex_config_only_mode_accepts_custom_reasoning_effort(monkeypatch, tmp_path: Path):
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider

    monkeypatch.setenv("CHUNKHOUND_CODEX_STDIN_FIRST", "0")
    monkeypatch.setenv("CHUNKHOUND_CODEX_CONFIG_OVERRIDE", "env")

    monkeypatch.setattr(CodexCLIProvider, "_get_base_codex_home", lambda self: None, raising=True)
    monkeypatch.setattr(CodexCLIProvider, "_codex_available", lambda self: True, raising=True)

    captured = {"env": None, "config_text": None}

    async def _fake_create_subprocess_exec(*args, **kwargs):  # noqa: ANN001
        env = kwargs.get("env", {})
        captured["env"] = env
        cfg_key = os.getenv("CHUNKHOUND_CODEX_CONFIG_ENV", "CODEX_CONFIG")
        cfg_path = env.get(cfg_key)
        if cfg_path:
            cfg = Path(cfg_path)
            if cfg.exists():
                captured["config_text"] = cfg.read_text()
        return _DummyProc(rc=0, out=b"OK", err=b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec, raising=True)

    prov = CodexCLIProvider(model="gpt-5-codex-pro", reasoning_effort="high")
    out = await prov._run_exec("ping", cwd=None, max_tokens=16, timeout=10, model="gpt-5-codex-pro")  # type: ignore[attr-defined]

    assert out.strip() == "OK"
    cfg_key = os.getenv("CHUNKHOUND_CODEX_CONFIG_ENV", "CODEX_CONFIG")
    cfg_path = Path(captured["env"][cfg_key])
    assert cfg_path.name == "config.toml"
    assert captured["env"].get("CODEX_HOME") == str(cfg_path.parent)
    assert captured["config_text"] is not None
    assert 'model_reasoning_effort = "high"' in captured["config_text"]


def test_codex_model_resolution_defaults(monkeypatch):
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider

    # Ensure env override is not set
    monkeypatch.delenv("CHUNKHOUND_CODEX_DEFAULT_MODEL", raising=False)

    prov = CodexCLIProvider(model="codex")
    assert prov._resolve_model_name("codex") == "gpt-5-codex"
    assert prov._resolve_model_name("gpt-5-codex-pro") == "gpt-5-codex-pro"


def test_codex_reasoning_effort_resolution(monkeypatch):
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider

    monkeypatch.delenv("CHUNKHOUND_CODEX_REASONING_EFFORT", raising=False)
    prov_default = CodexCLIProvider(model="codex")
    assert prov_default._reasoning_effort == "low"

    prov_custom = CodexCLIProvider(model="codex", reasoning_effort="High")
    assert prov_custom._reasoning_effort == "high"

    monkeypatch.setenv("CHUNKHOUND_CODEX_REASONING_EFFORT", "medium")
    prov_env = CodexCLIProvider(model="codex")
    assert prov_env._reasoning_effort == "medium"

    monkeypatch.setenv("CHUNKHOUND_CODEX_REASONING_EFFORT", "unknown")
    prov_invalid = CodexCLIProvider(model="codex")
    assert prov_invalid._reasoning_effort == "low"
