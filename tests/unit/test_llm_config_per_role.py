import asyncio
import os
from pathlib import Path

import pytest

from chunkhound.core.config.llm_config import LLMConfig
from chunkhound.llm_manager import LLMManager


def test_llm_config_per_role_provider_overrides():
    # Red test: fields not yet present, or not applied
    cfg = LLMConfig(
        provider="openai",
        utility_provider="openai",  # keep existing utility
        synthesis_provider="codex-cli",  # switch synthesis to codex
        utility_model="gpt-5-nano",
        synthesis_model="codex",
    )

    util_conf, synth_conf = cfg.get_provider_configs()

    assert util_conf["provider"] == "openai"
    assert util_conf["model"] == "gpt-5-nano"

    assert synth_conf["provider"] == "codex-cli"
    assert synth_conf["model"] == "codex"


def test_llm_config_codex_reasoning_effort_per_role():
    cfg = LLMConfig(
        provider="codex-cli",
        utility_provider="codex-cli",
        synthesis_provider="codex-cli",
        utility_model="codex",
        synthesis_model="codex",
        codex_reasoning_effort="medium",
        codex_reasoning_effort_synthesis="high",
    )

    utility_config, synthesis_config = cfg.get_provider_configs()

    assert utility_config["reasoning_effort"] == "medium"
    assert synthesis_config["reasoning_effort"] == "high"

    cfg2 = LLMConfig(
        provider="codex-cli",
        utility_model="codex",
        synthesis_model="codex",
        codex_reasoning_effort_utility="minimal",
    )

    util2, synth2 = cfg2.get_provider_configs()
    assert util2["reasoning_effort"] == "minimal"
    assert "reasoning_effort" not in synth2


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
async def test_llm_codex_cli_status_reflects_configured_model_and_effort(monkeypatch, tmp_path: Path):
    """End-to-end check: LLMConfig -> LLMManager -> CodexCLI overlay config."""
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider

    cfg = LLMConfig(
        provider="codex-cli",
        utility_provider="codex-cli",
        synthesis_provider="codex-cli",
        utility_model="gpt-5.1-codex",
        synthesis_model="gpt-5.1-codex",
        codex_reasoning_effort_utility="low",
        codex_reasoning_effort_synthesis="high",
    )

    utility_config, synthesis_config = cfg.get_provider_configs()

    # Ensure we never touch a real Codex home or binary
    monkeypatch.setenv("CHUNKHOUND_CODEX_STDIN_FIRST", "0")
    monkeypatch.setenv("CHUNKHOUND_CODEX_CONFIG_OVERRIDE", "env")
    monkeypatch.setattr(CodexCLIProvider, "_get_base_codex_home", lambda self: None, raising=True)
    monkeypatch.setattr(CodexCLIProvider, "_codex_available", lambda self: True, raising=True)

    captured: dict[str, object] = {"env": None, "config_text": None}

    async def _fake_create_subprocess_exec(*args, **kwargs):  # noqa: ANN001
        env = kwargs.get("env", {})
        captured["env"] = env

        cfg_key = os.getenv("CHUNKHOUND_CODEX_CONFIG_ENV", "CODEX_CONFIG")
        cfg_path_str = env.get(cfg_key)

        model_name = "<missing>"
        effort_value = "<missing>"

        if isinstance(cfg_path_str, str):
            cfg_path = Path(cfg_path_str)
            if cfg_path.exists():
                text = cfg_path.read_text(encoding="utf-8")
                captured["config_text"] = text
                for line in text.splitlines():
                    if line.startswith("model ="):
                        model_name = line.split("=", 1)[1].strip().strip('"')
                    if line.startswith("model_reasoning_effort ="):
                        effort_value = line.split("=", 1)[1].strip().strip('"')

        # Simulate a `/status`-style response from Codex
        status_text = f"MODEL={model_name};REASONING_EFFORT={effort_value}"
        return _DummyProc(rc=0, out=status_text.encode("utf-8"), err=b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec, raising=True)

    llm_manager = LLMManager(utility_config, synthesis_config)
    provider = llm_manager.get_synthesis_provider()

    response = await provider.complete(prompt="/status")

    assert "MODEL=gpt-5.1-codex" in response.content
    assert "REASONING_EFFORT=high" in response.content
