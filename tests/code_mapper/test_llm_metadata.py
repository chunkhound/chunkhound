from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chunkhound.code_mapper.llm import build_llm_metadata_and_assembly
from chunkhound.core.config.config import Config


class _FakeProvider:
    name = "fake-provider"
    model = "fake-model"


class _FakeLLMManager:
    def __init__(self) -> None:
        self.seen_configs: list[dict[str, Any]] = []

    def create_provider_for_config(self, config: dict[str, Any]) -> _FakeProvider:
        self.seen_configs.append(config)
        return _FakeProvider()


def _make_config(tmp_path: Path) -> Config:
    return Config(
        target_dir=tmp_path,
        llm={
            "provider": "openai",
            "api_key": "test",
            "synthesis_model": "synth-model",
            "utility_model": "util-model",
        },
    )


def test_build_llm_metadata_and_assembly_env_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _make_config(tmp_path)
    manager = _FakeLLMManager()

    monkeypatch.setenv("CH_AGENT_DOC_ASSEMBLY_PROVIDER", "codex-cli")
    monkeypatch.setenv("CH_AGENT_DOC_ASSEMBLY_MODEL", "assembly-model")
    monkeypatch.setenv("CH_AGENT_DOC_ASSEMBLY_REASONING_EFFORT", "HIGH")

    llm_meta, assembly_provider = build_llm_metadata_and_assembly(
        config=config, llm_manager=manager
    )

    assert assembly_provider is not None
    assert manager.seen_configs

    assembly_cfg = manager.seen_configs[0]
    assert assembly_cfg["provider"] == "codex-cli"
    assert assembly_cfg["model"] == "assembly-model"
    assert assembly_cfg["reasoning_effort"] == "high"

    assert llm_meta["assembly_synthesis_provider"] == "codex-cli"
    assert llm_meta["assembly_synthesis_model"] == "assembly-model"
    assert llm_meta["assembly_reasoning_effort"] == "high"


def test_build_llm_metadata_and_assembly_falls_back_to_synthesis(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)

    llm_meta, assembly_provider = build_llm_metadata_and_assembly(
        config=config, llm_manager=None
    )

    assert assembly_provider is None
    assert llm_meta["assembly_synthesis_provider"] == "openai"
    assert llm_meta["assembly_synthesis_model"] == "synth-model"
