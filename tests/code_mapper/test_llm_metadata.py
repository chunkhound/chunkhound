from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chunkhound.code_mapper.llm import build_llm_metadata_and_map_hyde
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


def test_build_llm_metadata_and_map_hyde_overrides(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    manager = _FakeLLMManager()

    config.llm.map_hyde_provider = "codex-cli"
    config.llm.map_hyde_model = "hyde-model"
    config.llm.map_hyde_reasoning_effort = "high"

    llm_meta, map_hyde_provider = build_llm_metadata_and_map_hyde(
        config=config, llm_manager=manager
    )

    assert map_hyde_provider is not None
    assert manager.seen_configs

    hyde_cfg = manager.seen_configs[0]
    assert hyde_cfg["provider"] == "codex-cli"
    assert hyde_cfg["model"] == "hyde-model"
    assert hyde_cfg["reasoning_effort"] == "high"

    assert llm_meta["map_hyde_provider"] == "codex-cli"
    assert llm_meta["map_hyde_model"] == "hyde-model"
    assert llm_meta["map_hyde_reasoning_effort"] == "high"


def test_build_llm_metadata_and_map_hyde_falls_back_to_synthesis(tmp_path: Path) -> None:
    config = _make_config(tmp_path)

    llm_meta, map_hyde_provider = build_llm_metadata_and_map_hyde(
        config=config, llm_manager=None
    )

    assert map_hyde_provider is None
    assert llm_meta["map_hyde_provider"] == "openai"
    assert llm_meta["map_hyde_model"] == "synth-model"
