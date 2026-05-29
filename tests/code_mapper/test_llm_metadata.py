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


def test_build_llm_metadata_and_map_hyde_falls_back_to_synthesis(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)

    llm_meta, map_hyde_provider = build_llm_metadata_and_map_hyde(
        config=config, llm_manager=None
    )

    assert map_hyde_provider is None
    assert llm_meta["map_hyde_provider"] == "openai"
    assert llm_meta["map_hyde_model"] == "synth-model"


def test_map_hyde_provider_override_drops_inherited_reasoning_effort(
    tmp_path: Path,
) -> None:
    config = Config(
        target_dir=tmp_path,
        llm={
            "provider": "opencode-cli",
            "synthesis_provider": "opencode-cli",
            "synthesis_model": "openai/gpt-5",
            "utility_model": "openai/gpt-5-nano",
            "codex_reasoning_effort_synthesis": "xhigh",
            "map_hyde_provider": "openai",
            "map_hyde_model": "gpt-5",
            "api_key": "test",
            "base_url": "https://source-provider.example/v1",
        },
    )
    manager = _FakeLLMManager()

    llm_meta, map_hyde_provider = build_llm_metadata_and_map_hyde(
        config=config, llm_manager=manager
    )

    assert map_hyde_provider is not None
    hyde_cfg = manager.seen_configs[0]
    assert hyde_cfg["provider"] == "openai"
    assert hyde_cfg["model"] == "gpt-5"
    assert "api_key" in hyde_cfg  # preserved for keyed provider
    assert hyde_cfg["base_url"] == "https://source-provider.example/v1"
    assert "reasoning_effort" not in hyde_cfg
    assert "map_hyde_reasoning_effort" not in llm_meta


def test_build_llm_metadata_and_map_hyde_preserves_custom_openai_base_url(
    tmp_path: Path,
) -> None:
    config = Config(
        target_dir=tmp_path,
        llm={
            "provider": "anthropic",
            "api_key": "test",
            "synthesis_model": "claude-sonnet-4-5-20250929",
            "map_hyde_provider": "openai",
            "map_hyde_model": "llama3.2",
            "base_url": "http://localhost:11434/v1",
        },
    )
    manager = _FakeLLMManager()

    llm_meta, map_hyde_provider = build_llm_metadata_and_map_hyde(
        config=config, llm_manager=manager
    )

    assert map_hyde_provider is not None
    assert manager.seen_configs[0]["provider"] == "openai"
    assert manager.seen_configs[0]["model"] == "llama3.2"
    assert manager.seen_configs[0]["base_url"] == "http://localhost:11434/v1"
    assert llm_meta["map_hyde_provider"] == "openai"
    assert llm_meta["map_hyde_model"] == "llama3.2"


def test_map_hyde_provider_override_requires_explicit_model_on_provider_switch(
    tmp_path: Path,
) -> None:
    """Provider switches require an explicit model — even within the same family."""
    with pytest.raises(
        ValueError,
        match=(
            "map_hyde provider override requires an explicit "
            "map_hyde_model"
        ),
    ):
        Config(
            target_dir=tmp_path,
            llm={
                "provider": "openai",
                "api_key": "test",
                "synthesis_model": "gpt-5",
                "utility_model": "gpt-5-nano",
                "map_hyde_provider": "deepseek",
            },
        )


def test_map_hyde_provider_override_drops_effort_for_non_reasoning_effort_provider(
    tmp_path: Path,
) -> None:
    config = Config(
        target_dir=tmp_path,
        llm={
            "provider": "codex-cli",
            "synthesis_model": "synth-model",
            "utility_model": "util-model",
            "codex_reasoning_effort_synthesis": "high",
            "map_hyde_provider": "anthropic",
            "map_hyde_model": "claude-opus-4-7",
            "map_hyde_reasoning_effort": "high",
        },
    )
    manager = _FakeLLMManager()

    build_llm_metadata_and_map_hyde(config=config, llm_manager=manager)

    hyde_cfg = manager.seen_configs[0]
    assert hyde_cfg["provider"] == "anthropic"
    assert "reasoning_effort" not in hyde_cfg


def test_build_llm_metadata_and_map_hyde_rejects_invalid_cross_family_override(
    tmp_path: Path,
) -> None:
    """Cross-family map_hyde override without model is caught at config
    validation time."""
    with pytest.raises(
        ValueError,
        match=(
            "map_hyde provider override requires an explicit "
            "map_hyde_model"
        ),
    ):
        Config(
            target_dir=tmp_path,
            llm={
                "provider": "openai",
                "api_key": "test",
                "synthesis_model": "gpt-5",
                "map_hyde_provider": "anthropic",
            },
        )


def test_map_hyde_provider_override_drops_inherited_structured_outputs_true(
    tmp_path: Path,
) -> None:
    config = Config(
        target_dir=tmp_path,
        llm={
            "provider": "openai",
            "api_key": "test",
            "synthesis_model": "gpt-5",
            "utility_model": "gpt-5-nano",
            "supports_structured_outputs": True,
            "map_hyde_provider": "deepseek",
            "map_hyde_model": "deepseek-v4-flash",
            "base_url": "https://api.openai.example/v1",
        },
    )
    manager = _FakeLLMManager()

    build_llm_metadata_and_map_hyde(config=config, llm_manager=manager)

    hyde_cfg = manager.seen_configs[0]
    assert hyde_cfg["provider"] == "deepseek"
    assert "api_key" in hyde_cfg  # preserved for keyed provider
    assert hyde_cfg["base_url"] == "https://api.openai.example/v1"
    assert "supports_structured_outputs" not in hyde_cfg


def test_map_hyde_provider_override_drops_inherited_structured_outputs_false(
    tmp_path: Path,
) -> None:
    config = Config(
        target_dir=tmp_path,
        llm={
            "provider": "deepseek",
            "api_key": "test",
            "synthesis_model": "deepseek-v4-flash",
            "utility_model": "deepseek-v4-flash",
            "supports_structured_outputs": False,
            "map_hyde_provider": "openai",
            "map_hyde_model": "gpt-5",
        },
    )
    manager = _FakeLLMManager()

    build_llm_metadata_and_map_hyde(config=config, llm_manager=manager)

    hyde_cfg = manager.seen_configs[0]
    assert hyde_cfg["provider"] == "openai"
    assert "supports_structured_outputs" not in hyde_cfg


def test_map_hyde_same_provider_preserves_explicit_structured_outputs(
    tmp_path: Path,
) -> None:
    config = Config(
        target_dir=tmp_path,
        llm={
            "provider": "openai",
            "api_key": "test",
            "synthesis_model": "gpt-5",
            "utility_model": "gpt-5-nano",
            "supports_structured_outputs": False,
            "map_hyde_provider": "openai",
            "map_hyde_model": "gpt-5-mini",
        },
    )
    manager = _FakeLLMManager()

    build_llm_metadata_and_map_hyde(config=config, llm_manager=manager)

    hyde_cfg = manager.seen_configs[0]
    assert hyde_cfg["provider"] == "openai"
    assert hyde_cfg["supports_structured_outputs"] is False


def test_map_hyde_same_provider_does_not_inherit_synthesis_reasoning_effort(
    tmp_path: Path,
) -> None:
    config = Config(
        target_dir=tmp_path,
        llm={
            "provider": "openai",
            "api_key": "test",
            "synthesis_model": "gpt-5",
            "utility_model": "gpt-5-nano",
            "codex_reasoning_effort_synthesis": "high",
            "map_hyde_provider": "openai",
            "map_hyde_model": "gpt-5-mini",
        },
    )
    manager = _FakeLLMManager()

    llm_meta, map_hyde_provider = build_llm_metadata_and_map_hyde(
        config=config, llm_manager=manager
    )

    assert map_hyde_provider is not None
    hyde_cfg = manager.seen_configs[0]
    assert hyde_cfg["provider"] == "openai"
    assert "reasoning_effort" not in hyde_cfg
    assert "map_hyde_reasoning_effort" not in llm_meta


def test_map_hyde_config_precedence_and_provider_switch_drop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)

    external_config = tmp_path / "external.json"
    external_config.write_text(
        """
{
  "llm": {
    "provider": "openai",
    "api_key": "external-test",
    "synthesis_model": "external-synth",
    "utility_model": "external-util",
    "supports_structured_outputs": true,
    "map_hyde_provider": "openai",
    "map_hyde_model": "external-hyde"
  }
}
""".strip(),
        encoding="utf-8",
    )
    (workspace_root / ".chunkhound.json").write_text(
        """
{
  "llm": {
    "supports_structured_outputs": false,
    "map_hyde_provider": "openai",
    "map_hyde_model": "local-hyde"
  }
}
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("CHUNKHOUND_LLM_SUPPORTS_STRUCTURED_OUTPUTS", "true")
    monkeypatch.setenv("CHUNKHOUND_LLM_MAP_HYDE_PROVIDER", "openai")
    monkeypatch.setenv("CHUNKHOUND_LLM_MAP_HYDE_MODEL", "env-hyde")

    class Args:
        def __init__(self) -> None:
            self.command = "search"
            self.path = workspace_root
            self.config = external_config
            self.llm_supports_structured_outputs = False
            self.llm_map_hyde_provider = "deepseek"
            self.llm_map_hyde_model = "deepseek-v4-flash"
            self.llm_map_hyde_reasoning_effort = None
            self.llm_utility_model = None
            self.llm_synthesis_model = None
            self.llm_api_key = None
            self.llm_base_url = None
            self.llm_provider = None
            self.llm_utility_provider = None
            self.llm_synthesis_provider = None
            self.llm_codex_reasoning_effort = None
            self.llm_codex_reasoning_effort_utility = None
            self.llm_codex_reasoning_effort_synthesis = None
            self.llm_autodoc_cleanup_provider = None
            self.llm_autodoc_cleanup_model = None
            self.llm_autodoc_cleanup_reasoning_effort = None
            self.verbose = False
            self.debug = False

    config = Config(args=Args())
    assert config.llm is not None
    assert config.llm.supports_structured_outputs is False
    assert config.llm.map_hyde_provider == "deepseek"
    assert config.llm.map_hyde_model == "deepseek-v4-flash"

    manager = _FakeLLMManager()
    build_llm_metadata_and_map_hyde(config=config, llm_manager=manager)

    hyde_cfg = manager.seen_configs[0]
    assert hyde_cfg["provider"] == "deepseek"
    assert hyde_cfg["model"] == "deepseek-v4-flash"
    assert "supports_structured_outputs" not in hyde_cfg
