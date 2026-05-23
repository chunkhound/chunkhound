from __future__ import annotations

import json
from types import SimpleNamespace

from chunkhound.api.cli.utils.config_factory import create_validated_config
from chunkhound.core.config.config import Config
from tests.site.tsx_runner import run_tsx_json


def _load_preset(provider_list: str, preset_id: str) -> dict:
    """Load a preset config from configurator-data.ts by provider list and id."""
    script = f"""
import {{ {provider_list} }} from './site/src/components/configurator-data.ts';

const option = {provider_list}.find((provider) => provider.id === '{preset_id}');
if (!option) {{
  throw new Error('missing {preset_id} option');
}}

console.log(JSON.stringify(option.config));
"""
    return run_tsx_json(script)


def test_ollama_llm_configurator_emits_explicit_local_model() -> None:
    config = _load_preset("llmProviders", "ollama-llm")

    assert config["provider"] == "openai"
    assert config["base_url"] == "http://localhost:11434/v1"
    assert config["model"] == "qwen3-coder:30b"


def test_ollama_embed_configurator_has_qwen3_reranker() -> None:
    config = _load_preset("embeddingProviders", "ollama-embed")

    assert config["model"] == "qwen3-embedding"
    assert config["rerank_model"] == "qwen3-reranker"
    assert config["rerank_format"] == "cohere"


def test_vllm_embed_configurator_has_qwen3_models() -> None:
    config = _load_preset("embeddingProviders", "vllm-embed")

    assert config["model"] == "Qwen/Qwen3-Embedding-0.6B"
    assert config["rerank_model"] == "Qwen/Qwen3-Reranker-0.6B"
    assert "rerank_url" not in config


def test_vllm_llm_configurator_has_qwen3_coder() -> None:
    config = _load_preset("llmProviders", "vllm-llm")

    assert config["model"] == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    assert config["base_url"] == "http://localhost:8000/v1"


def test_openai_embed_configurator_has_no_rerank_url() -> None:
    config = _load_preset("embeddingProviders", "openai-embed")

    assert "rerank_url" not in config


def _build_chunkhound_config(embedding_id: str, llm_id: str) -> dict:
    script = f"""
import {{
  buildChunkhoundConfig,
  embeddingProviders,
  llmProviders,
}} from './site/src/components/configurator-data.ts';

const embedding = embeddingProviders.find(
  (provider) => provider.id === '{embedding_id}'
);
const llm = llmProviders.find(
  (provider) => provider.id === '{llm_id}'
);
if (!embedding || !llm) {{
  throw new Error('missing provider');
}}

console.log(JSON.stringify(buildChunkhoundConfig(embedding, llm)));
"""
    return run_tsx_json(script)


def test_configurator_local_openai_compatible_presets_round_trip_through_backend_config(
) -> None:
    ollama_config = Config(
        **_build_chunkhound_config("ollama-embed", "ollama-llm")
    )
    vllm_config = Config(
        **_build_chunkhound_config("vllm-embed", "vllm-llm")
    )

    assert ollama_config.embedding is not None
    assert ollama_config.embedding.base_url == "http://localhost:11434/v1"
    assert ollama_config.embedding.rerank_url == "/rerank"
    assert ollama_config.embedding.rerank_model == "qwen3-reranker"
    assert ollama_config.llm is not None
    assert ollama_config.llm.base_url == "http://localhost:11434/v1"
    assert ollama_config.llm.model == "qwen3-coder:30b"

    assert vllm_config.embedding is not None
    assert vllm_config.embedding.base_url == "http://localhost:8000/v1"
    assert vllm_config.embedding.rerank_url == "/rerank"
    assert vllm_config.embedding.rerank_model == "Qwen/Qwen3-Reranker-0.6B"
    assert vllm_config.llm is not None
    assert vllm_config.llm.base_url == "http://localhost:8000/v1"
    assert vllm_config.llm.model == "Qwen/Qwen3-Coder-30B-A3B-Instruct"


def _write_config(tmp_path, embedding_id: str, llm_id: str) -> str:
    config_path = tmp_path / ".chunkhound.json"
    config_path.write_text(
        json.dumps(_build_chunkhound_config(embedding_id, llm_id)),
        encoding="utf-8",
    )
    return str(config_path)


def _validated_config_errors(
    tmp_path, command: str, embedding_id: str, llm_id: str
) -> list[str]:
    args = SimpleNamespace(
        command=command,
        config=_write_config(tmp_path, embedding_id, llm_id),
        path=str(tmp_path),
        no_embeddings=False,
        overview_only=False,
        assets_only=False,
    )
    _config, errors = create_validated_config(args, command)
    return errors


def test_ollama_generated_config_passes_index_validation(
    tmp_path, clean_environment
) -> None:
    errors = _validated_config_errors(tmp_path, "index", "ollama-embed", "ollama-llm")

    assert errors == []


def test_ollama_generated_config_passes_research_validation(
    tmp_path, clean_environment
) -> None:
    errors = _validated_config_errors(
        tmp_path, "research", "ollama-embed", "ollama-llm"
    )

    assert errors == []


def test_vllm_generated_config_passes_index_validation(
    tmp_path, clean_environment
) -> None:
    errors = _validated_config_errors(tmp_path, "index", "vllm-embed", "vllm-llm")

    assert errors == []


def test_vllm_generated_config_passes_research_validation(
    tmp_path, clean_environment
) -> None:
    errors = _validated_config_errors(tmp_path, "research", "vllm-embed", "vllm-llm")

    assert errors == []
