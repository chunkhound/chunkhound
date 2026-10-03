from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import get_args

import pytest
from pydantic import ValidationError

from chunkhound.api.cli.utils.config_factory import create_validated_config
from chunkhound.core.config.config import Config
from chunkhound.core.config.embedding_config import EmbeddingConfig
from tests.site.dom_helpers import browser_dom, dist_body
from tests.site.tsx_runner import run_tsx_json


def _load_preset(provider_list: str, preset_id: str) -> dict:
    """Load a preset config from the configurator modules by provider list and id."""
    script = f"""
import {{ {provider_list} }} from './site/src/components/configurator/index.ts';

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


def test_ollama_embed_configurator_has_only_embedding_configuration() -> None:
    config = _load_preset("embeddingProviders", "ollama-embed")

    assert config == {
        "provider": "openai",
        "model": "qwen3-embedding",
        "base_url": "http://localhost:11434/v1",
    }


def test_vllm_embed_configurator_uses_separate_rerank_endpoint() -> None:
    """vLLM serves one model per process, so the reranker gets its own endpoint.

    Regression guard: a shared base_url with no rerank_url sent rerank requests
    to :8000, which serves only the embedder."""
    config = _load_preset("embeddingProviders", "vllm-embed")

    assert config["model"] == "Qwen/Qwen3-Embedding-0.6B"
    assert config["base_url"] == "http://localhost:8000/v1"
    assert config["rerank_model"] == "Qwen/Qwen3-Reranker-0.6B"
    assert config["rerank_url"] == "http://localhost:8001/v1/rerank"
    assert config["rerank_format"] == "cohere"


def test_vllm_llm_configurator_has_dedicated_qwen3_endpoint() -> None:
    config = _load_preset("llmProviders", "vllm-llm")

    assert config["model"] == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    assert config["base_url"] == "http://localhost:8002/v1"


def test_openai_embed_configurator_has_no_rerank_url() -> None:
    config = _load_preset("embeddingProviders", "openai-embed")

    assert "rerank_url" not in config


def test_embedding_presets_declare_reranker_capabilities() -> None:
    script = """
import { embeddingProviders } from './site/src/components/configurator/index.ts';

console.log(JSON.stringify(embeddingProviders.map((provider) => ({
  id: provider.id,
  reranker: provider.reranker,
}))));
"""

    # Contract: every embedding preset tells the user whether a reranker is
    # included, and included ones explain how. Exact provider list/order is
    # data, not contract — new presets must not break this test.
    providers = run_tsx_json(script)
    ids = [provider["id"] for provider in providers]
    assert len(ids) == len(set(ids)), "duplicate embedding preset ids"
    for provider in providers:
        reranker = provider["reranker"]
        assert isinstance(reranker["included"], bool)
        if reranker["included"]:
            assert reranker.get("status"), f"{provider['id']} missing reranker status"


def _selection_defaults() -> dict:
    """Return the configurator's default route ids (agent/retrieval/research)."""
    script = """
import {
  DEFAULT_AGENT,
  DEFAULT_RESEARCH,
  DEFAULT_RETRIEVAL,
} from './site/src/components/configurator/index.ts';

console.log(JSON.stringify({
  defaultAgent: DEFAULT_AGENT,
  defaultRetrieval: DEFAULT_RETRIEVAL,
  defaultResearch: DEFAULT_RESEARCH,
}));
"""
    return run_tsx_json(script)


def test_default_route_renders_prerequisite_links_in_stage_order(built_site) -> None:
    """Each stage ships its default option's row already checked, and that row
    carries the prerequisite links — no script needed to show them."""
    script = (
        browser_dom(dist_body("docs/getting-started/index.html"))
        + "await import('./site/src/scripts/configurator.ts');\n"
        + """
const stages = [...window.document.querySelectorAll('[data-stage-panel]')].map(
  (panel) => {
    const role = panel.dataset.stagePanel;
    const row = panel.querySelector(`[data-${role}]:checked`)
      .closest('.provider-option');
    return {
      role,
      links: [...row.querySelectorAll('.option-requirements .prerequisite-link')]
        .map((link) => link.textContent),
    };
  },
);
console.log(JSON.stringify({ stages }));
"""
    )
    rendered = run_tsx_json(script)

    assert rendered["stages"] == [
        {
            "role": "retrieval",
            "links": [
                "VoyageAI API key",
                "Opt out of VoyageAI training for true privacy",
            ],
        },
        {"role": "research", "links": ["OpenRouter API key"]},
        {"role": "agent", "links": ["pi-mcp-adapter"]},
    ]


def test_pi_editor_has_the_recommended_local_mcp_contract() -> None:
    script = """
import { editors } from './site/src/components/configurator/index.ts';

const pi = editors.find((editor) => editor.id === 'pi');
if (!pi) throw new Error('missing Pi editor');
console.log(JSON.stringify(pi));
"""
    pi = run_tsx_json(script)

    assert pi["name"] == "Pi"
    assert pi["mcpFile"] == ".mcp.json"
    assert pi["installCommand"] == "pi install npm:pi-mcp-adapter"
    assert pi["gitignoreEntries"] == [".mcp.json"]
    assert pi["mcp"] == {
        "mcpServers": {
            "ChunkHound": {
                "command": "chunkhound",
                "args": ["mcp"],
                "directTools": True,
                "lifecycle": "eager",
                "requestTimeoutMs": 1200000,
            }
        },
        "settings": {"outputGuard": False},
    }


def test_every_provider_option_carries_picker_metadata() -> None:
    script = """
import {
  PROVIDER_GROUP_ORDER,
  embeddingProviders,
  llmProviders,
  editors,
  optionSearchText,
} from './site/src/components/configurator/index.ts';

console.log(JSON.stringify({
  groups: PROVIDER_GROUP_ORDER,
  options: [...embeddingProviders, ...llmProviders].map((option) => ({
    id: option.id,
    group: option.group,
    description: option.description,
    requirements: option.requirements,
    searchText: optionSearchText(option),
  })),
  agents: editors.map((agent) => ({
    id: agent.id,
    requirements: agent.requirements,
    searchText: optionSearchText(agent),
  })),
}));
"""
    data = run_tsx_json(script)

    # The catalog is data: assert shape and uniqueness, not an exact count,
    # so adding a provider preset never breaks this contract.
    ids = [option["id"] for option in data["options"]]
    assert ids and len(ids) == len(set(ids))
    for option in data["options"]:
        assert option["group"] in data["groups"], option
        assert option["description"], option
        assert option["requirements"], option
        assert option["searchText"] == option["searchText"].lower()
        for requirement in option["requirements"]:
            assert requirement["url"].startswith("https://"), option
            assert requirement["label"].lower() in option["searchText"]
    for agent in data["agents"]:
        assert agent["id"] and agent["requirements"], agent
        assert agent["searchText"] == agent["searchText"].lower()
        for requirement in agent["requirements"]:
            assert requirement["url"].startswith("https://"), agent
            assert requirement["label"].lower() in agent["searchText"]


def _build_chunkhound_config(
    embedding_id: str, llm_id: str, reranker: dict | None = None
) -> dict:
    script = f"""
import {{
  buildChunkhoundConfig,
  embeddingProviders,
  llmProviders,
}} from './site/src/components/configurator/index.ts';

const embedding = embeddingProviders.find(
  (provider) => provider.id === '{embedding_id}'
);
const llm = llmProviders.find(
  (provider) => provider.id === '{llm_id}'
);
if (!embedding || !llm) {{
  throw new Error('missing provider');
}}

console.log(JSON.stringify(buildChunkhoundConfig(
  embedding,
  llm,
  {json.dumps(reranker)},
)));
"""
    return run_tsx_json(script)


def test_reranker_override_is_trimmed_and_round_trips(
    tmp_path, clean_environment
) -> None:
    blank = _build_chunkhound_config("ollama-embed", "ollama-llm", {"url": " "})
    override = "https://reranker.example.com/rerank"
    config = _build_chunkhound_config(
        "ollama-embed", "ollama-llm", {"url": override, "format": "tei"}
    )

    assert "rerank_url" not in blank["embedding"]
    assert config["embedding"]["rerank_url"] == override
    assert config["embedding"]["rerank_format"] == "tei"
    assert Config(target_dir=tmp_path, **config).embedding.rerank_url == override


@pytest.mark.parametrize(
    ("reranker", "expected"),
    [
        (
            {"url": "https://tei.example.com/rerank", "format": "tei"},
            {"rerank_url": "https://tei.example.com/rerank", "rerank_format": "tei"},
        ),
        (
            {
                "url": "https://cohere.example.com/rerank",
                "format": "cohere",
                "model": "rerank-v3.5",
            },
            {
                "rerank_url": "https://cohere.example.com/rerank",
                "rerank_format": "cohere",
                "rerank_model": "rerank-v3.5",
            },
        ),
    ],
)
def test_openai_external_reranker_config_round_trips(
    tmp_path, clean_environment, reranker: dict, expected: dict
) -> None:
    config = _build_chunkhound_config("openai-embed", "ollama-llm", reranker)

    assert {key: config["embedding"][key] for key in expected} == expected
    assert Config(target_dir=tmp_path, **config).embedding is not None


def test_tei_reranker_without_url_round_trips(tmp_path, clean_environment) -> None:
    """TEI derives /rerank from a preset's base_url, so a URL-less TEI override
    must still emit rerank_format and let the backend derive rerank_url."""
    config = _build_chunkhound_config(
        "ollama-embed", "ollama-llm", {"format": "tei"}
    )

    assert config["embedding"]["rerank_format"] == "tei"
    assert "rerank_url" not in config["embedding"]
    assert "rerank_model" not in config["embedding"]

    validated = Config(target_dir=tmp_path, **config)
    # Backend auto-derives the rerank endpoint from base_url for TEI.
    assert validated.embedding.rerank_url == "/rerank"
    assert validated.embedding.rerank_format == "tei"


def test_url_less_tei_without_preset_base_url_is_rejected_by_backend(
    tmp_path, clean_environment
) -> None:
    """The configurator now refuses a URL-less TEI reranker when the preset has
    no base_url (openai-embed). Prove the UI and backend agree: the override the
    UI blocks is exactly the one the backend validator rejects."""
    config = _build_chunkhound_config(
        "openai-embed", "ollama-llm", {"format": "tei"}
    )

    assert config["embedding"]["rerank_format"] == "tei"
    assert "rerank_url" not in config["embedding"]
    assert "base_url" not in config["embedding"]
    with pytest.raises(
        ValidationError, match="requires base_url or explicit rerank_url"
    ):
        Config(target_dir=tmp_path, **config)


def test_voyageai_reranker_accepts_model_without_url(
    tmp_path, clean_environment
) -> None:
    """VoyageAI reranks via SDK with no rerank_url, so a URL-less model override
    must emit rerank_model and pass backend validation."""
    config = _build_chunkhound_config(
        "voyageai", "openrouter", {"model": "voyage-rerank-2.0"}
    )

    assert config["embedding"]["rerank_model"] == "voyage-rerank-2.0"
    assert "rerank_url" not in config["embedding"]
    assert "rerank_format" not in config["embedding"]

    validated = Config(target_dir=tmp_path, **config)
    assert validated.embedding.rerank_model == "voyage-rerank-2.0"
    assert validated.embedding.rerank_url is None


def test_local_openai_compatible_presets_round_trip_through_backend_config(
    tmp_path, clean_environment
) -> None:
    ollama_config = Config(
        target_dir=tmp_path,
        **_build_chunkhound_config("ollama-embed", "ollama-llm"),
    )
    vllm_config = Config(
        target_dir=tmp_path,
        **_build_chunkhound_config("vllm-embed", "vllm-llm"),
    )

    assert ollama_config.embedding is not None
    assert ollama_config.embedding.base_url == "http://localhost:11434/v1"
    assert ollama_config.embedding.rerank_url is None
    assert ollama_config.embedding.rerank_model is None
    assert ollama_config.llm is not None
    assert ollama_config.llm.base_url == "http://localhost:11434/v1"
    assert ollama_config.llm.model == "qwen3-coder:30b"

    assert vllm_config.embedding is not None
    assert vllm_config.embedding.base_url == "http://localhost:8000/v1"
    assert vllm_config.embedding.rerank_url == "http://localhost:8001/v1/rerank"
    assert vllm_config.embedding.rerank_model == "Qwen/Qwen3-Reranker-0.6B"
    assert vllm_config.llm is not None
    assert vllm_config.llm.base_url == "http://localhost:8002/v1"
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
    _config, errors = asyncio.run(create_validated_config(args, command))
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


@pytest.mark.parametrize(
    ("llm_id", "provider", "model"),
    [
        ("anthropic", "anthropic", None),
        ("openai-llm", "openai", None),
        ("codex-cli", "codex-cli", None),
        ("claude-code-cli", "claude-code-cli", None),
        # opencode-cli's backend validation requires a provider/model-formatted
        # model, so the preset must carry one (main's catalog omits it).
        ("opencode-cli", "opencode-cli", "opencode/deepseek-v4.1-flash"),
    ],
)
def test_every_supported_llm_preset_parses_through_backend_config(
    tmp_path, clean_environment, llm_id: str, provider: str, model: str | None
) -> None:
    """Every configurator LLM preset must survive Config() validation."""
    config = Config(
        target_dir=tmp_path,
        **_build_chunkhound_config("voyageai", llm_id),
    )

    assert config.llm is not None
    assert config.llm.provider == provider
    if model is not None:
        assert config.llm.model == model


def test_default_research_route_declares_openrouter_api_key_requirement() -> None:
    """The default research route (OpenRouter) must declare the API key it
    needs and emit the matching placeholder."""
    script = """
import {
  DEFAULT_RESEARCH,
  DEFAULT_RETRIEVAL,
  buildChunkhoundConfig,
  findEmbeddingProvider,
  findLlmProvider,
  requirementLabels,
} from './site/src/components/configurator/index.ts';

const llm = findLlmProvider(DEFAULT_RESEARCH);
const embedding = findEmbeddingProvider(DEFAULT_RETRIEVAL);

console.log(JSON.stringify({
  llmId: llm.id,
  requirementIds: llm.requirements.map(({ id }) => id),
  requirementLabels: requirementLabels(llm.requirements),
  apiKey: buildChunkhoundConfig(embedding, llm).llm.api_key,
}));
"""
    rendered = run_tsx_json(script)

    assert rendered["llmId"] == "openrouter"
    assert "openrouter-api-key" in rendered["requirementIds"]
    assert "OpenRouter API key" in rendered["requirementLabels"]
    assert rendered["apiKey"] == "<YOUR_OPENROUTER_API_KEY>"


def test_recommended_route_emits_voyage_4_lite_and_openrouter() -> None:
    defaults = _selection_defaults()
    config = _build_chunkhound_config(
        defaults["defaultRetrieval"], defaults["defaultResearch"]
    )

    assert config["embedding"] == {
        "provider": "voyageai",
        "model": "voyage-4-lite",
        "api_key": "<YOUR_VOYAGE_API_KEY>",
    }
    assert config["llm"] == {
        "provider": "openrouter",
        "model": "google/gemini-3.5-flash",
        "api_key": "<YOUR_OPENROUTER_API_KEY>",
    }


def test_recommended_route_config_round_trips_through_backend_config(
    tmp_path, clean_environment
) -> None:
    defaults = _selection_defaults()
    config = Config(
        target_dir=tmp_path,
        **_build_chunkhound_config(
            defaults["defaultRetrieval"], defaults["defaultResearch"]
        )
    )

    assert config.embedding is not None
    assert config.embedding.model == "voyage-4-lite"
    assert config.llm is not None
    assert config.llm.model == "google/gemini-3.5-flash"
    assert config.llm.provider == "openrouter"


@pytest.mark.parametrize("command", ["index", "research"])
def test_recommended_route_config_passes_cli_validation(
    tmp_path, clean_environment, command: str
) -> None:
    defaults = _selection_defaults()

    errors = _validated_config_errors(
        tmp_path, command, defaults["defaultRetrieval"], defaults["defaultResearch"]
    )

    assert errors == []


@pytest.mark.filterwarnings("ignore::UserWarning:.*configurator.*")
def test_deepseek_llm_configurator_emits_model() -> None:
    config = _load_preset("llmProviders", "deepseek")

    assert config["provider"] == "deepseek"
    assert config["model"] == "deepseek-v4-flash"


@pytest.mark.filterwarnings("ignore::UserWarning:.*configurator.*")
def test_openrouter_config_passes_research_validation(
    tmp_path, clean_environment
) -> None:
    errors = _validated_config_errors(tmp_path, "research", "voyageai", "openrouter")

    assert errors == []


@pytest.mark.filterwarnings("ignore::UserWarning:.*configurator.*")
def test_voyage_4_lite_config_passes_cli_validation(
    tmp_path, clean_environment
) -> None:
    errors = _validated_config_errors(tmp_path, "index", "voyageai", "openrouter")

    assert errors == []


@pytest.mark.filterwarnings("ignore::UserWarning:.*configurator.*")
def test_grok_llm_configurator_emits_model() -> None:
    config = _load_preset("llmProviders", "grok")

    assert config["provider"] == "grok"
    assert config["model"] == "grok-4.3"


@pytest.mark.filterwarnings("ignore::UserWarning:.*configurator.*")
def test_openrouter_llm_configurator_emits_model() -> None:
    config = _load_preset("llmProviders", "openrouter")

    assert config["provider"] == "openrouter"
    assert config["model"] == "google/gemini-3.5-flash"


@pytest.mark.filterwarnings("ignore::UserWarning:.*configurator.*")
def test_orcarouter_llm_configurator_emits_model() -> None:
    config = _load_preset("llmProviders", "orcarouter")

    assert config["provider"] == "orcarouter"
    assert config["model"] == "qwen/qwen3.7-flash"


@pytest.mark.filterwarnings("ignore::UserWarning:.*configurator.*")
def test_gemini_llm_configurator_emits_model() -> None:
    config = _load_preset("llmProviders", "gemini")

    assert config["provider"] == "gemini"
    assert config["model"] == "gemini-3.5-flash"


@pytest.mark.filterwarnings("ignore::UserWarning:.*configurator.*")
def test_deepseek_config_passes_research_validation(
    tmp_path, clean_environment
) -> None:
    errors = _validated_config_errors(tmp_path, "research", "voyageai", "deepseek")

    assert errors == []


@pytest.mark.filterwarnings("ignore::UserWarning:.*configurator.*")
def test_grok_config_passes_research_validation(tmp_path, clean_environment) -> None:
    errors = _validated_config_errors(tmp_path, "research", "voyageai", "grok")

    assert errors == []


@pytest.mark.filterwarnings("ignore::UserWarning:.*configurator.*")
def test_gemini_config_passes_research_validation(tmp_path, clean_environment) -> None:
    errors = _validated_config_errors(tmp_path, "research", "voyageai", "gemini")

    assert errors == []


def test_optional_requirement_renders_opt_in_class() -> None:
    """Gating contract: optional prerequisites opt-in visually, required do not."""
    script = """
import {
  requirements,
  renderRequirements,
} from './site/src/components/configurator/index.ts';

console.log(JSON.stringify({
  voyageKeyOptional: requirements.voyageApiKey.optional ?? false,
  voyageOptOutOptional: requirements.voyageTrainingOptOut.optional ?? false,
  html: renderRequirements([
    requirements.voyageApiKey,
    requirements.voyageTrainingOptOut,
  ]),
}));
"""
    rendered = run_tsx_json(script)

    assert rendered["voyageKeyOptional"] is False
    assert rendered["voyageOptOutOptional"] is True
    items = rendered["html"].split("</li>")
    assert "prerequisite-item-optional" not in items[0]
    assert "prerequisite-item-optional" in items[1]
    assert "VoyageAI API key" in items[0]
    assert "Opt out of VoyageAI training" in items[1]


def test_every_requirement_carries_label_and_docs_url() -> None:
    """Every catalogued requirement links its label to https docs."""
    script = """
import { requirements } from './site/src/components/configurator/index.ts';

console.log(JSON.stringify(Object.values(requirements).map((requirement) => ({
  id: requirement.id,
  label: requirement.label,
  url: requirement.url,
}))));
"""
    rendered = run_tsx_json(script)

    ids = [requirement["id"] for requirement in rendered]
    assert ids and len(ids) == len(set(ids))
    for requirement in rendered:
        assert requirement["label"].strip()
        assert requirement["url"].startswith("https://")


def test_powershell_quoting_covers_paths_with_spaces_and_quotes() -> None:
    """Shell-write contract: spaced paths stay one token, quotes never break."""
    script = """
import {
  quotePosix,
  quotePowerShell,
  jsonWriteScaffold,
  assembleJsonWrite,
} from './site/src/components/configurator/shell-write.ts';

console.log(JSON.stringify({
  spaced: quotePowerShell('my projects/.chunkhound.json'),
  quote: quotePowerShell("it's/.chunkhound.json"),
  home: quotePowerShell('$HOME/.chunkhound.json'),
  dollar: quotePowerShell('$HOME/my $dir/.chunkhound.json'),
  posixSpaced: quotePosix('my projects/.chunkhound.json'),
  posixQuote: quotePosix("it's/.chunkhound.json"),
  posixHome: quotePosix('~/.chunkhound.json'),
  posixSafe: quotePosix('.chunkhound.json'),
  posixHomeSpaced: quotePosix('~/my dir/.chunkhound.json'),
  posixHomeDollar: quotePosix('~/$dir/.chunkhound.json'),
  posixHomeQuote: quotePosix('~/"dir/.chunkhound.json'),
  posixHomeBackslash: quotePosix('~/a\\\\"b/.chunkhound.json'),
  posixWrite: assembleJsonWrite(
    jsonWriteScaffold('my dir/.chunkhound.json', 'posix'),
    '{}',
  ),
  powershellWrite: assembleJsonWrite(
    jsonWriteScaffold('my dir/.chunkhound.json', 'powershell'),
    '{}',
  ),
}));
"""
    rendered = run_tsx_json(script)

    assert rendered["spaced"] == "'my projects/.chunkhound.json'"
    assert rendered["quote"] == "'it''s/.chunkhound.json'"
    assert rendered["home"] == '"$HOME/.chunkhound.json"'
    # Double quotes interpolate `$`: a later `$` under $HOME must be escaped,
    # while the leading `$HOME` itself keeps expanding (that is why the
    # branch double-quotes at all).
    assert rendered["dollar"] == '"$HOME/my `$dir/.chunkhound.json"'
    # Tilde is literal inside any quotes: a safe tilde-lead stays bare (it
    # expands unquoted), while a tilde path needing quoting must be manually
    # expanded to "$HOME/…" and double-quoted.
    assert rendered["posixSpaced"] == "'my projects/.chunkhound.json'"
    assert rendered["posixQuote"] == "'it'\\''s/.chunkhound.json'"
    assert rendered["posixHome"] == "~/.chunkhound.json"
    assert rendered["posixHomeSpaced"] == '"$HOME/my dir/.chunkhound.json"'
    # The body is sliced past `~/`, so the `$` escape must cover its FIRST
    # character too — `(?!^)` would leave `$dir` unescaped and interpolating.
    assert rendered["posixHomeDollar"] == '"$HOME/\\$dir/.chunkhound.json"'
    # Backslash must be escaped FIRST: a body backslash before `"` would
    # otherwise render as `\\"` — the `\\` pair consumes itself inside the
    # double quotes and the bare `"` closes the string early.
    assert rendered["posixHomeQuote"] == '"$HOME/\\"dir/.chunkhound.json"'
    assert rendered["posixHomeBackslash"] == '"$HOME/a\\\\\\"b/.chunkhound.json"'
    # Shell-safe paths stay bare: quoting them adds noise with zero benefit.
    assert rendered["posixSafe"] == ".chunkhound.json"
    assert (
        "cat > 'my dir/.chunkhound.json' <<'CHUNKHOUND_EOF'"
        in rendered["posixWrite"]
    )
    assert "mkdir -p 'my dir'" in rendered["posixWrite"]
    powershell_write = rendered["powershellWrite"]
    assert "'@ | Set-Content -Path 'my dir/.chunkhound.json'" in powershell_write
    assert "New-Item -ItemType Directory -Force -Path 'my dir'" in powershell_write


def test_config_comment_states_provider_independence() -> None:
    """The copied artifact must say the embedding and LLM providers are
    independent — the root-cause fix for the coupling the UI implies."""
    script = """
import {
  buildCompactConfiguratorOutput,
  embeddingProviders,
  llmProviders,
} from './site/src/components/configurator/index.ts';

const embedding = embeddingProviders.find((provider) => provider.id === 'voyageai');
const llm = llmProviders.find((provider) => provider.id === 'openrouter');
if (!embedding || !llm) throw new Error('missing provider');
console.log(JSON.stringify(buildCompactConfiguratorOutput(embedding, llm, 'cursor')));
"""
    rendered = run_tsx_json(script)

    assert "Embedding and LLM providers are independent" in rendered["copy"]


def test_recommended_pill_marks_voyageai_only() -> None:
    """Gateway choice is not a recommendation (OpenRouter is merely the more
    common default); the recommended default is the retrieval provider."""
    script = """
import {
  embeddingProviders,
  llmProviders,
} from './site/src/components/configurator/index.ts';
const recommendedIds = (list) =>
  list.filter((provider) => provider.recommended).map((provider) => provider.id);
console.log(JSON.stringify({
  embedding: recommendedIds(embeddingProviders),
  llm: recommendedIds(llmProviders),
}));
"""
    rendered = run_tsx_json(script)

    assert rendered["embedding"] == ["voyageai"]
    assert rendered["llm"] == []


def test_llm_providers_recommend_their_cheapest_fastest_model() -> None:
    """Every recommendation names the provider's cheapest/fastest model, and
    the emitted config uses that model, so the default is never a compromise."""
    script = """
import { llmProviders } from './site/src/components/configurator/index.ts';
console.log(JSON.stringify(llmProviders.map((provider) => ({
  id: provider.id,
  model: provider.config.model ?? null,
  recommendation: provider.recommendation ?? null,
}))));
"""
    rendered = run_tsx_json(script)
    by_id = {row["id"]: row for row in rendered}

    expected_models = {
        "openrouter": "google/gemini-3.5-flash",
        "orcarouter": "qwen/qwen3.7-flash",
        "gemini": "gemini-3.5-flash",
        "grok": "grok-4.3",
        "deepseek": "deepseek-v4-flash",
        "ollama-llm": "qwen3-coder:30b",
        "vllm-llm": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "opencode-cli": "opencode/deepseek-v4.1-flash",
    }
    for provider_id, model in expected_models.items():
        assert by_id[provider_id]["model"] == model, provider_id
        assert by_id[provider_id]["recommendation"], provider_id

    # Providers whose config omits `model` defer to the product default.
    for provider_id in ("anthropic", "openai-llm"):
        assert by_id[provider_id]["model"] is None

    # The gateway default's rationale must state the default is not a compromise.
    assert "not a compromise" in by_id["openrouter"]["recommendation"]


def test_rerank_format_parity_between_ts_and_python() -> None:
    """The TS guard must accept exactly the backend's rerank formats minus 'auto'.

    'auto' is a backend runtime sentinel, so the configurator excludes it. The
    Python set is read from the field annotation (not source text) and the TS
    guard is executed, so renames or reordering on either side cannot silently
    desync them.
    """
    py_formats = set(get_args(EmbeddingConfig.model_fields["rerank_format"].annotation))
    candidates = sorted(py_formats | {"bogus"})
    script = (
        "const { isRerankFormat } = await import("
        "'./site/src/components/configurator/rerank-validate.ts');\n"
        f"const candidates = {json.dumps(candidates)};\n"
        "console.log(JSON.stringify(candidates.filter(isRerankFormat)));"
    )
    accepted = set(run_tsx_json(script))

    assert accepted == py_formats - {"auto"}, (
        f"Rerank formats desynced: Python accepts {py_formats}, "
        f"the TS guard accepts {accepted}."
    )
