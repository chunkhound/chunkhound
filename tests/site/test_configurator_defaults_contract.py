"""Cross-checks between the configurator's presets and the product/docs.

The configurator owns its model recommendations and must write them
explicitly: an emitted config never relies on the provider's runtime fallback,
so the site can move to newer recommended models without changing the behavior
of existing configs. Every "configurator defaults to X" claim in the docs has
to equal the preset the configurator actually emits. Kept out of
test_configurator_contract.py so this coupling is checked without pulling in the
heavy config-validation imports.
"""

from __future__ import annotations

import re

from chunkhound.core.config.provider_registry import (
    OPENAI_COMPATIBLE_PROVIDERS,
    OutputLimitCapability,
)
from chunkhound.core.constants import (
    VOYAGE_DEFAULT_MODEL,
    VOYAGE_DEFAULT_RERANK_MODEL,
)
from tests.site.tsx_runner import ROOT, run_tsx_json

_LLM_MODELS_SCRIPT = """
import { llmProviders } from './site/src/components/configurator/index.ts';
console.log(JSON.stringify(Object.fromEntries(
  llmProviders
    .filter((provider) => provider.config.model)
    .map((provider) => [provider.config.provider, provider.config.model])
)));
"""

_PROVIDER_IDS_SCRIPT = """
import { llmProviders } from './site/src/components/configurator/index.ts';
console.log(JSON.stringify(llmProviders.map((provider) => provider.config.provider)));
"""

# '| Name | `provider` | ... (configurator defaults to `model`) ...' rows in the
# LLM provider table; group 1 is the config value, group 2 the preset model.
_CONFIGURATOR_DEFAULT = re.compile(
    r"^\|\s*[^|]+\|\s*`([^`]+)`\s*\|.*?\(configurator defaults to `([^`]+)`\)",
    re.MULTILINE,
)

# The two places configuration.md promises the VoyageAI recommendation: the
# Embedding Providers table row and the fallback prose. Both must name the same
# pair the configurator emits.
_VOYAGE_TABLE_RECOMMENDATION = re.compile(
    r"^\|\s*VoyageAI\s*\|.*?configurator emits `([^`]+)`\s*\+\s*`([^`]+)`",
    re.MULTILINE,
)
_VOYAGE_PROSE_RECOMMENDATION = re.compile(
    r"recommends and emits `([^`]+)`\s*\+\s*`([^`]+)`"
)

_RECOMMENDED_VOYAGE_SCRIPT = """
import {
  RECOMMENDED_VOYAGE_EMBEDDING_MODEL,
  RECOMMENDED_VOYAGE_RERANK_MODEL,
} from './site/src/components/configurator/index.ts';
console.log(JSON.stringify({
  embedding: RECOMMENDED_VOYAGE_EMBEDDING_MODEL,
  rerank: RECOMMENDED_VOYAGE_RERANK_MODEL,
}));
"""


_RESEARCH_ROUTE_SCRIPT = """
import {
  CATALOG_VERIFIED,
  RESEARCH_MODEL,
  llmProviders,
} from './site/src/components/configurator/index.ts';
const preset = (id) => {
  const provider = llmProviders.find((entry) => entry.id === id);
  if (!provider) throw new Error(`missing ${id} preset`);
  return provider;
};
console.log(JSON.stringify({
  model: RESEARCH_MODEL,
  verified: CATALOG_VERIFIED,
  vercelModel: preset('vercel').config.model,
  openrouterModel: preset('openrouter').config.model,
  vercel: preset('vercel').recommendation,
  openrouter: preset('openrouter').recommendation,
}));
"""


def _recommended_voyage_models() -> dict[str, str]:
    return run_tsx_json(_RECOMMENDED_VOYAGE_SCRIPT)


def _changelog() -> str:
    return (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")


def _configuration_docs() -> str:
    return (ROOT / "site" / "src" / "pages" / "docs" / "configuration.md").read_text(
        encoding="utf-8"
    )


def _llm_models_by_provider() -> dict[str, str]:
    return run_tsx_json(_LLM_MODELS_SCRIPT)


def test_voyageai_preset_emits_explicit_recommended_models() -> None:
    """The configurator writes both VoyageAI models explicitly so a generated
    config never depends on the provider's runtime fallback. The emitted values
    are the site's own recommendation constants — one source, so a copied
    string cannot drift from the preset."""
    script = """
import {
  RECOMMENDED_VOYAGE_EMBEDDING_MODEL,
  RECOMMENDED_VOYAGE_RERANK_MODEL,
  embeddingProviders,
} from './site/src/components/configurator/index.ts';
const provider = embeddingProviders.find(({ id }) => id === 'voyageai');
if (!provider) throw new Error('missing voyageai preset');
console.log(JSON.stringify({
  config: provider.config,
  recommendedEmbedding: RECOMMENDED_VOYAGE_EMBEDDING_MODEL,
  recommendedRerank: RECOMMENDED_VOYAGE_RERANK_MODEL,
}));
"""
    rendered = run_tsx_json(script)
    config = rendered["config"]

    assert config["provider"] == "voyageai"
    assert config["model"] == rendered["recommendedEmbedding"]
    assert config["model"], "model must be explicit, not left to the runtime fallback"
    assert config["rerank_model"] == rendered["recommendedRerank"]
    assert config["rerank_model"], "rerank_model must be explicit"


def test_runtime_fallback_differs_from_configurator_recommendation() -> None:
    """The backend's frozen fallback must not track the site's recommendation:
    existing configs keep the fallback while new users get the recommended
    models explicitly. Both sides are read from their single source."""
    recommended = _recommended_voyage_models()

    assert VOYAGE_DEFAULT_MODEL != recommended["embedding"]
    assert VOYAGE_DEFAULT_RERANK_MODEL != recommended["rerank"]


def test_configuration_docs_promise_the_voyageai_recommendation() -> None:
    """Both VoyageAI promises in configuration.md — the provider table row and
    the fallback prose — must name the exact models the configurator emits.
    Renaming one place alone would tell users a config they get is not the one
    they read about."""
    docs = _configuration_docs()
    recommended = _recommended_voyage_models()
    expected = (recommended["embedding"], recommended["rerank"])

    table = _VOYAGE_TABLE_RECOMMENDATION.search(docs)
    prose = _VOYAGE_PROSE_RECOMMENDATION.search(docs)
    assert table, "the VoyageAI table row must state the configurator's models"
    assert prose, "the VoyageAI fallback prose must state the recommended models"
    assert table.groups() == expected, (
        f"configuration.md's VoyageAI row says {table.groups()}, "
        f"the configurator emits {expected}"
    )
    assert prose.groups() == expected, (
        f"configuration.md's VoyageAI prose says {prose.groups()}, "
        f"the configurator emits {expected}"
    )


def test_configuration_docs_list_every_omission_capable_provider() -> None:
    """configuration.md names the providers whose canonical endpoints omit the
    output cap. The registry owns that fact, so a new SUPPORTED entry must not
    leave the prose list stale (Vercel and Requesty had drifted)."""
    supported = {
        name
        for name, spec in OPENAI_COMPATIBLE_PROVIDERS.items()
        if spec.output_limit_omission is OutputLimitCapability.SUPPORTED
    }
    sentence = next(
        line
        for line in _configuration_docs().splitlines()
        if "authoritatively support omission" in line
    ).lower()
    missing = [name for name in sorted(supported) if name.lower() not in sentence]
    assert not missing, f"docs omission list misses providers: {missing}"


def test_every_registry_provider_has_a_configurator_preset() -> None:
    """Every OpenAI-compatible provider the backend registers must be
    selectable in the configurator; otherwise a provider ships with no way to
    pick it, and the two catalogs drift silently."""
    catalog_providers = set(run_tsx_json(_PROVIDER_IDS_SCRIPT))
    missing = set(OPENAI_COMPATIBLE_PROVIDERS) - catalog_providers
    assert not missing, (
        f"configurator has no preset for registered providers: {sorted(missing)}"
    )


def test_configuration_docs_match_configurator_llm_defaults() -> None:
    """configuration.md's 'configurator defaults to X' claims must equal the
    presets the configurator emits for that provider."""
    docs = _configuration_docs()
    documented = dict(_CONFIGURATOR_DEFAULT.findall(docs))
    assert documented, "no '(configurator defaults to ...)' rows found"

    presets = _llm_models_by_provider()

    for provider, model in documented.items():
        assert provider in presets, f"docs default for unknown provider {provider}"
        assert presets[provider] == model, (
            f"configuration.md says {provider} defaults to {model}, but the "
            f"configurator emits {presets[provider]}"
        )


def test_research_route_facts_stay_in_sync_across_docs_and_changelog() -> None:
    """The research-model pin lives in four files. constants.ts owns the ID and
    the verification month; both gateway presets interpolate them, and
    configuration.md plus CHANGELOG.md carry guarded copies. Anchored patterns
    keep an unrelated row from satisfying the promise, so a later model change
    cannot leave the docs silently stale."""
    facts = run_tsx_json(_RESEARCH_ROUTE_SCRIPT)
    model = facts["model"]
    verified = facts["verified"]

    for provider_id in ("vercel", "openrouter"):
        assert facts[f"{provider_id}Model"] == model, provider_id
        assert model in facts[provider_id], provider_id

    section = (
        _configuration_docs()
        .split("Vercel AI Gateway is a first-class", 1)[1]
        .split("### Local OpenAI-compatible servers", 1)[0]
    )
    assert f"`{model}`" in section
    assert f"catalogs checked {verified}" in section
    # The per-model cap workaround must not return: the gateway omits the cap.
    assert "output_limit_fallback" not in section

    changelog = _changelog()
    assert f"`{model}`" in changelog
    assert f"re-verified against live catalogs in {verified}" in changelog
