"""Drift guards for the generated configurator provider catalog.

The backend owns provider identity; the site derives its catalog with
scripts/generate_site_provider_catalog.py. These guards fail when the committed
file is stale or when the hand-authored presets stop covering the backend.
Every failure names the regeneration command.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from pydantic import SecretStr, ValidationError

import scripts.generate_site_provider_catalog as _generator
from chunkhound.core.config.llm_config import LLMConfig, LLMProviderLiteral
from tests.site.tsx_runner import run_tsx_json

# The generator is imported in-process: it is side-effect-free (logic in pure
# functions, CLI behind ``__main__``), avoiding the locale code-page hazard the
# site-test subprocess guard exists for.

_DRIFT_SCRIPT = """
import {
  catalogDrift,
  embeddingProviders,
  llmProviders,
} from './site/src/components/configurator/index.ts';
console.log(JSON.stringify(catalogDrift(llmProviders, embeddingProviders)));
"""

_SCENARIO_SCRIPT = """
import {
  catalogDrift,
  embeddingProviders,
  llmProviders,
} from './site/src/components/configurator/index.ts';

const without = (list, id) => list.filter((p) => p.config.provider !== id);
const ghost = { ...llmProviders[0], id: 'ghost' };
ghost.config = { ...ghost.config, provider: '__ghost__' };
const unpin = (p) => ({ ...p, config: { ...p.config, model: '' } });
const unpinned = llmProviders.map((p) =>
  p.config.provider === '__UNPINNED__' ? unpin(p) : p);
console.log(JSON.stringify({
  missingLlm: catalogDrift(without(llmProviders, '__LLM__'), embeddingProviders),
  missingEmbedding: catalogDrift(llmProviders, without(embeddingProviders, '__EMB__')),
  unknown: catalogDrift([...llmProviders, ghost], embeddingProviders),
  unpinned: catalogDrift(unpinned, embeddingProviders),
}));
"""

_BASE_URLS = (
    None,  # unset: the default official endpoint
    "https://api.openai.com/v1",
    "https://api.openai.com.evil.com/v1",  # lookalike must not pass as official
    "http://localhost:1234/v1",
)
_DRIFT_ONE = """
import { catalogDrift } from './site/src/components/configurator/index.ts';
const one = (s) => {
  const config = { provider: s.provider, model: '', base_url: s.baseUrl };
  return catalogDrift([{ id: s.id, config }], [])
    .filter((l) => l.includes(`preset '${s.id}'`));
};
console.log(JSON.stringify(Object.fromEntries(
  __SCENARIOS__.map((s) => [s.id, one(s)]))));
"""


def _preset_drift(scenarios: list[dict]) -> dict[str, list[str]]:
    return run_tsx_json(_DRIFT_ONE.replace("__SCENARIOS__", json.dumps(scenarios)))


def _backend_requires_model(provider: LLMProviderLiteral, base_url: str | None) -> bool:
    """LLMConfig verdict for an unpinned model, minus API-key noise."""
    try:
        config = LLMConfig(
            provider=provider, model=None, base_url=base_url, api_key=SecretStr("test")
        )
    except ValidationError as error:
        # Some providers reject a missing model eagerly; others must surface.
        if "model" not in str(error):
            raise
        return True
    missing = config.get_missing_config_for_roles(
        ("utility", "synthesis", "map_hyde", "autodoc_cleanup")
    )
    return any("explicit model selection" in entry for entry in missing)


def test_custom_endpoint_without_model_is_flagged() -> None:
    """A preset on a custom OpenAI endpoint with no model must drift."""
    scenario = {
        "id": "lmstudio",
        "provider": "openai",
        "baseUrl": "http://localhost:1234/v1",
    }
    lines = _preset_drift([scenario])["lmstudio"]
    assert any(
        "must pin a model" in line and scenario["baseUrl"] in line for line in lines
    ), lines


def test_model_pin_guard_matches_backend_for_every_preset() -> None:
    """The TS mirror and LLMConfig must agree on every preset and endpoint."""
    facts = json.loads(_generator.render_catalog())["llm"]
    assert all("customEndpointRequiresModel" in fact for fact in facts), facts
    scenarios = [
        {"id": f"{f['id']}-{i}", "provider": f["id"], "baseUrl": url}
        for f in facts
        for i, url in enumerate(_BASE_URLS)
    ]
    lines = _preset_drift(scenarios)
    for scenario in scenarios:
        flagged = any("must pin a model" in line for line in lines[scenario["id"]])
        expected = _backend_requires_model(scenario["provider"], scenario["baseUrl"])
        assert flagged == expected, scenario


def test_committed_catalog_matches_backend_registry() -> None:
    """The checked-in file must equal a fresh generation from the SSOT."""
    committed = _generator.OUTPUT_PATH.read_text(encoding="utf-8")
    assert committed == _generator.render_catalog(), (
        "provider-catalog.gen.json is out of date with the backend provider "
        "catalog.\nRegenerate and commit it:\n  " + _generator.REGENERATE
    )


def test_check_exits_clean_when_catalog_is_fresh(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The documented ``--check`` invocation must succeed and print nothing."""
    assert _generator.main(["--check"]) == 0
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("state", ["stale", "missing"])
def test_check_fails_when_catalog_is_stale(
    state: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Stale or absent catalogs fail ``--check`` with the fix command; never rewrite."""
    target = tmp_path / "provider-catalog.gen.json"
    if state == "stale":
        target.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(_generator, "OUTPUT_PATH", target)

    assert _generator.main(["--check"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert _generator.REGENERATE in captured.err and str(target) in captured.err
    untouched = state == "missing" and not target.exists()
    assert untouched or target.read_text(encoding="utf-8") == "{}\n"


def test_catalog_drift_emits_actionable_lines() -> None:
    """Each disagreement yields a per-provider line naming the id and the fix."""
    catalog = json.loads(_generator.render_catalog())
    requires_model = next(f for f in catalog["llm"] if f["requiresModel"])
    llm_id, emb_id = catalog["llm"][0]["id"], catalog["embedding"][0]
    script = (
        _SCENARIO_SCRIPT.replace("__LLM__", llm_id)
        .replace("__EMB__", emb_id)
        .replace("__UNPINNED__", requires_model["id"])
    )

    drift = run_tsx_json(script)

    expected = [
        ("missingLlm", llm_id, "no configurator preset"),
        ("missingEmbedding", emb_id, "no configurator preset"),
        ("unknown", "__ghost__", "targets unknown LLM provider"),
        ("unpinned", requires_model["id"], "no backend default model"),
    ]
    for case, offender, discriminator in expected:
        assert any(offender in s and discriminator in s for s in drift[case]), case


def test_unpinned_opencode_cli_preset_is_flagged() -> None:
    """opencode-cli has no backend default model: an unpinned preset must drift."""
    catalog = json.loads(_generator.render_catalog())
    script = (
        _SCENARIO_SCRIPT.replace("__LLM__", catalog["llm"][0]["id"])
        .replace("__EMB__", catalog["embedding"][0])
        .replace("__UNPINNED__", "opencode-cli")
    )

    drift = run_tsx_json(script)["unpinned"]
    assert any("opencode-cli" in s and "no backend default" in s for s in drift), drift


def test_presets_cover_every_backend_provider() -> None:
    """No backend provider ships unpickable; no preset targets a dead id; every
    provider without a backend default model carries a pinned model."""
    drift = cast(list[str], run_tsx_json(_DRIFT_SCRIPT))

    assert drift == [], (
        "The configurator presets no longer match the backend provider catalog:\n- "
        + "\n- ".join(drift)
        + "\n\nAfter fixing the presets, regenerate the derived catalog:\n  "
        + _generator.REGENERATE
    )
