"""Structural contract for the architecture SSOT (`site/src/lib/architecture.ts`).

The /docs/architecture explanation page and its diagrams render from this module,
so its shape is a real, user-visible contract: five pipeline stages in phase
order (Index → Exploration → Synthesis), every research phase carrying an id,
label and caption, and every system layer listing nodes shaped as
`{title}`. A separate page test asserts the rendered prose; this guards
the data the page consumes so a prose page cannot silently drift from the model.
"""

from __future__ import annotations

from functools import lru_cache

from tests.site.tsx_runner import run_tsx_json

_DATA = """
const architecture = await import('./site/src/lib/architecture.ts');
console.log(JSON.stringify({
  stages: architecture.STAGES,
  researchPhases: architecture.RESEARCH_PHASES,
  systemLayers: architecture.SYSTEM_LAYERS,
  llmRoles: architecture.LLM_ROLES,
  indexPhases: architecture.INDEX_PHASES,
}));
"""

# One TS import/execution per session: the module is static data.
@lru_cache(maxsize=1)
def _load() -> dict:
    return run_tsx_json(_DATA)


def test_stages_are_five_in_phase_order() -> None:
    stages = _load()["stages"]

    assert len(stages) == 5
    phases = [stage["phase"] for stage in stages]
    # Phases only move forward: Index before Exploration before Synthesis.
    assert phases == sorted(phases, key=("Index", "Exploration", "Synthesis").index)
    assert phases[0] == "Index"
    assert phases[-1] == "Synthesis"
    assert set(phases) == {"Index", "Exploration", "Synthesis"}
    for stage in stages:
        assert stage["icon"].startswith("ph-"), stage
        assert stage["title"].strip(), stage


def test_every_research_phase_has_id_label_caption() -> None:
    phases = _load()["researchPhases"]

    assert [phase["id"] for phase in phases] == ["exploration", "synthesis"]
    for phase in phases:
        assert phase["label"].strip()
        assert phase["caption"].strip()
        assert phase["mechanisms"], phase


def test_system_layers_expose_node_shape() -> None:
    layers = _load()["systemLayers"]

    assert len(layers) == 4
    for layer in layers:
        assert layer["id"].strip()
        assert layer["label"].strip()
        assert layer["caption"].strip()
        assert layer["nodes"], layer
        for node in layer["nodes"]:
            assert node["title"].strip(), layer


def test_llm_roles_and_index_phases_keep_their_fields() -> None:
    data = _load()

    assert len(data["llmRoles"]) == 2
    for role in data["llmRoles"]:
        assert role["name"].strip()
        assert role["calls"].strip()

    assert len(data["indexPhases"]) == 5
    for phase in data["indexPhases"]:
        assert phase["title"].strip()
        assert phase["detail"].strip()
        assert phase["language"] in {"Rust", "Python", "Rust + Python"}
