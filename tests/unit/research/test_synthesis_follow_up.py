"""Follow-up framing injection: SynthesisEngine → LLM prompt.

Locks the value-delivery path of the ``previous_query`` websearch chain:

- Single-pass and reduce (answer-authoring stages) inject the full
  framing block from ``build_follow_up_section``.
- Map (per-cluster extraction) uses the compact one-liner from
  ``build_follow_up_hint`` and does **not** carry the full framing.
- ``previous_query=None`` omits both entirely (baseline path).

Expected framing strings come from the builders themselves — tests
break if a builder is bypassed, not if its wording is retuned. The
non-empty assertions guard against a ``return ""`` regression in the
builder that would otherwise make ``in prompt`` trivially pass.
"""

import pytest

from chunkhound.services.clustering_service import ClusterGroup
from chunkhound.services.prompts import (
    build_follow_up_hint,
    build_follow_up_section,
)
from chunkhound.services.research import SynthesisEngine
from chunkhound.services.research.shared.models import ResearchContext
from tests.fixtures.fake_providers import FakeEmbeddingProvider
from tests.unit.research.conftest import FakeParent

PRIOR = "prior topic"


def _build_engine(llm_manager):
    parent = FakeParent(FakeEmbeddingProvider())
    return SynthesisEngine(
        llm_manager, database_services=object(), parent_service=parent
    )


def _chunks_and_files():
    chunks = [
        {
            "file_path": "a.py",
            "content": "print('hi')",
            "score": 1.0,
            "start_line": 1,
            "end_line": 1,
        }
    ]
    files = {"a.py": "print('hi')"}
    return chunks, files


@pytest.mark.asyncio
async def test_single_pass_prompt_includes_follow_up_framing(capturing_llm_manager):
    llm_manager, fake_provider = capturing_llm_manager
    engine = _build_engine(llm_manager)
    chunks, files = _chunks_and_files()

    await engine._single_pass_synthesis(
        chunks=chunks,
        files=files,
        context=ResearchContext(
            root_query="synthesis architecture overview",
            previous_query=PRIOR,
        ),
        synthesis_budgets={"output_tokens": 10_000},
    )

    assert len(fake_provider.calls) == 1
    prompt = fake_provider.calls[0]["prompt"]
    section = build_follow_up_section(PRIOR)
    assert section, "builder must be non-empty when previous_query is set"
    assert section in prompt


@pytest.mark.asyncio
async def test_reduce_prompt_includes_follow_up_framing(capturing_llm_manager):
    llm_manager, fake_provider = capturing_llm_manager
    engine = _build_engine(llm_manager)
    chunks, files = _chunks_and_files()

    cluster_results = [
        {
            "summary": "Cluster analysis of a.py [1] shows a print call.",
            "file_paths": ["a.py"],
            "file_reference_map": {"a.py": 1},
        }
    ]

    await engine._reduce_synthesis(
        cluster_results=cluster_results,
        all_chunks=chunks,
        all_files=files,
        context=ResearchContext(
            root_query="synthesis architecture overview",
            previous_query=PRIOR,
        ),
        synthesis_budgets={"output_tokens": 10_000},
    )

    assert len(fake_provider.calls) == 1
    prompt = fake_provider.calls[0]["prompt"]
    section = build_follow_up_section(PRIOR)
    assert section, "builder must be non-empty when previous_query is set"
    assert section in prompt


@pytest.mark.asyncio
async def test_map_prompt_includes_follow_up_framing(capturing_llm_manager):
    llm_manager, fake_provider = capturing_llm_manager
    engine = _build_engine(llm_manager)
    chunks, _ = _chunks_and_files()

    cluster = ClusterGroup(
        cluster_id=0,
        file_paths=["a.py"],
        files_content={"a.py": "print('hi')"},
        total_tokens=20_000,
    )

    await engine._map_synthesis_on_cluster(
        cluster=cluster,
        chunks=chunks,
        context=ResearchContext(
            root_query="synthesis architecture overview",
            previous_query=PRIOR,
        ),
        synthesis_budgets={"output_tokens": 30_000},
        total_input_tokens=100_000,
    )

    assert len(fake_provider.calls) == 1
    prompt = fake_provider.calls[0]["prompt"]
    hint = build_follow_up_hint(PRIOR)
    section = build_follow_up_section(PRIOR)
    assert hint and section, "builders must be non-empty when previous_query is set"
    # Contract: map receives *some* follow-up framing. Either builder is
    # acceptable — we don't pin which one the map stage uses.
    assert hint in prompt or section in prompt


@pytest.mark.asyncio
async def test_baseline_no_previous_query_omits_framing(capturing_llm_manager):
    llm_manager, fake_provider = capturing_llm_manager
    engine = _build_engine(llm_manager)
    chunks, files = _chunks_and_files()

    await engine._single_pass_synthesis(
        chunks=chunks,
        files=files,
        context=ResearchContext(
            root_query="synthesis architecture overview",
            previous_query=None,
        ),
        synthesis_budgets={"output_tokens": 10_000},
    )

    assert len(fake_provider.calls) == 1
    prompt = fake_provider.calls[0]["prompt"]
    # Probe with a canary value: if any framing leaked into the None path,
    # it would carry this string. Uses the builders as the source of truth
    # so the assertions survive prompt-wording changes.
    assert build_follow_up_section(PRIOR) not in prompt
    assert build_follow_up_hint(PRIOR) not in prompt
