"""Red-first contracts for invalid citation neutralization before delivery.

Invariant under test: no delivered answer (single-pass or reduce), and no map
summary fed into reduce, may contain an ``[N]`` reference that is absent from
the reference map used to build its Sources.
"""

from __future__ import annotations

from typing import Any

import pytest
from loguru import logger as _loguru_logger

from chunkhound.llm_manager import LLMManager
from chunkhound.services.clustering_service import ClusterGroup
from chunkhound.services.research import SynthesisEngine
from chunkhound.services.research.shared.citation_manager import CitationManager
from chunkhound.services.research.shared.models import ResearchContext
from tests.fixtures.fake_providers import FakeEmbeddingProvider, FakeLLMProvider
from tests.unit.research.conftest import FakeParent

_LONG_ENOUGH = (
    "This synthesized answer is deliberately long enough to satisfy the "
    "production synthesis minimum-length validation before delivery. "
)


class _RecordingFakeLLMProvider(FakeLLMProvider):
    """Fake provider that records every prompt it receives."""

    def __init__(self, responses: dict[str, str]):
        super().__init__(responses=responses)
        self.calls: list[str] = []

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ):
        self.calls.append(prompt)
        return await super().complete(
            prompt,
            system=system,
            max_completion_tokens=max_completion_tokens,
            timeout=timeout,
        )


def _engine_with(
    provider: FakeLLMProvider, monkeypatch: pytest.MonkeyPatch
) -> SynthesisEngine:
    """Build a real SynthesisEngine wired to a scripted fake provider."""

    def _fake_create_provider(self: LLMManager, config: dict[str, Any]):  # noqa: ANN001
        return provider

    monkeypatch.setattr(LLMManager, "_create_provider", _fake_create_provider)
    provider_config = {
        "provider": "fake",
        "model": "fake-gpt",
        "output_limits_enabled": True,
    }
    llm_manager = LLMManager(provider_config, provider_config)
    parent = FakeParent(FakeEmbeddingProvider())
    return SynthesisEngine(
        llm_manager, database_services=object(), parent_service=parent
    )


def _chunks() -> list[dict[str, Any]]:
    return [
        {
            "file_path": "a.py",
            "content": "def f():\n    return 1\n",
            "start_line": 1,
            "end_line": 2,
        }
    ]


def _files() -> dict[str, str]:
    return {"a.py": "def f():\n    return 1\n"}


class TestNeutralizeInvalidCitations:
    """Unit contracts for CitationManager.neutralize_invalid_citations."""

    def test_removes_only_invalid_markers_and_keeps_prose(self) -> None:
        manager = CitationManager()

        cleaned, invalid = manager.neutralize_invalid_citations(
            "Alpha [1] uses beta [2], while gamma [999] is invented.",
            {"a.py": 1, "b.py": 2},
        )

        assert invalid == [999]
        assert cleaned == "Alpha [1] uses beta [2], while gamma  is invented."

    def test_valid_citations_are_untouched(self) -> None:
        manager = CitationManager()
        text = "Valid [1] and [2] only."

        cleaned, invalid = manager.neutralize_invalid_citations(
            text, {"a.py": 1, "b.py": 2}
        )

        assert cleaned == text
        assert invalid == []

    def test_no_citations_is_a_noop(self) -> None:
        manager = CitationManager()
        text = "No references here."

        cleaned, invalid = manager.neutralize_invalid_citations(text, {"a.py": 1})

        assert cleaned == text
        assert invalid == []

    def test_empty_reference_map_drops_all_markers(self) -> None:
        manager = CitationManager()

        cleaned, invalid = manager.neutralize_invalid_citations(
            "Drop [1] and [12] here.", {}
        )

        assert "[1]" not in cleaned
        assert "[12]" not in cleaned
        assert invalid == [1, 12]

    def test_is_idempotent(self) -> None:
        manager = CitationManager()

        cleaned, invalid = manager.neutralize_invalid_citations(
            "Keep [1], drop [42].", {"a.py": 1}
        )
        again, invalid_again = manager.neutralize_invalid_citations(
            cleaned, {"a.py": 1}
        )

        assert invalid == [42]
        assert again == cleaned
        assert invalid_again == []


class TestSynthesisEngineNeutralization:
    """Engine-level contracts for each stage that can inject citations."""

    @pytest.mark.asyncio
    async def test_single_pass_drops_invalid_citations_before_footer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        answer = _LONG_ENOUGH + "The documented flow [1] is real, but [456] is not."
        provider = FakeLLMProvider(responses={"neutralize-single-pass": answer})
        engine = _engine_with(provider, monkeypatch)

        result = await engine._single_pass_synthesis(
            chunks=_chunks(),
            files=_files(),
            context=ResearchContext(root_query="NEUTRALIZE-SINGLE-PASS"),
            synthesis_budgets={"output_tokens": 30_000},
        )

        assert "documented flow [1]" in result
        assert "[456]" not in result
        assert "## Sources" in result

    @pytest.mark.asyncio
    async def test_single_pass_logs_neutralized_citations(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list[str] = []
        sink_id = _loguru_logger.add(
            lambda message: captured.append(message),
            level="WARNING",
            format="{message}",
        )
        try:
            answer = _LONG_ENOUGH + "Invented [456] reference."
            provider = FakeLLMProvider(responses={"neutralize-log": answer})
            engine = _engine_with(provider, monkeypatch)

            await engine._single_pass_synthesis(
                chunks=_chunks(),
                files=_files(),
                context=ResearchContext(root_query="NEUTRALIZE-LOG"),
                synthesis_budgets={"output_tokens": 30_000},
            )
        finally:
            _loguru_logger.remove(sink_id)

        assert any(
            "Neutralized" in message and "456" in message for message in captured
        )

    @pytest.mark.asyncio
    async def test_map_summary_is_neutralized_before_remap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = FakeLLMProvider(
            responses={
                "analyze the following sources": "Cluster analysis [1] plus [999]."
            }
        )
        engine = _engine_with(provider, monkeypatch)
        cluster = ClusterGroup(
            cluster_id=0,
            file_paths=["a.py"],
            files_content=_files(),
            total_tokens=10,
        )

        result = await engine._map_synthesis_on_cluster(
            cluster=cluster,
            chunks=_chunks(),
            context=ResearchContext(root_query="map neutralization"),
            synthesis_budgets={"output_tokens": 30_000},
            total_input_tokens=10,
        )

        assert result["summary"] == "Cluster analysis [1] plus ."
        assert result["file_reference_map"] == {"a.py": 1}

    @pytest.mark.asyncio
    async def test_invalid_cluster_refs_never_reach_reduce_or_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        answer = _LONG_ENOUGH + "Integrated answer cites [1] and invents [999]."
        provider = _RecordingFakeLLMProvider({"different code clusters": answer})
        engine = _engine_with(provider, monkeypatch)
        cluster_results = [
            {
                "cluster_id": 0,
                "summary": "Cluster view [1] and stale [7].",
                "sources": [],
                "file_paths": ["a.py"],
                "file_reference_map": {"a.py": 1},
            }
        ]

        final = await engine._reduce_synthesis(
            cluster_results,
            _chunks(),
            _files(),
            ResearchContext(root_query="reduce neutralization"),
            {"output_tokens": 30_000},
        )

        reduce_prompt = provider.calls[-1]
        assert "Cluster view [1]" in reduce_prompt
        assert "[7]" not in reduce_prompt
        assert "Integrated answer cites [1]" in final
        assert "[999]" not in final
        assert CitationManager().validate_citation_references(final, {"a.py": 1}) == []

    @pytest.mark.asyncio
    async def test_reduce_preserves_remapped_valid_citations(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = _RecordingFakeLLMProvider(
            {"different code clusters": _LONG_ENOUGH + "Final cites [1]."}
        )
        engine = _engine_with(provider, monkeypatch)
        cluster_results = [
            {
                "cluster_id": 0,
                "summary": "Cluster-local [2].",
                "sources": [],
                "file_paths": ["a.py"],
                "file_reference_map": {"a.py": 2},
            }
        ]

        final = await engine._reduce_synthesis(
            cluster_results,
            _chunks(),
            _files(),
            ResearchContext(root_query="valid remap"),
            {"output_tokens": 30_000},
        )

        assert "Cluster-local [1]" in provider.calls[-1]
        assert "[2]" not in final
