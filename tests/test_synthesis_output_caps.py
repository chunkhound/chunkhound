from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from chunkhound.services.research.synthesis_engine import SynthesisEngine
from tests.helpers.fake_llm_providers import SleepyDeterministicLLMProvider


class _DummyCitationManager:
    def filter_chunks_to_files(
        self, chunks: list[dict[str, Any]], files: dict[str, str]
    ) -> list[dict[str, Any]]:
        file_paths = set(files.keys())
        return [c for c in chunks if c.get("file_path") in file_paths]


class _DummyParent:
    def __init__(self) -> None:
        self._citation_manager = _DummyCitationManager()
        self._embedding_manager = SimpleNamespace(get_provider=lambda: None)

    async def _emit_event(self, *_args: Any, **_kwargs: Any) -> None:
        return

    def _build_file_reference_map(
        self, chunks: list[dict[str, Any]], files: dict[str, str]
    ) -> dict[str, int]:
        # Deterministic numbering by path
        paths = sorted(set(files.keys()) | {c.get("file_path", "") for c in chunks})
        paths = [p for p in paths if p]
        return {p: i + 1 for i, p in enumerate(paths)}

    def _format_reference_table(self, file_reference_map: dict[str, int]) -> str:
        lines = ["Source References:"]
        for p, n in sorted(file_reference_map.items(), key=lambda kv: kv[1]):
            lines.append(f"[{n}] {p}")
        return "\n".join(lines)

    def _build_sources_footer(
        self,
        _chunks: list[dict[str, Any]],
        _files: dict[str, str],
        _file_reference_map: dict[str, int],
    ) -> str:
        return "Sources:\n- [1] dummy.py"

    def _validate_citation_references(
        self, answer: str, _file_reference_map: dict[str, int]
    ) -> list[str]:
        # Keep this simple for unit test scope
        return [] if "[1]" in answer else ["missing [1]"]

    def _remap_cluster_citations(
        self,
        summary: str,
        _cluster_file_map: dict[str, int],
        _global_file_map: dict[str, int],
    ) -> str:
        return summary


class _DummyLLMManager:
    def __init__(self, synthesis_provider: Any) -> None:
        self._synthesis = synthesis_provider

    def get_synthesis_provider(self) -> Any:
        return self._synthesis

    def get_utility_provider(self) -> Any:
        return self._synthesis


@pytest.mark.asyncio
async def test_single_pass_synthesis_uses_full_output_budget_without_caps() -> None:
    llm = SleepyDeterministicLLMProvider(tokens_per_second=10_000_000.0)
    engine = SynthesisEngine(
        llm_manager=_DummyLLMManager(llm),  # type: ignore[arg-type]
        database_services=SimpleNamespace(),  # type: ignore[arg-type]
        parent_service=_DummyParent(),
    )

    files = {
        "dummy.py": "def f():\n    return 1\n",
    }
    chunks = [
        {
            "file_path": "dummy.py",
            "start_line": 1,
            "end_line": 2,
            "content": "def f():\n    return 1\n",
            "score": 1.0,
        }
    ]

    budgets = {"output_tokens": 30_000}

    answer = await engine._single_pass_synthesis(
        root_query="What does f do?",
        chunks=chunks,
        files=files,
        context=SimpleNamespace(),  # unused by engine
        synthesis_budgets=budgets,
    )

    assert llm.calls, "Expected the fake LLM provider to be called"
    observed_max = llm.calls[-1].max_completion_tokens

    # Without adaptive output capping, we should request the full configured budget.
    assert observed_max == 30_000
    assert "Sources:" in answer
