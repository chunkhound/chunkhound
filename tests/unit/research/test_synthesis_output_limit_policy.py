"""Synthesis output-limit routing and prompt-guidance contracts."""

from typing import Any

import pytest

from chunkhound.interfaces.llm_provider import (
    PROVIDER_MANAGED_OUTPUT,
    LLMResponse,
    OutputLimitMetadata,
    OutputLimitPolicy,
)
from chunkhound.services.clustering_service import ClusterGroup
from chunkhound.services.research import SynthesisEngine
from chunkhound.services.research.shared.citation_manager import CitationManager
from chunkhound.services.research.shared.models import (
    ResearchContext,
    build_output_guidance,
)

OUTPUT_TOKENS = 30_000
ANSWER = "Complete synthesis output with sufficient detail. " * 8


class _Provider:
    def __init__(self, *, output_limits_enabled: bool) -> None:
        self.synthesis_output_limit_policy = OutputLimitPolicy(
            output_limits_enabled=output_limits_enabled,
            fallback_tokens=OUTPUT_TOKENS,
            metadata=OutputLimitMetadata(),
        )
        self.calls: list[dict[str, Any]] = []

    async def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.calls.append({"prompt": prompt, **kwargs})
        return LLMResponse(
            content=ANSWER,
            tokens_used=100,
            model="contract-test",
            finish_reason="stop",
        )

    def estimate_tokens(self, text: str) -> int:
        return len(text.split())


class _Manager:
    def __init__(self, provider: _Provider) -> None:
        self.provider = provider

    def get_synthesis_provider(self) -> _Provider:
        return self.provider


class _Parent:
    def __init__(self) -> None:
        self._citation_manager = CitationManager()


def _cluster(cluster_id: int, path: str, total_tokens: int) -> ClusterGroup:
    content = f"def source_{cluster_id}():\n    return '{path}'\n"
    return ClusterGroup(
        cluster_id=cluster_id,
        file_paths=[path],
        files_content={path: content},
        total_tokens=total_tokens,
    )


def _chunks(clusters: list[ClusterGroup]) -> list[dict[str, Any]]:
    return [
        {
            "file_path": path,
            "content": content,
            "start_line": 1,
            "end_line": 2,
        }
        for cluster in clusters
        for path, content in cluster.files_content.items()
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("output_limits_enabled", [True, False])
async def test_synthesis_stages_route_allowance_without_changing_prompt_guidance(
    output_limits_enabled: bool,
) -> None:
    provider = _Provider(output_limits_enabled=output_limits_enabled)
    engine = SynthesisEngine(
        _Manager(provider),  # type: ignore[arg-type]
        database_services=object(),  # type: ignore[arg-type]
        parent_service=_Parent(),
    )
    clusters = [
        _cluster(0, "small.py", 10_000),
        _cluster(1, "large.py", 90_000),
    ]
    chunks = _chunks(clusters)
    files = {
        path: content
        for cluster in clusters
        for path, content in cluster.files_content.items()
    }
    budgets = {"output_tokens": OUTPUT_TOKENS}

    await engine._single_pass_synthesis(
        chunks=chunks,
        files=files,
        context=ResearchContext(root_query="Explain output-limit routing"),
        synthesis_budgets=budgets,
    )
    map_results = [
        await engine._map_synthesis_on_cluster(
            cluster=cluster,
            chunks=chunks,
            context=ResearchContext(root_query="Explain output-limit routing"),
            synthesis_budgets=budgets,
            total_input_tokens=100_000,
        )
        for cluster in clusters
    ]
    await engine._map_synthesis_on_cluster(
        cluster=clusters[0],
        chunks=chunks,
        context=ResearchContext(root_query="Explain zero-total fallback routing"),
        synthesis_budgets=budgets,
        total_input_tokens=0,
    )
    await engine._reduce_synthesis(
        cluster_results=map_results,
        all_chunks=chunks,
        all_files=files,
        context=ResearchContext(root_query="Explain output-limit routing"),
        synthesis_budgets=budgets,
    )

    allowances = [call["max_completion_tokens"] for call in provider.calls]
    if output_limits_enabled:
        assert allowances == [30_000, 5_000, 27_000, 30_000, 30_000]
    else:
        assert allowances == [PROVIDER_MANAGED_OUTPUT] * 5
        assert all(allowance is PROVIDER_MANAGED_OUTPUT for allowance in allowances)

    systems = [call["system"] for call in provider.calls]
    assert build_output_guidance(15_000) in systems[0]
    assert build_output_guidance(5_000) in systems[1]
    assert build_output_guidance(7_500) in systems[2]
    assert build_output_guidance(7_500) in systems[3]
    assert build_output_guidance(15_000) in systems[4]
