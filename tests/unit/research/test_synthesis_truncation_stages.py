"""Deterministic synthesis-stage truncation contracts over the real SDK wire."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

from chunkhound.llm_manager import LLMManager
from chunkhound.services.clustering_service import ClusterGroup
from chunkhound.services.research import SynthesisEngine
from chunkhound.services.research.shared.citation_manager import CitationManager
from chunkhound.services.research.shared.models import ResearchContext
from tests.fixtures.openai_compatible_server import (
    ChatCompletionScript,
    OpenAICompatibleTestServer,
)

OUTPUT_BUDGET = 30_000
LONG_ANSWER = "A complete synthesis answer. " * 8


class _SynthesisParent:
    def __init__(self) -> None:
        self._citation_manager = CitationManager()


class _Harness:
    """Real manager/provider/engine wired exclusively to the loopback fixture."""

    def __init__(self, server: OpenAICompatibleTestServer) -> None:
        base_url = server.base_url
        server.assert_loopback_url(base_url)
        config = {
            "provider": "grok",
            "api_key": "sk-local-fixture-not-a-real-credential",
            "model": "loopback-synthesis-model",
            "base_url": base_url,
            "max_retries": 0,
            "output_limits_enabled": True,
        }
        self.manager = LLMManager(config, config)
        self.engine = SynthesisEngine(
            self.manager,
            database_services=object(),
            parent_service=_SynthesisParent(),
        )

    async def close(self) -> None:
        providers = {
            self.manager.get_utility_provider(),
            self.manager.get_synthesis_provider(),
        }
        for provider in providers:
            await provider._client.close()  # noqa: SLF001 - test-owned SDK client


def _cluster(cluster_id: int, file_name: str, total_tokens: int) -> ClusterGroup:
    content = f"def source_{cluster_id}():\n    return '{file_name}'\n"
    return ClusterGroup(
        cluster_id=cluster_id,
        file_paths=[file_name],
        files_content={file_name: content},
        total_tokens=total_tokens,
    )


def _chunks(clusters: Sequence[ClusterGroup]) -> list[dict[str, Any]]:
    return [
        {
            "file_path": file_name,
            "content": content,
            "start_line": 1,
            "end_line": 2,
        }
        for cluster in clusters
        for file_name, content in cluster.files_content.items()
    ]


def _all_files(clusters: Sequence[ClusterGroup]) -> dict[str, str]:
    return {
        file_name: content
        for cluster in clusters
        for file_name, content in cluster.files_content.items()
    }


async def _run_synthesis(
    engine: SynthesisEngine,
    clusters: Sequence[ClusterGroup],
    *,
    query: str = "Explain the synthesis truncation contract",
) -> str:
    """Exercise the same single-pass or map/reduce business flow as research."""
    chunks = _chunks(clusters)
    files = _all_files(clusters)
    budgets = {"output_tokens": OUTPUT_BUDGET}

    if len(clusters) == 1:
        return await engine._single_pass_synthesis(
            chunks=chunks,
            files=files,
            context=ResearchContext(root_query=query),
            synthesis_budgets=budgets,
        )

    total_input_tokens = sum(cluster.total_tokens for cluster in clusters)
    tasks = [
        asyncio.create_task(
            engine._map_synthesis_on_cluster(
                cluster=cluster,
                chunks=chunks,
                context=ResearchContext(root_query=query),
                synthesis_budgets=budgets,
                total_input_tokens=total_input_tokens,
            )
        )
        for cluster in clusters
    ]
    try:
        cluster_results = await asyncio.gather(*tasks)
    except BaseException:
        # Settle siblings without asserting whether they completed before the failure.
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    return await engine._reduce_synthesis(
        cluster_results=cluster_results,
        all_chunks=chunks,
        all_files=files,
        context=ResearchContext(root_query=query),
        synthesis_budgets=budgets,
    )


async def _run_maps(
    engine: SynthesisEngine,
    clusters: Sequence[ClusterGroup],
) -> list[dict[str, Any]]:
    chunks = _chunks(clusters)
    total_input_tokens = sum(cluster.total_tokens for cluster in clusters)
    return await asyncio.gather(
        *(
            engine._map_synthesis_on_cluster(
                cluster=cluster,
                chunks=chunks,
                context=ResearchContext(root_query="Characterize floor allocation"),
                synthesis_budgets={"output_tokens": OUTPUT_BUDGET},
                total_input_tokens=total_input_tokens,
            )
            for cluster in clusters
        )
    )


def _allowances(server: OpenAICompatibleTestServer, stage: str) -> list[int]:
    requests = [request for request in server.requests if request["stage"] == stage]
    allowances = [request["max_completion_tokens"] for request in requests]
    assert all(isinstance(allowance, int) for allowance in allowances)
    return allowances


@pytest.mark.asyncio
async def test_single_cluster_success_is_one_30000_token_request() -> None:
    file_name = "single-success-unique.py"
    script = ChatCompletionScript(
        name="single-success",
        marker=file_name,
        stage="single",
        content=LONG_ANSWER,
    )
    with OpenAICompatibleTestServer([script]) as server:
        harness = _Harness(server)
        try:
            answer = await _run_synthesis(
                harness.engine, [_cluster(0, file_name, 10_000)]
            )
        finally:
            await harness.close()

        assert LONG_ANSWER in answer
        assert _allowances(server, "single") == [OUTPUT_BUDGET]
        assert len(server.requests) == 1
        server.assert_all_scripts_consumed()


@pytest.mark.asyncio
async def test_single_cluster_length_failure_is_not_retried_or_returned() -> None:
    file_name = "single-truncated-unique.py"
    script = ChatCompletionScript(
        name="single-truncated",
        marker=file_name,
        stage="single",
        content="partial answer that must be discarded",
        finish_reason="length",
    )
    with OpenAICompatibleTestServer([script]) as server:
        harness = _Harness(server)
        try:
            with pytest.raises(RuntimeError, match="token limit exceeded"):
                await _run_synthesis(harness.engine, [_cluster(0, file_name, 10_000)])
        finally:
            await harness.close()

        assert _allowances(server, "single") == [OUTPUT_BUDGET]
        assert len(server.requests) == 1
        server.assert_all_scripts_consumed()


@pytest.mark.asyncio
async def test_two_cluster_map_reduce_uses_expected_stage_budgets() -> None:
    small_file = "map-small-floor-unique.py"
    large_file = "map-large-proportional-unique.py"
    scripts = [
        ChatCompletionScript(
            name="small-map",
            marker=small_file,
            stage="map",
            content="SMALL_MAP_RESULT_UNIQUE [1]",
        ),
        ChatCompletionScript(
            name="large-map",
            marker=large_file,
            stage="map",
            content="LARGE_MAP_RESULT_UNIQUE [1]",
        ),
        ChatCompletionScript(
            name="reduce",
            stage="reduce",
            predicate=lambda _body, messages: (
                "SMALL_MAP_RESULT_UNIQUE" in messages
                and "LARGE_MAP_RESULT_UNIQUE" in messages
            ),
            content=LONG_ANSWER,
        ),
    ]
    clusters = [
        _cluster(0, small_file, 10_000),
        _cluster(1, large_file, 90_000),
    ]
    with OpenAICompatibleTestServer(scripts) as server:
        harness = _Harness(server)
        try:
            answer = await _run_synthesis(harness.engine, clusters)
        finally:
            await harness.close()

        assert LONG_ANSWER in answer
        assert sorted(_allowances(server, "map")) == [5_000, 27_000]
        assert _allowances(server, "reduce") == [OUTPUT_BUDGET]
        assert len(server.requests) == 3
        server.assert_all_scripts_consumed()


@pytest.mark.asyncio
async def test_targeted_map_truncation_fails_without_reduce() -> None:
    target_file = "target-map-truncation-unique.py"
    sibling_file = "sibling-map-success-unique.py"
    scripts = [
        ChatCompletionScript(
            name="target-map",
            marker=target_file,
            stage="map-target",
            content="partial targeted map result",
            finish_reason="length",
        ),
        ChatCompletionScript(
            name="sibling-map",
            marker=sibling_file,
            stage="map-sibling",
            content="SIBLING_MAP_RESULT_UNIQUE",
        ),
    ]
    clusters = [
        _cluster(0, target_file, 50_000),
        _cluster(1, sibling_file, 50_000),
    ]
    with OpenAICompatibleTestServer(scripts) as server:
        harness = _Harness(server)
        try:
            with pytest.raises(RuntimeError, match="token limit exceeded"):
                await _run_synthesis(harness.engine, clusters)
        finally:
            await harness.close()

        target_requests = [
            request
            for request in server.requests
            if request["matched_script"] == "target-map"
        ]
        assert len(target_requests) == 1
        assert target_requests[0]["max_completion_tokens"] == 15_000
        reduce_requests = [
            request for request in server.requests if request["stage"] == "reduce"
        ]
        unmatched_requests = [
            request for request in server.requests if request["matched_script"] is None
        ]
        assert not reduce_requests
        assert not unmatched_requests


@pytest.mark.asyncio
async def test_reduce_truncation_after_all_maps_fails_without_retry_or_answer() -> None:
    first_file = "reduce-map-first-unique.py"
    second_file = "reduce-map-second-unique.py"
    scripts = [
        ChatCompletionScript(
            name="first-map",
            marker=first_file,
            stage="map",
            content="FIRST_MAP_OUTPUT_MARKER_UNIQUE [1]",
        ),
        ChatCompletionScript(
            name="second-map",
            marker=second_file,
            stage="map",
            content="SECOND_MAP_OUTPUT_MARKER_UNIQUE [1]",
        ),
        ChatCompletionScript(
            name="truncated-reduce",
            stage="reduce",
            predicate=lambda _body, messages: (
                "FIRST_MAP_OUTPUT_MARKER_UNIQUE" in messages
                and "SECOND_MAP_OUTPUT_MARKER_UNIQUE" in messages
            ),
            content="partial final answer that must be discarded",
            finish_reason="length",
        ),
    ]
    clusters = [
        _cluster(0, first_file, 50_000),
        _cluster(1, second_file, 50_000),
    ]
    with OpenAICompatibleTestServer(scripts) as server:
        harness = _Harness(server)
        try:
            with pytest.raises(RuntimeError, match="token limit exceeded"):
                await _run_synthesis(harness.engine, clusters)
        finally:
            await harness.close()

        assert len([r for r in server.requests if r["stage"] == "map"]) == 2
        assert _allowances(server, "reduce") == [OUTPUT_BUDGET]
        assert len(server.requests) == 3
        server.assert_all_scripts_consumed()


@pytest.mark.asyncio
async def test_seven_equal_clusters_each_receive_5000_token_floor() -> None:
    clusters = [
        _cluster(index, f"floor-map-{index}-unique.py", 10_000) for index in range(7)
    ]
    scripts = [
        ChatCompletionScript(
            name=f"floor-map-{index}",
            marker=cluster.file_paths[0],
            stage="map",
            content=f"FLOOR_MAP_RESULT_{index}_UNIQUE",
        )
        for index, cluster in enumerate(clusters)
    ]
    with OpenAICompatibleTestServer(scripts) as server:
        harness = _Harness(server)
        try:
            results = await _run_maps(harness.engine, clusters)
        finally:
            await harness.close()

        allowances = _allowances(server, "map")
        assert len(results) == 7
        assert allowances == [5_000] * 7
        assert sum(allowances) == 35_000
        assert len(server.requests) == 7
        server.assert_all_scripts_consumed()
