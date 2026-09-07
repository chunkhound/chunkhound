"""Deterministic lifecycle contracts for concurrent synthesis maps."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, cast

import pytest

from chunkhound.services.clustering_service import ClusterGroup
from chunkhound.services.research.shared.models import ResearchContext
from chunkhound.services.research.v1.pluggable_research_service import (
    PluggableResearchService,
)


class _Ledger:
    def get_facts_map_prompt_context(
        self, _file_paths: set[str], *, cluster_id: int
    ) -> str:
        return f"facts-{cluster_id}"


class _Engine:
    def __init__(
        self,
        map_call: Callable[[int], Awaitable[dict[str, Any]]],
    ) -> None:
        self._map_call = map_call
        self.reduce_calls = 0

    async def _map_synthesis_on_cluster(
        self,
        cluster: ClusterGroup,
        *_args: Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        return await self._map_call(cluster.cluster_id)

    async def _reduce_synthesis(self, *_args: Any, **_kwargs: Any) -> str:
        self.reduce_calls += 1
        return "reduced"


def _clusters(count: int) -> list[ClusterGroup]:
    return [
        ClusterGroup(
            cluster_id=index,
            file_paths=[f"cluster-{index}.py"],
            files_content={f"cluster-{index}.py": "content"},
            total_tokens=10,
        )
        for index in range(count)
    ]


def _service(engine: _Engine) -> PluggableResearchService:
    service = PluggableResearchService.__new__(PluggableResearchService)
    service._synthesis_engine = cast(Any, engine)
    return service


async def _run_maps(
    service: PluggableResearchService,
    count: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    return await service._run_synthesis_maps(
        cluster_groups=_clusters(count),
        context=ResearchContext(root_query="query"),
        prioritized_chunks=[],
        synthesis_budgets={"output_tokens": 30_000},
        constants_context="constants",
        evidence_ledger=cast(Any, _Ledger()),
        max_concurrency=concurrency,
    )


def _unfinished_map_tasks() -> list[asyncio.Task[Any]]:
    current = asyncio.current_task()
    return [
        task
        for task in asyncio.all_tasks()
        if task is not current
        and task.get_name().startswith("synthesis-map-")
        and not task.done()
    ]


@pytest.mark.asyncio
async def test_success_preserves_input_order_with_out_of_order_completion() -> None:
    releases = [asyncio.Event() for _ in range(3)]
    started = [asyncio.Event() for _ in range(3)]
    completed = [asyncio.Event() for _ in range(3)]

    async def map_call(cluster_id: int) -> dict[str, Any]:
        started[cluster_id].set()
        await releases[cluster_id].wait()
        completed[cluster_id].set()
        return {"cluster_id": cluster_id}

    service = _service(_Engine(map_call))
    caller = asyncio.create_task(_run_maps(service, 3, 3))
    await asyncio.gather(*(event.wait() for event in started))

    for cluster_id in (2, 1, 0):
        releases[cluster_id].set()
        await completed[cluster_id].wait()

    assert await caller == [
        {"cluster_id": 0},
        {"cluster_id": 1},
        {"cluster_id": 2},
    ]
    assert not _unfinished_map_tasks()


@pytest.mark.asyncio
async def test_first_failure_aborts_queue_settles_siblings_and_skips_reduce() -> None:
    class TriggerError(RuntimeError):
        pass

    class CleanupError(RuntimeError):
        pass

    trigger = TriggerError("first temporal failure")
    fail_now = asyncio.Event()
    first_two_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()
    started: list[int] = []

    async def map_call(cluster_id: int) -> dict[str, Any]:
        started.append(cluster_id)
        if len(started) == 2:
            first_two_started.set()
        if cluster_id == 0:
            await fail_now.wait()
            raise trigger
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            sibling_cancelled.set()
            raise CleanupError("cleanup must not replace trigger")

    engine = _Engine(map_call)
    service = _service(engine)

    async def map_then_reduce() -> str:
        results = await _run_maps(service, 4, 2)
        return await engine._reduce_synthesis(results)

    caller = asyncio.create_task(map_then_reduce())
    await first_two_started.wait()
    fail_now.set()

    with pytest.raises(TriggerError) as raised:
        await caller

    assert raised.value is trigger
    assert started == [0, 1]
    assert sibling_cancelled.is_set()
    assert engine.reduce_calls == 0
    assert not _unfinished_map_tasks()


@pytest.mark.asyncio
async def test_external_caller_cancellation_wins_and_settles_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CleanupError(RuntimeError):
        pass

    both_started = asyncio.Event()
    cancelled: set[int] = set()
    started: set[int] = set()

    async def map_call(cluster_id: int) -> dict[str, Any]:
        started.add(cluster_id)
        if len(started) == 2:
            both_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.add(cluster_id)
            if cluster_id == 0:
                raise CleanupError("child cleanup failed")
            raise

    service = _service(_Engine(map_call))
    caller = asyncio.create_task(_run_maps(service, 4, 2))
    await both_started.wait()
    caller.cancel()

    # Python 3.10 tasks do not expose Task.cancelling(). Exercise the real
    # orchestration cleanup boundary with that supported runtime shape.
    current_task = asyncio.current_task
    monkeypatch.setattr(asyncio, "current_task", lambda: object())
    with pytest.raises(asyncio.CancelledError):
        await caller
    monkeypatch.setattr(asyncio, "current_task", current_task)

    assert started == {0, 1}
    assert cancelled == {0, 1}
    assert not _unfinished_map_tasks()


@pytest.mark.asyncio
async def test_unexpected_child_cancellation_surfaces_exact_object() -> None:
    cancellation = asyncio.CancelledError("unexpected map cancellation")
    map_started = asyncio.Event()

    async def map_call(cluster_id: int) -> dict[str, Any]:
        assert cluster_id == 0
        map_started.set()
        raise cancellation

    service = _service(_Engine(map_call))
    caller = asyncio.create_task(_run_maps(service, 3, 1))
    await map_started.wait()

    with pytest.raises(asyncio.CancelledError) as raised:
        await caller

    assert raised.value is cancellation
    assert not _unfinished_map_tasks()
