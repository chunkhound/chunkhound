#!/usr/bin/env python3
"""Unit tests for snapshot LLM label batching helpers."""

from __future__ import annotations

import asyncio

import pytest

from chunkhound.api.cli.commands.snapshot import (
    LABEL_MAX_CHARS,
    _BatchLabelRequest,
    _batch_label_schema,
    _chunk_system_group_prompt_body,
    _complete_structured_label_batch_with_replay,
    _pack_label_batches,
    _parse_validate_batch_results,
    _render_label_batch_prompt,
    _run_structured_label_batches_parallel,
)


def test_pack_label_batches_respects_max_items() -> None:
    items = [
        _BatchLabelRequest(item_id=f"x{i}", content="alpha beta gamma")
        for i in range(5)
    ]

    batches = _pack_label_batches(
        kind="test",
        items=items,
        estimate_tokens=lambda s: len(s),
        max_tokens=10**9,
        max_items=2,
    )

    assert [len(b) for b in batches] == [2, 2, 1]
    assert [it.item_id for it in batches[0]] == ["x0", "x1"]
    assert [it.item_id for it in batches[1]] == ["x2", "x3"]
    assert [it.item_id for it in batches[2]] == ["x4"]


def test_parse_validate_batch_results_happy_path() -> None:
    requested = ["a", "b"]
    response = {
        "results": [
            {"id": "b", "label": "Label B", "confidence": 0.7},
            {"id": "a", "label": "Label A", "confidence": 0.2},
        ]
    }

    parsed = _parse_validate_batch_results(requested_ids=requested, response=response)
    assert parsed["a"][0] == "Label A"
    assert parsed["b"][0] == "Label B"
    assert 0.0 <= parsed["a"][1] <= 1.0
    assert 0.0 <= parsed["b"][1] <= 1.0


def test_parse_validate_batch_results_strips_id_whitespace() -> None:
    requested = ["a"]
    response = {"results": [{"id": " a ", "label": "Label A", "confidence": 0.5}]}
    parsed = _parse_validate_batch_results(requested_ids=requested, response=response)
    assert parsed["a"][0] == "Label A"


def test_parse_validate_batch_results_accepts_int_ids() -> None:
    requested = ["16"]
    response = {"results": [{"id": 16, "label": "Label 16", "confidence": 0.5}]}
    parsed = _parse_validate_batch_results(requested_ids=requested, response=response)
    assert parsed["16"][0] == "Label 16"


def test_parse_validate_batch_results_missing_id_raises() -> None:
    requested = ["a", "b"]
    response = {"results": [{"id": "a", "label": "Label A", "confidence": 0.5}]}
    with pytest.raises(RuntimeError):
        _parse_validate_batch_results(requested_ids=requested, response=response)


def test_parse_validate_batch_results_extra_id_raises() -> None:
    requested = ["a"]
    response = {
        "results": [
            {"id": "a", "label": "Label A", "confidence": 0.5},
            {"id": "c", "label": "Label C", "confidence": 0.5},
        ]
    }
    with pytest.raises(RuntimeError):
        _parse_validate_batch_results(requested_ids=requested, response=response)


def test_render_label_batch_prompt_chunk_systems_includes_behavior_rubric() -> None:
    prompt = _render_label_batch_prompt(
        kind="chunk_systems",
        items=[_BatchLabelRequest(item_id="x1", content="Cluster ID: 1\nSize: 3\n")],
    )
    assert "behavior/responsibility" in prompt
    assert "noun phrase" in prompt
    assert "artifact/stage" in prompt
    assert "Tests:" in prompt


def test_render_label_batch_prompt_chunk_system_groups_includes_umbrella_guidance() -> None:
    prompt = _render_label_batch_prompt(
        kind="chunk_system_groups",
        items=[_BatchLabelRequest(item_id="g1", content="Resolution: 1.0\nGroup ID: 1\n")],
    )
    assert "umbrella" in prompt.lower()
    assert "dominant" in prompt.lower()


def test_batch_label_schema_enforces_label_caps_and_item_count() -> None:
    schema = _batch_label_schema(allowed_ids=["a", "b", "c"])
    results = schema["properties"]["results"]
    assert results["minItems"] == 3
    assert results["maxItems"] == 3
    item = results["items"]
    label = item["properties"]["label"]
    assert label["maxLength"] == LABEL_MAX_CHARS
    assert "\\n" in label["pattern"]
    assert item["properties"]["id"]["enum"] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_complete_structured_label_batch_with_replay_splits_on_truncation(
    tmp_path,
) -> None:
    class FakeProvider:
        async def complete_structured(
            self,
            *,
            prompt: str,
            json_schema: dict,
            system: str,
            max_completion_tokens: int,
        ) -> dict:
            del prompt, system, max_completion_tokens
            max_items = json_schema["properties"]["results"].get("maxItems")
            if max_items is not None and int(max_items) > 2:
                raise RuntimeError(
                    "LLM structured completion truncated - token limit exceeded "
                    "(prompt=7,177, completion=4,997)."
                )
            allowed = json_schema["properties"]["results"]["items"]["properties"]["id"]["enum"]
            return {
                "results": [
                    {"id": item_id, "label": f"Label {item_id}", "confidence": 0.5}
                    for item_id in allowed
                ]
            }

    items = [
        _BatchLabelRequest(item_id="a", content="x"),
        _BatchLabelRequest(item_id="b", content="x"),
        _BatchLabelRequest(item_id="c", content="x"),
        _BatchLabelRequest(item_id="d", content="x"),
    ]

    out = await _complete_structured_label_batch_with_replay(
        provider=FakeProvider(),
        kind="chunk_systems",
        items=items,
        prompt_path=tmp_path / "batch.md",
        system="test",
        max_completion_tokens=10,
    )

    assert out.llm_calls == 3
    assert out.min_successful_batch_items == 2
    assert sorted(out.parsed.keys()) == ["a", "b", "c", "d"]


@pytest.mark.asyncio
async def test_run_structured_label_batches_parallel_respects_concurrency(tmp_path) -> None:
    two_in_flight = asyncio.Event()
    release = asyncio.Event()
    in_flight = 0
    max_in_flight = 0

    class FakeProvider:
        async def complete_structured(
            self,
            *,
            prompt: str,
            json_schema: dict,
            system: str,
            max_completion_tokens: int,
        ) -> dict:
            del prompt, system, max_completion_tokens
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            if max_in_flight >= 2:
                two_in_flight.set()
            try:
                await release.wait()
            finally:
                in_flight -= 1

            allowed = json_schema["properties"]["results"]["items"]["properties"]["id"]["enum"]
            return {
                "results": [
                    {"id": item_id, "label": f"Label {item_id}", "confidence": 0.5}
                    for item_id in allowed
                ]
            }

    items = [
        _BatchLabelRequest(item_id="a", content="x"),
        _BatchLabelRequest(item_id="b", content="x"),
        _BatchLabelRequest(item_id="c", content="x"),
        _BatchLabelRequest(item_id="d", content="x"),
    ]

    async def run() -> int:
        return await _run_structured_label_batches_parallel(
            provider=FakeProvider(),
            kind="chunk_systems",
            items=items,
            out_dir=tmp_path,
            batch_prefix="batch",
            estimate_tokens=lambda s: 1,
            max_tokens=10**9,
            max_items=1,  # force 1 item per batch
            concurrency=2,
            system="test",
            max_completion_tokens=10,
        )

    task = asyncio.create_task(run())
    await asyncio.wait_for(two_in_flight.wait(), timeout=1.0)
    release.set()
    llm_calls = await asyncio.wait_for(task, timeout=1.0)

    assert llm_calls == 4
    assert max_in_flight >= 2


@pytest.mark.asyncio
async def test_complete_structured_label_batch_with_replay_retries_transient(tmp_path) -> None:
    calls = 0

    class FakeProvider:
        async def complete_structured(
            self,
            *,
            prompt: str,
            json_schema: dict,
            system: str,
            max_completion_tokens: int,
        ) -> dict:
            del prompt, system, max_completion_tokens
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("429 rate limit exceeded")
            allowed = json_schema["properties"]["results"]["items"]["properties"]["id"]["enum"]
            return {
                "results": [
                    {"id": item_id, "label": f"Label {item_id}", "confidence": 0.5}
                    for item_id in allowed
                ]
            }

    items = [
        _BatchLabelRequest(item_id="a", content="x"),
        _BatchLabelRequest(item_id="b", content="x"),
    ]

    out = await _complete_structured_label_batch_with_replay(
        provider=FakeProvider(),
        kind="chunk_systems",
        items=items,
        prompt_path=tmp_path / "batch.md",
        system="test",
        max_completion_tokens=10,
        retry_max_attempts=2,
        retry_base_delay_seconds=0.0,
        retry_max_delay_seconds=0.0,
        retry_jitter_ratio=0.0,
    )

    assert calls == 2
    assert out.llm_calls == 2
    assert sorted(out.parsed.keys()) == ["a", "b"]


def test_chunk_system_group_prompt_body_sorts_tests_last_and_hints_on_mixed_files() -> None:
    member_systems = [
        {
            "cluster_id": 1,
            "size": 100,  # Intentionally larger than non-test to prove ordering.
            "top_files": [{"path": "chunkhound/tests/test_a.py", "count": 10}],
            "top_symbols": [],
        },
        {
            "cluster_id": 2,
            "size": 5,
            "top_files": [{"path": "chunkhound/chunkhound/services/indexing_coordinator.py", "count": 50}],
            "top_symbols": [],
        },
    ]
    system_label_by_id = {1: "Tests: disk usage limit", 2: "Indexing coordinator safeguards"}

    lines = _chunk_system_group_prompt_body(
        resolution=3.0,
        group_id=4,
        member_systems=member_systems,
        system_label_by_id=system_label_by_id,
        is_test_only=False,
        is_test_heavy=False,
        max_member_systems=12,
        max_files=12,
        max_symbols=12,
        prompt_mode="systems_only",
    )

    members_idx = lines.index("Member systems:")
    member_lines = [ln for ln in lines[members_idx + 1 :] if ln.startswith("- #")][:2]
    assert member_lines[0].startswith("- #2:")
    assert member_lines[1].startswith("- #1:")

    assert not any("Top files" in ln for ln in lines)
    assert not any("Top symbols" in ln for ln in lines)


def test_chunk_system_group_prompt_body_full_mode_hints_on_mixed_files() -> None:
    member_systems = [
        {
            "cluster_id": 1,
            "size": 100,  # Intentionally larger than non-test to prove ordering.
            "top_files": [{"path": "chunkhound/tests/test_a.py", "count": 10}],
            "top_symbols": [],
        },
        {
            "cluster_id": 2,
            "size": 5,
            "top_files": [{"path": "chunkhound/chunkhound/services/indexing_coordinator.py", "count": 50}],
            "top_symbols": [],
        },
    ]
    system_label_by_id = {1: "Tests: disk usage limit", 2: "Indexing coordinator safeguards"}

    lines = _chunk_system_group_prompt_body(
        resolution=3.0,
        group_id=4,
        member_systems=member_systems,
        system_label_by_id=system_label_by_id,
        is_test_only=False,
        is_test_heavy=False,
        max_member_systems=12,
        max_files=12,
        max_symbols=12,
        prompt_mode="full",
    )

    members_idx = lines.index("Member systems (top 12):")
    member_lines = [ln for ln in lines[members_idx + 1 :] if ln.startswith("- #")][:2]
    assert member_lines[0].startswith("- #2:")
    assert member_lines[1].startswith("- #1:")

    assert any("Hint: mixed group" in ln for ln in lines)
    assert any("Top files" in ln for ln in lines)
