"""Contract tests for the MCP `handle_tool_call()` analytics hook.

Verifies exactly what the ChunkHound Per-User Analytics design requires of
this chokepoint: one `command_summary` event per tool call, the right
action fields, and the internal_error_type/provider-failure precedence
rule -- using the real chunkhound_native.AnalyticsRecorder (no mocking).
"""

import asyncio
import glob
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chunkhound.core.analytics.recorder import build_recorder
from chunkhound.core.config.analytics_config import AnalyticsConfig
from chunkhound.mcp_server.common import handle_tool_call
from chunkhound.mcp_server.tools import register_tool


def _read_events(buffer_dir: Path) -> list[dict]:
    events = []
    for path in glob.glob(str(buffer_dir / "buffer-*.jsonl")):
        for line in Path(path).read_text().splitlines():
            events.append(json.loads(line))
    return events


@pytest.fixture
def recorder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    buffer_dir = tmp_path / "analytics"
    monkeypatch.setattr("chunkhound.core.analytics.recorder._ANALYTICS_DIR", buffer_dir)
    config = AnalyticsConfig(
        enabled=True, flush_interval_seconds=999999, flush_batch_size=999999
    )
    return build_recorder(config, tmp_path), buffer_dir


@pytest.fixture(autouse=True)
def _register_test_tools():
    """Register throwaway tools for this module only, removed afterward so
    other test modules' TOOL_REGISTRY state is unaffected."""
    from chunkhound.mcp_server.tools import TOOL_REGISTRY

    @register_tool(name="_analytics_test_ok", description="test")
    async def _ok(query: str) -> dict:
        return {"query": query}

    @register_tool(name="_analytics_test_fail", description="test")
    async def _fail(query: str) -> dict:
        raise KeyError("boom")

    @register_tool(name="_analytics_test_cancel", description="test")
    async def _cancel(query: str) -> dict:
        raise asyncio.CancelledError()

    yield
    TOOL_REGISTRY.pop("_analytics_test_ok", None)
    TOOL_REGISTRY.pop("_analytics_test_fail", None)
    TOOL_REGISTRY.pop("_analytics_test_cancel", None)


async def _call(tool_name: str, arguments: dict, recorder, config=None) -> None:
    initialization_complete = asyncio.Event()
    initialization_complete.set()
    await handle_tool_call(
        tool_name=tool_name,
        arguments=arguments,
        services=None,
        embedding_manager=None,
        initialization_complete=initialization_complete,
        analytics_recorder=recorder,
        config=config,
    )


@pytest.mark.asyncio
async def test_successful_tool_call_records_one_command_summary(recorder) -> None:
    rec, buffer_dir = recorder
    await _call("_analytics_test_ok", {"query": "explain indexing"}, rec)

    events = _read_events(buffer_dir)
    assert len(events) == 1
    event = events[0]
    assert event["command"] == "_analytics_test_ok"
    assert event["source"] == "mcp"
    assert event["success"] is True
    assert event["internal_error_type"] is None


@pytest.mark.asyncio
async def test_search_action_fields_extract_the_query(recorder) -> None:
    rec, buffer_dir = recorder
    # "search" is a real registered tool name, so its action-field mapping
    # (query only, for a non-git-scoped call) applies. save_sensitive_data
    # is explicitly opted in here since this test is about field selection,
    # not redaction -- see the dedicated redaction tests below.
    config = SimpleNamespace(analytics=SimpleNamespace(save_sensitive_data=True))
    await _call(
        "search", {"query": "explain indexing", "unrelated": "drop me"}, rec, config
    )

    events = _read_events(buffer_dir)
    assert events[-1]["action"] == {"query": "explain indexing"}


@pytest.mark.asyncio
async def test_sensitive_action_fields_are_redacted_by_default(recorder) -> None:
    # No config passed -- save_sensitive_data defaults to False.
    rec, buffer_dir = recorder
    await _call("search", {"query": "explain indexing"}, rec)

    events = _read_events(buffer_dir)
    assert events[-1]["action"] == {"query": None}


@pytest.mark.asyncio
async def test_non_sensitive_action_fields_survive_redaction(recorder) -> None:
    rec, buffer_dir = recorder
    await _call(
        "search",
        {"query": "explain indexing", "commit_hash": "abc123"},
        rec,
    )

    events = _read_events(buffer_dir)
    assert events[-1]["action"] == {"query": None, "commit_hash": "abc123"}


@pytest.mark.asyncio
async def test_code_research_action_fields_extract_the_query(recorder) -> None:
    # Regression guard: _ANALYTICS_ACTION_FIELDS["code_research"] used to be
    # keyed on "question", but deep_research_impl's real parameter is
    # `query` -- that mapping silently recorded nothing. code_research
    # requires embeddings/llm/reranker (none wired here) so the tool call
    # itself fails after dispatch, but action-field extraction happens in
    # start_command(), before that capability check, so this still proves
    # the query is captured under the real key.
    rec, buffer_dir = recorder
    config = SimpleNamespace(analytics=SimpleNamespace(save_sensitive_data=True))
    await _call("code_research", {"query": "how does auth work?"}, rec, config)

    events = _read_events(buffer_dir)
    assert events[-1]["action"] == {"query": "how does auth work?"}


@pytest.mark.asyncio
async def test_code_research_query_is_redacted_by_default(recorder) -> None:
    rec, buffer_dir = recorder
    await _call("code_research", {"query": "how does auth work?"}, rec)

    events = _read_events(buffer_dir)
    assert events[-1]["action"] == {"query": None}


@pytest.mark.asyncio
async def test_failed_tool_call_records_internal_error_type(recorder) -> None:
    rec, buffer_dir = recorder
    await _call("_analytics_test_fail", {"query": "x"}, rec)

    events = _read_events(buffer_dir)
    assert len(events) == 1
    event = events[0]
    assert event["success"] is False
    assert event["internal_error_type"] == "KeyError"


@pytest.mark.asyncio
async def test_unknown_tool_still_records_a_failed_command(recorder) -> None:
    rec, buffer_dir = recorder
    await _call("_no_such_tool", {}, rec)

    events = _read_events(buffer_dir)
    assert len(events) == 1
    assert events[0]["success"] is False
    assert events[0]["internal_error_type"] == "ValueError"


@pytest.mark.asyncio
async def test_cancelled_tool_call_still_closes_the_handle(recorder) -> None:
    # Regression guard: asyncio.CancelledError is a BaseException, not an
    # Exception -- a naive `except Exception` would let it skip
    # end_command() entirely, leaking an open handle in the long-running
    # MCP server process for every client-cancelled call. Must also
    # propagate (not be swallowed) so real cancellation semantics survive.
    rec, buffer_dir = recorder
    with pytest.raises(asyncio.CancelledError):
        await _call("_analytics_test_cancel", {"query": "x"}, rec)

    events = _read_events(buffer_dir)
    assert len(events) == 1
    event = events[0]
    assert event["success"] is False
    assert event["internal_error_type"] == "CancelledError"


@pytest.mark.asyncio
async def test_none_recorder_is_a_silent_noop() -> None:
    # No analytics_recorder wired -- must not raise, and (nothing to assert
    # about buffered files since none exist).
    await _call("_analytics_test_ok", {"query": "x"}, None)
