"""Unit tests for the Python-side analytics construction/lifecycle glue.

These exercise the real `chunkhound_native.AnalyticsRecorder` (no mocking --
it's a hard dependency, same as the rest of the Rust-backed test suite) with
a disabled or locally-buffering-only config, never a real S3 endpoint.
"""

import glob
import json
from pathlib import Path

import pytest

from chunkhound.core.analytics.recorder import (
    _current,
    bind_current,
    build_recorder,
    end_command,
    get_current,
    record_internal_error,
    record_provider_call,
    redact_action_fields,
    shutdown,
    start_command,
    update_action,
)
from chunkhound.core.config.analytics_config import AnalyticsConfig


def _read_buffer_events(buffer_dir: Path) -> list[dict]:
    events = []
    for path in glob.glob(str(buffer_dir / "buffer-*.jsonl")):
        for line in Path(path).read_text().splitlines():
            events.append(json.loads(line))
    return events


@pytest.fixture(autouse=True)
def _reset_current_handle():
    token = _current.set(None)
    yield
    _current.reset(token)


def test_disabled_config_builds_a_safe_noop_recorder(tmp_path: Path) -> None:
    recorder = build_recorder(AnalyticsConfig(enabled=False), tmp_path)
    handle = start_command(recorder, "search", "mcp", {"query": "x"})
    record_provider_call("llm", "anthropic", "claude", True)
    end_command(recorder, handle, True)
    # No exception, and nothing to assert about buffered files -- disabled
    # means no file/thread is ever created.


def test_enabled_recorder_writes_a_well_formed_command_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    buffer_dir = tmp_path / "analytics"
    monkeypatch.setattr("chunkhound.core.analytics.recorder._ANALYTICS_DIR", buffer_dir)
    config = AnalyticsConfig(
        enabled=True, flush_interval_seconds=999999, flush_batch_size=999999
    )
    recorder = build_recorder(config, tmp_path)

    # save_sensitive_data=True: this test is about the well-formed shape of
    # the whole event, not redaction -- see the dedicated redaction tests.
    handle = start_command(
        recorder,
        "search",
        "mcp",
        {"query": "explain indexing"},
        save_sensitive_data=True,
    )
    record_provider_call(
        "llm", "anthropic", "claude", True, input_tokens=900, output_tokens=210
    )
    end_command(recorder, handle, True)

    events = _read_buffer_events(buffer_dir)
    assert len(events) == 1
    event = events[0]
    assert event["type"] == "command_summary"
    assert event["command"] == "search"
    assert event["source"] == "mcp"
    assert event["success"] is True
    assert event["action"] == {"query": "explain indexing"}
    llm = event["providers"]["llm"][0]
    assert llm["provider"] == "anthropic"
    assert llm["calls"] == 1
    assert llm["fails"] == 0
    assert llm["input_tokens"] == 900


def test_current_handle_is_cleared_after_end_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "chunkhound.core.analytics.recorder._ANALYTICS_DIR", tmp_path / "analytics"
    )
    recorder = build_recorder(AnalyticsConfig(enabled=True), tmp_path)
    handle = start_command(recorder, "search", "cli", {})
    assert _current.get() is not None
    end_command(recorder, handle, True)
    assert _current.get() is None


def test_record_provider_call_without_an_open_command_is_a_silent_noop() -> None:
    # No start_command was ever called in this test -- must not raise.
    record_provider_call("llm", "anthropic", "claude", True)
    record_internal_error("KeyError")


def test_internal_error_recorded_when_no_provider_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "chunkhound.core.analytics.recorder._ANALYTICS_DIR", tmp_path / "analytics"
    )
    recorder = build_recorder(AnalyticsConfig(enabled=True), tmp_path)
    handle = start_command(recorder, "search", "cli", {})
    record_internal_error("KeyError")
    end_command(recorder, handle, False)

    events = _read_buffer_events(tmp_path / "analytics")
    assert events[0]["internal_error_type"] == "KeyError"
    assert events[0]["success"] is False


def test_internal_error_suppressed_when_a_provider_already_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "chunkhound.core.analytics.recorder._ANALYTICS_DIR", tmp_path / "analytics"
    )
    recorder = build_recorder(AnalyticsConfig(enabled=True), tmp_path)
    handle = start_command(recorder, "search", "cli", {})
    record_provider_call("llm", "openai", "gpt", False, error_type="TimeoutError")
    record_internal_error("KeyError")
    end_command(recorder, handle, False)

    events = _read_buffer_events(tmp_path / "analytics")
    assert events[0]["internal_error_type"] is None
    assert events[0]["providers"]["llm"][0]["fails"] == 1


def test_update_action_merges_fields_known_only_after_the_command_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mirrors the index command: mode is known at start_command time,
    file_count/total_chunks only after indexing finishes."""
    monkeypatch.setattr(
        "chunkhound.core.analytics.recorder._ANALYTICS_DIR", tmp_path / "analytics"
    )
    recorder = build_recorder(AnalyticsConfig(enabled=True), tmp_path)
    handle = start_command(recorder, "index", "cli", {"mode": "initial"})
    update_action({"file_count": 42, "total_chunks": 1337})
    end_command(recorder, handle, True)

    events = _read_buffer_events(tmp_path / "analytics")
    assert events[0]["action"] == {
        "mode": "initial",
        "file_count": 42,
        "total_chunks": 1337,
    }


def test_update_action_without_an_open_command_is_a_silent_noop() -> None:
    update_action({"file_count": 1})


def test_update_action_redacts_sensitive_fields_on_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression guard for the design gap the independent review flagged:
    update_action() must enforce save_sensitive_data itself, using the value
    captured by start_command() -- callers (e.g. a future command merging a
    free-text field deep inside its own implementation) must not be able to
    bypass redaction just by going through update_action instead of the
    initial start_command action dict."""
    monkeypatch.setattr(
        "chunkhound.core.analytics.recorder._ANALYTICS_DIR", tmp_path / "analytics"
    )
    recorder = build_recorder(AnalyticsConfig(enabled=True), tmp_path)
    handle = start_command(recorder, "search", "cli", {}, save_sensitive_data=False)
    # A raw, unredacted dict -- update_action must redact this itself.
    update_action({"query": "explain indexing", "commit_hash": "abc123"})
    end_command(recorder, handle, True)

    events = _read_buffer_events(tmp_path / "analytics")
    assert events[0]["action"] == {"query": None, "commit_hash": "abc123"}


def test_get_current_returns_none_when_nothing_is_open() -> None:
    assert get_current() is None


def test_get_current_matches_what_start_command_set(tmp_path: Path) -> None:
    recorder = build_recorder(AnalyticsConfig(enabled=False), tmp_path)
    handle = start_command(recorder, "search", "cli", {})
    assert get_current() == (recorder, handle, False)


def test_bind_current_makes_the_binding_visible_to_record_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulates the cross-thread case: the CLI thread resolves
    get_current(), a *different* thread (standing in for a rayon worker)
    calls bind_current() with that value before its own provider call."""
    monkeypatch.setattr(
        "chunkhound.core.analytics.recorder._ANALYTICS_DIR", tmp_path / "analytics"
    )
    recorder = build_recorder(AnalyticsConfig(enabled=True), tmp_path)
    handle = start_command(recorder, "index", "cli", {})
    resolved = get_current()
    assert resolved is not None

    # Simulate landing on a fresh thread/context with nothing bound.
    token = _current.set(None)
    try:
        assert get_current() is None
        bind_current(*resolved)
        record_provider_call("embedding", "openai", "text-embedding-3", True)
    finally:
        _current.reset(token)

    end_command(recorder, handle, True)
    events = _read_buffer_events(tmp_path / "analytics")
    assert events[0]["providers"]["embedding"][0]["calls"] == 1


def test_bind_current_none_clears_any_stale_binding() -> None:
    _current.set(("stale-recorder", 999, False))
    bind_current(None, 0)
    assert get_current() is None


def test_shutdown_with_no_recorder_is_a_silent_noop() -> None:
    shutdown(None)


def test_shutdown_flushes_the_buffer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    buffer_dir = tmp_path / "analytics"
    monkeypatch.setattr("chunkhound.core.analytics.recorder._ANALYTICS_DIR", buffer_dir)
    config = AnalyticsConfig(
        enabled=True, flush_interval_seconds=999999, flush_batch_size=999999
    )
    recorder = build_recorder(config, tmp_path)
    handle = start_command(recorder, "search", "cli", {})
    end_command(recorder, handle, True)

    # No S3 endpoint configured, so this is a local-only flush: with nothing
    # to upload to, the buffer stays on the active file rather than rotating
    # into an orphaned ".pending-*" file (which would never be cleaned up on
    # a long-running process -- see recorder.rs's flush_active()). shutdown()
    # itself must still return promptly rather than raising or hanging.
    shutdown(recorder, timeout_ms=2000)

    pending = list(buffer_dir.glob("*.pending-*"))
    assert len(pending) == 0
    events = _read_buffer_events(buffer_dir)
    assert len(events) == 1


def test_redact_action_fields_nulls_only_the_sensitive_keys() -> None:
    fields = {
        "query": "explain indexing",
        "commit_range": "HEAD~5..HEAD",
        "commit_hash": "abc123",
    }
    assert redact_action_fields(fields, save_sensitive_data=False) == {
        "query": None,
        "commit_range": "HEAD~5..HEAD",
        "commit_hash": "abc123",
    }


def test_redact_action_fields_is_a_pass_through_when_enabled() -> None:
    fields = {"query": "explain indexing", "commit_range": "HEAD~5..HEAD"}
    result = redact_action_fields(fields, save_sensitive_data=True)
    assert result is fields
