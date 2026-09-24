"""Contract tests for the CLI dispatch analytics hook (async_main()).

Mirrors tests/unit/test_mcp_analytics_hook.py for the CLI side. Follows the
monkeypatch-create_parser/create_validated_config pattern already used in
tests/unit/test_cli_main_config_messages.py to avoid needing a real
embedding/LLM provider configured.
"""

import glob
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chunkhound.api.cli import main as cli_main
from chunkhound.core.config.analytics_config import AnalyticsConfig


class _Parser:
    def __init__(self, args: SimpleNamespace) -> None:
        self._args = args

    def parse_args(self) -> SimpleNamespace:
        return self._args

    def print_help(self) -> None:
        pass


def _read_events(buffer_dir: Path) -> list[dict]:
    # async_main()'s finally block calls ch_analytics.shutdown(), which
    # flushes and rotates the active buffer file to "*.pending-*" before
    # this helper runs -- glob both patterns, not just the active-file shape.
    events = []
    paths = glob.glob(str(buffer_dir / "buffer-*.jsonl")) + glob.glob(
        str(buffer_dir / "buffer-*.pending-*")
    )
    for path in paths:
        for line in Path(path).read_text().splitlines():
            events.append(json.loads(line))
    return events


def _patch_common(
    monkeypatch: pytest.MonkeyPatch,
    args,
    tmp_path: Path,
    save_sensitive_data: bool = False,
) -> Path:
    buffer_dir = tmp_path / "analytics"
    monkeypatch.setattr("chunkhound.core.analytics.recorder._ANALYTICS_DIR", buffer_dir)
    monkeypatch.setattr(cli_main, "create_parser", lambda: _Parser(args))
    monkeypatch.setattr(cli_main, "setup_logging", lambda _verbose: None)
    config = SimpleNamespace(
        analytics=AnalyticsConfig(
            enabled=True,
            flush_interval_seconds=999999,
            flush_batch_size=999999,
            save_sensitive_data=save_sensitive_data,
        ),
        target_dir=tmp_path,
    )
    monkeypatch.setattr(
        cli_main, "create_validated_config", lambda _a, _c: (config, [])
    )
    return buffer_dir


@pytest.mark.asyncio
async def test_successful_search_command_records_one_command_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = SimpleNamespace(command="search", verbose=False, query="explain indexing")
    # save_sensitive_data opted in here since this test is about the command
    # lifecycle/action extraction, not redaction -- see the dedicated
    # redaction tests below for the default (redacted) behavior.
    buffer_dir = _patch_common(monkeypatch, args, tmp_path, save_sensitive_data=True)

    from chunkhound.api.cli.commands import search as search_module

    monkeypatch.setattr(search_module, "search_command", _ok_command)

    await cli_main.async_main()

    events = _read_events(buffer_dir)
    assert len(events) == 1
    event = events[0]
    assert event["command"] == "search"
    assert event["source"] == "cli"
    assert event["success"] is True
    assert event["action"] == {"query": "explain indexing"}


@pytest.mark.asyncio
async def test_sensitive_action_fields_are_redacted_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = SimpleNamespace(command="search", verbose=False, query="explain indexing")
    buffer_dir = _patch_common(monkeypatch, args, tmp_path)  # save_sensitive_data=False

    from chunkhound.api.cli.commands import search as search_module

    monkeypatch.setattr(search_module, "search_command", _ok_command)

    await cli_main.async_main()

    events = _read_events(buffer_dir)
    assert events[0]["action"] == {"query": None}


@pytest.mark.asyncio
async def test_research_action_fields_extract_the_query(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Regression guard: _ANALYTICS_ACTION_ARGS["research"] used to be keyed
    # on "question", but the research subparser's real positional arg dest
    # is `query` -- that mapping silently recorded nothing.
    args = SimpleNamespace(
        command="research", verbose=False, query="how does auth work?"
    )
    buffer_dir = _patch_common(monkeypatch, args, tmp_path, save_sensitive_data=True)

    from chunkhound.api.cli.commands import research as research_module

    monkeypatch.setattr(research_module, "research_command", _ok_command)

    await cli_main.async_main()

    events = _read_events(buffer_dir)
    assert events[0]["action"] == {"query": "how does auth work?"}


@pytest.mark.asyncio
async def test_research_query_is_redacted_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = SimpleNamespace(
        command="research", verbose=False, query="how does auth work?"
    )
    buffer_dir = _patch_common(monkeypatch, args, tmp_path)  # save_sensitive_data=False

    from chunkhound.api.cli.commands import research as research_module

    monkeypatch.setattr(research_module, "research_command", _ok_command)

    await cli_main.async_main()

    events = _read_events(buffer_dir)
    assert events[0]["action"] == {"query": None}


@pytest.mark.asyncio
async def test_failed_command_records_internal_error_and_exits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = SimpleNamespace(command="search", verbose=False, query="x")
    buffer_dir = _patch_common(monkeypatch, args, tmp_path)

    from chunkhound.api.cli.commands import search as search_module

    monkeypatch.setattr(search_module, "search_command", _failing_command)

    with pytest.raises(SystemExit):
        await cli_main.async_main()

    events = _read_events(buffer_dir)
    assert len(events) == 1
    assert events[0]["success"] is False
    assert events[0]["internal_error_type"] == "KeyError"


@pytest.mark.asyncio
async def test_command_failing_via_sys_exit_is_recorded_as_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Regression guard: every real command (search.py, code_mapper.py, etc.)
    # handles its own errors internally and calls sys.exit(N) directly --
    # it never lets a bare exception propagate to async_main() the way
    # _failing_command above does. SystemExit is a BaseException, so this
    # exercises the actual failure path that used to skip `except Exception`
    # entirely and leave the command's event unrecorded.
    args = SimpleNamespace(command="search", verbose=False, query="x")
    buffer_dir = _patch_common(monkeypatch, args, tmp_path)

    from chunkhound.api.cli.commands import search as search_module

    monkeypatch.setattr(search_module, "search_command", _sys_exit_1_command)

    with pytest.raises(SystemExit) as exc_info:
        await cli_main.async_main()
    assert exc_info.value.code == 1

    events = _read_events(buffer_dir)
    assert len(events) == 1
    assert events[0]["success"] is False
    assert events[0]["internal_error_type"] == "SystemExit"


@pytest.mark.asyncio
async def test_command_exiting_zero_via_sys_exit_is_left_unrecorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # commands/run.py's own internal KeyboardInterrupt handler calls
    # sys.exit(0) after an interrupted (not successful) indexing run --
    # this must not be misrecorded as a successful command, matching the
    # top-level KeyboardInterrupt exemption above.
    args = SimpleNamespace(command="search", verbose=False, query="x")
    buffer_dir = _patch_common(monkeypatch, args, tmp_path)

    from chunkhound.api.cli.commands import search as search_module

    monkeypatch.setattr(search_module, "search_command", _sys_exit_0_command)

    with pytest.raises(SystemExit) as exc_info:
        await cli_main.async_main()
    assert exc_info.value.code == 0

    assert _read_events(buffer_dir) == []


@pytest.mark.asyncio
async def test_mcp_command_is_not_wrapped_by_the_cli_hook(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = SimpleNamespace(command="mcp", verbose=False)
    buffer_dir = _patch_common(monkeypatch, args, tmp_path)

    from chunkhound.api.cli.commands import mcp as mcp_module

    monkeypatch.setattr(mcp_module, "mcp_command", _ok_command)

    await cli_main.async_main()

    assert _read_events(buffer_dir) == []
    # "mcp" builds and owns its own recorder (one per server process, see
    # mcp_server/base.py) -- the CLI hook must not construct a second,
    # redundant one (background flush thread + reqwest client) just to
    # never use it. No recorder built means the buffer dir is never created.
    assert not buffer_dir.exists()


@pytest.mark.asyncio
async def test_daemon_command_is_not_wrapped_by_the_cli_hook(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = SimpleNamespace(command="_daemon", verbose=False)
    buffer_dir = _patch_common(monkeypatch, args, tmp_path)

    from chunkhound.api.cli.commands import daemon as daemon_module

    monkeypatch.setattr(daemon_module, "daemon_command", _ok_command)

    await cli_main.async_main()

    assert _read_events(buffer_dir) == []
    assert not buffer_dir.exists()


async def _ok_command(args, config) -> None:
    return None


async def _failing_command(args, config) -> None:
    raise KeyError("boom")


async def _sys_exit_1_command(args, config) -> None:
    import sys

    sys.exit(1)


async def _sys_exit_0_command(args, config) -> None:
    import sys

    sys.exit(0)
