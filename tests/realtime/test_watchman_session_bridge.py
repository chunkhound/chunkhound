from __future__ import annotations

import asyncio
import os
import stat
import sys
import textwrap
from pathlib import Path

import pytest

from chunkhound.watchman import PrivateWatchmanSidecar, WatchmanCliSession

pytestmark = pytest.mark.requires_native_watchman

_FAKE_WATCHMAN_CLI = """\
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def emit(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sockname")
    parser.add_argument("--unix-listener-path")
    parser.add_argument("--named-pipe-path")
    parser.add_argument("--pidfile")
    parser.add_argument("--statefile")
    parser.add_argument("--logfile")
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument("--json-command", action="store_true")
    parser.add_argument("--no-spawn", action="store_true")
    parser.add_argument("--no-pretty", action="store_true")
    parser.add_argument("--server-encoding")
    parser.add_argument("--output-encoding")
    args, extra = parser.parse_known_args(argv)

    if extra:
        print(f"unsupported args: {extra}", file=sys.stderr)
        return 64

    if not (
        (args.sockname or args.unix_listener_path or args.named_pipe_path)
        and args.json_command
        and args.no_spawn
        and args.no_pretty
        and args.server_encoding == "json"
        and args.output_encoding == "json"
    ):
        print("missing expected persistent client flags", file=sys.stderr)
        return 64

    socket_path = args.named_pipe_path or args.unix_listener_path or args.sockname
    socket_exists = False
    if socket_path and socket_path.startswith('\\\\\\\\.\\\\pipe\\\\'):
        socket_exists = True
    elif socket_path and Path(socket_path).exists():
        socket_exists = True
    if not socket_path or not socket_exists:
        print("socket missing", file=sys.stderr)
        return 69

    missing_capability = os.environ.get("CHUNKHOUND_TEST_WATCHMAN_MISSING_CAPABILITY")
    watch_root = os.environ.get("CHUNKHOUND_TEST_WATCHMAN_WATCH_ROOT")
    relative_path = os.environ.get("CHUNKHOUND_TEST_WATCHMAN_RELATIVE_PATH")
    emit_log_before_watch_project = (
        os.environ.get("CHUNKHOUND_TEST_WATCHMAN_EMIT_LOG_BEFORE_WATCH_PROJECT")
        == "1"
    )
    emit_pdu_after_subscribe = (
        os.environ.get("CHUNKHOUND_TEST_WATCHMAN_EMIT_PDU_AFTER_SUBSCRIBE") == "1"
    )
    exit_after_subscribe = (
        os.environ.get("CHUNKHOUND_TEST_WATCHMAN_EXIT_AFTER_SUBSCRIBE") == "1"
    )
    watch_project_logged = False

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        command = json.loads(line)
        name = command[0]
        if name == "version":
            emit(
                {
                    "version": "0.0.0-test",
                    "capabilities": {
                        "cmd-watch-project": missing_capability != "cmd-watch-project",
                        "relative_root": missing_capability != "relative_root",
                    },
                }
            )
            continue
        if name == "watch-project":
            if emit_log_before_watch_project and not watch_project_logged:
                emit({"log": "fake watchman log"})
                watch_project_logged = True
            emit(
                {
                    "version": "0.0.0-test",
                    "watch": watch_root or command[1],
                    **(
                        {"relative_path": relative_path}
                        if relative_path not in {None, "", "."}
                        else {}
                    ),
                }
            )
            continue
        if name == "subscribe":
            emit({"version": "0.0.0-test", "subscribe": command[2]})
            if emit_pdu_after_subscribe:
                emit(
                    {
                        "subscription": command[2],
                        "root": command[1],
                        "clock": "c:1:1",
                        "files": [
                            {
                                "name": "src/example.py",
                                "exists": True,
                                "new": True,
                                "type": "f",
                            }
                        ],
                    }
                )
            if exit_after_subscribe:
                return 75
            continue
        emit({"error": f"unsupported command {name}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
"""


def _prepend_poisoned_python_shims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shims_dir = tmp_path / "poisoned-python"
    shims_dir.mkdir()
    if os.name == "nt":
        for name in ("python.cmd", "python3.cmd"):
            (shims_dir / name).write_text(
                "@echo off\r\nexit /b 97\r\n", encoding="utf-8"
            )
    else:
        for name in ("python", "python3"):
            shim_path = shims_dir / name
            shim_path.write_text("#!/bin/sh\nexit 97\n", encoding="utf-8")
            shim_path.chmod(0o755)
    current_path = os.environ.get("PATH", "")
    monkeypatch.setenv(
        "PATH",
        str(shims_dir)
        if not current_path
        else f"{shims_dir}{os.pathsep}{current_path}",
    )


@pytest.mark.asyncio
async def test_watchman_cli_session_start_ignores_poisoned_python_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _prepend_poisoned_python_shims(tmp_path, monkeypatch)
    sidecar = PrivateWatchmanSidecar(repo_root)
    session: WatchmanCliSession | None = None

    try:
        metadata = await sidecar.start()
        session = WatchmanCliSession(
            binary_path=Path(metadata.binary_path),
            socket_path=sidecar.paths.listener_path,
            statefile_path=sidecar.paths.statefile_path,
            logfile_path=sidecar.paths.logfile_path,
            pidfile_path=sidecar.paths.pidfile_path,
            project_root=repo_root,
        )

        setup = await session.start(target_path=repo_root)

        assert setup.capabilities == {
            "cmd-watch-project": True,
            "relative_root": True,
        }
        assert setup.scope_plan.primary_scope.watch_root == repo_root.resolve()
        assert setup.scope_plan.primary_scope.relative_root is None
        assert session.get_health()["watchman_session_alive"] is True
    finally:
        if session is not None:
            await session.stop()
        await sidecar.stop()


def _write_fake_watchman_cli(tmp_path: Path) -> Path:
    script_path = tmp_path / "fake_watchman_cli.py"
    script_path.write_text(textwrap.dedent(_FAKE_WATCHMAN_CLI), encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR)
    return script_path


@pytest.mark.asyncio
async def test_watchman_cli_session_start_sets_scope_and_queues_pdus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_path = _write_fake_watchman_cli(tmp_path)
    watch_root = tmp_path / "repo"
    target_path = watch_root / "packages" / "api"
    target_path.mkdir(parents=True)
    socket_path = tmp_path / "watchman.sock"
    socket_path.write_text("socket ready\n", encoding="utf-8")
    statefile_path = tmp_path / "watchman.state"
    statefile_path.write_text("state ready\n", encoding="utf-8")
    logfile_path = tmp_path / "watchman.log"
    logfile_path.write_text("log ready\n", encoding="utf-8")
    pidfile_path = tmp_path / "watchman.pid"
    pidfile_path.write_text("123\n", encoding="utf-8")

    monkeypatch.setenv("CHUNKHOUND_TEST_WATCHMAN_WATCH_ROOT", str(watch_root))
    monkeypatch.setenv("CHUNKHOUND_TEST_WATCHMAN_RELATIVE_PATH", "packages/api")
    monkeypatch.setenv("CHUNKHOUND_TEST_WATCHMAN_EMIT_LOG_BEFORE_WATCH_PROJECT", "1")
    monkeypatch.setenv("CHUNKHOUND_TEST_WATCHMAN_EMIT_PDU_AFTER_SUBSCRIBE", "1")

    session = WatchmanCliSession(
        binary_path=script_path,
        socket_path=socket_path,
        statefile_path=statefile_path,
        logfile_path=logfile_path,
        pidfile_path=pidfile_path,
        project_root=tmp_path,
        command_prefix=[sys.executable, str(script_path)],
    )

    setup = await session.start(target_path=target_path)

    assert setup.subscription_name == "chunkhound-live-indexing"
    assert setup.capabilities == {
        "cmd-watch-project": True,
        "relative_root": True,
    }
    assert setup.scope_plan.primary_scope.watch_root == watch_root.resolve()
    assert setup.scope_plan.primary_scope.relative_root == "packages/api"

    pdu = await asyncio.wait_for(session.subscription_queue.get(), timeout=1.0)
    assert pdu["subscription"] == "chunkhound-live-indexing"
    assert pdu["root"] == str(watch_root)
    assert pdu["files"] == [
        {
            "name": "src/example.py",
            "exists": True,
            "new": True,
            "type": "f",
        }
    ]

    health = session.get_health()
    assert health["watchman_session_alive"] is True
    assert health["watchman_subscription_name"] == "chunkhound-live-indexing"
    assert health["watchman_watch_root"] == str(watch_root.resolve())
    assert health["watchman_relative_root"] == "packages/api"
    assert health["watchman_session_last_warning"] == "watchman log: fake watchman log"
    assert health["watchman_subscription_pdu_count"] == 1

    await session.stop()

    assert session.get_health()["watchman_session_alive"] is False


@pytest.mark.asyncio
async def test_watchman_cli_session_requires_relative_root_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_path = _write_fake_watchman_cli(tmp_path)
    target_path = tmp_path / "repo"
    target_path.mkdir()
    socket_path = tmp_path / "watchman.sock"
    socket_path.write_text("socket ready\n", encoding="utf-8")
    statefile_path = tmp_path / "watchman.state"
    statefile_path.write_text("state ready\n", encoding="utf-8")
    logfile_path = tmp_path / "watchman.log"
    logfile_path.write_text("log ready\n", encoding="utf-8")
    pidfile_path = tmp_path / "watchman.pid"
    pidfile_path.write_text("123\n", encoding="utf-8")

    monkeypatch.setenv("CHUNKHOUND_TEST_WATCHMAN_MISSING_CAPABILITY", "relative_root")

    session = WatchmanCliSession(
        binary_path=script_path,
        socket_path=socket_path,
        statefile_path=statefile_path,
        logfile_path=logfile_path,
        pidfile_path=pidfile_path,
        project_root=tmp_path,
        command_prefix=[sys.executable, str(script_path)],
    )

    with pytest.raises(RuntimeError, match="relative_root"):
        await session.start(target_path=target_path)

    await session.stop()


@pytest.mark.asyncio
async def test_watchman_cli_session_reports_unexpected_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_path = _write_fake_watchman_cli(tmp_path)
    target_path = tmp_path / "repo"
    target_path.mkdir()
    socket_path = tmp_path / "watchman.sock"
    socket_path.write_text("socket ready\n", encoding="utf-8")
    statefile_path = tmp_path / "watchman.state"
    statefile_path.write_text("state ready\n", encoding="utf-8")
    logfile_path = tmp_path / "watchman.log"
    logfile_path.write_text("log ready\n", encoding="utf-8")
    pidfile_path = tmp_path / "watchman.pid"
    pidfile_path.write_text("123\n", encoding="utf-8")

    monkeypatch.setenv("CHUNKHOUND_TEST_WATCHMAN_EXIT_AFTER_SUBSCRIBE", "1")

    session = WatchmanCliSession(
        binary_path=script_path,
        socket_path=socket_path,
        statefile_path=statefile_path,
        logfile_path=logfile_path,
        pidfile_path=pidfile_path,
        project_root=tmp_path,
        command_prefix=[sys.executable, str(script_path)],
    )

    await session.start(target_path=target_path)
    message = await asyncio.wait_for(session.wait_for_unexpected_exit(), timeout=1.0)

    assert message is not None
    assert "exited unexpectedly" in message
    assert session.get_health()["watchman_session_alive"] is False
    assert "exited unexpectedly" in (
        session.get_health()["watchman_session_last_error"] or ""
    )

    await session.stop()
