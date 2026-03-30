from __future__ import annotations

import asyncio
import errno
import os
import stat
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

import chunkhound.watchman.session as watchman_session_module
from chunkhound.watchman import (
    PrivateWatchmanSidecar,
    WatchmanCliSession,
    WatchmanSubscriptionScope,
)

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
        if name == "watch":
            emit({"version": "0.0.0-test", "watch": command[1]})
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


def test_watchman_cli_session_queue_overflow_reports_drop_and_calls_handler(
    tmp_path: Path,
) -> None:
    overflow_calls: list[tuple[dict[str, object], int, int]] = []
    session = WatchmanCliSession(
        binary_path=tmp_path / "watchman",
        socket_path=tmp_path / "watchman.sock",
        statefile_path=tmp_path / "watchman.state",
        logfile_path=tmp_path / "watchman.log",
        pidfile_path=tmp_path / "watchman.pid",
        project_root=tmp_path,
        subscription_overflow_handler=lambda payload, dropped, queue_maxsize: (
            overflow_calls.append((payload, dropped, queue_maxsize))
        ),
    )

    for index in range(session.subscription_queue.maxsize):
        session.subscription_queue.put_nowait({"subscription": f"baseline-{index}"})

    overflow_payload = {
        "subscription": "chunkhound-live-indexing",
        "clock": "c:1:1001",
        "files": [],
    }
    session._queue_subscription_pdu(overflow_payload)

    health = session.get_health()
    assert health["watchman_subscription_pdu_dropped"] == 1
    assert (
        health["watchman_session_last_warning"]
        == "Watchman subscription queue full; dropped a raw subscription PDU"
    )
    assert overflow_calls == [
        (overflow_payload, 1, session.subscription_queue.maxsize)
    ]


def test_watchman_cli_session_bounds_long_secondary_subscription_names(
    tmp_path: Path,
) -> None:
    session = WatchmanCliSession(
        binary_path=tmp_path / "watchman",
        socket_path=tmp_path / "watchman.sock",
        statefile_path=tmp_path / "watchman.state",
        logfile_path=tmp_path / "watchman.log",
        pidfile_path=tmp_path / "watchman.pid",
        project_root=tmp_path,
    )
    target_path = tmp_path / "workspace"
    target_path.mkdir()

    shared_parts = ["very-long-scope-name"] * 8
    scope_a = WatchmanSubscriptionScope(
        requested_path=target_path.joinpath(*shared_parts, "alpha"),
        watch_root=target_path,
        relative_root=None,
        scope_kind="nested_mount",
    )
    scope_b = WatchmanSubscriptionScope(
        requested_path=target_path.joinpath(*shared_parts, "beta"),
        watch_root=target_path,
        relative_root=None,
        scope_kind="nested_mount",
    )

    bounded_a = session._subscription_name_for_scope(
        base_name="chunkhound-live-indexing",
        target_path=target_path,
        scope=scope_a,
        scope_index=1,
    )
    bounded_a_repeat = session._subscription_name_for_scope(
        base_name="chunkhound-live-indexing",
        target_path=target_path,
        scope=scope_a,
        scope_index=1,
    )
    bounded_b = session._subscription_name_for_scope(
        base_name="chunkhound-live-indexing",
        target_path=target_path,
        scope=scope_b,
        scope_index=1,
    )

    assert len(bounded_a) <= WatchmanCliSession._SUBSCRIPTION_NAME_MAX_LENGTH
    assert len(bounded_b) <= WatchmanCliSession._SUBSCRIPTION_NAME_MAX_LENGTH
    assert bounded_a == bounded_a_repeat
    assert bounded_a != bounded_b
    assert bounded_a.startswith("chunkhound-live-indexing--very-long-scope-name")


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


def test_watchman_cli_session_prepared_startup_support_is_transport_aware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = WatchmanCliSession(
        binary_path=tmp_path / "watchman",
        socket_path=tmp_path / "watchman.sock",
        statefile_path=tmp_path / "watchman.state",
        logfile_path=tmp_path / "watchman.log",
        pidfile_path=tmp_path / "watchman.pid",
        project_root=tmp_path,
    )
    shimmed_session = WatchmanCliSession(
        binary_path=tmp_path / "watchman",
        socket_path=tmp_path / "watchman.sock",
        statefile_path=tmp_path / "watchman.state",
        logfile_path=tmp_path / "watchman.log",
        pidfile_path=tmp_path / "watchman.pid",
        project_root=tmp_path,
        command_prefix=[sys.executable, "fake-watchman"],
    )

    monkeypatch.setattr(
        watchman_session_module,
        "resolve_packaged_watchman_runtime",
        lambda: SimpleNamespace(listener_transport="unix_socket"),
    )
    assert session.supports_prepared_session_startup() is True
    assert shimmed_session.supports_prepared_session_startup() is False

    monkeypatch.setattr(
        watchman_session_module,
        "resolve_packaged_watchman_runtime",
        lambda: SimpleNamespace(listener_transport="named_pipe"),
    )
    assert session.supports_prepared_session_startup() is False


@pytest.mark.asyncio
async def test_watchman_cli_session_start_uses_one_shot_scope_planning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_path = _write_fake_watchman_cli(tmp_path)
    watch_root = tmp_path / "workspace_root"
    nested_mount = watch_root / "chunkhound_workspace"
    watch_root.mkdir(parents=True)
    nested_mount.mkdir(parents=True)
    socket_path = tmp_path / "watchman.sock"
    socket_path.write_text("socket ready\n", encoding="utf-8")
    statefile_path = tmp_path / "watchman.state"
    statefile_path.write_text("state ready\n", encoding="utf-8")
    logfile_path = tmp_path / "watchman.log"
    logfile_path.write_text("log ready\n", encoding="utf-8")
    pidfile_path = tmp_path / "watchman.pid"
    pidfile_path.write_text("123\n", encoding="utf-8")
    operations: list[tuple[object, ...]] = []

    class TrackingSession(WatchmanCliSession):
        async def watch_project(self, target_path: Path) -> dict[str, object]:
            raise AssertionError(
                "unexpected persistent watch-project during fallback start: "
                f"{target_path}"
            )

        async def watch_roots(self, roots) -> tuple[Path, ...]:
            raise AssertionError(
                f"unexpected persistent watch during fallback start: {roots!r}"
            )

        async def startup_watch_project_once(
            self, target_path: Path
        ) -> dict[str, object]:
            operations.append(
                ("startup_watch_project_once", str(target_path.resolve()))
            )
            return await super().startup_watch_project_once(target_path)

        async def startup_watch_roots_once(
            self, roots
        ) -> tuple[Path, ...]:
            operations.append(
                ("startup_watch_roots_once", tuple(str(root) for root in roots))
            )
            return await super().startup_watch_roots_once(roots)

    monkeypatch.setenv("CHUNKHOUND_TEST_WATCHMAN_WATCH_ROOT", str(watch_root))
    monkeypatch.setattr(
        watchman_session_module,
        "discover_nested_linux_mount_roots",
        lambda target_path: (nested_mount.resolve(),),
    )

    session = TrackingSession(
        binary_path=script_path,
        socket_path=socket_path,
        statefile_path=statefile_path,
        logfile_path=logfile_path,
        pidfile_path=pidfile_path,
        project_root=tmp_path,
        command_prefix=[sys.executable, str(script_path)],
    )

    setup = await session.start(target_path=watch_root)

    assert operations == [
        ("startup_watch_project_once", str(watch_root.resolve())),
        ("startup_watch_roots_once", (str(nested_mount.resolve()),)),
    ]
    assert setup.subscription_names == (
        "chunkhound-live-indexing",
        "chunkhound-live-indexing--chunkhound-workspace",
    )

    await session.stop()


@pytest.mark.asyncio
async def test_watchman_cli_session_falls_back_when_direct_socket_connect_races(
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

    connect_attempts = 0

    async def _missing_socket_connection(*, path: str):
        nonlocal connect_attempts
        connect_attempts += 1
        raise FileNotFoundError(errno.ENOENT, "socket missing", path)

    class DirectThenFallbackSession(WatchmanCliSession):
        def _use_direct_socket_session(self) -> bool:
            return True

    monkeypatch.setattr(
        watchman_session_module.asyncio,
        "open_unix_connection",
        _missing_socket_connection,
    )

    session = DirectThenFallbackSession(
        binary_path=script_path,
        socket_path=socket_path,
        statefile_path=statefile_path,
        logfile_path=logfile_path,
        pidfile_path=pidfile_path,
        project_root=tmp_path,
        command_prefix=[sys.executable, str(script_path)],
    )

    setup = await session.start(target_path=target_path)

    assert connect_attempts >= 1
    assert setup.capabilities == {
        "cmd-watch-project": True,
        "relative_root": True,
    }
    assert session.get_health()["watchman_session_alive"] is True

    await session.stop()


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
async def test_watchman_cli_session_start_adds_nested_mount_subscription(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_path = _write_fake_watchman_cli(tmp_path)
    watch_root = tmp_path / "workspace_root"
    nested_mount = watch_root / "chunkhound_workspace"
    watch_root.mkdir(parents=True)
    nested_mount.mkdir(parents=True)
    socket_path = tmp_path / "watchman.sock"
    socket_path.write_text("socket ready\n", encoding="utf-8")
    statefile_path = tmp_path / "watchman.state"
    statefile_path.write_text("state ready\n", encoding="utf-8")
    logfile_path = tmp_path / "watchman.log"
    logfile_path.write_text("log ready\n", encoding="utf-8")
    pidfile_path = tmp_path / "watchman.pid"
    pidfile_path.write_text("123\n", encoding="utf-8")

    monkeypatch.setenv("CHUNKHOUND_TEST_WATCHMAN_WATCH_ROOT", str(watch_root))
    monkeypatch.setenv("CHUNKHOUND_TEST_WATCHMAN_EMIT_PDU_AFTER_SUBSCRIBE", "1")
    monkeypatch.setattr(
        watchman_session_module,
        "discover_nested_linux_mount_roots",
        lambda target_path: (nested_mount.resolve(),),
    )

    session = WatchmanCliSession(
        binary_path=script_path,
        socket_path=socket_path,
        statefile_path=statefile_path,
        logfile_path=logfile_path,
        pidfile_path=pidfile_path,
        project_root=tmp_path,
        command_prefix=[sys.executable, str(script_path)],
    )

    setup = await session.start(target_path=watch_root)

    assert len(setup.scope_plan.scopes) == 2
    assert setup.subscription_names == (
        "chunkhound-live-indexing",
        "chunkhound-live-indexing--chunkhound-workspace",
    )

    first_pdu = await asyncio.wait_for(session.subscription_queue.get(), timeout=1.0)
    second_pdu = await asyncio.wait_for(session.subscription_queue.get(), timeout=1.0)

    assert {first_pdu["subscription"], second_pdu["subscription"]} == {
        "chunkhound-live-indexing",
        "chunkhound-live-indexing--chunkhound-workspace",
    }
    assert {first_pdu["root"], second_pdu["root"]} == {
        str(watch_root),
        str(nested_mount),
    }

    health = session.get_health()
    assert health["watchman_subscription_names"] == [
        "chunkhound-live-indexing",
        "chunkhound-live-indexing--chunkhound-workspace",
    ]
    assert health["watchman_scopes"] == [
        {
            "subscription_name": "chunkhound-live-indexing",
            "scope_kind": "primary",
            "requested_path": str(watch_root.resolve()),
            "watch_root": str(watch_root.resolve()),
            "relative_root": None,
        },
        {
            "subscription_name": "chunkhound-live-indexing--chunkhound-workspace",
            "scope_kind": "nested_mount",
            "requested_path": str(nested_mount.resolve()),
            "watch_root": str(nested_mount.resolve()),
            "relative_root": None,
        },
    ]

    await session.stop()


@pytest.mark.asyncio
async def test_watchman_cli_session_start_adds_nested_junction_subscription(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_path = _write_fake_watchman_cli(tmp_path)
    watch_root = tmp_path / "workspace_root"
    logical_junction = watch_root / "linked_workspace"
    physical_root = tmp_path / "external_workspace"
    watch_root.mkdir(parents=True)
    logical_junction.mkdir(parents=True)
    physical_root.mkdir(parents=True)
    socket_path = tmp_path / "watchman.sock"
    socket_path.write_text("socket ready\n", encoding="utf-8")
    statefile_path = tmp_path / "watchman.state"
    statefile_path.write_text("state ready\n", encoding="utf-8")
    logfile_path = tmp_path / "watchman.log"
    logfile_path.write_text("log ready\n", encoding="utf-8")
    pidfile_path = tmp_path / "watchman.pid"
    pidfile_path.write_text("123\n", encoding="utf-8")

    monkeypatch.setenv("CHUNKHOUND_TEST_WATCHMAN_WATCH_ROOT", str(watch_root))
    monkeypatch.setenv("CHUNKHOUND_TEST_WATCHMAN_EMIT_PDU_AFTER_SUBSCRIBE", "1")
    monkeypatch.setattr(
        watchman_session_module,
        "discover_nested_windows_junction_scopes",
        lambda target_path: (
            WatchmanSubscriptionScope(
                requested_path=logical_junction,
                watch_root=physical_root.resolve(),
                relative_root=None,
                scope_kind="nested_junction",
            ),
        ),
    )

    session = WatchmanCliSession(
        binary_path=script_path,
        socket_path=socket_path,
        statefile_path=statefile_path,
        logfile_path=logfile_path,
        pidfile_path=pidfile_path,
        project_root=tmp_path,
        command_prefix=[sys.executable, str(script_path)],
    )

    setup = await session.start(target_path=watch_root)

    assert len(setup.scope_plan.scopes) == 2
    assert setup.subscription_names == (
        "chunkhound-live-indexing",
        "chunkhound-live-indexing--linked-workspace",
    )

    first_pdu = await asyncio.wait_for(session.subscription_queue.get(), timeout=1.0)
    second_pdu = await asyncio.wait_for(session.subscription_queue.get(), timeout=1.0)

    assert {first_pdu["subscription"], second_pdu["subscription"]} == {
        "chunkhound-live-indexing",
        "chunkhound-live-indexing--linked-workspace",
    }
    assert {first_pdu["root"], second_pdu["root"]} == {
        str(watch_root),
        str(physical_root.resolve()),
    }

    health = session.get_health()
    assert health["watchman_scopes"] == [
        {
            "subscription_name": "chunkhound-live-indexing",
            "scope_kind": "primary",
            "requested_path": str(watch_root.resolve()),
            "watch_root": str(watch_root.resolve()),
            "relative_root": None,
        },
        {
            "subscription_name": "chunkhound-live-indexing--linked-workspace",
            "scope_kind": "nested_junction",
            "requested_path": str(logical_junction),
            "watch_root": str(physical_root.resolve()),
            "relative_root": None,
        },
    ]

    await session.stop()


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
