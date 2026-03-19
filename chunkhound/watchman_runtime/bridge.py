from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers.polling import PollingObserver

_VERSION = "watchman-runtime-bridge-2026-03"
_COMMAND_POLL_SECONDS = 0.1
_WATCH_POLL_SECONDS = 0.2
_COMMAND_SENTINEL = object()


def _emit(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _stderr(message: str) -> None:
    print(message, file=sys.stderr)


def _normalize_relative_root(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    normalized = value.strip().replace("\\", "/")
    if normalized in {"", "."}:
        return None
    candidate = PurePosixPath(normalized)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        return None
    return candidate.as_posix()


def _default_subscription_payload(
    *, subscription_name: str, watch_root: str
) -> dict[str, object]:
    return {
        "subscription": subscription_name,
        "root": watch_root,
        "clock": "c:0:1",
        "files": [
            {
                "name": "src/example.py",
                "exists": True,
                "new": True,
                "type": "f",
            }
        ],
    }


def _capability_state(name: str, missing_capability: str | None) -> bool:
    return missing_capability != name


@dataclass(slots=True)
class RuntimeSubscription:
    subscription_name: str
    watch_root: Path
    relative_root: str | None
    event_queue: queue.Queue[dict[str, object]]
    _clock: int = 0
    _observer: PollingObserver | None = None

    @property
    def requested_root(self) -> Path:
        if self.relative_root is None:
            return self.watch_root
        return self.watch_root.joinpath(*PurePosixPath(self.relative_root).parts)

    def start(self) -> None:
        handler = _RuntimeEventHandler(subscription=self)
        observer = PollingObserver(timeout=_WATCH_POLL_SECONDS)
        observer.schedule(handler, str(self.requested_root), recursive=True)
        observer.start()
        self._observer = observer

    def stop(self) -> None:
        observer = self._observer
        self._observer = None
        if observer is None:
            return
        observer.stop()
        observer.join(timeout=5.0)

    def emit_file_event(
        self, *, absolute_path: Path, exists: bool, is_new: bool
    ) -> None:
        try:
            relative_name = absolute_path.resolve(strict=False).relative_to(
                self.requested_root.resolve(strict=False)
            )
        except ValueError:
            return

        relative_posix = PurePosixPath(relative_name).as_posix()
        if not relative_posix or relative_posix == ".":
            return

        self._clock += 1
        self.event_queue.put(
            {
                "subscription": self.subscription_name,
                "root": str(self.watch_root),
                "clock": f"c:0:{self._clock}",
                "files": [
                    {
                        "name": relative_posix,
                        "exists": exists,
                        "new": is_new,
                        "type": "f",
                    }
                ],
            }
        )


class _RuntimeEventHandler(FileSystemEventHandler):
    def __init__(self, *, subscription: RuntimeSubscription) -> None:
        self._subscription = subscription

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._subscription.emit_file_event(
            absolute_path=Path(os.fsdecode(event.src_path)),
            exists=True,
            is_new=True,
        )

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._subscription.emit_file_event(
            absolute_path=Path(os.fsdecode(event.src_path)),
            exists=True,
            is_new=False,
        )

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._subscription.emit_file_event(
            absolute_path=Path(os.fsdecode(event.src_path)),
            exists=False,
            is_new=False,
        )

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._subscription.emit_file_event(
            absolute_path=Path(os.fsdecode(event.src_path)),
            exists=False,
            is_new=False,
        )
        self._subscription.emit_file_event(
            absolute_path=Path(os.fsdecode(event.dest_path)),
            exists=True,
            is_new=True,
        )


@dataclass(slots=True)
class RuntimeClient:
    sockname: Path
    version: str
    command_queue: queue.Queue[object] = field(default_factory=queue.Queue)
    event_queue: queue.Queue[dict[str, object]] = field(default_factory=queue.Queue)
    subscriptions: dict[str, RuntimeSubscription] = field(default_factory=dict)

    def run(self) -> int:
        if not self.sockname.exists():
            _stderr("chunkhound watchman runtime: private socket is missing")
            return 69

        reader = threading.Thread(target=self._command_reader, daemon=True)
        reader.start()
        try:
            return self._main_loop()
        finally:
            self._stop_all_subscriptions()

    def _command_reader(self) -> None:
        try:
            for raw_line in sys.stdin:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    self.command_queue.put(error)
                    return
                self.command_queue.put(payload)
        finally:
            self.command_queue.put(_COMMAND_SENTINEL)

    def _main_loop(self) -> int:
        while True:
            command_or_signal = self._poll_next_command()
            if command_or_signal is _COMMAND_SENTINEL:
                return 0
            if isinstance(command_or_signal, json.JSONDecodeError):
                _stderr("chunkhound watchman runtime: invalid JSON command")
                return 65
            if command_or_signal is not None:
                result = self._handle_command(command_or_signal)
                if result is not None:
                    return result
            self._drain_event_queue()

    def _poll_next_command(self) -> object | None:
        try:
            return self.command_queue.get(timeout=_COMMAND_POLL_SECONDS)
        except queue.Empty:
            return None

    def _drain_event_queue(self) -> None:
        while True:
            try:
                payload = self.event_queue.get_nowait()
            except queue.Empty:
                return
            _emit(payload)

    def _handle_command(self, command: object) -> int | None:
        if not isinstance(command, list) or not command:
            _emit({"error": "expected command array"})
            return None

        command_name = command[0]
        if not isinstance(command_name, str):
            _emit({"error": "expected command name string"})
            return None

        if command_name == "version":
            self._handle_version(command)
            return None
        if command_name == "watch-project":
            self._handle_watch_project(command)
            return None
        if command_name == "subscribe":
            self._handle_subscribe(command)
            return None

        _emit({"error": f"unsupported command {command_name}"})
        return None

    def _handle_version(self, command: list[object]) -> None:
        requested: list[str] = []
        if len(command) > 1 and isinstance(command[1], dict):
            raw_requested = command[1].get("required")
            if isinstance(raw_requested, list):
                requested = [str(item) for item in raw_requested]

        missing_capability = os.environ.get(
            "CHUNKHOUND_FAKE_WATCHMAN_MISSING_CAPABILITY"
        )
        capabilities = {
            "cmd-watch-project": _capability_state(
                "cmd-watch-project", missing_capability
            ),
            "relative_root": _capability_state("relative_root", missing_capability),
        }
        for name in requested:
            capabilities.setdefault(name, _capability_state(name, missing_capability))

        _emit({"version": self.version, "capabilities": capabilities})

    def _handle_watch_project(self, command: list[object]) -> None:
        if len(command) < 2:
            _emit({"error": "watch-project requires a target path"})
            return

        target_path = Path(str(command[1])).resolve(strict=False)
        watch_root_override = os.environ.get("CHUNKHOUND_FAKE_WATCHMAN_WATCH_ROOT")
        relative_path_override = _normalize_relative_root(
            os.environ.get("CHUNKHOUND_FAKE_WATCHMAN_RELATIVE_PATH")
        )
        response: dict[str, object] = {
            "version": self.version,
            "watch": str(Path(watch_root_override).resolve(strict=False))
            if watch_root_override
            else str(target_path),
        }
        if relative_path_override is not None:
            response["relative_path"] = relative_path_override
        _emit(response)

    def _handle_subscribe(self, command: list[object]) -> None:
        if len(command) < 4:
            _emit({"error": "subscribe requires a root, name, and payload"})
            return

        watch_root = Path(str(command[1])).resolve(strict=False)
        subscription_name = str(command[2])
        payload = command[3] if isinstance(command[3], dict) else {}
        relative_root = _normalize_relative_root(payload.get("relative_root"))
        subscription = RuntimeSubscription(
            subscription_name=subscription_name,
            watch_root=watch_root,
            relative_root=relative_root,
            event_queue=self.event_queue,
        )
        previous = self.subscriptions.pop(subscription_name, None)
        if previous is not None:
            previous.stop()

        requested_root = subscription.requested_root
        requested_root.mkdir(parents=True, exist_ok=True)
        subscription.start()
        self.subscriptions[subscription_name] = subscription

        _emit({"version": self.version, "subscribe": subscription_name})

        emit_log = os.environ.get("CHUNKHOUND_FAKE_WATCHMAN_EMIT_LOG_AFTER_SUBSCRIBE")
        if isinstance(emit_log, str) and emit_log:
            _emit({"log": emit_log})

        emit_pdu = os.environ.get("CHUNKHOUND_FAKE_WATCHMAN_EMIT_SUBSCRIPTION_PDU")
        if isinstance(emit_pdu, str) and emit_pdu:
            try:
                rendered = json.loads(emit_pdu)
            except json.JSONDecodeError:
                rendered = _default_subscription_payload(
                    subscription_name=subscription_name,
                    watch_root=str(watch_root),
                )
            if not isinstance(rendered, dict):
                rendered = _default_subscription_payload(
                    subscription_name=subscription_name,
                    watch_root=str(watch_root),
                )
            rendered.setdefault("subscription", subscription_name)
            rendered.setdefault("root", str(watch_root))
            _emit(rendered)

    def _stop_all_subscriptions(self) -> None:
        while self.subscriptions:
            _, subscription = self.subscriptions.popitem()
            subscription.stop()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--no-save-state", action="store_true")
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument("--json-command", action="store_true")
    parser.add_argument("--no-spawn", action="store_true")
    parser.add_argument("--no-pretty", action="store_true")
    parser.add_argument("--sockname")
    parser.add_argument("--statefile")
    parser.add_argument("--logfile")
    parser.add_argument("--server-encoding")
    parser.add_argument("--output-encoding")
    return parser


def _run_sidecar(args: argparse.Namespace) -> int:
    if (
        not args.foreground
        or not args.sockname
        or not args.statefile
        or not args.logfile
    ):
        _stderr("chunkhound watchman runtime: missing sidecar path flags")
        return 64

    delay_seconds = float(
        os.environ.get("CHUNKHOUND_FAKE_WATCHMAN_START_DELAY_SECONDS", "0") or "0"
    )
    if delay_seconds > 0:
        time.sleep(delay_seconds)

    logfile_path = Path(args.logfile)
    socket_path = Path(args.sockname)
    statefile_path = Path(args.statefile)

    logfile_path.parent.mkdir(parents=True, exist_ok=True)
    with logfile_path.open("a", encoding="utf-8") as handle:
        handle.write("watchman runtime sidecar start\n")

    if os.environ.get("CHUNKHOUND_FAKE_WATCHMAN_FAIL_BEFORE_READY") == "1":
        return 70

    socket_path.parent.mkdir(parents=True, exist_ok=True)
    statefile_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.write_text("socket ready\n", encoding="utf-8")
    statefile_path.write_text("state ready\n", encoding="utf-8")

    stop_requested = threading.Event()

    def _request_stop(_signum: int, _frame: Any) -> None:
        stop_requested.set()

    for name in ("SIGTERM", "SIGINT"):
        signum = getattr(signal, name, None)
        if signum is not None:
            signal.signal(signum, _request_stop)

    while not stop_requested.wait(0.2):
        continue
    return 0


def _run_client(args: argparse.Namespace) -> int:
    if (
        args.foreground
        or args.statefile
        or args.logfile
        or not args.sockname
        or not args.persistent
        or not args.json_command
        or not args.no_spawn
        or not args.no_pretty
        or args.server_encoding != "json"
        or args.output_encoding != "json"
    ):
        _stderr("chunkhound watchman runtime: missing client session flags")
        return 64

    client = RuntimeClient(sockname=Path(args.sockname), version=_VERSION)
    return client.run()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args, extra = parser.parse_known_args(argv)

    if extra:
        _stderr(
            "chunkhound watchman runtime: unsupported flag "
            + " ".join(str(item) for item in extra)
        )
        return 64

    if args.version:
        print(f"watchman {_VERSION}")
        return 0

    client_mode_requested = any(
        [
            args.persistent,
            args.json_command,
            args.no_spawn,
            args.no_pretty,
            args.server_encoding is not None,
            args.output_encoding is not None,
        ]
    )
    if client_mode_requested:
        return _run_client(args)
    return _run_sidecar(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
