from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import pytest

from scripts import verify_watchman_runtime_resources as watchman_verifier

_SYNTHETIC_WHEEL_FILES: tuple[str, ...] = (
    "chunkhound/watchman_runtime/loader.py",
    *watchman_verifier._REQUIRED_WHEEL_PATHS,
)
_SYNTHETIC_TEXT_FILES: dict[str, str] = {
    "chunkhound/__init__.py": '"""Synthetic ChunkHound test package."""\n',
}
_BROKEN_LIVE_MUTATION_BRIDGE = """\
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path

VERSION = "watchman-runtime-bridge-2026-03"


def emit(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
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
    args = parser.parse_args(argv)

    if args.version:
        print(f"watchman {VERSION}")
        return 0

    client_mode = any(
        [
            args.persistent,
            args.json_command,
            args.no_spawn,
            args.no_pretty,
            args.server_encoding is not None,
            args.output_encoding is not None,
        ]
    )
    if client_mode:
        if not args.sockname or not Path(args.sockname).exists():
            return 69
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            command = json.loads(line)
            name = command[0]
            if name == "version":
                emit(
                    {
                        "version": VERSION,
                        "capabilities": {
                            "cmd-watch-project": True,
                            "relative_root": True,
                        },
                    }
                )
                continue
            if name == "watch-project":
                emit({"version": VERSION, "watch": str(Path(command[1]).resolve())})
                continue
            if name == "subscribe":
                emit({"version": VERSION, "subscribe": str(command[2])})
                continue
            emit({"error": f"unsupported command {name}"})
        return 0

    Path(args.sockname).parent.mkdir(parents=True, exist_ok=True)
    Path(args.statefile).parent.mkdir(parents=True, exist_ok=True)
    Path(args.logfile).parent.mkdir(parents=True, exist_ok=True)
    Path(args.sockname).write_text("socket ready\\n", encoding="utf-8")
    Path(args.statefile).write_text("state ready\\n", encoding="utf-8")
    Path(args.logfile).write_text("watchman runtime sidecar start\\n", encoding="utf-8")

    stop_requested = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_requested.set()

    for signal_name in ("SIGTERM", "SIGINT"):
        signum = getattr(signal, signal_name, None)
        if signum is not None:
            signal.signal(signum, request_stop)

    while not stop_requested.wait(0.2):
        continue
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_synthetic_watchman_wheel(
    tmp_path: Path,
    *,
    wheel_name: str,
    excluded_paths: set[str] | None = None,
    overridden_text_files: dict[str, str] | None = None,
) -> Path:
    repo_root = _repo_root()
    wheel_path = tmp_path / wheel_name
    excluded = excluded_paths or set()
    overrides = overridden_text_files or {}

    with zipfile.ZipFile(wheel_path, "w") as zf:
        for relative_path, content in _SYNTHETIC_TEXT_FILES.items():
            info = zipfile.ZipInfo(relative_path)
            info.create_system = 3
            info.external_attr = 0o644 << 16
            zf.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED)

        for relative_path in _SYNTHETIC_WHEEL_FILES:
            if relative_path in excluded:
                continue
            overridden_text = overrides.get(relative_path)
            if overridden_text is not None:
                info = zipfile.ZipInfo(relative_path)
                info.create_system = 3
                info.external_attr = 0o644 << 16
                zf.writestr(
                    info,
                    overridden_text,
                    compress_type=zipfile.ZIP_DEFLATED,
                )
                continue
            source_path = repo_root / relative_path
            info = zipfile.ZipInfo(relative_path)
            info.create_system = 3
            info.external_attr = (source_path.stat().st_mode & 0xFFFF) << 16
            zf.writestr(
                info,
                source_path.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
            )

    return wheel_path


def test_main_accepts_synthetic_platform_wheel(tmp_path: Path) -> None:
    wheel_path = _build_synthetic_watchman_wheel(
        tmp_path,
        wheel_name="chunkhound-0.0.0-py3-none-manylinux_2_36_x86_64.whl",
    )

    assert watchman_verifier.main([str(wheel_path)]) == 0


def test_main_rejects_universal_wheel_tag(tmp_path: Path) -> None:
    wheel_path = _build_synthetic_watchman_wheel(
        tmp_path,
        wheel_name="chunkhound-0.0.0-py3-none-any.whl",
    )

    with pytest.raises(RuntimeError, match="py3-none-platform"):
        watchman_verifier.main([str(wheel_path)])


def test_main_rejects_missing_required_runtime_resource(tmp_path: Path) -> None:
    wheel_path = _build_synthetic_watchman_wheel(
        tmp_path,
        wheel_name="chunkhound-0.0.0-py3-none-manylinux_2_36_x86_64.whl",
        excluded_paths={
            "chunkhound/watchman_runtime/platforms/linux-x86_64/bin/watchman"
        },
    )

    with pytest.raises(RuntimeError, match="missing required Watchman runtime"):
        watchman_verifier.main([str(wheel_path)])


def test_main_rejects_runtime_that_never_emits_live_subscription_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel_path = _build_synthetic_watchman_wheel(
        tmp_path,
        wheel_name="chunkhound-0.0.0-py3-none-manylinux_2_36_x86_64.whl",
        overridden_text_files={
            "chunkhound/watchman_runtime/bridge.py": _BROKEN_LIVE_MUTATION_BRIDGE
        },
    )
    monkeypatch.setenv(
        watchman_verifier._LIVE_MUTATION_TIMEOUT_ENV,
        "0.2",
    )

    with pytest.raises(
        RuntimeError, match="timed out waiting for live subscription payload"
    ):
        watchman_verifier.main([str(wheel_path)])


def test_remove_tree_with_retries_retries_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    locked_root = tmp_path / "locked-root"
    locked_root.mkdir()
    (locked_root / "payload.txt").write_text("payload", encoding="utf-8")
    original_rmtree = shutil.rmtree
    attempts = {"count": 0}

    def flaky_rmtree(path: Path) -> None:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise PermissionError("simulated Windows handle delay")
        original_rmtree(path)

    monkeypatch.setattr(watchman_verifier.shutil, "rmtree", flaky_rmtree)
    monkeypatch.setattr(watchman_verifier.time, "sleep", lambda *_args: None)

    watchman_verifier._remove_tree_with_retries(locked_root, attempts=2)

    assert attempts["count"] == 2
    assert not locked_root.exists()
