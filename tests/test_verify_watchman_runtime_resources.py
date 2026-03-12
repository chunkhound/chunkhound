from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest

import hatch_build
from scripts import verify_watchman_runtime_resources as watchman_verifier

pytestmark = pytest.mark.requires_native_watchman

_SYNTHETIC_WHEEL_FILES: tuple[str, ...] = (
    "chunkhound/watchman_runtime/loader.py",
    "chunkhound/watchman_runtime/bridge.py",
    *watchman_verifier._REQUIRED_WHEEL_PATHS,
)
_SYNTHETIC_TEXT_FILES: dict[str, str] = {
    "chunkhound/__init__.py": '"""Synthetic ChunkHound test package."""\n',
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _host_runtime_binary_path() -> str:
    platform_tag = hatch_build._host_watchman_platform()
    manifest_path = (
        _repo_root()
        / "chunkhound"
        / "watchman_runtime"
        / "platforms"
        / platform_tag
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binary_path = manifest.get("binary")
    if not isinstance(binary_path, str) or not binary_path:
        raise AssertionError(f"Invalid manifest binary path in {manifest_path}")
    return f"chunkhound/watchman_runtime/platforms/{platform_tag}/{binary_path}"


def _hydrated_runtime_sources() -> dict[str, Path]:
    return {
        destination_path: Path(source_path)
        for source_path, destination_path in hatch_build._hydrate_runtime_for_build().items()
    }


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
    hydrated_runtime_sources = _hydrated_runtime_sources()

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
            hydrated_source_path = hydrated_runtime_sources.get(relative_path)
            if hydrated_source_path is not None:
                payload = hydrated_source_path.read_bytes()
                info.external_attr = (hydrated_source_path.stat().st_mode & 0xFFFF) << 16
            elif source_path.exists():
                payload = source_path.read_bytes()
                info.external_attr = (source_path.stat().st_mode & 0xFFFF) << 16
            else:
                raise FileNotFoundError(source_path)
            zf.writestr(
                info,
                payload,
                compress_type=zipfile.ZIP_DEFLATED,
            )

    return wheel_path


def test_main_accepts_synthetic_platform_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel_path = _build_synthetic_watchman_wheel(
        tmp_path,
        wheel_name="chunkhound-0.0.0-py3-none-manylinux_2_36_x86_64.whl",
    )
    calls: list[Path] = []
    monkeypatch.setattr(
        watchman_verifier,
        "_verify_runtime_reads",
        lambda *, wheel_path: calls.append(wheel_path),
    )

    assert watchman_verifier.main([str(wheel_path)]) == 0
    assert calls == [wheel_path]


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
        excluded_paths={_host_runtime_binary_path()},
    )

    with pytest.raises(RuntimeError, match="missing required Watchman runtime"):
        watchman_verifier.main([str(wheel_path)])


def test_main_surfaces_runtime_verification_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel_path = _build_synthetic_watchman_wheel(
        tmp_path,
        wheel_name="chunkhound-0.0.0-py3-none-manylinux_2_36_x86_64.whl",
    )
    monkeypatch.setattr(
        watchman_verifier,
        "_verify_runtime_reads",
        lambda *, wheel_path: (_ for _ in ()).throw(RuntimeError("native daemon")),
    )

    with pytest.raises(RuntimeError, match="native daemon"):
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


def test_remove_tree_with_retries_terminates_windows_processes_using_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    locked_root = tmp_path / "locked-root"
    locked_root.mkdir()
    terminated: list[int] = []
    original_rmtree = shutil.rmtree
    attempts = {"count": 0}

    class FakeProcess:
        def __init__(self, pid: int, cwd: str | None, cmdline: list[str]) -> None:
            self.info = {"pid": pid, "cwd": cwd, "cmdline": cmdline}

    def flaky_rmtree(path: Path) -> None:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise PermissionError("simulated Windows handle delay")
        original_rmtree(path)

    monkeypatch.setattr(watchman_verifier.os, "name", "nt", raising=False)
    monkeypatch.setattr(watchman_verifier.shutil, "rmtree", flaky_rmtree)
    monkeypatch.setattr(watchman_verifier.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(
        watchman_verifier.psutil,
        "process_iter",
        lambda *_args, **_kwargs: iter(
            [
                FakeProcess(101, str(locked_root), []),
                FakeProcess(202, None, [str(locked_root / "child.py")]),
                FakeProcess(303, str(tmp_path / "other"), []),
            ]
        ),
    )
    monkeypatch.setattr(
        watchman_verifier, "_terminate_process_tree", lambda pid: terminated.append(pid)
    )

    watchman_verifier._remove_tree_with_retries(locked_root, attempts=2)

    assert attempts["count"] == 2
    assert terminated == [101, 202]
