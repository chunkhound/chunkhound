from __future__ import annotations

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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_synthetic_watchman_wheel(
    tmp_path: Path,
    *,
    wheel_name: str,
    excluded_paths: set[str] | None = None,
) -> Path:
    repo_root = _repo_root()
    wheel_path = tmp_path / wheel_name
    excluded = excluded_paths or set()

    with zipfile.ZipFile(wheel_path, "w") as zf:
        for relative_path, content in _SYNTHETIC_TEXT_FILES.items():
            info = zipfile.ZipInfo(relative_path)
            info.create_system = 3
            info.external_attr = 0o644 << 16
            zf.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED)

        for relative_path in _SYNTHETIC_WHEEL_FILES:
            if relative_path in excluded:
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
