from __future__ import annotations

import hashlib
import importlib.resources
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from platform import machine as current_machine
from platform import system as current_system

_RUNTIME_PACKAGE = "chunkhound.watchman_runtime"
_DEFAULT_RUNTIME_DIRNAME = "chunkhound-watchman-runtime"
_MACHINE_ALIASES = {
    "amd64": "x86_64",
    "arm64e": "arm64",
    "aarch64": "arm64",
    "x64": "x86_64",
}
_SUPPORTED_PLATFORM_ROOTS = {
    ("linux", "x86_64"): PurePosixPath("platforms/linux-x86_64"),
    ("darwin", "arm64"): PurePosixPath("platforms/macos-arm64"),
    ("darwin", "x86_64"): PurePosixPath("platforms/macos-x86_64"),
    ("windows", "x86_64"): PurePosixPath("platforms/windows-x86_64"),
}


class UnsupportedWatchmanRuntimePlatformError(RuntimeError):
    """Raised when no packaged Watchman payload exists for the requested platform."""


@dataclass(frozen=True)
class PackagedWatchmanRuntime:
    """Resolved packaged Watchman runtime metadata."""

    platform_tag: str
    runtime_version: str
    relative_root: PurePosixPath
    relative_binary_path: PurePosixPath
    probe_args: tuple[str, ...]
    packaging_decision: str
    source_digest: str
    source_size: int

    @property
    def packaged_binary_path(self) -> PurePosixPath:
        return self.relative_root / self.relative_binary_path


def _validate_relative_path(relative_path: str) -> PurePosixPath:
    candidate = PurePosixPath(relative_path)
    if candidate.is_absolute():
        raise ValueError(f"Asset path must be relative: {relative_path}")
    if ".." in candidate.parts:
        raise ValueError(f"Asset path must not traverse parents: {relative_path}")
    if not candidate.parts:
        raise ValueError("Asset path must not be empty")
    return candidate


def _normalize_platform_key(
    *, system_name: str | None = None, machine_name: str | None = None
) -> tuple[str, str]:
    normalized_system = (system_name or current_system()).strip().lower()
    normalized_machine = (machine_name or current_machine()).strip().lower()
    normalized_machine = _MACHINE_ALIASES.get(normalized_machine, normalized_machine)
    return normalized_system, normalized_machine


def _read_packaged_bytes(relative_path: PurePosixPath) -> bytes:
    with (
        importlib.resources.files(_RUNTIME_PACKAGE)
        .joinpath(*relative_path.parts)
        .open("rb") as handle
    ):
        return handle.read()


def _read_packaged_json(relative_path: PurePosixPath) -> dict[str, object]:
    with (
        importlib.resources.files(_RUNTIME_PACKAGE)
        .joinpath(*relative_path.parts)
        .open("r", encoding="utf-8") as handle
    ):
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {relative_path}")
    return loaded


def _require_manifest_string(manifest: dict[str, object], key: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest field {key!r} must be a non-empty string")
    return value


def _require_manifest_args(
    manifest: dict[str, object], key: str
) -> tuple[str, ...]:
    value = manifest.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Manifest field {key!r} must be a non-empty list")
    parsed: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(f"Manifest field {key!r} must contain strings")
        parsed.append(item)
    return tuple(parsed)


def resolve_packaged_watchman_runtime(
    *, system_name: str | None = None, machine_name: str | None = None
) -> PackagedWatchmanRuntime:
    """Resolve the packaged Watchman runtime for the requested platform."""

    platform_key = _normalize_platform_key(
        system_name=system_name, machine_name=machine_name
    )
    relative_root = _SUPPORTED_PLATFORM_ROOTS.get(platform_key)
    if relative_root is None:
        available = ", ".join(
            f"{system_key}/{machine_key}"
            for system_key, machine_key in sorted(_SUPPORTED_PLATFORM_ROOTS)
        )
        raise UnsupportedWatchmanRuntimePlatformError(
            "No packaged Watchman runtime for "
            f"{platform_key[0]}/{platform_key[1]}. Available: {available}"
        )

    manifest = _read_packaged_json(relative_root / "manifest.json")
    relative_binary_path = _validate_relative_path(
        _require_manifest_string(manifest, "binary")
    )
    payload = _read_packaged_bytes(relative_root / relative_binary_path)
    return PackagedWatchmanRuntime(
        platform_tag=_require_manifest_string(manifest, "platform"),
        runtime_version=_require_manifest_string(manifest, "runtime_version"),
        relative_root=relative_root,
        relative_binary_path=relative_binary_path,
        probe_args=_require_manifest_args(manifest, "probe_args"),
        packaging_decision=_require_manifest_string(manifest, "packaging_decision"),
        source_digest=hashlib.sha256(payload).hexdigest(),
        source_size=len(payload),
    )


def materialize_watchman_binary(
    *,
    destination_root: Path | None = None,
    system_name: str | None = None,
    machine_name: str | None = None,
) -> Path:
    """Copy the packaged Watchman payload to a stable executable path."""

    runtime = resolve_packaged_watchman_runtime(
        system_name=system_name, machine_name=machine_name
    )
    root = destination_root
    if root is None:
        root = Path(tempfile.gettempdir()) / _DEFAULT_RUNTIME_DIRNAME
    destination_root = root.expanduser().resolve()
    destination_path = (
        destination_root
        / runtime.platform_tag
        / runtime.runtime_version
        / Path(*runtime.relative_binary_path.parts)
    )
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    expected_payload = _read_packaged_bytes(runtime.packaged_binary_path)
    needs_write = True
    if destination_path.is_file():
        current_payload = destination_path.read_bytes()
        current_digest = hashlib.sha256(current_payload).hexdigest()
        needs_write = current_digest != runtime.source_digest

    if needs_write:
        temp_path = destination_path.with_name(f"{destination_path.name}.tmp")
        temp_path.write_bytes(expected_payload)
        os.replace(temp_path, destination_path)

    if os.name != "nt":
        mode = destination_path.stat().st_mode
        os.chmod(
            destination_path,
            mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
        )

    return destination_path
