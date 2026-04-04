from __future__ import annotations

import hashlib
import http.client
import importlib.abc
import importlib.resources
import io
import json
import os
import shutil
import socket
import stat
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from platform import machine as current_machine
from platform import system as current_system
from typing import Literal, cast

if os.name == "nt":
    import msvcrt
else:
    import fcntl

_RUNTIME_PACKAGE = "chunkhound.watchman_runtime"
_PACKAGE_ROOT = Path(__file__).resolve().parent
_DEFAULT_RUNTIME_DIRNAME = "chunkhound-watchman-runtime"
_RUNTIME_CACHE_DIR_ENV = "CHUNKHOUND_WATCHMAN_RUNTIME_CACHE_DIR"
_RUNTIME_DOWNLOAD_TIMEOUT_SECONDS_ENV = (
    "CHUNKHOUND_WATCHMAN_RUNTIME_DOWNLOAD_TIMEOUT_SECONDS"
)
_RUNTIME_DOWNLOAD_RETRIES_ENV = "CHUNKHOUND_WATCHMAN_RUNTIME_DOWNLOAD_RETRIES"
_HYDRATION_MARKER_NAME = ".chunkhound-hydrated.json"
_DEFAULT_RUNTIME_DOWNLOAD_TIMEOUT_SECONDS = 30.0
_DEFAULT_RUNTIME_DOWNLOAD_RETRIES = 2
_MACHINE_ALIASES = {
    "amd64": "x86_64",
    "arm64e": "arm64",
    "aarch64": "arm64",
    "x64": "x86_64",
}
# macOS intentionally ships no packaged Watchman runtime slot in this rollout.
# Keep macOS on the fallback backends until a separate macOS-native follow-up
# exists.
_SUPPORTED_PLATFORM_ROOTS = {
    ("linux", "x86_64"): PurePosixPath("platforms/linux-x86_64"),
    ("windows", "x86_64"): PurePosixPath("platforms/windows-x86_64"),
}
_PYTHON_BRIDGE_LAUNCH_MODE = "python_bridge"
_NATIVE_BINARY_LAUNCH_MODE = "native_binary"
_SOCKNAME_LISTENER_TRANSPORT = "sockname"
_UNIX_SOCKET_LISTENER_TRANSPORT = "unix_socket"
_NAMED_PIPE_LISTENER_TRANSPORT = "named_pipe"
_ZIP_SOURCE_ARCHIVE_FORMAT = "zip"
_DEB_SOURCE_ARCHIVE_FORMAT = "deb"
_AR_GLOBAL_HEADER = b"!<arch>\n"
_AR_HEADER_SIZE = 60

WatchmanRuntimeLaunchMode = Literal["python_bridge", "native_binary"]
WatchmanRuntimeListenerTransport = Literal[
    "sockname",
    "unix_socket",
    "named_pipe",
]
WatchmanRuntimeSourceArchiveFormat = Literal["zip", "deb"]


class UnsupportedWatchmanRuntimePlatformError(RuntimeError):
    """Raised when no packaged Watchman payload exists for the requested platform."""


@dataclass(frozen=True)
class WatchmanRuntimeSourceFile:
    """One staged runtime file sourced from a pinned upstream archive."""

    destination_relative_path: PurePosixPath
    source_relative_path: PurePosixPath


@dataclass(frozen=True)
class WatchmanRuntimeSource:
    """Pinned upstream archive metadata for a runtime source artifact."""

    source_url: str
    source_sha256: str
    source_archive_format: WatchmanRuntimeSourceArchiveFormat
    source_root_prefix: PurePosixPath | None
    source_files: tuple[WatchmanRuntimeSourceFile, ...]


@dataclass(frozen=True)
class _WatchmanRuntimeSourceDescriptor:
    """Manifest descriptor for a single pinned upstream archive."""

    source_url: str
    source_sha256: str
    source_archive_format: WatchmanRuntimeSourceArchiveFormat
    source_root_prefix: PurePosixPath | None
    source_relative_path: PurePosixPath | None


@dataclass(frozen=True)
class PackagedWatchmanRuntime:
    """Resolved packaged Watchman runtime metadata."""

    platform_tag: str
    runtime_version: str
    relative_root: PurePosixPath
    relative_binary_path: PurePosixPath
    relative_support_paths: tuple[PurePosixPath, ...]
    launch_mode: WatchmanRuntimeLaunchMode
    listener_transport: WatchmanRuntimeListenerTransport
    probe_args: tuple[str, ...]
    wheel_platform_tags: tuple[str, ...]
    env_path_entries: dict[str, tuple[PurePosixPath, ...]]
    packaging_decision: str
    source_archives: tuple[WatchmanRuntimeSource, ...]
    hydrated_payload_root: Path | None
    manifest_fingerprint: str
    source_digest: str
    source_size: int

    @property
    def packaged_binary_path(self) -> PurePosixPath:
        return self.relative_root / self.relative_binary_path

    @property
    def packaged_support_paths(self) -> tuple[PurePosixPath, ...]:
        return tuple(
            self.relative_root / relative_path
            for relative_path in self.relative_support_paths
        )

    def materialized_root(self, binary_path: Path) -> Path:
        root = binary_path
        for _ in self.relative_binary_path.parts:
            root = root.parent
        return root


def _validate_relative_path(relative_path: str) -> PurePosixPath:
    candidate = PurePosixPath(relative_path)
    if candidate.is_absolute():
        raise ValueError(f"Asset path must be relative: {relative_path}")
    if ".." in candidate.parts:
        raise ValueError(f"Asset path must not traverse parents: {relative_path}")
    if not candidate.parts:
        raise ValueError("Asset path must not be empty")
    return candidate


def _validate_optional_relative_path(relative_path: str) -> PurePosixPath | None:
    if not relative_path:
        return None
    return _validate_relative_path(relative_path)


def _normalize_safe_archive_path(path: PurePosixPath | str) -> PurePosixPath | None:
    candidate = PurePosixPath(path)
    if candidate.is_absolute():
        return None

    normalized_parts: list[str] = []
    for part in candidate.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not normalized_parts:
                return None
            normalized_parts.pop()
            continue
        normalized_parts.append(part)

    if not normalized_parts:
        return None
    return PurePosixPath(*normalized_parts)


def _normalize_platform_key(
    *, system_name: str | None = None, machine_name: str | None = None
) -> tuple[str, str]:
    normalized_system = (system_name or current_system()).strip().lower()
    normalized_machine = (machine_name or current_machine()).strip().lower()
    normalized_machine = _MACHINE_ALIASES.get(normalized_machine, normalized_machine)
    return normalized_system, normalized_machine


def _resource(relative_path: PurePosixPath) -> importlib.abc.Traversable:
    return importlib.resources.files(_RUNTIME_PACKAGE).joinpath(*relative_path.parts)


def _packaged_resource_exists(relative_path: PurePosixPath) -> bool:
    try:
        return _resource(relative_path).is_file()
    except FileNotFoundError:
        return False


def _read_packaged_bytes(relative_path: PurePosixPath) -> bytes:
    return _resource(relative_path).read_bytes()


def _read_packaged_text(relative_path: PurePosixPath) -> str:
    return _resource(relative_path).read_text(encoding="utf-8")


def _read_packaged_json(relative_path: PurePosixPath) -> dict[str, object]:
    loaded = json.loads(_read_packaged_text(relative_path))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {relative_path}")
    return loaded


def _require_manifest_string(manifest: dict[str, object], key: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest field {key!r} must be a non-empty string")
    return value


def _require_optional_manifest_string(manifest: dict[str, object], key: str) -> str:
    value = manifest.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"Manifest field {key!r} must be a string when present")
    return value


def _require_https_url(url: str, *, context: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise ValueError(
            f"{context} must be an https URL with a network location: {url!r}"
        )
    return url


def _require_manifest_source_url(manifest: dict[str, object], key: str) -> str:
    value = _require_manifest_string(manifest, key)
    return _require_https_url(value, context=f"Manifest field {key!r}")


def _require_manifest_args(manifest: dict[str, object], key: str) -> tuple[str, ...]:
    value = manifest.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Manifest field {key!r} must be a non-empty list")
    parsed: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(f"Manifest field {key!r} must contain strings")
        parsed.append(item)
    return tuple(parsed)


def _require_manifest_wheel_platform_tags(
    manifest: dict[str, object], key: str
) -> tuple[str, ...]:
    value = manifest.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Manifest field {key!r} must be a non-empty list")

    parsed: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Manifest field {key!r} must contain strings")
        parsed.append(item.strip())
    return tuple(parsed)


def _require_optional_manifest_paths(
    manifest: dict[str, object], key: str
) -> tuple[PurePosixPath, ...]:
    value = manifest.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"Manifest field {key!r} must be a list when present")

    parsed: list[PurePosixPath] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(f"Manifest field {key!r} must contain strings")
        parsed.append(_validate_relative_path(item))
    return tuple(parsed)


def _require_manifest_env_path_entries(
    manifest: dict[str, object], key: str
) -> dict[str, tuple[PurePosixPath, ...]]:
    value = manifest.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Manifest field {key!r} must be an object when present")

    parsed: dict[str, tuple[PurePosixPath, ...]] = {}
    for env_key, relative_paths in value.items():
        if not isinstance(env_key, str) or not env_key.strip():
            raise ValueError(f"Manifest field {key!r} must use non-empty string keys")
        if not isinstance(relative_paths, list) or not relative_paths:
            raise ValueError(
                f"Manifest field {key!r}.{env_key!r} must contain a non-empty list"
            )
        parsed[env_key] = tuple(
            _validate_relative_path(path_value)
            for path_value in _require_manifest_args(
                {env_key: relative_paths},
                env_key,
            )
        )
    return parsed


def _require_manifest_launch_mode(
    manifest: dict[str, object], key: str
) -> WatchmanRuntimeLaunchMode:
    value = _require_manifest_string(manifest, key)
    if value not in {
        _PYTHON_BRIDGE_LAUNCH_MODE,
        _NATIVE_BINARY_LAUNCH_MODE,
    }:
        raise ValueError(
            "Manifest field "
            f"{key!r} must be one of {_PYTHON_BRIDGE_LAUNCH_MODE!r} or "
            f"{_NATIVE_BINARY_LAUNCH_MODE!r}"
        )
    return cast(WatchmanRuntimeLaunchMode, value)


def _require_manifest_listener_transport(
    manifest: dict[str, object], key: str
) -> WatchmanRuntimeListenerTransport:
    value = _require_manifest_string(manifest, key)
    allowed = {
        _SOCKNAME_LISTENER_TRANSPORT,
        _UNIX_SOCKET_LISTENER_TRANSPORT,
        _NAMED_PIPE_LISTENER_TRANSPORT,
    }
    if value not in allowed:
        rendered = ", ".join(repr(item) for item in sorted(allowed))
        raise ValueError(
            f"Manifest field {key!r} must be one of {rendered}; got {value!r}"
        )
    return cast(WatchmanRuntimeListenerTransport, value)


def _require_manifest_source_archive_format(
    manifest: dict[str, object], key: str
) -> WatchmanRuntimeSourceArchiveFormat:
    value = _require_manifest_string(manifest, key)
    allowed = {
        _ZIP_SOURCE_ARCHIVE_FORMAT,
        _DEB_SOURCE_ARCHIVE_FORMAT,
    }
    if value not in allowed:
        rendered = ", ".join(repr(item) for item in sorted(allowed))
        raise ValueError(
            f"Manifest field {key!r} must be one of {rendered}; got {value!r}"
        )
    return cast(WatchmanRuntimeSourceArchiveFormat, value)


def _require_manifest_source_sha256(manifest: dict[str, object], key: str) -> str:
    value = _require_manifest_string(manifest, key).lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(
            f"Manifest field {key!r} must be a lowercase 64-character SHA256 hex string"
        )
    return value


def _require_manifest_source_files(
    manifest: dict[str, object], key: str
) -> tuple[WatchmanRuntimeSourceFile, ...]:
    value = manifest.get(key)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Manifest field {key!r} must be a non-empty object")

    parsed: list[WatchmanRuntimeSourceFile] = []
    for destination_path, source_path in value.items():
        if not isinstance(destination_path, str) or not destination_path:
            raise ValueError(
                f"Manifest field {key!r} must use non-empty destination paths"
            )
        if not isinstance(source_path, str) or not source_path:
            raise ValueError(
                f"Manifest field {key!r} must map to non-empty source paths"
            )
        parsed.append(
            WatchmanRuntimeSourceFile(
                destination_relative_path=_validate_relative_path(destination_path),
                source_relative_path=_validate_relative_path(source_path),
            )
        )
    return tuple(parsed)


def _require_manifest_sources(
    manifest: dict[str, object], key: str
) -> tuple[WatchmanRuntimeSource, ...]:
    value = manifest.get(key)
    if value is None:
        return ()
    if not isinstance(value, list) or not value:
        raise ValueError(f"Manifest field {key!r} must be a non-empty list")

    parsed: list[WatchmanRuntimeSource] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"Manifest field {key!r} must contain objects")
        parsed.append(
            WatchmanRuntimeSource(
                source_url=_require_manifest_source_url(item, "source_url"),
                source_sha256=_require_manifest_source_sha256(item, "source_sha256"),
                source_archive_format=_require_manifest_source_archive_format(
                    item,
                    "source_archive_format",
                ),
                source_root_prefix=_validate_optional_relative_path(
                    _require_optional_manifest_string(item, "source_root_prefix")
                ),
                source_files=_require_manifest_source_files(item, "source_files"),
            )
        )
    return tuple(parsed)


def _require_manifest_source_descriptor(
    manifest: dict[str, object],
) -> _WatchmanRuntimeSourceDescriptor:
    return _WatchmanRuntimeSourceDescriptor(
        source_url=_require_manifest_source_url(manifest, "source_url"),
        source_sha256=_require_manifest_source_sha256(manifest, "source_sha256"),
        source_archive_format=_require_manifest_source_archive_format(
            manifest, "source_archive_format"
        ),
        source_root_prefix=_validate_relative_path(
            _require_manifest_string(manifest, "source_root_prefix")
        ),
        source_relative_path=_validate_optional_relative_path(
            _require_optional_manifest_string(manifest, "source_relative_path")
        ),
    )


def _require_manifest_support_file_source_descriptors(
    manifest: dict[str, object], key: str
) -> dict[PurePosixPath, _WatchmanRuntimeSourceDescriptor]:
    value = manifest.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Manifest field {key!r} must be an object when present")

    parsed: dict[PurePosixPath, _WatchmanRuntimeSourceDescriptor] = {}
    for destination_path, descriptor in value.items():
        if not isinstance(destination_path, str) or not destination_path:
            raise ValueError(
                f"Manifest field {key!r} must use non-empty destination paths"
            )
        if not isinstance(descriptor, dict):
            raise ValueError(f"Manifest field {key!r} must contain source objects")
        parsed[_validate_relative_path(destination_path)] = (
            _require_manifest_source_descriptor(cast(dict[str, object], descriptor))
        )
    return parsed


def _manifest_sources_from_source_metadata(
    manifest: dict[str, object],
    *,
    relative_binary_path: PurePosixPath,
    relative_support_paths: tuple[PurePosixPath, ...],
) -> tuple[WatchmanRuntimeSource, ...]:
    if "sources" in manifest:
        return _require_manifest_sources(manifest, "sources")

    primary_source = _require_manifest_source_descriptor(manifest)
    support_overrides = _require_manifest_support_file_source_descriptors(
        manifest,
        "support_file_sources",
    )
    declared_support_paths = set(relative_support_paths)
    unexpected_support_paths = sorted(
        path.as_posix()
        for path in support_overrides
        if path not in declared_support_paths
    )
    if unexpected_support_paths:
        raise ValueError(
            "Manifest field 'support_file_sources' references undeclared support "
            f"files: {', '.join(unexpected_support_paths)}"
        )

    grouped_sources: dict[
        tuple[str, str, WatchmanRuntimeSourceArchiveFormat, PurePosixPath | None],
        list[WatchmanRuntimeSourceFile],
    ] = {}

    def add_source_file(
        descriptor: _WatchmanRuntimeSourceDescriptor,
        *,
        destination_relative_path: PurePosixPath,
        source_relative_path: PurePosixPath,
    ) -> None:
        key = (
            descriptor.source_url,
            descriptor.source_sha256,
            descriptor.source_archive_format,
            descriptor.source_root_prefix,
        )
        grouped_sources.setdefault(key, []).append(
            WatchmanRuntimeSourceFile(
                destination_relative_path=destination_relative_path,
                source_relative_path=source_relative_path,
            )
        )

    add_source_file(
        primary_source,
        destination_relative_path=relative_binary_path,
        source_relative_path=(
            primary_source.source_relative_path or relative_binary_path
        ),
    )
    for support_relative_path in relative_support_paths:
        override = support_overrides.get(support_relative_path)
        if override is None:
            add_source_file(
                primary_source,
                destination_relative_path=support_relative_path,
                source_relative_path=support_relative_path,
            )
            continue
        add_source_file(
            override,
            destination_relative_path=support_relative_path,
            source_relative_path=override.source_relative_path or support_relative_path,
        )

    parsed: list[WatchmanRuntimeSource] = []
    for (
        source_url,
        source_sha256,
        source_archive_format,
        source_root_prefix,
    ), source_files in grouped_sources.items():
        parsed.append(
            WatchmanRuntimeSource(
                source_url=source_url,
                source_sha256=source_sha256,
                source_archive_format=source_archive_format,
                source_root_prefix=source_root_prefix,
                source_files=tuple(source_files),
            )
        )
    return tuple(parsed)


def build_watchman_runtime_command_prefix(
    *, runtime: PackagedWatchmanRuntime, binary_path: Path
) -> list[str]:
    """Build the ChunkHound-owned command prefix for the resolved runtime."""

    if runtime.launch_mode == _PYTHON_BRIDGE_LAUNCH_MODE:
        return [sys.executable, "-m", "chunkhound.watchman_runtime.bridge"]
    if runtime.launch_mode == _NATIVE_BINARY_LAUNCH_MODE:
        return [str(binary_path)]
    raise ValueError(f"Unsupported Watchman runtime launch mode: {runtime.launch_mode}")


def _socket_flag(runtime: PackagedWatchmanRuntime) -> str:
    if runtime.listener_transport == _UNIX_SOCKET_LISTENER_TRANSPORT:
        return "--unix-listener-path"
    if runtime.listener_transport == _NAMED_PIPE_LISTENER_TRANSPORT:
        return "--named-pipe-path"
    return "--sockname"


def build_watchman_probe_command(
    *, runtime: PackagedWatchmanRuntime, binary_path: Path
) -> list[str]:
    return [
        *build_watchman_runtime_command_prefix(
            runtime=runtime,
            binary_path=binary_path,
        ),
        *runtime.probe_args,
    ]


def build_watchman_sidecar_command(
    *,
    runtime: PackagedWatchmanRuntime,
    binary_path: Path,
    socket_path: str | Path,
    statefile_path: Path,
    logfile_path: Path,
    pidfile_path: Path,
) -> list[str]:
    command = [
        *build_watchman_runtime_command_prefix(
            runtime=runtime,
            binary_path=binary_path,
        ),
        "--foreground",
        _socket_flag(runtime),
        str(socket_path),
    ]
    if runtime.launch_mode == _NATIVE_BINARY_LAUNCH_MODE:
        command.extend(["--pidfile", str(pidfile_path)])
    command.extend(
        [
            "--statefile",
            str(statefile_path),
            "--logfile",
            str(logfile_path),
            "--no-save-state",
        ]
    )
    return command


def build_watchman_client_command(
    *,
    runtime: PackagedWatchmanRuntime,
    binary_path: Path,
    socket_path: str | Path,
    statefile_path: Path,
    logfile_path: Path,
    pidfile_path: Path,
    persistent: bool = True,
) -> list[str]:
    command = [
        *build_watchman_runtime_command_prefix(
            runtime=runtime,
            binary_path=binary_path,
        ),
        _socket_flag(runtime),
        str(socket_path),
    ]
    if runtime.launch_mode == _NATIVE_BINARY_LAUNCH_MODE:
        command.extend(
            [
                "--statefile",
                str(statefile_path),
                "--logfile",
                str(logfile_path),
            ]
        )
    command.extend(["--no-spawn", "--no-pretty"])
    if persistent:
        command.append("--persistent")
    command.extend(
        [
            "--server-encoding",
            "json",
            "--output-encoding",
            "json",
            "--json-command",
        ]
    )
    return command


def listener_path_is_filesystem(
    runtime: PackagedWatchmanRuntime,
) -> bool:
    return runtime.listener_transport != _NAMED_PIPE_LISTENER_TRANSPORT


def _prepend_env_paths(existing: str | None, additions: list[str]) -> str:
    entries: list[str] = []
    seen: set[str] = set()
    for entry in additions:
        if entry and entry not in seen:
            entries.append(entry)
            seen.add(entry)
    if existing:
        for entry in existing.split(os.pathsep):
            if entry and entry not in seen:
                entries.append(entry)
                seen.add(entry)
    return os.pathsep.join(entries)


def build_watchman_runtime_environment(
    *,
    runtime: PackagedWatchmanRuntime,
    binary_path: Path,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(base_env or os.environ)
    if not runtime.env_path_entries:
        return env

    materialized_root = runtime.materialized_root(binary_path)
    for env_key, relative_paths in runtime.env_path_entries.items():
        additions = [
            str((materialized_root / Path(*relative_path.parts)).resolve())
            for relative_path in relative_paths
        ]
        env[env_key] = _prepend_env_paths(env.get(env_key), additions)
    return env


def is_packaged_watchman_runtime_supported(
    *, system_name: str | None = None, machine_name: str | None = None
) -> bool:
    platform_key = _normalize_platform_key(
        system_name=system_name,
        machine_name=machine_name,
    )
    return platform_key in _SUPPORTED_PLATFORM_ROOTS


def is_packaged_watchman_runtime_available(
    *, system_name: str | None = None, machine_name: str | None = None
) -> bool:
    platform_key = _normalize_platform_key(
        system_name=system_name,
        machine_name=machine_name,
    )
    relative_root = _SUPPORTED_PLATFORM_ROOTS.get(platform_key)
    if relative_root is None:
        return False

    manifest_path = relative_root / "manifest.json"
    if not _packaged_resource_exists(manifest_path):
        return False
    manifest = _read_packaged_json(manifest_path)
    relative_binary_path = _validate_relative_path(
        _require_manifest_string(manifest, "binary")
    )
    relative_support_paths = _require_optional_manifest_paths(manifest, "support_files")
    return _packaged_payloads_available(
        relative_root=relative_root,
        relative_binary_path=relative_binary_path,
        relative_support_paths=relative_support_paths,
    )


def default_realtime_backend_for_platform(
    *, system_name: str | None = None, machine_name: str | None = None
) -> str:
    if is_packaged_watchman_runtime_supported(
        system_name=system_name,
        machine_name=machine_name,
    ):
        return "watchman"
    return "watchdog"


def default_realtime_backend_for_current_install() -> Literal["watchman", "watchdog"]:
    if is_packaged_watchman_runtime_available():
        return "watchman"
    return "watchdog"


def _running_from_source_tree() -> bool:
    return (_PACKAGE_ROOT.parent.parent / "pyproject.toml").is_file()


def _packaged_payloads_available(
    *,
    relative_root: PurePosixPath,
    relative_binary_path: PurePosixPath,
    relative_support_paths: tuple[PurePosixPath, ...],
) -> bool:
    if _running_from_source_tree():
        return False
    paths = (
        relative_root / relative_binary_path,
        *(relative_root / path for path in relative_support_paths),
    )
    return all(_packaged_resource_exists(path) for path in paths)


def _default_runtime_cache_dir() -> Path:
    override = os.environ.get(_RUNTIME_CACHE_DIR_ENV)
    if override:
        return Path(override).expanduser()

    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            return Path(local_appdata) / "ChunkHound" / "watchman-runtime"
        return Path.home() / "AppData" / "Local" / "ChunkHound" / "watchman-runtime"

    cache_root = os.environ.get("XDG_CACHE_HOME")
    if cache_root:
        return Path(cache_root) / "chunkhound" / "watchman-runtime"
    return Path.home() / ".cache" / "chunkhound" / "watchman-runtime"


def _relative_root_path(relative_root: PurePosixPath) -> Path:
    return Path(*relative_root.parts)


def _runtime_hydration_root(
    *,
    platform_tag: str,
    runtime_version: str,
    manifest_fingerprint: str,
) -> Path:
    return (
        _default_runtime_cache_dir()
        / "payloads"
        / platform_tag
        / runtime_version
        / manifest_fingerprint[:16]
    )


def _download_cache_path(source: WatchmanRuntimeSource) -> Path:
    parsed = urllib.parse.urlparse(source.source_url)
    filename = Path(urllib.parse.unquote(parsed.path)).name or "watchman-runtime-asset"
    return (
        _default_runtime_cache_dir()
        / "downloads"
        / f"{source.source_sha256}-{filename}"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def _runtime_download_timeout_seconds() -> float:
    raw = os.environ.get(_RUNTIME_DOWNLOAD_TIMEOUT_SECONDS_ENV)
    if raw is None:
        return _DEFAULT_RUNTIME_DOWNLOAD_TIMEOUT_SECONDS
    try:
        timeout = float(raw)
    except ValueError:
        return _DEFAULT_RUNTIME_DOWNLOAD_TIMEOUT_SECONDS
    if timeout <= 0:
        return _DEFAULT_RUNTIME_DOWNLOAD_TIMEOUT_SECONDS
    return timeout


def _runtime_download_retries() -> int:
    raw = os.environ.get(_RUNTIME_DOWNLOAD_RETRIES_ENV)
    if raw is None:
        return _DEFAULT_RUNTIME_DOWNLOAD_RETRIES
    try:
        retries = int(raw)
    except ValueError:
        return _DEFAULT_RUNTIME_DOWNLOAD_RETRIES
    if retries < 1:
        return _DEFAULT_RUNTIME_DOWNLOAD_RETRIES
    return retries


def _cleanup_partial_download(temp_path: Path) -> None:
    try:
        temp_path.unlink()
    except FileNotFoundError:
        pass


def _should_retry_source_download(error: Exception) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return False
    if isinstance(
        error,
        (
            TimeoutError,
            socket.timeout,
            http.client.IncompleteRead,
            urllib.error.URLError,
        ),
    ):
        return True
    return False


def _download_source_archive_once(
    *,
    source: WatchmanRuntimeSource,
    temp_path: Path,
    timeout_seconds: float,
) -> None:
    with urllib.request.urlopen(
        source.source_url,
        timeout=timeout_seconds,
    ) as response, temp_path.open("wb") as handle:
        response_geturl = getattr(response, "geturl", None)
        if callable(response_geturl):
            final_url = response_geturl()
            if isinstance(final_url, str) and final_url:
                try:
                    _require_https_url(
                        final_url,
                        context="Watchman runtime download final URL",
                    )
                except ValueError as error:
                    raise RuntimeError(str(error)) from error
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def _ensure_downloaded_source_archive(source: WatchmanRuntimeSource) -> Path:
    archive_path = _download_cache_path(source)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.is_file() and _file_sha256(archive_path) == source.source_sha256:
        return archive_path

    temp_path = archive_path.with_name(f"{archive_path.name}.tmp")
    timeout_seconds = _runtime_download_timeout_seconds()
    max_attempts = _runtime_download_retries()
    for attempt in range(1, max_attempts + 1):
        try:
            _cleanup_partial_download(temp_path)
            _download_source_archive_once(
                source=source,
                temp_path=temp_path,
                timeout_seconds=timeout_seconds,
            )
            break
        except Exception as error:
            _cleanup_partial_download(temp_path)
            if attempt >= max_attempts or not _should_retry_source_download(error):
                if _should_retry_source_download(error):
                    raise RuntimeError(
                        "Watchman runtime archive download failed after "
                        f"{attempt} attempt(s) with timeout "
                        f"{timeout_seconds}s: {source.source_url}"
                    ) from error
                raise
            time.sleep(float(attempt))

    actual_sha256 = _file_sha256(temp_path)
    if actual_sha256 != source.source_sha256:
        _cleanup_partial_download(temp_path)
        raise RuntimeError(
            "Watchman runtime archive checksum mismatch for "
            f"{source.source_url}: expected {source.source_sha256}, got {actual_sha256}"
        )
    os.replace(temp_path, archive_path)
    return archive_path


def _iter_ar_members(archive_path: Path) -> Generator[tuple[str, bytes], None, None]:
    payload = archive_path.read_bytes()
    if not payload.startswith(_AR_GLOBAL_HEADER):
        raise RuntimeError(f"Unsupported ar archive header: {archive_path}")

    offset = len(_AR_GLOBAL_HEADER)
    long_name_table = b""
    while offset < len(payload):
        header = payload[offset : offset + _AR_HEADER_SIZE]
        if len(header) < _AR_HEADER_SIZE:
            raise RuntimeError(f"Truncated ar archive entry in {archive_path}")
        if header[58:60] != b"`\n":
            raise RuntimeError(f"Invalid ar archive entry trailer in {archive_path}")

        raw_name = header[:16].decode("utf-8", errors="replace")
        size = int(header[48:58].decode("ascii").strip())
        offset += _AR_HEADER_SIZE
        data = payload[offset : offset + size]
        offset += size + (size % 2)

        name = raw_name.rstrip()
        if name == "//":
            long_name_table = data
            continue
        if name.startswith("#1/"):
            name_length = int(name[3:].strip())
            actual_name = data[:name_length].decode("utf-8", errors="replace")
            yield actual_name.rstrip("/"), data[name_length:]
            continue
        if name.startswith("/") and name not in {"/", "//"}:
            if not long_name_table:
                raise RuntimeError(f"Missing ar long-name table in {archive_path}")
            table_offset = int(name[1:].strip())
            name_end = long_name_table.find(b"/\n", table_offset)
            if name_end == -1:
                raise RuntimeError(f"Corrupt ar long-name table in {archive_path}")
            resolved_name = long_name_table[table_offset:name_end].decode(
                "utf-8",
                errors="replace",
            )
            yield resolved_name.rstrip("/"), data
            continue
        yield name.rstrip("/"), data


def _read_deb_member_bytes(
    archive_path: Path,
    *,
    source_root_prefix: PurePosixPath | None,
    source_relative_path: PurePosixPath,
) -> bytes:
    data_member_bytes: bytes | None = None
    data_member_name: str | None = None
    for member_name, member_bytes in _iter_ar_members(archive_path):
        if member_name.startswith("data.tar"):
            data_member_bytes = member_bytes
            data_member_name = member_name
            break
    if data_member_bytes is None:
        raise RuntimeError(f"Unable to locate data.tar payload in {archive_path}")
    if data_member_name is None:
        raise RuntimeError(f"Unable to identify data.tar payload in {archive_path}")

    expected_path = (
        source_root_prefix / source_relative_path
        if source_root_prefix is not None
        else source_relative_path
    )
    if data_member_name.endswith(".zst"):
        try:
            import zstandard
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Watchman runtime hydration requires the 'zstandard' package to "
                f"read {archive_path.name}"
            ) from error
        decompressor = zstandard.ZstdDecompressor()
        with decompressor.stream_reader(io.BytesIO(data_member_bytes)) as reader:
            data_member_bytes = reader.read()

    with tarfile.open(fileobj=io.BytesIO(data_member_bytes), mode="r:*") as handle:
        members_by_path = {
            normalized_path: member
            for member in handle.getmembers()
            if (
                normalized_path := _normalize_safe_archive_path(member.name)
            )
            is not None
        }
        current_path = expected_path
        visited_paths: set[PurePosixPath] = set()

        while True:
            member = members_by_path.get(current_path)
            if member is None:
                raise RuntimeError(
                    "Watchman runtime archive is missing required payload "
                    f"{expected_path.as_posix()} in {archive_path}"
                )

            if member.isfile():
                extracted_file = handle.extractfile(member)
                if extracted_file is None:
                    raise RuntimeError(
                        "Watchman runtime archive member could not be read: "
                        f"{expected_path.as_posix()} in {archive_path}"
                    )
                return extracted_file.read()

            if member.issym() or member.islnk():
                if current_path in visited_paths:
                    raise RuntimeError(
                        "Watchman runtime archive member link cycle detected: "
                        f"{expected_path.as_posix()} in {archive_path}"
                    )
                visited_paths.add(current_path)
                link_base = current_path.parent if member.issym() else PurePosixPath()
                linked_path = _normalize_safe_archive_path(link_base / member.linkname)
                if linked_path is None:
                    raise RuntimeError(
                        "Watchman runtime archive member link target is unsafe: "
                        f"{expected_path.as_posix()} -> {member.linkname} in "
                        f"{archive_path}"
                    )
                current_path = linked_path
                continue

            raise RuntimeError(
                "Watchman runtime archive member is not a regular file: "
                f"{expected_path.as_posix()} in {archive_path}"
            )


def _read_zip_member_bytes(
    archive_path: Path,
    *,
    source_root_prefix: PurePosixPath | None,
    source_relative_path: PurePosixPath,
) -> bytes:
    expected_path = (
        source_root_prefix / source_relative_path
        if source_root_prefix is not None
        else source_relative_path
    )
    with zipfile.ZipFile(archive_path) as handle:
        try:
            return handle.read(expected_path.as_posix())
        except KeyError as error:
            raise RuntimeError(
                "Watchman runtime archive is missing required payload "
                f"{expected_path.as_posix()} in {archive_path}"
            ) from error


def _read_source_member_bytes(
    archive_path: Path,
    *,
    source: WatchmanRuntimeSource,
    source_relative_path: PurePosixPath,
) -> bytes:
    if source.source_archive_format == _ZIP_SOURCE_ARCHIVE_FORMAT:
        return _read_zip_member_bytes(
            archive_path,
            source_root_prefix=source.source_root_prefix,
            source_relative_path=source_relative_path,
        )
    if source.source_archive_format == _DEB_SOURCE_ARCHIVE_FORMAT:
        return _read_deb_member_bytes(
            archive_path,
            source_root_prefix=source.source_root_prefix,
            source_relative_path=source_relative_path,
        )
    raise ValueError(
        f"Unsupported Watchman runtime source format: {source.source_archive_format}"
    )


def _write_materialized_payload(
    *, destination_path: Path, expected_payload: bytes, executable: bool
) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    needs_write = True
    if destination_path.is_file():
        current_payload = destination_path.read_bytes()
        current_digest = hashlib.sha256(current_payload).hexdigest()
        expected_digest = hashlib.sha256(expected_payload).hexdigest()
        needs_write = current_digest != expected_digest

    if needs_write:
        temp_path = destination_path.with_name(f"{destination_path.name}.tmp")
        temp_path.write_bytes(expected_payload)
        os.replace(temp_path, destination_path)

    if executable and os.name != "nt":
        mode = destination_path.stat().st_mode
        os.chmod(
            destination_path,
            mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
        )


def _expected_source_destinations(
    *,
    relative_binary_path: PurePosixPath,
    relative_support_paths: tuple[PurePosixPath, ...],
) -> set[PurePosixPath]:
    return {
        relative_binary_path,
        *relative_support_paths,
    }


def _load_hydration_marker(marker_path: Path) -> dict[str, object] | None:
    if not marker_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(marker, dict):
        return None
    return marker


def _hydrated_root_matches(
    *,
    destination_root: Path,
    manifest_fingerprint: str,
    expected_destinations: set[PurePosixPath],
) -> bool:
    marker_path = destination_root / _HYDRATION_MARKER_NAME
    marker = _load_hydration_marker(marker_path)
    if marker is None or marker.get("manifest_fingerprint") != manifest_fingerprint:
        return False
    files = marker.get("files")
    if not isinstance(files, dict):
        return False
    try:
        actual_destinations = {
            _validate_relative_path(path_key)
            for path_key in files.keys()
            if isinstance(path_key, str)
        }
    except ValueError:
        return False
    if actual_destinations != expected_destinations:
        return False
    expected_files = {
        *expected_destinations,
        PurePosixPath(_HYDRATION_MARKER_NAME),
    }
    expected_directories: set[PurePosixPath] = set()
    for relative_path in expected_destinations:
        for parent in relative_path.parents:
            if parent == PurePosixPath("."):
                break
            expected_directories.add(parent)
    for existing_path in destination_root.rglob("*"):
        relative_existing_path = PurePosixPath(
            existing_path.relative_to(destination_root).as_posix()
        )
        if existing_path.is_dir():
            if relative_existing_path not in expected_directories:
                return False
            continue
        if relative_existing_path not in expected_files:
            return False
    for relative_path in expected_destinations:
        expected_digest = files.get(relative_path.as_posix())
        if not isinstance(expected_digest, str):
            return False
        destination_path = destination_root / Path(*relative_path.parts)
        if not destination_path.is_file():
            return False
        if _file_sha256(destination_path) != expected_digest:
            return False
    return True


def _remove_runtime_tree(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)


def _acquire_runtime_lock(handle: io.BufferedRandom) -> None:
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    if os.name == "nt":
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _release_runtime_lock(handle: io.BufferedRandom) -> None:
    handle.seek(0)
    if os.name == "nt":
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _locked_runtime_hydration_root(
    destination_root: Path,
) -> Generator[None, None, None]:
    lock_path = destination_root.with_name(f"{destination_root.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        _acquire_runtime_lock(handle)
        try:
            yield
        finally:
            _release_runtime_lock(handle)


def _stage_hydrated_runtime_tree(
    *,
    platform_tag: str,
    runtime_version: str,
    manifest_fingerprint: str,
    source_archives: tuple[WatchmanRuntimeSource, ...],
    expected_destinations: set[PurePosixPath],
    destination_root: Path,
) -> None:
    staged_digests: dict[str, str] = {}
    seen_destinations: set[PurePosixPath] = set()
    for source in source_archives:
        archive_path = _ensure_downloaded_source_archive(source)
        for source_file in source.source_files:
            payload = _read_source_member_bytes(
                archive_path,
                source=source,
                source_relative_path=source_file.source_relative_path,
            )
            destination_path = destination_root / Path(
                *source_file.destination_relative_path.parts
            )
            _write_materialized_payload(
                destination_path=destination_path,
                expected_payload=payload,
                executable=False,
            )
            staged_digests[source_file.destination_relative_path.as_posix()] = (
                hashlib.sha256(payload).hexdigest()
            )
            seen_destinations.add(source_file.destination_relative_path)

    if seen_destinations != expected_destinations:
        missing = sorted(
            path.as_posix() for path in (expected_destinations - seen_destinations)
        )
        extra = sorted(
            path.as_posix() for path in (seen_destinations - expected_destinations)
        )
        detail_lines = []
        if missing:
            detail_lines.append("missing: " + ", ".join(missing))
        if extra:
            detail_lines.append("extra: " + ", ".join(extra))
        raise RuntimeError(
            "Watchman runtime source metadata does not stage the declared payload "
            f"for {platform_tag}: {'; '.join(detail_lines)}"
        )

    _write_json_atomic(
        destination_root / _HYDRATION_MARKER_NAME,
        {
            "manifest_fingerprint": manifest_fingerprint,
            "platform_tag": platform_tag,
            "runtime_version": runtime_version,
            "files": staged_digests,
        },
    )


def _hydrate_runtime_tree_from_sources(
    *,
    platform_tag: str,
    runtime_version: str,
    manifest_fingerprint: str,
    source_archives: tuple[WatchmanRuntimeSource, ...],
    relative_binary_path: PurePosixPath,
    relative_support_paths: tuple[PurePosixPath, ...],
    destination_root: Path,
) -> Path:
    if not source_archives:
        raise RuntimeError(
            "Watchman runtime hydration requires manifest source metadata for "
            f"{platform_tag}"
        )

    destination_root = destination_root.expanduser().resolve()
    expected_destinations = _expected_source_destinations(
        relative_binary_path=relative_binary_path,
        relative_support_paths=relative_support_paths,
    )
    destination_root.parent.mkdir(parents=True, exist_ok=True)
    with _locked_runtime_hydration_root(destination_root):
        if _hydrated_root_matches(
            destination_root=destination_root,
            manifest_fingerprint=manifest_fingerprint,
            expected_destinations=expected_destinations,
        ):
            return destination_root

        _remove_runtime_tree(destination_root)
        staging_root = Path(
            tempfile.mkdtemp(
                dir=destination_root.parent,
                prefix=f"{destination_root.name}.tmp-",
            )
        )
        try:
            _stage_hydrated_runtime_tree(
                platform_tag=platform_tag,
                runtime_version=runtime_version,
                manifest_fingerprint=manifest_fingerprint,
                source_archives=source_archives,
                expected_destinations=expected_destinations,
                destination_root=staging_root,
            )
            os.replace(staging_root, destination_root)
        except Exception:
            _remove_runtime_tree(staging_root)
            raise
    return destination_root


def _read_runtime_payload_bytes(
    runtime: PackagedWatchmanRuntime, relative_path: PurePosixPath
) -> bytes:
    if runtime.hydrated_payload_root is not None:
        return (
            runtime.hydrated_payload_root / Path(*relative_path.parts)
        ).read_bytes()
    return _read_packaged_bytes(runtime.relative_root / relative_path)


def resolve_packaged_watchman_runtime(
    *,
    system_name: str | None = None,
    machine_name: str | None = None,
    _hydrate_if_missing: bool = True,
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

    manifest_text = _read_packaged_text(relative_root / "manifest.json")
    manifest = json.loads(manifest_text)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected JSON object in {relative_root / 'manifest.json'}")
    relative_binary_path = _validate_relative_path(
        _require_manifest_string(manifest, "binary")
    )
    relative_support_paths = _require_optional_manifest_paths(manifest, "support_files")
    source_archives = _manifest_sources_from_source_metadata(
        manifest,
        relative_binary_path=relative_binary_path,
        relative_support_paths=relative_support_paths,
    )
    expected_destinations = _expected_source_destinations(
        relative_binary_path=relative_binary_path,
        relative_support_paths=relative_support_paths,
    )
    actual_source_destinations = {
        source_file.destination_relative_path
        for source in source_archives
        for source_file in source.source_files
    }
    if source_archives and actual_source_destinations != expected_destinations:
        missing = sorted(
            path.as_posix()
            for path in (expected_destinations - actual_source_destinations)
        )
        extra = sorted(
            path.as_posix()
            for path in (actual_source_destinations - expected_destinations)
        )
        detail_lines = []
        if missing:
            detail_lines.append("missing: " + ", ".join(missing))
        if extra:
            detail_lines.append("extra: " + ", ".join(extra))
        raise ValueError(
            "Manifest sources must stage exactly the declared Watchman payload for "
            f"{relative_root}: {'; '.join(detail_lines)}"
        )

    hydrated_payload_root: Path | None = None
    payload = b""
    if _packaged_payloads_available(
        relative_root=relative_root,
        relative_binary_path=relative_binary_path,
        relative_support_paths=relative_support_paths,
    ):
        payload = _read_packaged_bytes(relative_root / relative_binary_path)
    elif _hydrate_if_missing:
        manifest_fingerprint = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
        hydrated_payload_root = _hydrate_runtime_tree_from_sources(
            platform_tag=_require_manifest_string(manifest, "platform"),
            runtime_version=_require_manifest_string(manifest, "runtime_version"),
            manifest_fingerprint=manifest_fingerprint,
            source_archives=source_archives,
            relative_binary_path=relative_binary_path,
            relative_support_paths=relative_support_paths,
            destination_root=_runtime_hydration_root(
                platform_tag=_require_manifest_string(manifest, "platform"),
                runtime_version=_require_manifest_string(manifest, "runtime_version"),
                manifest_fingerprint=manifest_fingerprint,
            ),
        )
        payload = (
            hydrated_payload_root / Path(*relative_binary_path.parts)
        ).read_bytes()

    return PackagedWatchmanRuntime(
        platform_tag=_require_manifest_string(manifest, "platform"),
        runtime_version=_require_manifest_string(manifest, "runtime_version"),
        relative_root=relative_root,
        relative_binary_path=relative_binary_path,
        relative_support_paths=relative_support_paths,
        launch_mode=_require_manifest_launch_mode(manifest, "launch_mode"),
        listener_transport=_require_manifest_listener_transport(
            manifest, "listener_transport"
        ),
        probe_args=_require_manifest_args(manifest, "probe_args"),
        wheel_platform_tags=_require_manifest_wheel_platform_tags(
            manifest, "wheel_platform_tags"
        ),
        env_path_entries=_require_manifest_env_path_entries(manifest, "path_env"),
        packaging_decision=_require_manifest_string(manifest, "packaging_decision"),
        source_archives=source_archives,
        hydrated_payload_root=hydrated_payload_root,
        manifest_fingerprint=hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
        source_digest=hashlib.sha256(payload).hexdigest() if payload else "",
        source_size=len(payload),
    )


def hydrate_watchman_runtime_for_build(
    *,
    system_name: str | None = None,
    machine_name: str | None = None,
) -> Path:
    """Hydrate the current host payload into cache for wheel builds."""

    runtime = resolve_packaged_watchman_runtime(
        system_name=system_name,
        machine_name=machine_name,
    )
    if runtime.hydrated_payload_root is None:
        raise RuntimeError(
            "Watchman runtime build hydration requires a hydrated payload root for "
            f"{runtime.platform_tag}"
        )
    return runtime.hydrated_payload_root


def build_watchman_runtime_force_include_entries(
    *,
    system_name: str | None = None,
    machine_name: str | None = None,
) -> dict[str, str]:
    """Return hatch force-include entries for the current host runtime payload."""

    runtime = resolve_packaged_watchman_runtime(
        system_name=system_name,
        machine_name=machine_name,
    )
    destination_root = hydrate_watchman_runtime_for_build(
        system_name=system_name,
        machine_name=machine_name,
    )
    package_root = (
        PurePosixPath("chunkhound")
        / "watchman_runtime"
    )
    manifest_relative_path = runtime.relative_root / "manifest.json"
    entries: dict[str, str] = {
        str(_PACKAGE_ROOT / Path(*manifest_relative_path.parts)): (
            package_root / manifest_relative_path
        ).as_posix()
    }
    package_root = package_root / runtime.relative_root
    for relative_path in (
        runtime.relative_binary_path,
        *runtime.relative_support_paths,
    ):
        source_path = destination_root / Path(*relative_path.parts)
        entries[str(source_path)] = (package_root / relative_path).as_posix()
    return entries


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
    _write_materialized_payload(
        destination_path=destination_path,
        expected_payload=_read_runtime_payload_bytes(
            runtime,
            runtime.relative_binary_path,
        ),
        executable=True,
    )
    for relative_support_path in runtime.relative_support_paths:
        support_destination_path = (
            destination_root
            / runtime.platform_tag
            / runtime.runtime_version
            / Path(*relative_support_path.parts)
        )
        _write_materialized_payload(
            destination_path=support_destination_path,
            expected_payload=_read_runtime_payload_bytes(
                runtime,
                relative_support_path,
            ),
            executable=False,
        )

    return destination_path
