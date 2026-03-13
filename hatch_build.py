from __future__ import annotations

import sys
from pathlib import Path
from platform import machine as current_machine
from platform import system as current_system

from packaging import tags

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 build env
    import tomli as tomllib

try:
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface
except ModuleNotFoundError:  # pragma: no cover - local test env without hatchling
    class BuildHookInterface:
        pass


_PYPROJECT_PATH = Path(__file__).with_name("pyproject.toml")
_MACHINE_ALIASES = {
    "amd64": "x86_64",
    "arm64e": "arm64",
    "aarch64": "arm64",
    "x64": "x86_64",
}
_SYSTEM_ALIASES = {
    "darwin": "macos",
}


def _hydrate_runtime_for_build() -> dict[str, str]:
    from chunkhound.watchman_runtime.loader import (
        build_watchman_runtime_force_include_entries,
    )

    return build_watchman_runtime_force_include_entries()


def _load_supported_watchman_platforms() -> set[str]:
    with _PYPROJECT_PATH.open("rb") as handle:
        loaded = tomllib.load(handle)

    supported_platforms = (
        loaded.get("tool", {})
        .get("chunkhound", {})
        .get("watchman_runtime", {})
        .get("supported_platforms")
    )
    if not isinstance(supported_platforms, list) or not supported_platforms:
        raise RuntimeError(
            "tool.chunkhound.watchman_runtime.supported_platforms must declare at "
            "least one supported platform"
        )

    parsed: set[str] = set()
    for platform_name in supported_platforms:
        if not isinstance(platform_name, str) or not platform_name.strip():
            raise RuntimeError(
                "tool.chunkhound.watchman_runtime.supported_platforms must contain "
                "non-empty platform strings"
            )
        parsed.add(platform_name.strip())
    return parsed


def _host_watchman_platform(
    *, system_name: str | None = None, machine_name: str | None = None
) -> str:
    normalized_system = (system_name or current_system()).strip().lower()
    normalized_system = _SYSTEM_ALIASES.get(normalized_system, normalized_system)
    if normalized_system.startswith("win"):
        normalized_system = "windows"

    normalized_machine = (machine_name or current_machine()).strip().lower()
    normalized_machine = _MACHINE_ALIASES.get(normalized_machine, normalized_machine)
    return f"{normalized_system}-{normalized_machine}"


def _require_supported_build_host(
    supported_platforms: set[str],
    *,
    system_name: str | None = None,
    machine_name: str | None = None,
) -> str:
    host_platform = _host_watchman_platform(
        system_name=system_name,
        machine_name=machine_name,
    )
    if host_platform not in supported_platforms:
        rendered = ", ".join(sorted(supported_platforms))
        raise RuntimeError(
            "No packaged Watchman runtime payload is declared for build host "
            f"{host_platform}. Supported platforms: {rendered}"
        )
    return host_platform


def _should_skip_native_runtime_for_build_version(version: str) -> bool:
    return version == "editable"


def _platform_only_tag() -> str:
    for tag in tags.sys_tags():
        if tag.interpreter == "py3" and tag.abi == "none" and tag.platform != "any":
            return str(tag)

    raise RuntimeError(
        "Unable to determine a host-native py3-none-platform wheel tag for the "
        "packaged Watchman runtime."
    )


class CustomBuildHook(BuildHookInterface):
    """Force platform-specific wheel tags for Watchman-carrying artifacts."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        supported_platforms = _load_supported_watchman_platforms()
        try:
            _require_supported_build_host(supported_platforms)
        except RuntimeError:
            if _should_skip_native_runtime_for_build_version(version):
                return
            raise
        force_include = build_data.setdefault("force_include", {})
        if not isinstance(force_include, dict):
            raise RuntimeError("hatch build_data.force_include must be a mapping")
        force_include.update(_hydrate_runtime_for_build())
        build_data["pure_python"] = False
        build_data["tag"] = _platform_only_tag()
