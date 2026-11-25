from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from platform import machine as current_machine
from platform import system as current_system

from packaging import tags
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 build env
    import tomli as tomllib

try:
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface
    from hatchling.metadata.plugin.interface import MetadataHookInterface
except ModuleNotFoundError:  # pragma: no cover - local test env without hatchling

    class BuildHookInterface:
        pass

    class MetadataHookInterface:
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
_PACKAGED_RUNTIME_BUILD_ENV = "CHUNKHOUND_BUILD_PACKAGED_WATCHMAN_RUNTIME"
_ALPHA_RELEASE_BUILD_ENV = "CHUNKHOUND_BUILD_ALPHA_RELEASE"


def _alpha_native_requirement(version: str) -> str:
    parsed = Version(version)
    if (
        parsed.pre is None
        or parsed.pre[0] != "a"
        or parsed.dev is not None
        or parsed.local
    ):
        raise RuntimeError("Alpha release builds require an alpha release version")
    return f"chunkhound-native=={version}"


_SOURCE_COMMIT_URL_LABEL = "Source commit"
_REPOSITORY_URL = "https://github.com/chunkhound/chunkhound"


def _source_commit(root: Path) -> str | None:
    """Return the checked-out commit SHA, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    sha = result.stdout.strip()
    if result.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", sha):
        return None
    return sha


def _add_source_commit_url(metadata: dict, root: Path) -> None:
    """Link the package page to the exact commit the wheel was built from."""
    sha = _source_commit(root)
    if sha is None:
        return
    urls = metadata.setdefault("urls", {})
    urls[_SOURCE_COMMIT_URL_LABEL] = f"{_REPOSITORY_URL}/commit/{sha}"


class CustomMetadataHook(MetadataHookInterface):
    """Keep stable dependencies, pairing alpha releases with their native build."""

    PLUGIN_NAME = "custom"

    def update(self, metadata: dict) -> None:
        _add_source_commit_url(metadata, _REPO_ROOT)
        dependencies = list(self.config["dependencies"])
        if os.environ.get(_ALPHA_RELEASE_BUILD_ENV) == "1":
            pin = _alpha_native_requirement(metadata["version"])
            native = [
                index
                for index, value in enumerate(dependencies)
                if canonicalize_name(Requirement(value).name) == "chunkhound-native"
            ]
            if len(native) != 1:
                raise RuntimeError(
                    "Exactly one chunkhound-native dependency is required"
                )
            dependencies[native[0]] = pin
        metadata["dependencies"] = dependencies


def _hydrate_runtime_for_build() -> dict[str, str]:
    from chunkhound.watchman_runtime.loader import (
        build_watchman_runtime_force_include_entries,
    )

    return build_watchman_runtime_force_include_entries()


def _manifest_force_include_entries() -> dict[str, str]:
    package_root = Path("chunkhound") / "watchman_runtime"
    entries: dict[str, str] = {}
    for platform_name in sorted(_load_supported_watchman_platforms()):
        manifest_path = (
            _REPO_ROOT
            / "chunkhound"
            / "watchman_runtime"
            / "platforms"
            / platform_name
            / "manifest.json"
        )
        entries[str(manifest_path)] = (
            package_root / "platforms" / platform_name / "manifest.json"
        ).as_posix()
    return entries


def _allowed_wheel_platform_tags_for_build_host(
    *, system_name: str | None = None, machine_name: str | None = None
) -> set[str]:
    from chunkhound.watchman_runtime.loader import resolve_packaged_watchman_runtime

    runtime = resolve_packaged_watchman_runtime(
        system_name=system_name,
        machine_name=machine_name,
        _hydrate_if_missing=False,
    )
    return set(runtime.wheel_platform_tags)


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


def _should_build_packaged_runtime(version: str) -> bool:
    if _should_skip_native_runtime_for_build_version(version):
        return False
    value = os.environ.get(_PACKAGED_RUNTIME_BUILD_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _platform_only_tag(allowed_platform_tags: set[str] | None = None) -> str:
    for tag in tags.sys_tags():
        if tag.interpreter != "py3" or tag.abi != "none" or tag.platform == "any":
            continue
        if (
            allowed_platform_tags is not None
            and tag.platform not in allowed_platform_tags
        ):
            continue
        return str(tag)

    if allowed_platform_tags:
        rendered = ", ".join(sorted(allowed_platform_tags))
        raise RuntimeError(
            "Unable to determine a host-native py3-none-platform wheel tag matching "
            f"the declared Watchman runtime contract. Allowed tags: {rendered}"
        )
    raise RuntimeError(
        "Unable to determine a host-native py3-none-platform wheel tag for the "
        "packaged Watchman runtime."
    )


class CustomBuildHook(BuildHookInterface):
    """Persist alpha sdist pins and tag Watchman-carrying wheels by platform."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        if getattr(self, "target_name", None) == "sdist":
            if os.environ.get(_ALPHA_RELEASE_BUILD_ENV) == "1":
                # Change only the archived config, never the checkout. This makes
                # source rebuilds independent of CI flags (even without PKG-INFO).
                pin = _alpha_native_requirement(self.metadata.version)
                source = _PYPROJECT_PATH.read_text(encoding="utf-8")
                dependencies = self.metadata.hatch.metadata.hooks["custom"].config[
                    "dependencies"
                ]
                native = [
                    value
                    for value in dependencies
                    if canonicalize_name(Requirement(value).name) == "chunkhound-native"
                ]
                if len(native) != 1 or source.count(json.dumps(native[0])) != 1:
                    raise RuntimeError("Cannot persist the alpha native dependency pin")
                source = source.replace(json.dumps(native[0]), json.dumps(pin))
                self._alpha_sdist_directory = tempfile.TemporaryDirectory(
                    prefix="chunkhound-alpha-sdist-"
                )
                config_path = Path(self._alpha_sdist_directory.name) / "pyproject.toml"
                config_path.write_text(source, encoding="utf-8")
                force_include = build_data["force_include"]
                for path, destination in list(force_include.items()):
                    if destination == "pyproject.toml":
                        del force_include[path]
                force_include[str(config_path)] = "pyproject.toml"
            return
        if _should_skip_native_runtime_for_build_version(version):
            return
        if not _should_build_packaged_runtime(version):
            force_include = build_data.setdefault("force_include", {})
            if not isinstance(force_include, dict):
                raise RuntimeError("hatch build_data.force_include must be a mapping")
            force_include.update(_manifest_force_include_entries())
            return
        supported_platforms = _load_supported_watchman_platforms()
        _require_supported_build_host(supported_platforms)
        allowed_platform_tags = _allowed_wheel_platform_tags_for_build_host()
        force_include = build_data.setdefault("force_include", {})
        if not isinstance(force_include, dict):
            raise RuntimeError("hatch build_data.force_include must be a mapping")
        force_include.update(_hydrate_runtime_for_build())
        build_data["pure_python"] = False
        build_data["tag"] = _platform_only_tag(allowed_platform_tags)

    def finalize(
        self, version: str, build_data: dict[str, object], artifact_path: str
    ) -> None:
        directory = getattr(self, "_alpha_sdist_directory", None)
        if directory is not None:
            directory.cleanup()
