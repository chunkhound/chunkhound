from __future__ import annotations

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from packaging import tags


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
        del version
        build_data["pure_python"] = False
        build_data["tag"] = _platform_only_tag()
