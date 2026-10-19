"""Shared user-scoped platform cache root for ChunkHound features."""

import os
import sys
from pathlib import Path


def platform_cache_root() -> Path:
    """Select the platform-specific ChunkHound cache directory."""
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA")
        return (
            Path(local_appdata) / "ChunkHound"
            if local_appdata
            else Path.home() / "AppData" / "Local" / "ChunkHound"
        )

    cache_root = os.environ.get("XDG_CACHE_HOME")
    return (
        Path(cache_root) / "chunkhound"
        if cache_root
        else Path.home() / ".cache" / "chunkhound"
    )
