"""Owned index-format compatibility, independent of package release versions."""

INDEX_FORMAT_VERSION = 1
LEGACY_INDEX_FORMAT_VERSION = 1


class IndexFormatError(RuntimeError):
    """An index cannot be safely opened by this version of ChunkHound."""


def require_supported(version: int) -> None:
    """Reject unrecognized formats before any mutation-capable database open."""
    if type(version) is not int or version != INDEX_FORMAT_VERSION:
        raise IndexFormatError(
            "Unsupported index format; upgrade ChunkHound or rebuild"
        )
