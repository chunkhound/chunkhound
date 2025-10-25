import xxhash
from pathlib import Path


def compute_file_hash(path: Path) -> str:
    """Compute a fast full-file checksum using xxHash3-64.

    This hash is used for change detection to skip reindexing unchanged files.
    xxHash3-64 is chosen for speed (~10-20x faster than SHA-256) while providing
    sufficient collision resistance for file deduplication (64-bit hash space).

    Note: This is NOT cryptographically secure - it's optimized for fast change
    detection, not security. The hash detects file content changes when size and
    mtime alone are insufficient.

    Args:
        path: File path to hash

    Returns:
        Hex digest string (16 hex characters, 64 bits)

    Raises:
        ValueError: If path is not a file
        OSError: If file cannot be read
    """
    if not path.is_file():
        raise ValueError(f"Path must be a file, not directory: {path}")
    h = xxhash.xxh3_64()
    with path.open('rb') as f:
        # Read in 1MB chunks for efficiency
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
