"""Chunk ID generation using content-based hashing.

Provides deterministic, collision-resistant chunk IDs based on file ID, content,
and source position. Uses xxHash3-64 for fast hashing with negligible collision
probability.
"""

import xxhash

from chunkhound.utils.normalization import normalize_content


def generate_chunk_id(
    file_id: int,
    content: str,
    concept: str | None = None,
    *,
    start_line: int | None = None,
    end_line: int | None = None,
) -> int:
    """Generate deterministic 64-bit chunk ID from file, content, and position.

    Uses xxHash3-64 for fast, collision-resistant hashing. Hash includes:

    - ``file_id`` so the same text in different files gets different IDs
    - normalized content
    - optional concept type (Vue/Haskell semantic disambiguation)
    - optional ``start_line`` / ``end_line`` so **identical content at different
      locations in the same file** gets different IDs

    Position is required for Lance append (deferred write): content-only IDs
    collide for copy-pasted blocks / empty chunks, producing duplicate primary
    keys. Later ``merge_insert("id")`` then fails with ambiguous match.

    The content is normalized before hashing to ignore insignificant whitespace
    differences (e.g., line endings, trailing whitespace).

    Collision probability: ~1.5 × 10^-12 for 1M chunks (negligible in practice).

    Args:
        file_id: File ID from database (for per-file uniqueness)
        content: Raw chunk code content
        concept: Optional concept type (DEFINITION, BLOCK, etc.)
        start_line: Optional 1-based start line (positional uniqueness)
        end_line: Optional 1-based end line

    Returns:
        64-bit signed integer suitable for database storage
    """
    # Normalize content to ignore insignificant whitespace differences
    # This ensures CRLF vs LF, trailing spaces, etc. don't create different IDs
    normalized = normalize_content(content)

    # Use xxHash3-64 for fast, collision-resistant hashing
    h = xxhash.xxh3_64()

    # Include file_id for per-file uniqueness
    # (same content in different files gets different IDs)
    h.update(str(file_id).encode("utf-8"))

    # Hash the normalized content
    h.update(normalized.encode("utf-8"))

    # Include concept type if provided (for Vue/Haskell semantic disambiguation)
    # This ensures identical content with different semantic meanings gets different IDs
    if concept is not None:
        h.update(concept.encode("utf-8"))

    # Positional salt: identical blocks at different lines must not share a PK
    # (Lance append + later merge_insert on id).
    if start_line is not None:
        h.update(b"|sl:")
        h.update(str(int(start_line)).encode("utf-8"))
    if end_line is not None:
        h.update(b"|el:")
        h.update(str(int(end_line)).encode("utf-8"))

    # Get unsigned 64-bit hash
    unsigned_hash = h.intdigest()

    # Convert to signed 64-bit integer (compatible with database INTEGER types)
    # xxHash3 returns unsigned (0 to 2^64-1), databases expect signed (-2^63 to 2^63-1)
    if unsigned_hash >= 2**63:
        return unsigned_hash - 2**64
    return unsigned_hash
