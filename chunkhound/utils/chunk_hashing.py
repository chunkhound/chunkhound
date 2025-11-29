"""Chunk ID generation using content-based hashing.

Provides deterministic, collision-resistant chunk IDs based on file ID and content.
Uses xxHash3-64 for fast hashing with negligible collision probability.
"""

import xxhash

from chunkhound.utils.normalization import normalize_content


def generate_chunk_id(file_id: int, content: str, concept: str | None = None) -> int:
    """Generate deterministic 64-bit chunk ID from file, content, and concept.

    Uses xxHash3-64 for fast, collision-resistant hashing. Hash includes file_id
    to maintain per-file uniqueness while enabling content-based deduplication
    within files. Optionally includes concept type to disambiguate identical
    content with different semantic meanings (e.g., Vue directives vs elements).

    The content is normalized before hashing to ignore insignificant whitespace
    differences (e.g., line endings, trailing whitespace).

    Collision probability: ~1.5 × 10^-12 for 1M chunks (negligible in practice).

    Args:
        file_id: File ID from database (for per-file uniqueness)
        content: Raw chunk code content
        concept: Optional concept type (DEFINITION, BLOCK, etc.) to disambiguate
                 identical content with different semantic meanings. Used for
                 Vue/Haskell where same content may be extracted as multiple
                 semantic concepts.

    Returns:
        64-bit signed integer suitable for database storage

    Example:
        >>> generate_chunk_id(123, "def foo(): pass")
        -5247198712345678901
        >>> # Same content, same file → same ID (deterministic)
        >>> generate_chunk_id(123, "def foo(): pass")
        -5247198712345678901
        >>> # Same content, different file → different ID
        >>> generate_chunk_id(456, "def foo(): pass")
        8765432198765432109
        >>> # Same content, different concept → different ID (Vue/Haskell)
        >>> generate_chunk_id(123, "def foo(): pass", concept="DEFINITION")
        -1234567890123456789
        >>> generate_chunk_id(123, "def foo(): pass", concept="BLOCK")
        9876543210987654321
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

    # Get unsigned 64-bit hash
    unsigned_hash = h.intdigest()

    # Convert to signed 64-bit integer (compatible with database INTEGER types)
    # xxHash3 returns unsigned (0 to 2^64-1), databases expect signed (-2^63 to 2^63-1)
    if unsigned_hash >= 2**63:
        return unsigned_hash - 2**64
    return unsigned_hash
