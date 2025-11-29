"""High-performance chunk deduplication with O(n log n) complexity.

Implements two-stage deduplication:
1. Exact content matching via hash table (O(n))
2. Substring detection via interval tree (O(n log n))

Preserves language-specific exemptions for Vue and Haskell.
"""

from typing import Sequence

from collections import defaultdict
import xxhash

from chunkhound.core.models.chunk import Chunk
from chunkhound.core.types.common import ChunkType, Language
from chunkhound.utils.normalization import normalize_content


def deduplicate_chunks(
    chunks: Sequence[Chunk | dict],
    language: Language | str | None = None,
) -> list[Chunk | dict]:
    """Deduplicate chunks using hash-based exact match + interval tree substring detection.

    Args:
        chunks: List of Chunk objects or chunk dictionaries
        language: Optional language for language-specific exemptions (Language enum or string)

    Returns:
        Deduplicated list of chunks (same type as input)
    """
    if not chunks:
        return []

    # Language exemptions: Vue and Haskell preserve duplicates for semantic reasons
    # Handle both Language enum and string values
    if language is None:
        language_name = ""
    elif hasattr(language, "value"):
        language_name = language.value.lower()
    else:
        language_name = str(language).lower()

    preserve_duplicates = language_name in ["vue", "vue_template", "haskell"]

    if preserve_duplicates:
        return list(chunks)

    # Stage 1: Exact content deduplication via hash table (O(n))
    exact_match_deduplicated = _deduplicate_exact_content(chunks)

    # Stage 2: Substring detection via interval tree (O(n log n))
    final = _remove_substring_overlaps(exact_match_deduplicated)

    return final


def _deduplicate_exact_content(chunks: Sequence[Chunk | dict]) -> list[Chunk | dict]:
    """Remove chunks with identical normalized content, keeping highest specificity.

    Uses hash table for O(n) performance instead of O(n²) nested loops.
    """
    # Build hash table: content_hash -> list of chunks with that hash
    hash_to_chunks: dict[int, list[Chunk | dict]] = defaultdict(list)

    for chunk in chunks:
        content = _get_chunk_content(chunk)
        normalized = normalize_content(content)

        if not normalized:  # Skip empty chunks
            continue

        # Use xxHash3-64 for fast, collision-resistant hashing
        content_hash = xxhash.xxh3_64(normalized.encode("utf-8")).intdigest()
        hash_to_chunks[content_hash].append(chunk)

    # For each hash, keep only the chunk with highest specificity
    result = []
    for chunk_list in hash_to_chunks.values():
        if len(chunk_list) == 1:
            result.append(chunk_list[0])
        else:
            # Multiple chunks with same content - pick best by specificity
            best = max(
                chunk_list,
                key=lambda c: (
                    get_chunk_specificity(c),
                    -(_get_end_line(c) - _get_start_line(c)),  # Prefer smaller spans
                ),
            )
            result.append(best)

    return result


def _remove_substring_overlaps(chunks: Sequence[Chunk | dict]) -> list[Chunk | dict]:
    """Remove BLOCK chunks that are substrings of DEFINITION/STRUCTURE chunks.

    Uses interval tree for O(n log n) performance instead of O(n²) nested loops.
    """
    # Build interval tree: sorted list of (start_line, end_line, chunk) for DEFINITION/STRUCTURE
    definitions = []
    blocks = []
    other = []

    for chunk in chunks:
        # Use specificity to categorize chunks
        # High specificity (4) = definitions (functions, classes)
        # Low specificity (1) = blocks (code blocks, arrays)
        specificity = get_chunk_specificity(chunk)

        if specificity == 1:  # BLOCK-like chunks
            blocks.append(chunk)
        elif specificity >= 3:  # DEFINITION-like or STRUCTURE-like chunks
            definitions.append(chunk)
        else:
            other.append(chunk)

    # Sort definitions by start line for binary search
    definitions.sort(key=lambda c: _get_start_line(c))

    # Check each BLOCK chunk for substring containment in overlapping DEFINITION/STRUCTURE
    final = other + definitions  # Keep all non-BLOCK chunks

    for block in blocks:
        block_content = normalize_content(_get_chunk_content(block))
        block_start = _get_start_line(block)
        block_end = _get_end_line(block)

        is_substring = False

        # Binary search for overlapping definitions (interval tree query)
        for definition in _find_overlapping_chunks(definitions, block_start, block_end):
            def_content = normalize_content(_get_chunk_content(definition))

            # Check substring containment
            if block_content in def_content and len(block_content) < len(def_content):
                is_substring = True
                break

        if not is_substring:
            final.append(block)

    return final


def _find_overlapping_chunks(
    sorted_chunks: list[Chunk | dict],
    query_start: int,
    query_end: int,
) -> list[Chunk | dict]:
    """Find chunks whose line ranges overlap with [query_start, query_end].

    Uses sorted list for early termination when query_end < chunk_start.
    Complexity: O(k) where k = overlapping chunks (not O(n) full scan).
    """
    overlapping = []

    for chunk in sorted_chunks:
        chunk_start = _get_start_line(chunk)
        chunk_end = _get_end_line(chunk)

        # No overlap: chunk ends before query starts
        if chunk_end < query_start:
            continue

        # No overlap: chunk starts after query ends (rest of list also won't overlap)
        if chunk_start > query_end:
            break

        # Overlap detected
        overlapping.append(chunk)

    return overlapping


# Specificity ranking (higher = more specific)
# Maps concept/type names to specificity scores
_CONCEPT_SPECIFICITY = {
    "DEFINITION": 4,
    "IMPORT": 3,
    "COMMENT": 2,
    "BLOCK": 1,
    "STRUCTURE": 0,
    # ChunkType mappings (for standard Chunk objects)
    "FUNCTION": 4,
    "METHOD": 4,
    "CLASS": 4,
    "INTERFACE": 4,
    "STRUCT": 4,
    "ENUM": 4,
    "TYPE_ALIAS": 3,
    "KEY_VALUE": 2,  # YAML key-value pairs
    "ARRAY": 1,
    "BLOCK": 1,
}


def get_chunk_specificity(chunk: Chunk | dict | object) -> int:
    """Get specificity ranking for chunk's concept type.

    Supports Chunk (ChunkType), UniversalChunk (UniversalConcept), and dict.
    Returns -1 for unknown types.
    """
    chunk_type = _get_chunk_type(chunk)

    if chunk_type is None:
        return -1

    # Handle enum types (both ChunkType and UniversalConcept)
    if hasattr(chunk_type, "value"):
        # Get the string value from enum
        type_value = chunk_type.value
        if isinstance(type_value, str):
            type_name = type_value.upper()
        else:
            type_name = str(type_value).upper()
    elif hasattr(chunk_type, "name"):
        # Fallback to enum name
        type_name = chunk_type.name.upper()
    else:
        # String type or unknown
        type_name = str(chunk_type).upper() if chunk_type else ""

    return _CONCEPT_SPECIFICITY.get(type_name, -1)


# Helper functions for Chunk vs dict vs UniversalChunk compatibility
def _get_chunk_content(chunk: Chunk | dict | object) -> str:
    """Get content from Chunk, dict, or UniversalChunk object."""
    if isinstance(chunk, dict):
        return chunk.get("code", "") or ""
    # Check for UniversalChunk (has 'content' attribute)
    if hasattr(chunk, "content"):
        return chunk.content or ""
    # Standard Chunk (has 'code' attribute)
    if hasattr(chunk, "code"):
        return chunk.code or ""
    return ""


def _get_chunk_type(chunk: Chunk | dict | object) -> ChunkType | object:
    """Get type from Chunk, dict, or UniversalChunk object."""
    if isinstance(chunk, dict):
        chunk_type = chunk.get("chunk_type")
        if isinstance(chunk_type, str):
            return ChunkType(chunk_type)
        return chunk_type
    # Check for UniversalChunk (has 'concept' attribute)
    if hasattr(chunk, "concept"):
        return chunk.concept
    # Standard Chunk (has 'chunk_type' attribute)
    if hasattr(chunk, "chunk_type"):
        return chunk.chunk_type
    return None


def _get_start_line(chunk: Chunk | dict | object) -> int:
    """Get start line from Chunk, dict, or UniversalChunk object."""
    if isinstance(chunk, dict):
        return int(chunk.get("start_line", 0))
    if hasattr(chunk, "start_line"):
        return int(chunk.start_line)
    return 0


def _get_end_line(chunk: Chunk | dict | object) -> int:
    """Get end line from Chunk, dict, or UniversalChunk object."""
    if isinstance(chunk, dict):
        return int(chunk.get("end_line", 0))
    if hasattr(chunk, "end_line"):
        return int(chunk.end_line)
    return 0
