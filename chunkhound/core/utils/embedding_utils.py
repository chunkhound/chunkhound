"""Utilities for embedding generation."""

# Maximum constants to include in embedding header (prevents bloat)
MAX_CONSTANTS_IN_HEADER = 5


def format_chunk_for_embedding(
    code: str,
    file_path: str | None = None,
    language: str | None = None,
    constants: list[dict[str, str]] | None = None,
) -> str:
    """Prepend path, language, and constants metadata to chunk content for embedding.

    This improves semantic search recall by ~35% (Anthropic Contextual Retrieval)
    by allowing path-related queries to match chunks from relevant directories.
    Constants in the header enable searching for specific constant values.

    Args:
        code: The chunk code content.
        file_path: Relative file path (e.g., "src/auth/handler.py").
        language: Programming language (e.g., "python").
        constants: List of constant dicts with "name" and "value" keys.

    Returns:
        Formatted text with metadata header prepended.

    Examples:
        >>> format_chunk_for_embedding("def foo(): pass", "src/main.py", "python")
        '# src/main.py (python)\\ndef foo(): pass'

        >>> format_chunk_for_embedding("MAX=100", "config.py", "python",
        ...                            [{"name": "MAX", "value": "100"}])
        '# config.py (python) [MAX=100]\\nMAX=100'

        >>> format_chunk_for_embedding("def foo(): pass")
        'def foo(): pass'
    """
    if not file_path and not language and not constants:
        return code

    if file_path and language:
        header = f"# {file_path} ({language})"
    elif file_path:
        header = f"# {file_path}"
    elif language:
        header = f"# ({language})"
    else:
        header = "#"

    # Append constants summary to header (limited to avoid bloat)
    if constants:
        const_items = [
            f"{c['name']}={c['value']}" for c in constants[:MAX_CONSTANTS_IN_HEADER]
        ]
        const_str = ", ".join(const_items)
        if len(constants) > MAX_CONSTANTS_IN_HEADER:
            const_str += f", +{len(constants) - MAX_CONSTANTS_IN_HEADER} more"
        header += f" [{const_str}]"

    return f"{header}\n{code}"
