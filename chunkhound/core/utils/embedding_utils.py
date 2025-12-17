"""Utilities for embedding generation."""


def format_chunk_for_embedding(
    code: str,
    file_path: str | None = None,
    language: str | None = None,
) -> str:
    """Prepend path and language metadata to chunk content for embedding.

    This improves semantic search recall by ~35% (Anthropic Contextual Retrieval)
    by allowing path-related queries to match chunks from relevant directories.

    Args:
        code: The chunk code content.
        file_path: Relative file path (e.g., "src/auth/handler.py").
        language: Programming language (e.g., "python").

    Returns:
        Formatted text with metadata header prepended.

    Examples:
        >>> format_chunk_for_embedding("def foo(): pass", "src/main.py", "python")
        '# src/main.py (python)\\ndef foo(): pass'

        >>> format_chunk_for_embedding("def foo(): pass")
        'def foo(): pass'
    """
    if not file_path and not language:
        return code

    if file_path and language:
        header = f"# {file_path} ({language})"
    elif file_path:
        header = f"# {file_path}"
    else:
        header = f"# ({language})"

    return f"{header}\n{code}"
