"""Utilities for embedding generation."""

import re
from pathlib import PurePath
from typing import Any

# Maximum constants to include in embedding header (prevents bloat)
MAX_CONSTANTS_IN_HEADER = 5


def format_chunk_for_embedding(
    code: str,
    file_path: str | None = None,
    language: str | None = None,
    constants: list[dict[str, str]] | None = None,
    rule_target: str | None = None,
) -> str:
    """Prepend path, language, and constants metadata to chunk content for embedding.

    This improves semantic search recall by ~35% (Anthropic Contextual Retrieval)
    by allowing path-related queries to match chunks from relevant directories.
    Constants in the header enable searching for specific constant values.
    rule_target embeds the Makefile rule target name into the header for all rule parts,
    enabling target-aware search across continuation chunks.

    Args:
        code: The chunk code content.
        file_path: Relative file path (e.g., "src/auth/handler.py").
        language: Programming language (e.g., "python").
        constants: List of constant dicts with "name" and "value" keys.
        rule_target: Makefile rule target name; set on all rule parts
            (including Part 1).

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
    if not file_path and not language and not constants and not rule_target:
        return code

    if file_path and language:
        header = f"# {file_path} ({language})"
    elif file_path:
        header = f"# {file_path}"
    elif language:
        header = f"# ({language})"
    else:
        header = "#"

    # Append rule_target context (set on all Makefile rule parts for consistent search)
    if rule_target:
        header += f" [target: {rule_target}]"

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


# Document templates let OpenAI-compatible models that expect a per-role input
# format (e.g. EmbeddingGemma's ``title: ... | text: ...``) receive it.
DEFAULT_DOCUMENT_TEMPLATE = "{text}"
DOCUMENT_TEMPLATE_PLACEHOLDERS = ("text", "file_path", "filename")
_PLACEHOLDER_RE = re.compile(r"\{([^{}]*)\}")


def validate_document_template(template: str) -> str:
    """Validate a document template and return it unchanged.

    Raises:
        ValueError: If the template names an unknown placeholder or has no
            ``{text}`` placeholder (the chunk would never be embedded).
    """
    names = _PLACEHOLDER_RE.findall(template)
    allowed = ", ".join("{" + n + "}" for n in DOCUMENT_TEMPLATE_PLACEHOLDERS)
    for name in names:
        if name not in DOCUMENT_TEMPLATE_PLACEHOLDERS:
            raise ValueError(
                f"document_template contains unknown placeholder {{{name}}}; "
                f"allowed placeholders: {allowed}"
            )
    if "text" not in names:
        raise ValueError("document_template must contain the {text} placeholder")
    return template


def render_document_text(template: str, text: str, file_path: str | None = None) -> str:
    """Render a document template for one chunk.

    Placeholders are substituted in a single pass over the template, so braces
    inside ``text`` (source code) stay literal and are never re-expanded.
    ``{file_path}`` is the repository-relative path with ``/`` separators and
    ``{filename}`` its basename; both resolve to ``none`` without file metadata.
    """
    if template == DEFAULT_DOCUMENT_TEMPLATE:
        return text

    path = file_path.replace("\\", "/") if file_path else None
    values = {
        "text": text,
        "file_path": path or "none",
        "filename": PurePath(path).name if path else "none",
    }
    return _PLACEHOLDER_RE.sub(lambda m: values.get(m.group(1), m.group(0)), template)


def format_document_for_provider(
    provider: Any, text: str, file_path: str | None = None
) -> str:
    """Apply the provider's document template to a chunk's embedding text.

    Providers without a string ``document_template`` (Voyage, test doubles)
    keep ``text`` unchanged.
    """
    template = getattr(provider, "document_template", None)
    if not isinstance(template, str):
        return text
    return render_document_text(template, text, file_path)
