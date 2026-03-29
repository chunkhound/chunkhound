"""Shared helpers for CSS-family language mappings (CSS, SCSS, HTML).

Provides tree-sitter node utilities used across CssMapping, ScssMapping, and
HtmlMapping.  Keeping them here eliminates the duplication that would otherwise
exist between sibling modules.

Note: ``node_text`` intentionally accepts ``bytes`` (not ``str``), unlike
``BaseMapping.get_node_text`` which accepts ``str`` source.  The byte-oriented
variant is needed by the concept-extraction path where ``content_bytes`` is
the raw UTF-8 encoding of the original source.
"""

from tree_sitter import Node

_MAX_SELECTOR_LEN = 60


def node_text(node: Node, content: bytes) -> str:
    """Decode the source span of node from raw UTF-8 bytes."""
    return content[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def selector_text(node: Node, content: bytes) -> str:
    """Extract the selector string from a CSS/SCSS rule_set node.

    Returns the raw selectors text truncated to _MAX_SELECTOR_LEN, or a
    line-based fallback when no ``selectors`` child is present (e.g. for
    malformed or edge-case nodes).
    """
    for child in node.children:
        if child.type == "selectors":
            raw = node_text(child, content).strip()
            if len(raw) > _MAX_SELECTOR_LEN:
                raw = raw[:_MAX_SELECTOR_LEN] + "..."
            return raw
    return f"rule_line{node.start_point[0] + 1}"


def resolve_capture(captures: dict[str, Node]) -> Node | None:
    """Return the primary captured node from a tree-sitter captures dict.

    Tries the canonical ``"definition"`` key first, then falls back to the
    first value in the dict.  Returns ``None`` if the dict is empty.
    """
    return captures.get("definition") or (
        next(iter(captures.values()), None) if captures else None
    )


def extract_at_rule_name(node: Node, content: bytes) -> str:
    """Extract a human-readable name from a ``@media`` or ``@keyframes`` node.

    Shared between CssMapping and ScssMapping to avoid duplicating the same
    child-traversal logic.
    """
    if node.type == "media_statement":
        for child in node.children:
            if child.type not in ("@media", "block"):
                cond = node_text(child, content).strip()
                return f"@media {cond[:40]}"
        return f"@media_line{node.start_point[0] + 1}"
    if node.type == "keyframes_statement":
        for child in node.children:
            if child.type == "keyframes_name":
                name = node_text(child, content).strip()
                return f"@keyframes {name}"
        return f"@keyframes_line{node.start_point[0] + 1}"
    # Fallback for unexpected at-rule types; use @ prefix for naming consistency.
    at_name = node.type.replace("_statement", "")
    return f"@{at_name}_line{node.start_point[0] + 1}"
