"""Shared helpers for CSS-family language mappings (CSS, SCSS).

Provides the selector-text extractor and the raw node-bytes decoder used by
both CssMapping and ScssMapping.  Keeping them here eliminates the duplication
that would otherwise exist between the two sibling modules.
"""

from tree_sitter import Node

_MAX_SELECTOR_LEN = 60


def node_text(node: Node, content: bytes) -> str:
    """Decode the source span of *node* from raw UTF-8 bytes."""
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
                raw = raw[:_MAX_SELECTOR_LEN] + "…"
            return raw
    return f"rule_line{node.start_point[0] + 1}"
