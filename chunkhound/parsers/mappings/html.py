"""HTML language mapping for unified parser architecture.

Provides semantic chunking for HTML documents. Semantic landmark elements
(section, article, main, header, footer, nav, aside, form, table, etc.)
and custom elements (tag names containing '-') are extracted as BLOCK chunks.
Script/style blocks, comments, and imports are also captured.
"""

from pathlib import Path
from typing import Any

from tree_sitter import Node

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.base import BaseMapping
from chunkhound.parsers.universal_engine import UniversalConcept

# HTML semantic landmark tags to extract as BLOCK chunks
SEMANTIC_TAGS = frozenset(
    {
        "section",
        "article",
        "main",
        "header",
        "footer",
        "nav",
        "aside",
        "form",
        "table",
        "details",
        "dialog",
        "figure",
        "fieldset",
    }
)


def _is_semantic_element(node: Node, content: bytes) -> bool:
    """Return True if node is a semantic landmark or custom element."""
    if node.type != "element":
        return False
    start_tag = node.child_by_field_name("start_tag") or (
        node.children[0] if node.children else None
    )
    if start_tag is None or start_tag.type != "start_tag":
        return False
    tag_name_node = None
    for child in start_tag.children:
        if child.type == "tag_name":
            tag_name_node = child
            break
    if tag_name_node is None:
        return False
    tag = content[tag_name_node.start_byte : tag_name_node.end_byte].decode(
        "utf-8", errors="replace"
    ).strip().lower()
    if not tag:
        return False
    return tag in SEMANTIC_TAGS or "-" in tag


def _get_tag_name(node: Node, content: bytes) -> str:
    """Extract the tag name from an element node."""
    start_tag = None
    for child in node.children:
        if child.type == "start_tag":
            start_tag = child
            break
    if start_tag is None:
        return ""
    for child in start_tag.children:
        if child.type == "tag_name":
            return content[child.start_byte : child.end_byte].decode(
                "utf-8", errors="replace"
            ).lower()
    return ""


def _get_attribute(start_tag: Node, attr_name: str, content: bytes) -> str:
    """Extract a specific attribute value from a start_tag node."""
    for child in start_tag.children:
        if child.type != "attribute":
            continue
        name_node = None
        value_node = None
        for attr_child in child.children:
            if attr_child.type == "attribute_name":
                name_node = attr_child
            elif attr_child.type in ("quoted_attribute_value", "attribute_value"):
                value_node = attr_child
        if name_node is None:
            continue
        name = content[name_node.start_byte : name_node.end_byte].decode(
            "utf-8", errors="replace"
        ).lower()
        if name == attr_name and value_node is not None:
            raw = content[value_node.start_byte : value_node.end_byte].decode(
                "utf-8", errors="replace"
            )
            # Strip surrounding quotes
            return raw.strip("\"'")
    return ""


def _extract_element_name(node: Node, content: bytes) -> str:
    """Derive a human-readable name for an HTML element."""
    tag = _get_tag_name(node, content)
    # Find start_tag to query attributes
    start_tag = None
    for child in node.children:
        if child.type == "start_tag":
            start_tag = child
            break
    if start_tag is not None:
        # Prefer id > class > aria-label
        id_val = _get_attribute(start_tag, "id", content)
        if id_val:
            return f"{tag}#{id_val}"
        class_val = _get_attribute(start_tag, "class", content)
        if class_val:
            class_parts = class_val.split()
            first_class = class_parts[0] if class_parts else ""
            if first_class:
                return f"{tag}.{first_class}"
        aria_val = _get_attribute(start_tag, "aria-label", content)
        if aria_val:
            return f"{tag}[{aria_val[:30]}]"
    if tag:
        return f"{tag}_line{node.start_point[0] + 1}"
    return f"element_line{node.start_point[0] + 1}"


class HtmlMapping(BaseMapping):
    """HTML-specific mapping for universal concepts."""

    def __init__(self) -> None:
        super().__init__(Language.HTML)

    def get_function_query(self) -> str:
        """HTML has no function definitions to extract."""
        return ""

    def get_class_query(self) -> str:
        """HTML has no class definitions to extract."""
        return ""

    def get_comment_query(self) -> str:
        return "(comment) @definition"

    def extract_function_name(self, node: Node | None, source: str) -> str:
        """HTML has no function definitions; always returns empty string."""
        return ""

    def extract_class_name(self, node: Node | None, source: str) -> str:
        """HTML has no class definitions; always returns empty string."""
        return ""

    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        if concept == UniversalConcept.BLOCK:
            # Capture all elements; filtering to semantic/custom happens in extract_name
            return """
                (element) @definition
                (script_element) @definition
                (style_element) @definition
            """
        elif concept == UniversalConcept.COMMENT:
            return "(comment) @definition"
        elif concept == UniversalConcept.STRUCTURE:
            return "(doctype) @definition"
        elif concept == UniversalConcept.IMPORT:
            return """
                (element) @definition
            """
        elif concept == UniversalConcept.DEFINITION:
            return ""
        return None

    def extract_name(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        node = captures.get("definition") or (
            next(iter(captures.values()), None) if captures else None
        )
        if node is None:
            return "unnamed"

        if concept == UniversalConcept.BLOCK:
            if node.type == "script_element":
                # Try to get src attribute
                start_tag = None
                for child in node.children:
                    if child.type == "start_tag":
                        start_tag = child
                        break
                if start_tag is not None:
                    src = _get_attribute(start_tag, "src", content)
                    if src:
                        return f"script[src={src}]"
                return f"script_line{node.start_point[0] + 1}"
            if node.type == "style_element":
                return f"style_line{node.start_point[0] + 1}"
            if node.type == "element":
                return _extract_element_name(node, content)

        elif concept == UniversalConcept.COMMENT:
            return f"comment_line{node.start_point[0] + 1}"

        elif concept == UniversalConcept.STRUCTURE:
            return "doctype"

        elif concept == UniversalConcept.IMPORT:
            if node.type == "element":
                tag = _get_tag_name(node, content)
                start_tag = None
                for child in node.children:
                    if child.type == "start_tag":
                        start_tag = child
                        break
                if tag == "link" and start_tag is not None:
                    rel = _get_attribute(start_tag, "rel", content)
                    if rel == "stylesheet":
                        href = _get_attribute(start_tag, "href", content)
                        return href or "link_stylesheet"
                if tag == "script" and start_tag is not None:
                    src = _get_attribute(start_tag, "src", content)
                    if src:
                        return src

        return "unnamed"

    def extract_content(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        node = captures.get("definition") or (
            next(iter(captures.values()), None) if captures else None
        )
        if node is None:
            return ""

        # Filter BLOCK: only emit semantic/custom elements, script, style
        if concept == UniversalConcept.BLOCK:
            if node.type not in ("script_element", "style_element"):
                if not _is_semantic_element(node, content):
                    return ""

        # Filter IMPORT: only emit link[rel=stylesheet] and script[src=...]
        if concept == UniversalConcept.IMPORT:
            if node.type != "element":
                return ""
            tag = _get_tag_name(node, content)
            start_tag = None
            for child in node.children:
                if child.type == "start_tag":
                    start_tag = child
                    break
            if tag == "link" and start_tag is not None:
                rel = _get_attribute(start_tag, "rel", content)
                if rel != "stylesheet":
                    return ""
            elif tag == "script" and start_tag is not None:
                src = _get_attribute(start_tag, "src", content)
                if not src:
                    return ""
            else:
                return ""

        return content[node.start_byte : node.end_byte].decode(
            "utf-8", errors="replace"
        )

    def extract_metadata(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> dict[str, Any]:
        node = captures.get("definition") or (
            next(iter(captures.values()), None) if captures else None
        )
        metadata: dict[str, Any] = {}
        if node is not None:
            metadata["node_type"] = node.type
            if node.type == "element":
                tag = _get_tag_name(node, content)
                metadata["tag_name"] = tag
                metadata["is_custom_element"] = "-" in tag
                metadata["is_semantic"] = tag in SEMANTIC_TAGS
        return metadata

    def resolve_import_paths(
        self, import_text: str, base_dir: Path, source_file: Path
    ) -> list[Path]:
        # Resolve relative hrefs/srcs to actual paths
        candidate = base_dir / import_text
        if candidate.exists():
            return [candidate]
        return []

    def extract_constants(
        self,
        concept: Any,
        captures: dict[str, Any],
        content: bytes,
    ) -> list[dict[str, str]] | None:
        return None

