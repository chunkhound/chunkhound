"""HTML and Jinja language mappings for unified parser architecture.

Provides semantic chunking for HTML documents. Semantic landmark elements
(section, article, main, header, footer, nav, aside, form, table, etc.)
and custom elements (tag names containing '-') are extracted as BLOCK chunks.
Script/style blocks, comments, and imports are also captured.

``JinjaMapping`` is a thin subclass of ``HtmlMapping`` that overrides the
language label to ``Language.JINJA``, ensuring chunks produced from ``.jinja``,
``.j2``, and ``.njk`` files are tagged correctly.
"""

import re
from pathlib import Path
from typing import Any

# Matches Jinja/template expressions: {{ }}, {% %}, {# #}
# Used to sanitize element attribute values before using them as chunk symbols.
_JINJA_EXPR_RE = re.compile(r"\{\{.*?\}\}|\{%.*?%\}|\{#.*?#\}", re.DOTALL)

from tree_sitter import Node

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings._shared.css_family_helpers import (
    # node_text lives in css_family_helpers because it is a generic tree-sitter
    # byte-slice utility shared by the web-language family (CSS, SCSS, HTML).
    node_text,
    resolve_capture,
)
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


class HtmlMapping(BaseMapping):
    """HTML-specific mapping for universal concepts."""

    def __init__(self, language: Language = Language.HTML) -> None:
        super().__init__(language)

    def get_function_query(self) -> str:
        """Get tree-sitter query for function definitions.

        Returns:
            Empty string — HTML has no function definitions.
        """
        return ""

    def get_class_query(self) -> str:
        """Get tree-sitter query for class definitions.

        Returns:
            Empty string — HTML has no class definitions.
        """
        return ""

    def get_comment_query(self) -> str:
        """Get tree-sitter query for HTML comments."""
        return "(comment) @definition"

    def extract_function_name(self, node: Node | None, source: str) -> str:
        """HTML has no function definitions; always returns empty string."""
        return ""

    def extract_class_name(self, node: Node | None, source: str) -> str:
        """HTML has no class definitions; always returns empty string."""
        return ""

    # --- private helpers ---

    def _is_semantic_element(self, node: Node, content: bytes) -> bool:
        """Return True if node is a semantic landmark or custom element."""
        if node.type != "element":
            return False
        tag = self._get_tag_name(node, content)
        if not tag:
            return False
        return tag in SEMANTIC_TAGS or "-" in tag

    def _get_tag_name(self, node: Node, content: bytes) -> str:
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
                return node_text(child, content).lower()
        return ""

    def _get_attribute(
        self, start_tag: Node, attr_name: str, content: bytes
    ) -> str:
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
            name = node_text(name_node, content).lower()
            if name == attr_name and value_node is not None:
                # Strip surrounding quotes
                return node_text(value_node, content).strip("\"'")
        return ""

    def _get_start_tag(self, node: Node) -> Node | None:
        """Return the start_tag child of an element node, or None."""
        for child in node.children:
            if child.type == "start_tag":
                return child
        return None

    def _extract_element_name(self, node: Node, content: bytes) -> str:
        """Derive a human-readable name for an HTML element.

        Strips Jinja/template expressions (``{{ }}``, ``{% %}``, ``{# #}``)
        from attribute values before using them as symbol names, so that
        dynamic attributes like ``id="{{ item.id }}"`` don't produce noisy
        symbols such as ``section#{{ item.id }}``.
        """
        tag = self._get_tag_name(node, content)
        start_tag = self._get_start_tag(node)
        if start_tag is not None:
            # Prefer id > class > aria-label; strip template expressions first
            id_val = _JINJA_EXPR_RE.sub("", self._get_attribute(start_tag, "id", content)).strip()
            if id_val:
                return f"{tag}#{id_val}"
            class_val = _JINJA_EXPR_RE.sub("", self._get_attribute(start_tag, "class", content)).strip()
            if class_val:
                first_class = class_val.split()[0]
                if first_class:
                    return f"{tag}.{first_class}"
            aria_val = _JINJA_EXPR_RE.sub("", self._get_attribute(start_tag, "aria-label", content)).strip()
            if aria_val:
                return f"{tag}[{aria_val[:30]}]"
        if tag:
            return f"{tag}_line{node.start_point[0] + 1}"
        return f"element_line{node.start_point[0] + 1}"

    # --- universal concept interface ---

    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        """Get tree-sitter query for a universal concept in HTML."""
        if concept == UniversalConcept.BLOCK:
            # Use #match? predicates to filter at the tree-sitter C layer so
            # Python only sees semantic landmarks and custom elements (those
            # containing '-').  extract_content still acts as a correctness
            # guard but won't be hit in practice for generic elements.
            return r"""
                (element
                  (start_tag
                    (tag_name) @tag_name
                    (#match? @tag_name "^(section|article|main|header|footer|nav|aside|form|table|details|dialog|figure|fieldset)$")
                  )
                ) @definition
                (element
                  (start_tag
                    (tag_name) @tag_name
                    (#match? @tag_name "-")
                  )
                ) @definition
                (script_element) @definition
                (style_element) @definition
            """
        elif concept == UniversalConcept.COMMENT:
            return "(comment) @definition"
        elif concept == UniversalConcept.STRUCTURE:
            return "(doctype) @definition"
        elif concept == UniversalConcept.IMPORT:
            # #eq? predicate is placed inside the pattern so tree-sitter
            # applies it as a filter at the C layer — only <link> elements
            # and script_element nodes are returned.
            # rel="stylesheet" / src attribute filtering is handled in extract_content.
            return r"""
                (element
                  (start_tag
                    (tag_name) @tag_name
                    (#eq? @tag_name "link")
                  )
                ) @definition
                (script_element) @definition
            """
        elif concept == UniversalConcept.DEFINITION:
            return ""
        return None

    def extract_name(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract a human-readable name for a captured HTML node."""
        node = resolve_capture(captures)
        if node is None:
            return "unnamed"

        if concept == UniversalConcept.BLOCK:
            if node.type == "script_element":
                start_tag = self._get_start_tag(node)
                if start_tag is not None:
                    src = self._get_attribute(start_tag, "src", content)
                    if src:
                        return f"script[src={src}]"
                return f"script_line{node.start_point[0] + 1}"
            if node.type == "style_element":
                return f"style_line{node.start_point[0] + 1}"
            if node.type == "element":
                return self._extract_element_name(node, content)

        elif concept == UniversalConcept.COMMENT:
            return f"comment_line{node.start_point[0] + 1}"

        elif concept == UniversalConcept.STRUCTURE:
            return "doctype"

        elif concept == UniversalConcept.IMPORT:
            if node.type == "script_element":
                start_tag = self._get_start_tag(node)
                if start_tag is not None:
                    src = self._get_attribute(start_tag, "src", content)
                    if src:
                        return src.split("?")[0].split("#")[0]
                return "unnamed"
            if node.type == "element":
                tag = self._get_tag_name(node, content)
                start_tag = self._get_start_tag(node)
                if tag == "link" and start_tag is not None:
                    rel = self._get_attribute(start_tag, "rel", content)
                    if rel and "stylesheet" in rel.lower().split():
                        href = self._get_attribute(start_tag, "href", content)
                        if href:
                            # Strip cache-busting query strings / fragments
                            href = href.split("?")[0].split("#")[0]
                        return href or "link_stylesheet"

        return "unnamed"

    def extract_content(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract raw source text for a captured HTML node, or '' to skip it."""
        node = resolve_capture(captures)
        if node is None:
            return ""

        # Filter BLOCK: only emit semantic/custom elements, inline scripts, style.
        # External scripts (<script src="...">) become IMPORT chunks instead.
        if concept == UniversalConcept.BLOCK:
            if node.type == "script_element":
                start_tag = self._get_start_tag(node)
                if start_tag is not None and self._get_attribute(start_tag, "src", content):
                    return ""  # external script → captured as IMPORT
            elif node.type != "style_element":
                if not self._is_semantic_element(node, content):
                    return ""

        # Filter IMPORT: emit link[rel=stylesheet] and script[src].
        if concept == UniversalConcept.IMPORT:
            if node.type == "script_element":
                start_tag = self._get_start_tag(node)
                if start_tag is None:
                    return ""
                return node_text(node, content) if self._get_attribute(start_tag, "src", content) else ""
            if node.type != "element":
                return ""
            tag = self._get_tag_name(node, content)
            start_tag = self._get_start_tag(node)
            if tag != "link" or start_tag is None:
                return ""
            rel = self._get_attribute(start_tag, "rel", content)
            if not rel or "stylesheet" not in rel.lower().split():
                return ""

        return node_text(node, content)

    def extract_metadata(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> dict[str, Any]:
        """Build metadata dict for a captured HTML node."""
        node = resolve_capture(captures)
        metadata: dict[str, Any] = {}
        if node is not None:
            metadata["node_type"] = node.type
            if node.type == "element":
                tag = self._get_tag_name(node, content)
                metadata["tag_name"] = tag
                metadata["is_custom_element"] = "-" in tag
                metadata["is_semantic"] = tag in SEMANTIC_TAGS
        return metadata

    def resolve_import_paths(
        self, import_text: str, base_dir: Path, source_file: Path
    ) -> list[Path]:
        """Resolve a relative href or src to an absolute filesystem path.

        Handles both ``<link rel="stylesheet" href="...">`` and
        ``<script src="...">`` import forms.  Supports both quoted and
        unquoted attribute values (e.g. ``href=style.css``).

        Args:
            import_text: The full element text (link or script_element).
            base_dir: Project root directory.
            source_file: Path of the importing file — used to resolve relative
                paths from the importing file's directory.

        Returns:
            List with a single resolved Path if it exists, otherwise empty list.
        """
        # Resolve relative to the importing file's directory, not the project root.
        resolve_dir = (
            source_file.parent
            if source_file.is_absolute()
            else (base_dir / source_file).parent
        )
        # Try href (stylesheet link) — support both quoted and unquoted values.
        m = re.search(r'href=(?:["\']([^"\']+)["\']|([^\s>/"\']+))', import_text)
        if m:
            raw = m.group(1) or m.group(2)
            path = raw.split("?")[0].split("#")[0]
            candidate = resolve_dir / path
            if candidate.exists():
                return [candidate.resolve()]
        # Try src (external script) — same attribute format.
        m = re.search(r'src=(?:["\']([^"\']+)["\']|([^\s>/"\']+))', import_text)
        if m:
            raw = m.group(1) or m.group(2)
            path = raw.split("?")[0].split("#")[0]
            candidate = resolve_dir / path
            if candidate.exists():
                return [candidate.resolve()]
        return []

    def extract_constants(
        self,
        concept: UniversalConcept,
        captures: dict[str, Any],
        content: bytes,
    ) -> list[dict[str, str]] | None:
        """HTML does not define constants via this interface; always returns None."""
        return None


class JinjaMapping(HtmlMapping):
    """Jinja/template mapping — identical to HTML but labels chunks as JINJA.

    Jinja ``{{ }}``, ``{% %}``, and ``{# #}`` tokens are treated as plain
    text by the tree-sitter HTML grammar, so the HTML mapping is reused as a
    best-effort approximation.  Overriding ``__init__`` is sufficient to
    ensure all output chunks carry ``Language.JINJA`` rather than
    ``Language.HTML``.
    """

    def __init__(self) -> None:
        super().__init__(Language.JINJA)
