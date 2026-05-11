"""CSS language mapping for unified parser architecture.

Maps CSS AST nodes to semantic chunks:
- rule_set            → DEFINITION (selector string as name)
- @media/@keyframes   → BLOCK
- :root / * with vars → STRUCTURE
- @import             → IMPORT
- comment             → COMMENT

Design note — dual-query for DEFINITION and STRUCTURE:
Both concepts use ``(rule_set) @definition`` as their tree-sitter query.
``extract_content`` acts as the discriminator: it returns ``""`` (skip) for
``DEFINITION`` when the rule set is a ``:root``/``*`` var block, and returns
``""`` for ``STRUCTURE`` when it is not.  This ensures each ``rule_set`` node
produces exactly one chunk with no duplicates.
"""

import re
from pathlib import Path
from typing import Any

from tree_sitter import Node

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings._shared.css_family_helpers import (
    extract_at_rule_name,
    node_text,
    resolve_capture,
    selector_text,
)
from chunkhound.parsers.mappings.base import BaseMapping
from chunkhound.parsers.universal_engine import UniversalConcept


class CssMapping(BaseMapping):
    """CSS-specific mapping for universal concepts."""

    def __init__(self) -> None:
        super().__init__(Language.CSS)

    def get_function_query(self) -> str:
        """Get tree-sitter query for function definitions.

        Returns:
            Empty string — CSS has no function definitions.
        """
        return ""

    def get_class_query(self) -> str:
        """Get tree-sitter query for class definitions.

        Returns:
            Empty string — CSS has no class definitions.
        """
        return ""

    def get_comment_query(self) -> str:
        """Get tree-sitter query for CSS comments."""
        return "(comment) @definition"

    def extract_function_name(self, node: Node | None, source: str) -> str:
        """CSS has no function definitions; always returns empty string."""
        return ""

    def extract_class_name(self, node: Node | None, source: str) -> str:
        """CSS has no class definitions; always returns empty string."""
        return ""

    # --- private helpers ---

    def _is_root_vars(self, node: Node, content: bytes) -> bool:
        """Return True if rule_set is :root or * containing custom properties.

        Handles comma-separated selector lists such as ``:root, [data-bs-theme=light]``
        (Bootstrap theme tokens) by checking whether any selector part is ``:root`` or ``*``.

        Note: reads the raw selectors node text directly instead of calling
        ``selector_text()`` because that helper truncates at 60 characters — a selector
        list with ``:root`` after the cutoff would otherwise be misclassified.
        """
        sel = ""
        for child in node.children:
            if child.type == "selectors":
                sel = node_text(child, content).strip()
                break
        if not sel:
            return False
        parts = [s.strip() for s in sel.split(",")]
        if not any(p in (":root", "*") for p in parts):
            return False
        # Walk direct block children looking for a declaration whose property
        # name starts with '--'.  This is more precise than a substring match
        # on the whole block text (avoids false positives from comments like
        # /* -- separator */ or calc values).
        for child in node.children:
            if child.type == "block":
                for block_child in child.children:
                    if block_child.type == "declaration":
                        for prop_child in block_child.children:
                            if prop_child.type == "property_name":
                                prop = node_text(prop_child, content).strip()
                                if prop.startswith("--"):
                                    return True
        return False

    # --- universal concept interface ---

    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        """Get tree-sitter query for a universal concept in CSS."""
        if concept == UniversalConcept.DEFINITION:
            return "(rule_set) @definition"
        elif concept == UniversalConcept.BLOCK:
            return """
                (media_statement) @definition
                (keyframes_statement) @definition
                (supports_statement) @definition
            """
        elif concept == UniversalConcept.STRUCTURE:
            # Intentionally the same query as DEFINITION — both scan rule_set nodes.
            # extract_content filters them to non-overlapping sets:
            #   DEFINITION → rule sets that are NOT :root/:* var blocks
            #   STRUCTURE  → rule sets that ARE :root/:* var blocks
            return "(rule_set) @definition"
        elif concept == UniversalConcept.IMPORT:
            return "(import_statement) @definition"
        elif concept == UniversalConcept.COMMENT:
            return "(comment) @definition"
        return None

    def extract_name(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract a human-readable name for a captured CSS node."""
        node = resolve_capture(captures)
        if node is None:
            return "unnamed"

        if concept == UniversalConcept.DEFINITION:
            return selector_text(node, content)

        elif concept == UniversalConcept.BLOCK:
            if node.type == "supports_statement":
                return f"@supports_line{node.start_point[0] + 1}"
            return extract_at_rule_name(node, content)

        elif concept == UniversalConcept.STRUCTURE:
            sel = selector_text(node, content)
            return f"{sel}_vars_line{node.start_point[0] + 1}"

        elif concept == UniversalConcept.IMPORT:
            raw = node_text(node, content).strip()
            # Strip '@import ' prefix and trailing semicolon
            raw = raw.removeprefix("@import").strip().rstrip(";").strip()
            # Take only the first whitespace-delimited token — anything after
            # a space is a media query (e.g. ``"reset.css" screen, print``).
            raw = raw.split()[0] if raw else raw
            return raw[:60]

        elif concept == UniversalConcept.COMMENT:
            return f"comment_line{node.start_point[0] + 1}"

        return "unnamed"

    def extract_content(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract raw source text for a captured CSS node, or '' to skip it."""
        node = resolve_capture(captures)
        if node is None:
            return ""
        # STRUCTURE: only :root/:* blocks with --variables
        if concept == UniversalConcept.STRUCTURE:
            if not self._is_root_vars(node, content):
                return ""
        # DEFINITION: exclude :root/:* var blocks (those are STRUCTURE)
        if concept == UniversalConcept.DEFINITION:
            if self._is_root_vars(node, content):
                return ""
        return node_text(node, content)

    def extract_metadata(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> dict[str, Any]:
        """Build metadata dict for a captured CSS node."""
        node = resolve_capture(captures)
        metadata: dict[str, Any] = {}
        if node is not None:
            metadata["node_type"] = node.type
            if node.type == "rule_set":
                metadata["selector"] = selector_text(node, content)
                is_root_vars = self._is_root_vars(node, content)
                metadata["is_root_vars"] = is_root_vars
                # :root/:* var blocks are STRUCTURE (namespace); plain rule sets are blocks.
                metadata["chunk_type_hint"] = "namespace" if is_root_vars else "block"
                if is_root_vars:
                    # Prevent :root/:* var blocks from merging with adjacent rule-set chunks.
                    # DEFINITION↔STRUCTURE is a valid pair for OOP languages (class+method);
                    # CSS needs the per-chunk override instead of a global exclusion.
                    metadata["prevent_merge_across_concepts"] = True
        return metadata

    def resolve_import_paths(
        self, import_text: str, base_dir: Path, source_file: Path
    ) -> list[Path]:
        """Resolve a CSS @import path to an absolute filesystem path.

        Strips surrounding quotes and ``url(...)`` wrappers before resolving.

        Args:
            import_text: The full @import statement text.
            base_dir: Project root directory.
            source_file: Path of the importing file — used to resolve relative imports
                from the importing file's directory rather than the project root.

        Returns:
            List with a single resolved Path if it exists, otherwise empty list.
        """
        # Resolve relative to the importing file's directory, not the project root.
        resolve_dir = (
            source_file.parent
            if source_file.is_absolute()
            else (base_dir / source_file).parent
        )
        # Strip @import prefix and trailing semicolon
        text = import_text.removeprefix("@import").strip().rstrip(";").strip()
        # Unwrap url(...) before splitting on whitespace so that
        # ``url( path )`` with internal spaces is handled correctly.
        url_m = re.match(r'^url\(\s*(["\']?)(.+?)\1\s*\)', text, re.IGNORECASE)
        if url_m:
            text = url_m.group(2)
        else:
            # Take only the first whitespace-delimited token (rest may be a
            # media query, e.g. ``"reset.css" screen, print``).
            text = text.split()[0] if text else text
        # Strip surrounding quotes
        path = text.strip("\"'")
        # Strip query strings (?...) and fragments (#...) — they are not part of the
        # file path and would cause resolution to fail silently.
        path = re.split(r"[?#]", path)[0]
        candidate = resolve_dir / path
        if candidate.exists():
            return [candidate.resolve()]
        return []

    def extract_constants(
        self,
        concept: Any,
        captures: dict[str, Any],
        content: bytes,
    ) -> list[dict[str, str]] | None:
        """CSS does not define constants; always returns None."""
        return None
