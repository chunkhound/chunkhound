"""CSS language mapping for unified parser architecture.

Maps CSS AST nodes to semantic chunks:
- rule_set            → DEFINITION (selector string as name)
- @media/@keyframes   → BLOCK
- :root / * with vars → STRUCTURE
- @import             → IMPORT
- comment             → COMMENT
"""

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
        """Return True if rule_set is :root or * containing custom properties."""
        sel = selector_text(node, content)
        if sel not in (":root", "*"):
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
            return ":root_vars"

        elif concept == UniversalConcept.IMPORT:
            raw = node_text(node, content).strip()
            # Strip '@import ' prefix and trailing semicolon
            raw = raw.removeprefix("@import").strip().rstrip(";").strip()
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
                metadata["is_root_vars"] = self._is_root_vars(node, content)
                metadata["chunk_type_hint"] = "block"
        return metadata

    def resolve_import_paths(
        self, import_text: str, base_dir: Path, source_file: Path
    ) -> list[Path]:
        """Resolve a CSS @import path to an absolute filesystem path.

        Strips surrounding quotes and ``url(...)`` wrappers before resolving.

        Args:
            import_text: The import value extracted from the @import statement.
            base_dir: Directory of the importing file.
            source_file: Path of the importing file (unused, for API compat).

        Returns:
            List with a single resolved Path if it exists, otherwise empty list.
        """
        # Strip quotes and url()
        path = import_text.strip("\"'")
        if path.startswith("url("):
            path = path[4:].rstrip(")").strip("\"'")
        candidate = base_dir / path
        if candidate.exists():
            return [candidate]
        return []

    def extract_constants(
        self,
        concept: Any,
        captures: dict[str, Any],
        content: bytes,
    ) -> list[dict[str, str]] | None:
        """CSS does not define constants; always returns None."""
        return None
