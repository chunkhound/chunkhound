"""SCSS language mapping for unified parser architecture.

Extends CSS parsing with SCSS-specific constructs:
- @mixin/@function       → DEFINITION (with name)
- $variable declarations → STRUCTURE
- @include               → BLOCK (inline call)
- rule_set               → DEFINITION (selector string)
- @media/@keyframes      → BLOCK
- @import/@use/@forward  → IMPORT
- comment                → COMMENT
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

# Matches SCSS #{...} interpolations for preprocessing.
_INTERP_RE = re.compile(r"#\{[^}]*\}")


class ScssMapping(BaseMapping):
    """SCSS-specific mapping for universal concepts."""

    def __init__(self) -> None:
        super().__init__(Language.SCSS)

    def preprocess_for_ast(self, content: str) -> str:
        """Replace SCSS interpolations with same-length placeholders.

        The tree-sitter SCSS grammar cannot parse ``--#{$var}name`` (interpolated
        CSS custom property names). Replacing every ``#{...}`` with an equal-length
        run of ``x`` characters keeps byte offsets intact while producing a
        grammar-valid token, so AST positions remain aligned with the original
        source for text extraction.
        """
        return _INTERP_RE.sub(lambda m: "x" * len(m.group().encode("utf-8")), content)

    def get_function_query(self) -> str:
        """Get tree-sitter query for SCSS mixin and function definitions."""
        return "(mixin_statement) @definition (function_statement) @definition"

    def get_class_query(self) -> str:
        """Get tree-sitter query for class definitions.

        Returns:
            Empty string — SCSS has no class definitions.
        """
        return ""

    def get_comment_query(self) -> str:
        """Get tree-sitter query for SCSS comments."""
        return "(comment) @definition"

    def extract_function_name(self, node: Node | None, source: str) -> str:
        """Extract mixin/function name from a mixin or function statement node."""
        if node is None:
            return ""
        # The AST was built from the preprocessed source, so byte offsets are
        # aligned with the preprocessed bytes — not the original source.  Only
        # run the regex when the source actually contains interpolations; for
        # the common case (no #{...}) the original bytes are identical.
        if "#{" in source:
            source_bytes = self.preprocess_for_ast(source).encode("utf-8")
        else:
            source_bytes = source.encode("utf-8")
        return self._identifier_name(node, source_bytes)

    def extract_class_name(self, node: Node | None, source: str) -> str:
        """SCSS has no class definitions; always returns empty string."""
        return ""

    # --- private helpers ---

    def _identifier_name(self, node: Node, content: bytes) -> str:
        """Get the identifier child text from a mixin/function statement."""
        for child in node.children:
            if child.type == "identifier":
                return node_text(child, content).strip()
        return f"unknown_line{node.start_point[0] + 1}"

    def _property_name(self, node: Node, content: bytes) -> str:
        """Get property_name from a declaration (SCSS $variable)."""
        for child in node.children:
            if child.type == "property_name":
                return node_text(child, content).strip()
        return f"var_line{node.start_point[0] + 1}"

    # --- universal concept interface ---

    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        """Get tree-sitter query for a universal concept in SCSS."""
        if concept == UniversalConcept.DEFINITION:
            return """
                (mixin_statement) @definition
                (function_statement) @definition
                (rule_set) @definition
            """
        elif concept == UniversalConcept.BLOCK:
            return """
                (media_statement) @definition
                (keyframes_statement) @definition
                (include_statement) @definition
                (each_statement) @definition
                (for_statement) @definition
                (while_statement) @definition
                (if_statement) @definition
            """
        elif concept == UniversalConcept.STRUCTURE:
            # $variable declarations (top-level declarations starting with $)
            return "(declaration) @definition"
        elif concept == UniversalConcept.IMPORT:
            return """
                (import_statement) @definition
                (use_statement) @definition
                (forward_statement) @definition
            """
        elif concept == UniversalConcept.COMMENT:
            return "(comment) @definition"
        return None

    def extract_name(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract a human-readable name for a captured SCSS node."""
        node = resolve_capture(captures)
        if node is None:
            return "unnamed"

        if concept == UniversalConcept.DEFINITION:
            if node.type == "mixin_statement":
                return f"@mixin {self._identifier_name(node, content)}"
            elif node.type == "function_statement":
                return f"@function {self._identifier_name(node, content)}"
            elif node.type == "rule_set":
                return selector_text(node, content)
            # Unexpected node type — fall through to final return "unnamed".

        elif concept == UniversalConcept.BLOCK:
            if node.type == "include_statement":
                return f"@include_line{node.start_point[0] + 1}"
            elif node.type in (
                "media_statement",
                "keyframes_statement",
            ):
                return extract_at_rule_name(node, content)
            else:
                type_name = node.type.replace("_statement", "")
                return f"@{type_name}_line{node.start_point[0] + 1}"

        elif concept == UniversalConcept.STRUCTURE:
            return self._property_name(node, content)

        elif concept == UniversalConcept.IMPORT:
            raw = node_text(node, content).strip()
            # Strip @import/@use/@forward prefix and trailing semicolon
            for prefix in ("@forward", "@import", "@use"):
                if raw.startswith(prefix):
                    raw = raw[len(prefix) :].strip().rstrip(";").strip()
                    break
            return raw.strip("\"'")[:60]

        elif concept == UniversalConcept.COMMENT:
            return f"comment_line{node.start_point[0] + 1}"

        return "unnamed"

    def extract_content(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract raw source text for a captured SCSS node, or '' to skip it."""
        node = resolve_capture(captures)
        if node is None:
            return ""
        # STRUCTURE: only $variable declarations
        if concept == UniversalConcept.STRUCTURE:
            if node.type != "declaration":
                return ""
            if not self._property_name(node, content).startswith("$"):
                return ""
        return node_text(node, content)

    def extract_metadata(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> dict[str, Any]:
        """Build metadata dict for a captured SCSS node."""
        node = resolve_capture(captures)
        metadata: dict[str, Any] = {}
        if node is not None:
            metadata["node_type"] = node.type
            if node.type in ("mixin_statement", "function_statement"):
                metadata["name"] = self._identifier_name(node, content)
                metadata["chunk_type_hint"] = "function"
            elif node.type == "rule_set":
                metadata["selector"] = selector_text(node, content)
                metadata["chunk_type_hint"] = "block"
            elif node.type == "declaration":
                prop = self._property_name(node, content)
                metadata["property"] = prop
                if prop.startswith("$"):
                    metadata["kind"] = "variable"
        return metadata

    def resolve_import_paths(
        self, import_text: str, base_dir: Path, source_file: Path
    ) -> list[Path]:
        """Resolve an SCSS @import/@use/@forward path to an absolute path.

        Handles SCSS partial conventions: ``@import 'colors'`` also tries
        ``_colors.scss`` (underscore-prefixed partials).

        Args:
            import_text: The import value extracted from the statement.
            base_dir: Directory of the importing file.
            source_file: Path of the importing file (unused, for API compat).

        Returns:
            List with a single resolved Path if it exists, otherwise empty list.
        """
        path = import_text.strip("\"'")
        # Try with and without leading underscore (SCSS partials)
        candidates = [base_dir / path]
        stem = Path(path).stem
        parent = Path(path).parent
        candidates.append(base_dir / parent / f"_{stem}.scss")
        candidates.append(base_dir / f"{path}.scss")
        for c in candidates:
            if c.exists():
                return [c]
        return []

    def extract_constants(
        self,
        concept: Any,
        captures: dict[str, Any],
        content: bytes,
    ) -> list[dict[str, str]] | None:
        """SCSS does not define constants via this interface; always returns None."""
        return None
