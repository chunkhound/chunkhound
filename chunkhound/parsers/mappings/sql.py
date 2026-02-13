"""SQL language mapping for unified parser architecture.

This module provides SQL-specific tree-sitter queries and extraction logic
for the universal concept system. It maps SQL's AST nodes to universal
semantic concepts used by the unified parser.

Supported constructs: CREATE TABLE, CREATE VIEW, CREATE FUNCTION,
CREATE INDEX, CREATE TRIGGER, ALTER TABLE, comments, and BEGIN...END blocks.

Note: CREATE PROCEDURE is not supported because tree-sitter-sql lacks a
grammar rule for it (produces ERROR nodes).
Tracked upstream: https://github.com/DerekStride/tree-sitter-sql/issues/354
"""

from typing import Any

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.base import BaseMapping
from chunkhound.parsers.universal_engine import UniversalConcept

try:
    from tree_sitter import Node as TSNode

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    TSNode = Any  # type: ignore


class SqlMapping(BaseMapping):
    """SQL-specific tree-sitter mapping for universal concepts."""

    def __init__(self) -> None:
        """Initialize SQL mapping."""
        super().__init__(Language.SQL)

    # BaseMapping required methods
    def get_function_query(self) -> str:
        """Get tree-sitter query pattern for function definitions."""
        return """
        (create_function
            name: (object_reference
                (identifier) @func_name
            )
        ) @func_def
        """

    def get_class_query(self) -> str:
        """Get tree-sitter query pattern for class definitions (not applicable to SQL)."""
        return ""

    def get_comment_query(self) -> str:
        """Get tree-sitter query pattern for comments.

        Delegates to get_query_for_concept to avoid capture name inconsistency.
        """
        return self.get_query_for_concept(UniversalConcept.COMMENT) or ""

    def extract_function_name(self, node: TSNode | None, source: str) -> str:
        """Extract function name from a function definition node."""
        if node is None:
            return self.get_fallback_name(node, "function")

        name = self._extract_object_name(node, source)
        if name:
            return name

        return self.get_fallback_name(node, "function")

    def extract_class_name(self, node: TSNode | None, source: str) -> str:
        """Extract class name (not applicable to SQL)."""
        return ""

    def _extract_object_name(self, node: TSNode, source: str) -> str:
        """Extract the full name from an object_reference child.

        Handles schema-qualified names like dbo.Users by returning
        the full object_reference text (e.g. 'dbo.Users' -> 'dbo.Users').
        """
        obj_ref = self.find_child_by_type(node, "object_reference")
        if obj_ref:
            return self.get_node_text(obj_ref, source).strip()
        # Some constructs put the identifier directly on the node
        ident = self.find_child_by_type(node, "identifier")
        if ident:
            return self.get_node_text(ident, source).strip()
        return ""

    # LanguageMapping protocol methods
    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        """Get tree-sitter query for universal concept in SQL."""

        if concept == UniversalConcept.DEFINITION:
            return """
            (create_table) @definition

            (create_view) @definition

            (create_function) @definition

            (create_index) @definition

            (create_trigger) @definition
            """

        elif concept == UniversalConcept.BLOCK:
            return """
            (block) @block
            """

        elif concept == UniversalConcept.COMMENT:
            return """
            (comment) @definition

            (marginalia) @definition
            """

        elif concept == UniversalConcept.IMPORT:
            # SQL has no import concept
            return None

        elif concept == UniversalConcept.STRUCTURE:
            return """
            (alter_table) @definition
            """

        return None

    def extract_name(
        self, concept: UniversalConcept, captures: dict[str, TSNode], content: bytes
    ) -> str:
        """Extract name from captures for this concept."""

        source = content.decode("utf-8")

        if concept == UniversalConcept.DEFINITION:
            node = captures.get("definition")
            if node is None:
                return "unnamed_definition"

            node_type = node.type
            name = self._extract_object_name(node, source)

            if node_type == "create_table":
                return f"table_{name}" if name else self.get_fallback_name(node, "table")
            elif node_type == "create_view":
                return f"view_{name}" if name else self.get_fallback_name(node, "view")
            elif node_type == "create_function":
                return f"function_{name}" if name else self.get_fallback_name(node, "function")
            elif node_type == "create_index":
                # Index name is a direct identifier child, not inside object_reference
                idx_name = self.find_child_by_type(node, "identifier")
                if idx_name:
                    return f"index_{self.get_node_text(idx_name, source).strip()}"
                return self.get_fallback_name(node, "index")
            elif node_type == "create_trigger":
                return f"trigger_{name}" if name else self.get_fallback_name(node, "trigger")
            else:
                return name if name else self.get_fallback_name(node, "definition")

        elif concept == UniversalConcept.BLOCK:
            node = captures.get("block")
            if node:
                line = node.start_point[0] + 1
                return f"block_line_{line}"
            return "unnamed_block"

        elif concept == UniversalConcept.COMMENT:
            node = captures.get("definition")
            if node:
                line = node.start_point[0] + 1
                return f"comment_line_{line}"
            return "unnamed_comment"

        elif concept == UniversalConcept.STRUCTURE:
            node = captures.get("definition")
            if node is None:
                return "unnamed_structure"

            if node.type == "alter_table":
                name = self._extract_object_name(node, source)
                if name:
                    return f"alter_{name}"
                return self.get_fallback_name(node, "alter")

            return "unnamed_structure"

        return "unnamed"

    def extract_content(
        self, concept: UniversalConcept, captures: dict[str, TSNode], content: bytes
    ) -> str:
        """Extract content from captures for this concept."""

        source = content.decode("utf-8")

        if concept == UniversalConcept.BLOCK and "block" in captures:
            node = captures["block"]
            return self.get_node_text(node, source)
        elif "definition" in captures:
            node = captures["definition"]
            return self.get_node_text(node, source)
        elif captures:
            node = list(captures.values())[0]
            return self.get_node_text(node, source)

        return ""

    def extract_metadata(
        self, concept: UniversalConcept, captures: dict[str, TSNode], content: bytes
    ) -> dict[str, Any]:
        """Extract SQL-specific metadata."""

        source = content.decode("utf-8")
        metadata: dict[str, Any] = {}

        if concept == UniversalConcept.DEFINITION:
            def_node = captures.get("definition")
            if def_node:
                metadata["node_type"] = def_node.type

                if def_node.type == "create_table":
                    metadata["kind"] = "table"
                    # Count columns
                    col_defs = self.find_child_by_type(def_node, "column_definitions")
                    if col_defs:
                        cols = self.find_children_by_type(col_defs, "column_definition")
                        metadata["column_count"] = len(cols)

                elif def_node.type == "create_view":
                    metadata["kind"] = "view"

                elif def_node.type == "create_function":
                    metadata["kind"] = "function"

                elif def_node.type == "create_index":
                    metadata["kind"] = "index"
                    # Extract target table from object_reference
                    obj_ref = self.find_child_by_type(def_node, "object_reference")
                    if obj_ref:
                        metadata["target_table"] = self.get_node_text(
                            obj_ref, source
                        ).strip()

                elif def_node.type == "create_trigger":
                    metadata["kind"] = "trigger"

        elif concept == UniversalConcept.BLOCK:
            if "block" in captures:
                block_node = captures["block"]
                metadata["block_type"] = "begin_end"
                # Count statements
                stmts = self.find_children_by_type(block_node, "statement")
                metadata["statement_count"] = len(stmts)

        elif concept == UniversalConcept.COMMENT:
            if "definition" in captures:
                comment_node = captures["definition"]
                comment_text = self.get_node_text(comment_node, source)

                clean_text = self.clean_comment_text(comment_text)

                comment_type = "regular"
                if comment_node.type == "marginalia":
                    comment_type = "block"
                elif clean_text:
                    upper_text = clean_text.upper()
                    if any(
                        prefix in upper_text
                        for prefix in ["TODO:", "FIXME:", "HACK:", "NOTE:", "WARNING:"]
                    ):
                        comment_type = "annotation"

                metadata["comment_type"] = comment_type

        elif concept == UniversalConcept.STRUCTURE:
            node = captures.get("definition")
            if node and node.type == "alter_table":
                metadata["kind"] = "alter_table"

        return metadata
