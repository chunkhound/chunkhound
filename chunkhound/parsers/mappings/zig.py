"""Zig language mapping for unified parser architecture.

This module provides Zig-specific tree-sitter queries and extraction logic
for the universal concept system. It maps Zig's AST nodes to universal
semantic concepts used by the unified parser.
"""

from typing import Any

from tree_sitter import Node

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.base import BaseMapping
from chunkhound.parsers.universal_engine import UniversalConcept


class ZigMapping(BaseMapping):
    """Zig-specific tree-sitter mapping for universal concepts."""

    def __init__(self) -> None:
        """Initialize Zig mapping."""
        super().__init__(Language.ZIG)

    # BaseMapping required methods
    def get_function_query(self) -> str:
        """Get tree-sitter query pattern for function definitions."""
        return """
        (function_declaration
            name: (identifier) @func_name
        ) @func_def
        """

    def get_class_query(self) -> str:
        """Get tree-sitter query pattern for type definitions.

        Zig defines types using variable_declaration with struct/enum/union/opaque.
        """
        return """
        (variable_declaration
            (identifier) @struct_name
            "="
            [
                (struct_declaration)
                (enum_declaration)
                (union_declaration)
                (opaque_declaration)
            ]
        ) @struct_def
        """

    def get_comment_query(self) -> str:
        """Get tree-sitter query pattern for comments."""
        return """
        (comment) @comment
        """

    def extract_function_name(self, node: Node | None, source: str) -> str:
        """Extract function name from a function definition node."""
        if node is None:
            return self.get_fallback_name(node, "function")

        # Find the function name child
        name_node = self.find_child_by_type(node, "identifier")
        if name_node:
            return self.get_node_text(name_node, source).strip()

        return self.get_fallback_name(node, "function")

    def extract_class_name(self, node: Node | None, source: str) -> str:
        """Extract container name from a definition node."""
        if node is None:
            return self.get_fallback_name(node, "container")

        # Find the identifier child
        name_node = self.find_child_by_type(node, "identifier")
        if name_node:
            return self.get_node_text(name_node, source).strip()

        return self.get_fallback_name(node, "container")

    # LanguageMapping protocol methods
    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        """Get tree-sitter query for universal concept in Zig."""

        if concept == UniversalConcept.DEFINITION:
            return """
            (function_declaration
                name: (identifier) @name
            ) @definition

            (test_declaration
                [
                    (string) @name
                    (identifier) @name
                ]
            ) @definition

            (source_file
                (variable_declaration
                    (identifier) @name
                    "="
                    [
                        (struct_declaration)
                        (enum_declaration)
                        (union_declaration)
                        (opaque_declaration)
                    ]
                ) @definition
            )

            (source_file
                (variable_declaration
                    (identifier) @name
                ) @definition
            )
            """

        elif concept == UniversalConcept.BLOCK:
            return """
            (if_statement) @definition

            (while_statement) @definition

            (for_statement) @definition

            (switch_expression) @definition

            (block) @definition
            """

        elif concept == UniversalConcept.COMMENT:
            return """
            (comment) @definition
            """

        elif concept == UniversalConcept.IMPORT:
            return """
            (variable_declaration
                (identifier) @name
                (builtin_function
                    (builtin_identifier) @import_builtin
                ) @import_call
            ) @definition
            """

        # STRUCTURE not needed for Zig - only for structured text formats
        # elif concept == UniversalConcept.STRUCTURE:
        #     return """
        #     (source_file) @definition
        #     """

        return None

    def extract_name(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract name from captures for this concept."""

        # Convert bytes to string for processing
        source = content.decode("utf-8")

        if concept == UniversalConcept.DEFINITION:
            # Try to get the name from various capture groups
            if "name" in captures:
                name_node = captures["name"]
                name = self.get_node_text(name_node, source).strip()

                # For test declarations with string names, strip quotes
                if (
                    "definition" in captures
                    and captures["definition"].type == "test_declaration"
                ):
                    if name_node.type == "string":
                        name = name.strip('"')

                return name

            return "unnamed_definition"

        elif concept == UniversalConcept.BLOCK:
            # Use location-based naming for blocks
            if "definition" in captures:
                node = captures["definition"]
                line = node.start_point[0] + 1
                return f"block_line_{line}"

            return "unnamed_block"

        elif concept == UniversalConcept.COMMENT:
            # Use location-based naming for comments
            if "definition" in captures:
                node = captures["definition"]
                line = node.start_point[0] + 1
                return f"comment_line_{line}"

            return "unnamed_comment"

        elif concept == UniversalConcept.IMPORT:
            if "name" in captures:
                name_node = captures["name"]
                name = self.get_node_text(name_node, source).strip()
                return f"import_{name}"

            if "import_builtin" in captures:
                builtin_node = captures["import_builtin"]
                builtin_name = self.get_node_text(builtin_node, source).strip()
                return f"import_{builtin_name}"

            return "unnamed_import"

        # STRUCTURE not needed for Zig - only for structured text formats
        # elif concept == UniversalConcept.STRUCTURE:
        #     return "file_structure"

        return "unnamed"

    def extract_content(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract content from captures for this concept."""

        # Convert bytes to string for processing
        source = content.decode("utf-8")

        if "definition" in captures:
            node = captures["definition"]
            return self.get_node_text(node, source)
        elif captures:
            # Use the first available capture
            node = list(captures.values())[0]
            return self.get_node_text(node, source)

        return ""

    def extract_metadata(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> dict[str, Any]:
        """Extract Zig-specific metadata."""

        source = content.decode("utf-8")
        metadata = {}

        if concept == UniversalConcept.DEFINITION:
            # Extract definition-specific metadata
            def_node = captures.get("definition")
            if def_node:
                metadata["node_type"] = def_node.type

                # For functions, extract parameters and return type
                if def_node.type == "function_declaration":
                    metadata["kind"] = "function"

                    # Check for pub visibility
                    if self._is_public(def_node, source):
                        metadata["visibility"] = "pub"

                # For test declarations
                elif def_node.type == "test_declaration":
                    metadata["kind"] = "test"

                # For variable declarations, check if it's a type definition or regular variable
                elif def_node.type == "variable_declaration":
                    # Check if it's a container type definition
                    has_container = False
                    for child in self.walk_tree(def_node):
                        if child and child.type in [
                            "struct_declaration",
                            "enum_declaration",
                            "union_declaration",
                            "opaque_declaration",
                        ]:
                            metadata["kind"] = child.type.replace("_declaration", "")
                            has_container = True
                            break

                    if not has_container:
                        metadata["kind"] = "variable"

                    # Check if const or var
                    if self._is_const(def_node, source):
                        metadata["mutability"] = "const"
                    else:
                        metadata["mutability"] = "var"

                    # Check for pub visibility
                    if self._is_public(def_node, source):
                        metadata["visibility"] = "pub"

        elif concept == UniversalConcept.IMPORT:
            if "import_call" in captures:
                call_node = captures["import_call"]
                # Extract import path from string literal in call
                for child in self.walk_tree(call_node):
                    if child and child.type == "string_literal":
                        import_path = (
                            self.get_node_text(child, source).strip().strip('"')
                        )
                        metadata["import_path"] = import_path
                        break

            if "import_builtin" in captures:
                builtin_node = captures["import_builtin"]
                builtin_name = self.get_node_text(builtin_node, source).strip()
                metadata["builtin"] = builtin_name

        elif concept == UniversalConcept.COMMENT:
            if "definition" in captures:
                comment_node = captures["definition"]
                comment_text = self.get_node_text(comment_node, source)

                # Determine comment type
                if comment_text.startswith("///"):
                    metadata["comment_type"] = "doc"
                    metadata["is_doc_comment"] = True
                elif comment_text.startswith("//!"):
                    metadata["comment_type"] = "module_doc"
                    metadata["is_doc_comment"] = True
                elif comment_text.startswith("//"):
                    metadata["comment_type"] = "line"

        return metadata

    def _is_public(self, node: Node, source: str) -> bool:
        """Check if a declaration has pub visibility."""
        # For function_declaration and variable_declaration, pub is a direct child
        for child in node.children:
            if (
                child.type == "pub"
                or self.get_node_text(child, source).strip() == "pub"
            ):
                return True
        return False

    def _is_const(self, node: Node, source: str) -> bool:
        """Check if a variable declaration is const."""
        # Check first 3 children (pub, const/var, identifier) without walking full tree
        # Don't walk entire tree to avoid picking up const from other merged variables
        max_children_to_check = min(3, len(node.children)) if node.children else 0
        for i in range(max_children_to_check):
            child = node.children[i]
            if child and (
                child.type == "const"
                or self.get_node_text(child, source).strip() == "const"
            ):
                return True
        return False
