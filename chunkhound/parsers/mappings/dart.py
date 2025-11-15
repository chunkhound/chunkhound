"""Dart language mapping for unified parser architecture.

This module provides Dart-specific tree-sitter queries and extraction logic
for mapping Dart AST nodes to semantic chunks.
"""

from typing import TYPE_CHECKING, Any

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.base import BaseMapping

if TYPE_CHECKING:
    from tree_sitter import Node as TSNode

try:
    from tree_sitter import Node as TSNode

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    TSNode = Any  # type: ignore


class DartMapping(BaseMapping):
    """Dart-specific tree-sitter mapping implementation.

    Handles Dart's unique language features including:
    - Function definitions with async support
    - Class definitions with inheritance and mixins
    - Method definitions within classes
    - Dart-specific constructs (async/await, extensions, mixins)
    - Comments and documentation
    - Import statements
    """

    def __init__(self) -> None:
        """Initialize Dart mapping."""
        super().__init__(Language.DART)

    def get_function_query(self) -> str:
        """Get tree-sitter query pattern for Dart function definitions.

        Handles:
        - Function signatures with function bodies
        - Standalone function signatures

        Returns:
            Tree-sitter query string for finding Dart function definitions
        """
        return """
            (function_signature
                name: (identifier) @function_name
            ) @function_def
        """

    def get_class_query(self) -> str:
        """Get tree-sitter query pattern for Dart class definitions.

        Handles:
        - Regular class definitions
        - Classes with inheritance
        - Classes with mixins
        - Abstract classes
        - Enum declarations
        - Mixin declarations

        Returns:
            Tree-sitter query string for finding Dart class definitions
        """
        return """
            (class_definition
                name: (identifier) @class_name
            ) @class_def

            (enum_declaration
                name: (identifier) @enum_name
            ) @enum_def

            (mixin_declaration
                name: (identifier) @mixin_name
            ) @mixin_def
        """

    def get_method_query(self) -> str:
        """Get tree-sitter query pattern for Dart method definitions.

        Handles:
        - Method signatures within class bodies
        - Constructor methods

        Returns:
            Tree-sitter query string for finding Dart method definitions
        """
        return """
            (class_definition
                (class_body
                    (method_signature
                        (function_signature
                            name: (identifier) @method_name
                        )
                    ) @method_def
                )
            )
        """

    def get_comment_query(self) -> str:
        """Get tree-sitter query pattern for Dart comments.

        Handles:
        - Single-line comments: // comment
        - Multi-line comments: /* comment */
        - Documentation comments: /// comment and /** comment */

        Returns:
            Tree-sitter query string for finding Dart comments
        """
        return """
            (comment) @comment
        """

    def get_docstring_query(self) -> str:
        """Get tree-sitter query pattern for Dart documentation comments.

        Dart uses /// for doc comments and /** */ for block doc comments.

        Returns:
            Tree-sitter query string for finding Dart documentation comments
        """
        return """
            (comment) @doc_comment
            (#match? @doc_comment "^///|^/\\*\\*")
        """

    def get_import_query(self) -> str:
        """Get tree-sitter query pattern for Dart import statements.

        Handles:
        - Regular imports: import 'package:...'
        - Prefixed imports: import 'package:...' as prefix
        - Show/hide imports: import 'package:...' show X hide Y
        - Deferred imports: import 'package:...' deferred as prefix

        Returns:
            Tree-sitter query string for finding Dart import statements
        """
        return """
            (import_declaration) @import
        """

    def extract_function_name(self, node: "TSNode | None", source: str) -> str:
        """Extract function name from a Dart function definition node.

        Args:
            node: Tree-sitter function definition node
            source: Source code string

        Returns:
            Function name or fallback name if extraction fails
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return self.get_fallback_name(node, "function")

        # Try to find identifier child node for function name
        name_node = self.find_child_by_type(node, "identifier")
        if name_node:
            name = self.get_node_text(name_node, source).strip()
            if name:
                return name

        # Handle function signatures separately
        if node.type == "function_signature":
            name_node = self.find_child_by_type(node, "identifier")
            if name_node:
                name = self.get_node_text(name_node, source).strip()
                if name:
                    return name

        return self.get_fallback_name(node, "function")

    def extract_class_name(self, node: "TSNode | None", source: str) -> str:
        """Extract class name from a Dart class definition node.

        Args:
            node: Tree-sitter class definition node
            source: Source code string

        Returns:
            Class name or fallback name if extraction fails
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return self.get_fallback_name(node, "class")

        # Handle regular classes, enums, and mixins
        if node.type in ["class_definition", "enum_declaration", "mixin_declaration"]:
            name_node = self.find_child_by_type(node, "identifier")
            if name_node:
                name = self.get_node_text(name_node, source).strip()
                if name:
                    return name

        return self.get_fallback_name(node, "class")

    def extract_method_name(self, node: "TSNode | None", source: str) -> str:
        """Extract method name from a Dart method definition node.

        Args:
            node: Tree-sitter method definition node
            source: Source code string

        Returns:
            Method name or fallback name if extraction fails
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return self.get_fallback_name(node, "method")

        # Handle method signatures within classes
        if node.type == "method_signature":
            # Look for the function signature within the method signature
            func_sig_node = self.find_child_by_type(node, "function_signature")
            if func_sig_node:
                name_node = self.find_child_by_type(func_sig_node, "identifier")
                if name_node:
                    name = self.get_node_text(name_node, source).strip()
                    if name:
                        return name

        # Fallback to looking for identifier directly
        name_node = self.find_child_by_type(node, "identifier")
        if name_node:
            name = self.get_node_text(name_node, source).strip()
            if name:
                return name

        return self.get_fallback_name(node, "method")

    def extract_parameters(self, node: "TSNode | None", source: str) -> list[str]:
        """Extract parameter names from a Dart function/method node.

        Handles:
        - Regular parameters
        - Optional positional parameters
        - Named parameters
        - Default values

        Args:
            node: Tree-sitter function/method definition node
            source: Source code string

        Returns:
            List of parameter names
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return []

        parameters: list[str] = []

        # Find the formal parameter list
        params_node = self.find_child_by_type(node, "formal_parameter_list")
        if not params_node:
            return parameters

        # Extract parameters from the parameter list
        for child in self.walk_tree(params_node):
            if child and child.type == "formal_parameter":
                # Get the parameter name
                name_node = self.find_child_by_type(child, "identifier")
                if name_node:
                    param_name = self.get_node_text(name_node, source).strip()
                    if param_name:
                        parameters.append(param_name)

        return parameters

    def should_include_node(self, node: "TSNode | None", source: str) -> bool:
        """Determine if a node should be included as a chunk.

        Override to add Dart-specific filtering logic.

        Args:
            node: Tree-sitter node
            source: Source code string

        Returns:
            True if node should be included, False otherwise
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return False

        # Exclude synthetic nodes or nodes without meaningful content
        node_text = self.get_node_text(node, source).strip()
        if len(node_text) < 3:  # Too short to be meaningful
            return False

        return True
