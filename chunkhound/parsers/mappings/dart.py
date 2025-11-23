"""Dart language mapping for unified parser architecture.

This module provides Dart-specific tree-sitter queries and extraction logic
for mapping Dart AST nodes to semantic chunks.
"""

from typing import TYPE_CHECKING, Any

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.base import BaseMapping

if TYPE_CHECKING:
    from chunkhound.parsers.universal_engine import UniversalConcept
try:
    from tree_sitter import Node as TSNode

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    TSNode = Any  # type: ignore

# Import UniversalConcept at runtime
from chunkhound.parsers.universal_engine import UniversalConcept


class DartMapping(BaseMapping):
    """Dart-specific tree-sitter mapping implementation.

    Handles Dart's unique language features including:
    - Function definitions with async support
    - Class definitions with inheritance and mixins
    - Method definitions within classes
    - Dart-specific constructs (async/await, extensions, mixins)
    - Comments and documentation
    - Import statements

    Implements both BaseMapping and LanguageMapping protocols for universal
    concept extraction and rich metadata generation.
    """

    def __init__(self) -> None:
        """Initialize Dart mapping."""
        super().__init__(Language.DART)

    def get_function_query(self) -> str:
        """Get tree-sitter query pattern for Dart function definitions.

        Delegates to UniversalConcept.DEFINITION query to avoid duplication.
        """
        return self.get_query_for_concept(UniversalConcept.DEFINITION) or ""

    def get_class_query(self) -> str:
        """Get tree-sitter query pattern for Dart class definitions.

        Delegates to UniversalConcept.DEFINITION query to avoid duplication.
        """
        return self.get_query_for_concept(UniversalConcept.DEFINITION) or ""

    def get_method_query(self) -> str:
        """Get tree-sitter query pattern for Dart method definitions.

        Delegates to UniversalConcept.DEFINITION query to avoid duplication.
        """
        return self.get_query_for_concept(UniversalConcept.DEFINITION) or ""

    def get_comment_query(self) -> str:
        """Get tree-sitter query pattern for Dart comments.

        Delegates to UniversalConcept.COMMENT query to avoid duplication.
        """
        return self.get_query_for_concept(UniversalConcept.COMMENT) or ""

    def get_docstring_query(self) -> str:
        """Get tree-sitter query pattern for Dart documentation comments.

        Delegates to UniversalConcept.COMMENT query to avoid duplication.
        """
        return self.get_query_for_concept(UniversalConcept.COMMENT) or ""

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
            (library_import) @import
        """

    def extract_function_name(self, node: Any, source: str) -> str:
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

    def extract_class_name(self, node: Any, source: str) -> str:
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

    def extract_method_name(self, node: Any, source: str) -> str:
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

    def extract_parameters(self, node: Any, source: str) -> list[str]:
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

    def should_include_node(self, node: Any, source: str) -> bool:
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

    # UniversalConcept interface methods

    def get_query_for_concept(self, concept: "UniversalConcept") -> str | None:
        """Get tree-sitter query for universal concept in Dart."""

        if concept == UniversalConcept.DEFINITION:
            return """
            (function_signature
                name: (identifier) @name
            ) @definition

            (class_definition
                name: (identifier) @name
            ) @definition

            (enum_declaration
                name: (identifier) @name
            ) @definition

            (mixin_declaration
                name: (identifier) @name
            ) @definition

            (extension_declaration
                name: (identifier) @name
            ) @definition
            """

        elif concept == UniversalConcept.COMMENT:
            return """
            (comment) @definition
            """

        elif concept == UniversalConcept.IMPORT:
            return """
            (library_import) @definition
            """

        elif concept == UniversalConcept.BLOCK:
            return """
            (block) @definition

            (if_statement
                consequence: (block) @block
            ) @definition

            (if_statement
                alternative: (block) @block
            ) @definition

            (for_statement
                body: (block) @block
            ) @definition

            (while_statement
                body: (block) @block
            ) @definition

            (do_statement
                body: (block) @block
            ) @definition

            (try_statement
                (block) @block
            ) @definition

            (finally_clause
                (block) @block
            ) @definition
            """

        return None

    def extract_name(
        self, concept: "UniversalConcept", captures: dict[str, Any], content: bytes
    ) -> str:
        """Extract name from captures for this concept."""
        source = content.decode("utf-8")

        if concept == UniversalConcept.DEFINITION:
            if "name" in captures:
                name_node = captures["name"]
                name = self.get_node_text(name_node, source).strip()
                if name:
                    return name
            return "unnamed_definition"

        elif concept == UniversalConcept.COMMENT:
            if "definition" in captures:
                node = captures["definition"]
                line = node.start_point[0] + 1
                return f"comment_line_{line}"
            return "unnamed_comment"

        elif concept == UniversalConcept.IMPORT:
            if "definition" in captures:
                node = captures["definition"]
                uri_node = self.find_child_by_type(node, "configurable_uri")
                if uri_node:
                    uri_text = self.get_node_text(uri_node, source).strip().strip("'\"")
                    return f"import {uri_text}"
            return "unnamed_import"

        elif concept == UniversalConcept.BLOCK:
            # Use location-based naming for blocks
            if "block" in captures:
                node = captures["block"]
                line = node.start_point[0] + 1
                return f"block_line_{line}"
            elif "definition" in captures:
                node = captures["definition"]
                line = node.start_point[0] + 1
                return f"block_line_{line}"

            return "unnamed_block"

        return "unnamed"

    def extract_content(
        self, concept: "UniversalConcept", captures: dict[str, Any], content: bytes
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
        self, concept: "UniversalConcept", captures: dict[str, Any], content: bytes
    ) -> dict[str, Any]:
        """Extract Dart-specific metadata."""
        source = content.decode("utf-8")
        metadata: dict[str, Any] = {}

        if concept == UniversalConcept.DEFINITION:
            def_node = captures.get("definition")
            if def_node:
                metadata["node_type"] = def_node.type

                # Determine kind based on node type
                if def_node.type == "function_signature":
                    metadata["kind"] = "function"
                    # Check for async functions
                    if self._is_async_function(def_node, source):
                        metadata["async"] = True

                elif def_node.type == "class_definition":
                    metadata["kind"] = "class"
                    # Extract inheritance and mixins
                    inheritance = self._extract_inheritance(def_node, source)
                    if inheritance:
                        metadata["inherits"] = inheritance

                elif def_node.type == "enum_declaration":
                    metadata["kind"] = "enum"

                elif def_node.type == "mixin_declaration":
                    metadata["kind"] = "mixin"

                elif def_node.type == "extension_declaration":
                    metadata["kind"] = "extension"
                    extended_type = self._extract_extended_type(def_node, source)
                    if extended_type:
                        metadata["extends"] = extended_type

        elif concept == UniversalConcept.COMMENT:
            if "definition" in captures:
                comment_node = captures["definition"]
                comment_text = self.get_node_text(comment_node, source)

                if comment_text.startswith("///"):
                    metadata["comment_type"] = "doc"
                    metadata["is_doc_comment"] = True
                elif comment_text.startswith("//"):
                    metadata["comment_type"] = "line"
                elif comment_text.startswith("/*") and comment_text.endswith("*/"):
                    metadata["comment_type"] = "block"

        elif concept == UniversalConcept.IMPORT:
            if "definition" in captures:
                import_node = captures["definition"]
                import_text = self.get_node_text(import_node, source).strip()

                if import_text.startswith("import"):
                    module_name = import_text[6:].strip()
                    metadata["module"] = module_name

        elif concept == UniversalConcept.BLOCK:
            block_node = captures.get("block") or captures.get("definition")
            if block_node:
                metadata["node_type"] = block_node.type

                # Determine block context based on parent
                parent = block_node.parent
                if parent:
                    if parent.type == "if_statement":
                        # Check if this is consequence or alternative
                        for i, child in enumerate(parent.children):
                            if child == block_node:
                                if (
                                    i < len(parent.children) - 1
                                    and parent.children[i + 1].type == "else"
                                ):
                                    metadata["block_context"] = "if_consequence"
                                elif parent.children[0] == block_node:
                                    metadata["block_context"] = "if_consequence"
                                else:
                                    metadata["block_context"] = "if_alternative"
                                break
                    elif parent.type == "for_statement":
                        metadata["block_context"] = "for_body"
                    elif parent.type == "while_statement":
                        metadata["block_context"] = "while_body"
                    elif parent.type == "do_statement":
                        metadata["block_context"] = "do_body"
                    elif parent.type == "try_statement":
                        metadata["block_context"] = "try_block"
                    elif parent.type == "finally_clause":
                        metadata["block_context"] = "finally_block"
                    else:
                        metadata["block_context"] = "standalone"

        return metadata

    # Helper methods for metadata extraction
    def _is_async_function(self, node: Any, source: str) -> bool:
        """Check if function is async."""
        # Look for 'async' keyword in the function signature
        for child in node.children:
            if child.type == "async":
                return True
        return False

    def _extract_inheritance(self, node: Any, source: str) -> list[str]:
        """Extract inheritance and mixin information from class definition."""
        inheritance = []

        # Look for extends clause
        extends_clause = self.find_child_by_type(node, "extends_clause")
        if extends_clause:
            type_node = self.find_child_by_type(extends_clause, "type_identifier")
            if type_node:
                inheritance.append(self.get_node_text(type_node, source))

        # Look for with clause (mixins)
        with_clause = self.find_child_by_type(node, "with_clause")
        if with_clause:
            for child in with_clause.children:
                if child.type == "type_identifier":
                    inheritance.append(self.get_node_text(child, source))

        return inheritance

    def _extract_extended_type(self, node: Any, source: str) -> str:
        """Extract the type being extended from extension declaration."""
        type_node = self.find_child_by_type(node, "type_identifier")
        if type_node:
            return self.get_node_text(type_node, source)
        return ""
