"""TypeScript language mapping for unified parser architecture.

This module provides TypeScript-specific tree-sitter queries and extraction logic
for the unified parser system. It handles TypeScript's unique features including
type annotations, generics, interfaces, enums, namespaces, and decorators.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from chunkhound.core.types.common import ChunkType, Language
from chunkhound.parsers.mappings.base import BaseMapping
from chunkhound.parsers.mappings._shared.js_family_extraction import (
    JSFamilyExtraction,
)
from chunkhound.parsers.mappings._shared.js_query_patterns import (
    TOP_LEVEL_LEXICAL_CONFIG,
    COMMONJS_MODULE_EXPORTS,
    COMMONJS_NESTED_EXPORTS,
    COMMONJS_EXPORTS_SHORTHAND,
)
from chunkhound.parsers.universal_engine import UniversalConcept

if TYPE_CHECKING:
    from tree_sitter import Node as TSNode

try:
    from tree_sitter import Node as TSNode

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    # TSNode is already defined in TYPE_CHECKING block


class TypeScriptMapping(BaseMapping, JSFamilyExtraction):
    """TypeScript language mapping for tree-sitter parsing.

    This mapping handles TypeScript-specific AST patterns including:
    - Function declarations and arrow functions
    - Class declarations with access modifiers
    - Interface declarations
    - Type aliases and generics
    - Enum declarations
    - Namespace declarations
    - Decorators
    - TSDoc comments
    """

    def __init__(self) -> None:
        """Initialize TypeScript mapping."""
        super().__init__(Language.TYPESCRIPT)

    def get_function_query(self) -> str:
        """Get tree-sitter query pattern for TypeScript function definitions.

        Returns:
            Tree-sitter query string for finding function definitions
        """
        return """
            (function_declaration
                name: (identifier) @function.name
            ) @function.def

            (variable_declarator
                name: (identifier) @arrow_function.name
                value: (arrow_function) @arrow_function.def
            )

            (function_expression
                name: (identifier) @function_expr.name
            ) @function_expr.def

            (method_definition
                name: (_) @method.name
            ) @method.def

            (method_signature
                name: (_) @method_sig.name
            ) @method_sig.def
        """

    def get_class_query(self) -> str:
        """Get tree-sitter query pattern for TypeScript class definitions.

        Returns:
            Tree-sitter query string for finding class definitions
        """
        return """
            (class_declaration
                name: (type_identifier) @class.name
            ) @class.def
        """

    def get_comment_query(self) -> str:
        """Get tree-sitter query pattern for TypeScript comments.

        Returns:
            Tree-sitter query string for finding comments
        """
        return """
            (comment) @comment
        """

    # Universal Concept integration -------------------------------------------------
    def get_query_for_concept(self, concept: "UniversalConcept") -> str | None:  # type: ignore[override]
        """Provide a richer DEFINITION query including top-level config patterns.

        - Keep standard function/class definitions (duplicated here for completeness)
        - Add export statements and top-level declarations/assignments, so TS config
          modules that export object literals are chunked.
        """
        if concept == UniversalConcept.DEFINITION:
            return ("\n".join([
                """
                ; Standard definitions
                (function_declaration
                    name: (identifier) @name
                ) @definition

                (class_declaration
                    name: (type_identifier) @name
                ) @definition

                ; Top-level export (default or named)
                (export_statement) @definition
                """,
                TOP_LEVEL_LEXICAL_CONFIG,
                # Top-level function/arrow declarators
                """
                (program
                    (lexical_declaration
                        (variable_declarator
                            name: (identifier) @name
                            value: (function_expression)
                        ) @definition
                    )
                )
                (program
                    (lexical_declaration
                        (variable_declarator
                            name: (identifier) @name
                            value: (arrow_function)
                        ) @definition
                    )
                )
                """,
                COMMONJS_MODULE_EXPORTS,
                COMMONJS_NESTED_EXPORTS,
                COMMONJS_EXPORTS_SHORTHAND,
            ]))
        elif concept == UniversalConcept.COMMENT:
            return """
            (comment) @definition
            """
        return None

    # extract_name / extract_metadata / extract_content are inherited
    # from JSFamilyExtraction (de-duplicated across JS/TS/JSX)

    def get_interface_query(self) -> str:
        """Get tree-sitter query pattern for TypeScript interface definitions.

        Returns:
            Tree-sitter query string for finding interface definitions
        """
        return """
            (interface_declaration
                name: (type_identifier) @interface.name
            ) @interface.def
        """

    def get_enum_query(self) -> str:
        """Get tree-sitter query pattern for TypeScript enum definitions.

        Returns:
            Tree-sitter query string for finding enum definitions
        """
        return """
            (enum_declaration
                name: (identifier) @enum.name
            ) @enum.def
        """

    def get_type_alias_query(self) -> str:
        """Get tree-sitter query pattern for TypeScript type alias definitions.

        Returns:
            Tree-sitter query string for finding type alias definitions
        """
        return """
            (type_alias_declaration
                name: (type_identifier) @type_alias.name
            ) @type_alias.def
        """

    def get_namespace_query(self) -> str:
        """Get tree-sitter query pattern for TypeScript namespace definitions.

        Returns:
            Tree-sitter query string for finding namespace definitions
        """
        return """
            (module_declaration
                name: (identifier) @namespace.name
            ) @namespace.def

            (namespace_declaration
                name: (identifier) @namespace.name
            ) @namespace.def
        """

    def get_decorator_query(self) -> str:
        """Get tree-sitter query pattern for TypeScript decorators.

        Returns:
            Tree-sitter query string for finding decorators
        """
        return """
            (decorator
                (identifier) @decorator.name
            ) @decorator.def

            (decorator
                (call_expression
                    function: (identifier) @decorator.name
                )
            ) @decorator.def
        """

    def get_docstring_query(self) -> str:
        """Get tree-sitter query pattern for TSDoc comments.

        Returns:
            Tree-sitter query string for finding TSDoc-style comments
        """
        return """
            (comment) @tsdoc
        """

    def extract_function_name(self, node: "TSNode | None", source: str) -> str:
        """Extract function name from a TypeScript function definition node.

        Args:
            node: Tree-sitter function definition node
            source: Source code string

        Returns:
            Function name or fallback name if extraction fails
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return self.get_fallback_name(node, "function")

        try:
            # Handle different function types
            if node.type == "function_declaration":
                name_node = self.find_child_by_type(node, "identifier")
                if name_node:
                    return self.get_node_text(name_node, source)
            elif node.type == "variable_declarator":
                # Arrow function assigned to variable
                name_node = self.find_child_by_type(node, "identifier")
                if name_node:
                    return self.get_node_text(name_node, source)
            elif node.type == "method_definition":
                # Find the method name (can be identifier, string, computed_property_name)
                for i in range(node.child_count):
                    child = node.child(i)
                    if child and child.type in [
                        "identifier",
                        "string",
                        "computed_property_name",
                    ]:
                        return self.get_node_text(child, source)

            return self.get_fallback_name(node, "function")
        except Exception as e:
            logger.error(f"Failed to extract TypeScript function name: {e}")
            return self.get_fallback_name(node, "function")

    def extract_class_name(self, node: "TSNode | None", source: str) -> str:
        """Extract class name from a TypeScript class definition node.

        Args:
            node: Tree-sitter class definition node
            source: Source code string

        Returns:
            Class name or fallback name if extraction fails
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return self.get_fallback_name(node, "class")

        try:
            name_node = self.find_child_by_type(node, "type_identifier")
            if name_node:
                return self.get_node_text(name_node, source)

            return self.get_fallback_name(node, "class")
        except Exception as e:
            logger.error(f"Failed to extract TypeScript class name: {e}")
            return self.get_fallback_name(node, "class")

    def extract_interface_name(self, node: "TSNode | None", source: str) -> str:
        """Extract interface name from a TypeScript interface definition node.

        Args:
            node: Tree-sitter interface definition node
            source: Source code string

        Returns:
            Interface name or fallback name if extraction fails
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return self.get_fallback_name(node, "interface")

        try:
            name_node = self.find_child_by_type(node, "type_identifier")
            if name_node:
                return self.get_node_text(name_node, source)

            return self.get_fallback_name(node, "interface")
        except Exception as e:
            logger.error(f"Failed to extract TypeScript interface name: {e}")
            return self.get_fallback_name(node, "interface")

    def extract_enum_name(self, node: "TSNode | None", source: str) -> str:
        """Extract enum name from a TypeScript enum definition node.

        Args:
            node: Tree-sitter enum definition node
            source: Source code string

        Returns:
            Enum name or fallback name if extraction fails
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return self.get_fallback_name(node, "enum")

        try:
            name_node = self.find_child_by_type(node, "identifier")
            if name_node:
                return self.get_node_text(name_node, source)

            return self.get_fallback_name(node, "enum")
        except Exception as e:
            logger.error(f"Failed to extract TypeScript enum name: {e}")
            return self.get_fallback_name(node, "enum")

    def extract_type_alias_name(self, node: "TSNode | None", source: str) -> str:
        """Extract type alias name from a TypeScript type alias definition node.

        Args:
            node: Tree-sitter type alias definition node
            source: Source code string

        Returns:
            Type alias name or fallback name if extraction fails
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return self.get_fallback_name(node, "type")

        try:
            name_node = self.find_child_by_type(node, "type_identifier")
            if name_node:
                return self.get_node_text(name_node, source)

            return self.get_fallback_name(node, "type")
        except Exception as e:
            logger.error(f"Failed to extract TypeScript type alias name: {e}")
            return self.get_fallback_name(node, "type")

    def extract_namespace_name(self, node: "TSNode | None", source: str) -> str:
        """Extract namespace name from a TypeScript namespace definition node.

        Args:
            node: Tree-sitter namespace definition node
            source: Source code string

        Returns:
            Namespace name or fallback name if extraction fails
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return self.get_fallback_name(node, "namespace")

        try:
            name_node = self.find_child_by_type(node, "identifier")
            if name_node:
                return self.get_node_text(name_node, source)

            return self.get_fallback_name(node, "namespace")
        except Exception as e:
            logger.error(f"Failed to extract TypeScript namespace name: {e}")
            return self.get_fallback_name(node, "namespace")

    def extract_parameters(self, node: "TSNode | None", source: str) -> list[str]:
        """Extract parameter names and types from a TypeScript function/method node.

        Args:
            node: Tree-sitter function/method definition node
            source: Source code string

        Returns:
            List of parameter strings with types
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return []

        parameters: list[str] = []

        try:
            # Find the parameters node
            params_node = None
            for child_type in ["formal_parameters", "parameters"]:
                params_node = self.find_child_by_type(node, child_type)
                if params_node:
                    break

            if not params_node:
                return parameters

            # Extract each parameter with its type annotation
            for i in range(params_node.child_count):
                child = params_node.child(i)
                if child and child.type in [
                    "required_parameter",
                    "optional_parameter",
                    "rest_parameter",
                ]:
                    param_text = self.get_node_text(child, source).strip()
                    if param_text and param_text not in [",", "(", ")"]:
                        parameters.append(param_text)

        except Exception as e:
            logger.error(f"Failed to extract TypeScript parameters: {e}")

        return parameters

    def extract_return_type(self, node: "TSNode | None", source: str) -> str | None:
        """Extract return type annotation from a TypeScript function.

        Args:
            node: Tree-sitter function definition node
            source: Source code string

        Returns:
            Return type string or None if not found
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return None

        try:
            # Look for type_annotation nodes
            type_nodes = self.find_children_by_type(node, "type_annotation")
            for type_node in type_nodes:
                # Get the type annotation text, excluding the colon
                type_text = self.get_node_text(type_node, source).strip()
                if type_text.startswith(":"):
                    return type_text[1:].strip()
                return type_text

        except Exception as e:
            logger.error(f"Failed to extract TypeScript return type: {e}")

        return None

    def extract_type_parameters(self, node: "TSNode | None", source: str) -> str | None:
        """Extract generic type parameters from a TypeScript declaration.

        Args:
            node: Tree-sitter declaration node
            source: Source code string

        Returns:
            Type parameters string or None if not found
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return None

        try:
            type_params_node = self.find_child_by_type(node, "type_parameters")
            if type_params_node:
                return self.get_node_text(type_params_node, source).strip()

        except Exception as e:
            logger.error(f"Failed to extract TypeScript type parameters: {e}")

        return None

    def extract_access_modifiers(self, node: "TSNode | None", source: str) -> list[str]:
        """Extract access modifiers from a TypeScript class member.

        Args:
            node: Tree-sitter class member node
            source: Source code string

        Returns:
            List of access modifiers (public, private, protected, static, readonly, etc.)
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return []

        modifiers: list[str] = []

        try:
            # Look for accessibility_modifier and other modifier nodes
            for i in range(node.child_count):
                child = node.child(i)
                if child and child.type in [
                    "accessibility_modifier",  # public, private, protected
                    "override_modifier",  # override
                    "static",  # static
                    "readonly",  # readonly
                    "abstract",  # abstract
                    "async",  # async
                ]:
                    modifier_text = self.get_node_text(child, source).strip()
                    if modifier_text:
                        modifiers.append(modifier_text)

        except Exception as e:
            logger.error(f"Failed to extract TypeScript access modifiers: {e}")

        return modifiers

    def extract_decorators(self, node: "TSNode | None", source: str) -> list[str]:
        """Extract decorators from a TypeScript declaration.

        Args:
            node: Tree-sitter declaration node
            source: Source code string

        Returns:
            List of decorator names
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return []

        decorators: list[str] = []

        try:
            decorator_nodes = self.find_children_by_type(node, "decorator")
            for decorator_node in decorator_nodes:
                decorator_text = self.get_node_text(decorator_node, source).strip()
                if decorator_text:
                    decorators.append(decorator_text)

        except Exception as e:
            logger.error(f"Failed to extract TypeScript decorators: {e}")

        return decorators

    def is_tsdoc_comment(self, node: "TSNode | None", source: str) -> bool:
        """Check if a comment node is a TSDoc comment.

        Args:
            node: Tree-sitter comment node
            source: Source code string

        Returns:
            True if the comment is TSDoc-style (starts with /**)
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return False

        try:
            comment_text = self.get_node_text(node, source).strip()
            return comment_text.startswith("/**") and comment_text.endswith("*/")
        except Exception as e:
            logger.error(f"Failed to check TSDoc comment: {e}")
            return False

    def clean_comment_text(self, text: str) -> str:
        """Clean TypeScript comment text by removing comment markers and TSDoc tags.

        Args:
            text: Raw comment text

        Returns:
            Cleaned comment text
        """
        cleaned = super().clean_comment_text(text)

        # Additional TypeScript-specific cleaning for TSDoc
        lines = cleaned.split("\n")
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            # Remove TSDoc comment markers
            if line.startswith("*"):
                line = line[1:].strip()
            # Remove common TSDoc tags at start of line for readability
            for tag in [
                "@param",
                "@returns",
                "@return",
                "@throws",
                "@example",
                "@see",
                "@since",
            ]:
                if line.startswith(tag):
                    line = line.replace(tag, "", 1).strip()
                    break
            if line:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def should_include_node(self, node: "TSNode | None", source: str) -> bool:
        """Determine if a TypeScript node should be included as a chunk.

        Args:
            node: Tree-sitter node
            source: Source code string

        Returns:
            True if node should be included, False otherwise
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return False

        try:
            # Skip very small nodes
            node_text = self.get_node_text(node, source)
            if len(node_text.strip()) < 10:
                return False

            # Skip nodes that are just punctuation or keywords
            if node_text.strip() in ["{", "}", "(", ")", ";", ",", "export", "import"]:
                return False

            # For functions, check if they're likely React components starting with uppercase
            if (
                node.type in ["function_declaration", "variable_declarator"]
                and node_text
            ):
                # Extract function name to check if it's a component
                if node.type == "function_declaration":
                    name_node = self.find_child_by_type(node, "identifier")
                elif node.type == "variable_declarator":
                    name_node = self.find_child_by_type(node, "identifier")
                else:
                    name_node = None

                if name_node:
                    func_name = self.get_node_text(name_node, source).strip()
                    # Include React components (start with uppercase)
                    if func_name and func_name[0].isupper():
                        return True

            return True

        except Exception as e:
            logger.error(f"Failed to evaluate TypeScript node inclusion: {e}")
            return False

    def create_enhanced_chunk(
        self,
        node: "TSNode | None",
        source: str,
        file_path: Path,
        chunk_type: ChunkType,
        name: str,
        **extra_fields: Any,
    ) -> dict[str, Any]:
        """Create an enhanced chunk dictionary with TypeScript-specific metadata.

        Args:
            node: Tree-sitter node
            source: Source code string
            file_path: Path to source file
            chunk_type: Type of chunk
            name: Chunk name/symbol
            **extra_fields: Additional fields to include

        Returns:
            Enhanced chunk dictionary with TypeScript metadata
        """
        # Start with base chunk
        display_name = name

        # Add TypeScript-specific enhancements
        if node and TREE_SITTER_AVAILABLE:
            try:
                # Add type parameters for generics
                type_params = self.extract_type_parameters(node, source)
                if type_params:
                    display_name = f"{name}{type_params}"
                    extra_fields["type_parameters"] = type_params

                # Add parameters for functions/methods
                if chunk_type in [ChunkType.FUNCTION, ChunkType.METHOD]:
                    parameters = self.extract_parameters(node, source)
                    param_str = ", ".join(parameters) if parameters else ""
                    display_name = f"{name}({param_str})"

                    # Add return type
                    return_type = self.extract_return_type(node, source)
                    if return_type:
                        display_name += f": {return_type}"
                        extra_fields["return_type"] = return_type

                    if parameters:
                        extra_fields["parameters"] = parameters

                # Add access modifiers for class members
                if chunk_type in [ChunkType.METHOD, ChunkType.PROPERTY]:
                    modifiers = self.extract_access_modifiers(node, source)
                    if modifiers:
                        extra_fields["access_modifiers"] = modifiers

                # Add decorators
                decorators = self.extract_decorators(node, source)
                if decorators:
                    extra_fields["decorators"] = decorators

            except Exception as e:
                logger.error(f"Failed to enhance TypeScript chunk: {e}")

        # Create base chunk with enhanced display name
        chunk = self.create_chunk_dict(
            node=node,
            source=source,
            file_path=file_path,
            chunk_type=chunk_type,
            name=name,
            display_name=display_name,
            **extra_fields,
        )

        return chunk
