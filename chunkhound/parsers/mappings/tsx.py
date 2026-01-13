"""TSX language mapping for unified parser architecture.

This module provides TSX-specific tree-sitter queries and extraction logic
extending TypeScript functionality for React-specific patterns with type safety.
Handles JSX elements, React components with types, hooks with generics, and
TSX expressions with type annotations.
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from chunkhound.core.types.common import ChunkType, Language
from chunkhound.parsers.mappings.base import BaseMapping
from chunkhound.parsers.mappings.typescript import TypeScriptMapping

if TYPE_CHECKING:
    from tree_sitter import Node as TSNode

try:
    from tree_sitter import Node as TSNode

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    TSNode = None


class TSXMapping(TypeScriptMapping):
    """TSX language mapping extending TypeScript mapping.

    Provides TSX-specific queries and extraction methods for:
    - React functional and class components with TypeScript types
    - JSX elements and fragments with type safety
    - JSX expressions and attributes with type annotations
    - React hooks with generics
    - Component props interfaces
    - TSX comments and documentation
    """

    def __init__(self):
        """Initialize TSX mapping."""
        # Initialize with TSX language instead of TypeScript
        BaseMapping.__init__(self, Language.TSX)

    def get_function_query(self) -> str:
        """Get tree-sitter query for TSX function definitions including typed React components.

        Extends TypeScript function query with React component patterns.

        Returns:
            Tree-sitter query string for finding function definitions and React components
        """
        # Get base TypeScript function query
        base_query = super().get_function_query()

        # Add TSX-specific patterns for typed React components
        tsx_specific_query = """
        ; React functional components (JSX return)
        (function_declaration
            name: (identifier) @component.name
            body: (statement_block
                (return_statement
                    (jsx_element) @jsx_return
                )
            )
        ) @component.definition

        (variable_declarator
            name: (identifier) @component.name
            value: (arrow_function
                body: (jsx_element) @jsx_return
            )
        ) @component.definition

        (variable_declarator
            name: (identifier) @component.name  
            value: (arrow_function
                body: (statement_block
                    (return_statement
                        (jsx_element) @jsx_return
                    )
                )
            )
        ) @component.definition

        ; React.FC and React.FunctionComponent typed components
        (variable_declarator
            name: (identifier) @fc_component.name
            value: (arrow_function
                body: (jsx_element) @jsx_return
            )
        ) @fc_component.definition

        """

        return base_query + tsx_specific_query

    def get_jsx_element_query(self) -> str:
        """Get tree-sitter query for JSX elements with TypeScript context.

        Returns:
            Tree-sitter query string for finding JSX elements
        """
        return """
        (jsx_element
            open_tag: (jsx_opening_element 
                name: (_) @jsx.element_name
            )
        ) @jsx.element

        (jsx_self_closing_element
            name: (_) @jsx.self_closing_name
        ) @jsx.self_closing

        """

    def get_jsx_expression_query(self) -> str:
        """Get tree-sitter query for JSX expressions with TypeScript types.

        Returns:
            Tree-sitter query string for finding JSX expressions
        """
        return """
        (jsx_expression
            (_) @jsx.expression_content
        ) @jsx.expression
        """

    def get_hook_query(self) -> str:
        """Get tree-sitter query for React hooks with TypeScript generics.

        Returns:
            Tree-sitter query string for finding React hook usage with types
        """
        return """
        ; Typed hook calls with generics
        (call_expression
            function: (identifier) @hook.name
            (#match? @hook.name "^use[A-Z]")
            type_arguments: (type_arguments) @hook.type_args
        ) @hook.typed_call

        ; Regular hook calls
        (call_expression
            function: (identifier) @hook.name
            (#match? @hook.name "^use[A-Z]")
        ) @hook.call

        ; Typed hook variable declarations
        (variable_declarator
            name: (_) @hook.variable
            value: (call_expression
                function: (identifier) @hook.function
                (#match? @hook.function "^use[A-Z]")
            )
        ) @hook.typed_declaration

        ; Regular hook variable declarations  
        (variable_declarator
            name: (_) @hook.variable
            value: (call_expression
                function: (identifier) @hook.function
                (#match? @hook.function "^use[A-Z]")
                type_arguments: (type_arguments) @hook.type_args
            )
        ) @hook.declaration
        """

    def get_props_interface_query(self) -> str:
        """Get tree-sitter query for component props interfaces.

        Returns:
            Tree-sitter query string for finding props interface definitions
        """
        return """
        (interface_declaration
            name: (type_identifier) @props.interface_name
            (#match? @props.interface_name "Props$")
        ) @props.interface

        (type_alias_declaration
            name: (type_identifier) @props.type_name
            (#match? @props.type_name "Props$")
        ) @props.type_alias
        """

    def get_comment_query(self) -> str:
        """Get tree-sitter query for TSX comments including JSX comment syntax.

        Returns:
            Tree-sitter query string for finding comments including TSDoc and JSX comments
        """
        base_query = super().get_comment_query()

        # Add TSX-specific comment patterns
        tsx_comment_query = """
        ; JSX comments {/* comment */}
        (jsx_expression
            (comment) @jsx.comment
        ) @jsx.comment_expression
        """

        return base_query + tsx_comment_query

    def extract_component_name(self, node: "TSNode | None", source: str) -> str:
        """Extract React component name from a typed function definition.

        Args:
            node: Tree-sitter function/component definition node
            source: Source code string

        Returns:
            Component name or fallback name if extraction fails
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return "unknown_component"

        # Use base TypeScript function name extraction
        name = self.extract_function_name(node, source)

        # Check if this appears to be a React component (starts with uppercase)
        if name and name[0].isupper():
            return name

        return name

    def extract_jsx_element_name(self, node: "TSNode | None", source: str) -> str:
        """Extract JSX element name.

        Args:
            node: Tree-sitter JSX element node
            source: Source code string

        Returns:
            JSX element name or fallback
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return "unknown_element"

        # Look for opening tag name
        if node.type == "jsx_element":
            opening_tag = self.find_child_by_type(node, "jsx_opening_element")
            if opening_tag:
                name_node = opening_tag.child(1)  # Skip '<'
                if name_node:
                    return self.get_node_text(name_node, source)

        # Self-closing element
        elif node.type == "jsx_self_closing_element":
            name_node = node.child(1)  # Skip '<'
            if name_node:
                return self.get_node_text(name_node, source)

        return self.get_fallback_name(node, "jsx_element")

    def extract_hook_name(self, node: "TSNode | None", source: str) -> str:
        """Extract React hook name from a typed hook call.

        Args:
            node: Tree-sitter hook call node
            source: Source code string

        Returns:
            Hook name or fallback
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return "unknown_hook"

        # For hook calls
        if node.type == "call_expression":
            func_node = self.find_child_by_type(node, "identifier")
            if func_node:
                return self.get_node_text(func_node, source)

        # For hook variable declarations
        elif node.type == "variable_declarator":
            value_node = self.find_child_by_type(node, "call_expression")
            if value_node:
                func_node = self.find_child_by_type(value_node, "identifier")
                if func_node:
                    return self.get_node_text(func_node, source)

        return self.get_fallback_name(node, "hook")

    def extract_component_props_type(
        self, node: "TSNode | None", source: str
    ) -> str | None:
        """Extract component props type annotation.

        Args:
            node: Tree-sitter component definition node
            source: Source code string

        Returns:
            Props type string or None if not found
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return None

        try:
            # Look for function parameter with type annotation
            params_node = self.find_child_by_type(node, "formal_parameters")
            if params_node and params_node.child_count > 0:
                first_param = params_node.child(1)  # Skip opening parenthesis
                if first_param and first_param.type == "required_parameter":
                    type_annotation = self.find_child_by_type(
                        first_param, "type_annotation"
                    )
                    if type_annotation:
                        type_text = self.get_node_text(type_annotation, source).strip()
                        if type_text.startswith(":"):
                            return type_text[1:].strip()
                        return type_text

            # Look for React.FC type annotation
            if node.type == "variable_declarator":
                type_annotation = self.find_child_by_type(node, "type_annotation")
                if type_annotation:
                    type_text = self.get_node_text(type_annotation, source).strip()
                    if "FC<" in type_text or "FunctionComponent<" in type_text:
                        # Extract the generic type parameter
                        start = type_text.find("<")
                        end = type_text.rfind(">")
                        if start != -1 and end != -1:
                            return type_text[start + 1 : end].strip()

        except Exception as e:
            logger.error(f"Failed to extract TSX component props type: {e}")

        return None

    def extract_hook_types(self, node: "TSNode | None", source: str) -> dict[str, str]:
        """Extract type information from a typed React hook.

        Args:
            node: Tree-sitter hook call node
            source: Source code string

        Returns:
            Dictionary with type information
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return {}

        types = {}

        try:
            # Look for type arguments in hook call
            if node.type == "call_expression":
                type_args = self.find_child_by_type(node, "type_arguments")
                if type_args:
                    types["generic_types"] = self.get_node_text(
                        type_args, source
                    ).strip()

            # Look for variable type annotation
            elif node.type == "variable_declarator":
                type_annotation = self.find_child_by_type(node, "type_annotation")
                if type_annotation:
                    types["variable_type"] = self.get_node_text(
                        type_annotation, source
                    ).strip()

                # Also check the call expression for generic types
                call_expr = self.find_child_by_type(node, "call_expression")
                if call_expr:
                    type_args = self.find_child_by_type(call_expr, "type_arguments")
                    if type_args:
                        types["generic_types"] = self.get_node_text(
                            type_args, source
                        ).strip()

        except Exception as e:
            logger.error(f"Failed to extract TSX hook types: {e}")

        return types

    def is_react_component(self, node: "TSNode | None", source: str) -> bool:
        """Check if a function is a typed React component.

        Args:
            node: Tree-sitter function definition node
            source: Source code string

        Returns:
            True if the function appears to be a React component
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return False

        # Check if function returns JSX
        node_text = self.get_node_text(node, source)
        if any(
            jsx_indicator in node_text
            for jsx_indicator in ["<", "jsx", "React.createElement"]
        ):
            # Check if function name starts with uppercase (React convention)
            name = self.extract_function_name(node, source)
            if name and len(name) > 0 and name[0].isupper():
                return True

            # Check for React.FC type annotation
            if "React.FC" in node_text or "FunctionComponent" in node_text:
                return True

        return False

    def extract_jsx_props(self, node: "TSNode | None", source: str) -> list[str]:
        """Extract JSX props from a JSX element.

        Args:
            node: Tree-sitter JSX element node
            source: Source code string

        Returns:
            List of prop names
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return []

        props = []

        # Find JSX opening element or self-closing element
        opening_element = None
        if node.type == "jsx_element":
            opening_element = self.find_child_by_type(node, "jsx_opening_element")
        elif node.type == "jsx_self_closing_element":
            opening_element = node

        if opening_element:
            # Look for jsx_attribute nodes
            for i in range(opening_element.child_count):
                child = opening_element.child(i)
                if child and child.type == "jsx_attribute":
                    name_node = child.child(0)  # First child is the attribute name
                    if name_node:
                        prop_name = self.get_node_text(name_node, source)
                        if prop_name:
                            props.append(prop_name)

        return props

    def should_include_node(self, node: "TSNode | None", source: str) -> bool:
        """Determine if a TSX node should be included as a chunk.

        Extends TypeScript logic with TSX-specific considerations.

        Args:
            node: Tree-sitter node
            source: Source code string

        Returns:
            True if node should be included, False otherwise
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return False

        # Use base TypeScript filtering first
        if not super().should_include_node(node, source):
            return False

        # TSX-specific filtering
        node_text = self.get_node_text(node, source)

        # Always include React components
        if self.is_react_component(node, source):
            return True

        # Include JSX elements that are substantial
        if node.type in ["jsx_element", "jsx_self_closing_element"]:
            return len(node_text.strip()) > 20

        # Include hook usage
        if node.type == "call_expression":
            hook_name = self.extract_hook_name(node, source)
            if (
                hook_name.startswith("use")
                and len(hook_name) > 3
                and hook_name[3].isupper()
            ):
                return True

        # Include props interfaces
        if node.type in ["interface_declaration", "type_alias_declaration"]:
            name = (
                self.extract_interface_name(node, source)
                if node.type == "interface_declaration"
                else self.extract_type_alias_name(node, source)
            )
            if name.endswith("Props"):
                return True

        return True

    def clean_jsx_text(self, text: str) -> str:
        """Clean JSX text by removing JSX-specific syntax artifacts.

        Args:
            text: Raw JSX text

        Returns:
            Cleaned text
        """
        # Remove JSX comment syntax
        text = text.replace("{/*", "").replace("*/}", "")

        # Clean up JSX expressions
        text = text.replace("{", " ").replace("}", " ")

        # Use base TypeScript cleaning
        return self.clean_comment_text(text)

    def create_enhanced_chunk(
        self,
        node: "TSNode | None",
        source: str,
        file_path: Path,
        chunk_type: ChunkType,
        name: str,
        **extra_fields: Any,
    ) -> dict[str, Any]:
        """Create an enhanced chunk dictionary with TSX-specific metadata.

        Args:
            node: Tree-sitter node
            source: Source code string
            file_path: Path to source file
            chunk_type: Type of chunk
            name: Chunk name/symbol
            **extra_fields: Additional fields to include

        Returns:
            Enhanced chunk dictionary with TSX metadata
        """
        # Start with base TypeScript enhanced chunk
        chunk = super().create_enhanced_chunk(
            node, source, file_path, chunk_type, name, **extra_fields
        )

        # Add TSX-specific enhancements
        if node and TREE_SITTER_AVAILABLE:
            try:
                # Add component props type for React components
                if chunk_type == ChunkType.FUNCTION and self.is_react_component(
                    node, source
                ):
                    props_type = self.extract_component_props_type(node, source)
                    if props_type:
                        extra_fields["props_type"] = props_type

                # Add hook type information
                if "hook" in name.lower():
                    hook_types = self.extract_hook_types(node, source)
                    if hook_types:
                        extra_fields.update(hook_types)

                # Add JSX props for JSX elements
                if chunk_type == ChunkType.OTHER and node.type in [
                    "jsx_element",
                    "jsx_self_closing_element",
                ]:
                    jsx_props = self.extract_jsx_props(node, source)
                    if jsx_props:
                        extra_fields["jsx_props"] = jsx_props

            except Exception as e:
                logger.error(f"Failed to enhance TSX chunk: {e}")

        return chunk

    def resolve_import_path(self, import_text: str, base_dir: Path, source_file: Path) -> Path | None:
        """Resolve relative import path to absolute file path.

        Args:
            import_text: Import statement text
            base_dir: Base directory for resolution
            source_file: Source file containing the import

        Returns:
            Resolved absolute path or None if not resolvable
        """
        match = re.search(r'''from\s+['"](.+?)['"]''', import_text)
        if not match:
            return None
        import_path = match.group(1)
        if not import_path or not import_path.startswith('.'):
            return None
        source_dir = self._resolve_source_dir(source_file, base_dir)
        resolved = (source_dir / import_path).resolve()
        if resolved.exists() and resolved.is_file():
            return resolved
        for ext in ['.ts', '.tsx', '.d.ts', '.js', '.jsx']:
            with_ext = resolved.with_suffix(ext)
            if with_ext.exists():
                return with_ext
        for index in ['index.ts', 'index.tsx', 'index.js']:
            index_path = resolved / index
            if index_path.exists():
                return index_path
        return None
