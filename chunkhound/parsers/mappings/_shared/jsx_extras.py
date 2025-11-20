"""JSX extras mixin for JSX and TSX language mappings.

This module provides a mixin class containing JSX/React-specific methods that are
shared between JSXMapping and TSXMapping. It includes queries and extraction logic
for:
- JSX elements and fragments
- React functional and class components
- React hooks
- JSX expressions and attributes
- Component props types (TypeScript)
"""

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from tree_sitter import Node as TSNode

try:
    from tree_sitter import Node as TSNode

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    TSNode = None


class JSXExtras:
    """Mixin class providing JSX/React-specific functionality.

    This mixin is designed to be used with JSXMapping and TSXMapping classes.
    It provides common methods for handling JSX elements, React components,
    hooks, and related patterns.

    The methods in this mixin expect the following methods to be available
    from the parent class (BaseMapping or its subclasses):
    - get_node_text(node, source)
    - find_child_by_type(node, node_type)
    - get_fallback_name(node, prefix)
    - extract_function_name(node, source)
    - clean_comment_text(text)
    """

    # ========================================================================
    # JSX Element Queries
    # ========================================================================

    def get_jsx_element_query(self) -> str:
        """Get tree-sitter query for JSX elements.

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
        """Get tree-sitter query for JSX expressions.

        Returns:
            Tree-sitter query string for finding JSX expressions
        """
        return """
        (jsx_expression
            (_) @jsx.expression_content
        ) @jsx.expression
        """

    def get_hook_query(self) -> str:
        """Get tree-sitter query for React hooks.

        This query works for both JSX and TSX grammars. For TypeScript,
        it includes patterns with type arguments.

        Returns:
            Tree-sitter query string for finding React hook usage
        """
        return """
        ; Typed hook calls with generics (TSX)
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

        ; Typed hook variable declarations (TSX)
        (variable_declarator
            name: (_) @hook.variable
            value: (call_expression
                function: (identifier) @hook.function
                (#match? @hook.function "^use[A-Z]")
            )
        ) @hook.typed_declaration

        ; Hook variable declarations with generics
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

        This is primarily for TypeScript/TSX but the query is safe to use
        with JSX grammars (will simply not match).

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

    # ========================================================================
    # Name Extraction Methods
    # ========================================================================

    def extract_component_name(self, node: "TSNode | None", source: str) -> str:
        """Extract React component name from a function definition.

        Args:
            node: Tree-sitter function/component definition node
            source: Source code string

        Returns:
            Component name or fallback name if extraction fails
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return "unknown_component"

        # Use base function name extraction
        name = self.extract_function_name(node, source)  # type: ignore[attr-defined]

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
            opening_tag = self.find_child_by_type(node, "jsx_opening_element")  # type: ignore[attr-defined]
            if opening_tag:
                name_node = opening_tag.child(1)  # Skip '<'
                if name_node:
                    return self.get_node_text(name_node, source)  # type: ignore[attr-defined]

        # Self-closing element
        elif node.type == "jsx_self_closing_element":
            name_node = node.child(1)  # Skip '<'
            if name_node:
                return self.get_node_text(name_node, source)  # type: ignore[attr-defined]

        return self.get_fallback_name(node, "jsx_element")  # type: ignore[attr-defined]

    def extract_hook_name(self, node: "TSNode | None", source: str) -> str:
        """Extract React hook name from a hook call.

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
            func_node = self.find_child_by_type(node, "identifier")  # type: ignore[attr-defined]
            if func_node:
                return self.get_node_text(func_node, source)  # type: ignore[attr-defined]

        # For hook variable declarations
        elif node.type == "variable_declarator":
            value_node = self.find_child_by_type(node, "call_expression")  # type: ignore[attr-defined]
            if value_node:
                func_node = self.find_child_by_type(value_node, "identifier")  # type: ignore[attr-defined]
                if func_node:
                    return self.get_node_text(func_node, source)  # type: ignore[attr-defined]

        return self.get_fallback_name(node, "hook")  # type: ignore[attr-defined]

    # ========================================================================
    # TypeScript-Specific Extraction Methods
    # ========================================================================

    def extract_component_props_type(
        self, node: "TSNode | None", source: str
    ) -> str | None:
        """Extract component props type annotation.

        This method is primarily for TypeScript/TSX components.

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
            params_node = self.find_child_by_type(node, "formal_parameters")  # type: ignore[attr-defined]
            if params_node and params_node.child_count > 0:
                first_param = params_node.child(1)  # Skip opening parenthesis
                if first_param and first_param.type == "required_parameter":
                    type_annotation = self.find_child_by_type(  # type: ignore[attr-defined]
                        first_param, "type_annotation"
                    )
                    if type_annotation:
                        type_text = self.get_node_text(type_annotation, source).strip()  # type: ignore[attr-defined]
                        if type_text.startswith(":"):
                            return type_text[1:].strip()
                        return type_text

            # Look for React.FC type annotation
            if node.type == "variable_declarator":
                type_annotation = self.find_child_by_type(node, "type_annotation")  # type: ignore[attr-defined]
                if type_annotation:
                    type_text = self.get_node_text(type_annotation, source).strip()  # type: ignore[attr-defined]
                    if "FC<" in type_text or "FunctionComponent<" in type_text:
                        # Extract the generic type parameter
                        start = type_text.find("<")
                        end = type_text.rfind(">")
                        if start != -1 and end != -1:
                            return type_text[start + 1 : end].strip()

        except Exception as e:
            logger.error(f"Failed to extract component props type: {e}")

        return None

    def extract_hook_types(self, node: "TSNode | None", source: str) -> dict[str, str]:
        """Extract type information from a typed React hook.

        This method is primarily for TypeScript/TSX hooks with generics.

        Args:
            node: Tree-sitter hook call node
            source: Source code string

        Returns:
            Dictionary with type information
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return {}

        types: dict[str, str] = {}

        try:
            # Look for type arguments in hook call
            if node.type == "call_expression":
                type_args = self.find_child_by_type(node, "type_arguments")  # type: ignore[attr-defined]
                if type_args:
                    types["generic_types"] = self.get_node_text(  # type: ignore[attr-defined]
                        type_args, source
                    ).strip()

            # Look for variable type annotation
            elif node.type == "variable_declarator":
                type_annotation = self.find_child_by_type(node, "type_annotation")  # type: ignore[attr-defined]
                if type_annotation:
                    types["variable_type"] = self.get_node_text(  # type: ignore[attr-defined]
                        type_annotation, source
                    ).strip()

                # Also check the call expression for generic types
                call_expr = self.find_child_by_type(node, "call_expression")  # type: ignore[attr-defined]
                if call_expr:
                    type_args = self.find_child_by_type(call_expr, "type_arguments")  # type: ignore[attr-defined]
                    if type_args:
                        types["generic_types"] = self.get_node_text(  # type: ignore[attr-defined]
                            type_args, source
                        ).strip()

        except Exception as e:
            logger.error(f"Failed to extract hook types: {e}")

        return types

    # ========================================================================
    # Component Detection and Filtering
    # ========================================================================

    def is_react_component(self, node: "TSNode | None", source: str) -> bool:
        """Check if a function is a React component.

        Detects React components by:
        1. Checking if the function returns JSX
        2. Verifying the function name starts with uppercase (React convention)
        3. Checking for React.FC type annotations (TypeScript)

        Args:
            node: Tree-sitter function definition node
            source: Source code string

        Returns:
            True if the function appears to be a React component
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return False

        # Check if function returns JSX
        node_text = self.get_node_text(node, source)  # type: ignore[attr-defined]
        if any(
            jsx_indicator in node_text
            for jsx_indicator in ["<", "jsx", "React.createElement"]
        ):
            # Check if function name starts with uppercase (React convention)
            name = self.extract_function_name(node, source)  # type: ignore[attr-defined]
            if name and len(name) > 0 and name[0].isupper():
                return True

            # Check for React.FC type annotation (TypeScript)
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

        props: list[str] = []

        # Find JSX opening element or self-closing element
        opening_element = None
        if node.type == "jsx_element":
            opening_element = self.find_child_by_type(node, "jsx_opening_element")  # type: ignore[attr-defined]
        elif node.type == "jsx_self_closing_element":
            opening_element = node

        if opening_element:
            # Look for jsx_attribute nodes
            for i in range(opening_element.child_count):
                child = opening_element.child(i)
                if child and child.type == "jsx_attribute":
                    name_node = child.child(0)  # First child is the attribute name
                    if name_node:
                        prop_name = self.get_node_text(name_node, source)  # type: ignore[attr-defined]
                        if prop_name:
                            props.append(prop_name)

        return props

    # ========================================================================
    # Text Cleaning Methods
    # ========================================================================

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

        # Use base comment cleaning
        return self.clean_comment_text(text)  # type: ignore[attr-defined]

    # ========================================================================
    # Node Filtering (JSX-specific parts)
    # ========================================================================

    def should_include_jsx_node(self, node: "TSNode | None", source: str) -> bool:
        """Determine if a JSX-specific node should be included as a chunk.

        This method provides JSX-specific inclusion logic that can be called
        from the should_include_node method of JSX/TSX mappings.

        Args:
            node: Tree-sitter node
            source: Source code string

        Returns:
            True if node should be included based on JSX-specific criteria,
            None if this method doesn't apply to the node type
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return False

        node_text = self.get_node_text(node, source)  # type: ignore[attr-defined]

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

        # Include props interfaces (TypeScript)
        if node.type in ["interface_declaration", "type_alias_declaration"]:
            # Check if name ends with "Props"
            name_node = self.find_child_by_type(node, "type_identifier")  # type: ignore[attr-defined]
            if name_node:
                name = self.get_node_text(name_node, source)  # type: ignore[attr-defined]
                if name.endswith("Props"):
                    return True

        return True
