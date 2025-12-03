"""JSX language mapping for unified parser architecture.

This module provides JSX-specific tree-sitter queries and extraction logic
extending JavaScript functionality for React-specific patterns like JSX elements,
components, hooks, and JSX expressions.
"""

from typing import TYPE_CHECKING

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.javascript import JavaScriptMapping
from chunkhound.parsers.universal_engine import UniversalConcept
from chunkhound.parsers.mappings._shared.js_query_patterns import (
    TOP_LEVEL_LEXICAL_CONFIG,
    TOP_LEVEL_VAR_CONFIG,
    COMMONJS_MODULE_EXPORTS,
    COMMONJS_NESTED_EXPORTS,
    COMMONJS_EXPORTS_SHORTHAND,
)
from chunkhound.parsers.mappings._shared.js_query_patterns import (
    TOP_LEVEL_LEXICAL_CONFIG,
    TOP_LEVEL_VAR_CONFIG,
    COMMONJS_MODULE_EXPORTS,
    COMMONJS_NESTED_EXPORTS,
    COMMONJS_EXPORTS_SHORTHAND,
)

if TYPE_CHECKING:
    from tree_sitter import Node as TSNode

try:
    from tree_sitter import Node as TSNode

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    TSNode = None


class JSXMapping(JavaScriptMapping):
    """JSX language mapping extending JavaScript mapping.

    Provides JSX-specific queries and extraction methods for:
    - React functional and class components
    - JSX elements and fragments
    - JSX expressions and attributes
    - React hooks
    - JSX comments
    """

    def __init__(self):
        """Initialize JSX mapping."""
        # Initialize with JSX language instead of JavaScript
        super().__init__()
        self.language = Language.JSX

    def get_function_query(self) -> str:
        """Get tree-sitter query for JSX function definitions including React components.

        Extends JavaScript function query with React component patterns.

        Returns:
            Tree-sitter query string for finding function definitions and React components
        """
        # Get base JavaScript function query
        base_query = super().get_function_query()

        # Add JSX-specific patterns for React components
        jsx_specific_query = """
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

        """

        return base_query + jsx_specific_query

    def get_class_query(self) -> str:
        """Get tree-sitter query for JSX class definitions using TSX grammar.

        Overrides JavaScript mapping to use TSX node types.

        Returns:
            Tree-sitter query string for finding class definitions
        """
        return """
        (class_declaration
            name: (type_identifier) @class_name
        ) @class_def

        (variable_declarator
            name: (identifier) @var_class_name
            value: (class) @class_expr
        ) @var_class_def
        """

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

        Returns:
            Tree-sitter query string for finding React hook usage
        """
        return """
        (call_expression
            function: (identifier) @hook.name
            (#match? @hook.name "^use[A-Z]")
        ) @hook.call

        (variable_declarator
            name: (_) @hook.variable
            value: (call_expression
                function: (identifier) @hook.function
                (#match? @hook.function "^use[A-Z]")
            )
        ) @hook.declaration
        """

    def get_comment_query(self) -> str:
        """Get tree-sitter query for JSX comments including JSX comment syntax.

        Returns:
            Tree-sitter query string for finding comments
        """
        base_query = super().get_comment_query()

        # Add JSX-specific comment patterns
        jsx_comment_query = """
        ; JSX comments {/* comment */}
        (jsx_expression
            (comment) @jsx.comment
        ) @jsx.comment_expression
        """

        return base_query + jsx_comment_query

    # Universal Concept integration: override to TSX-friendly patterns
    def get_query_for_concept(self, concept: "UniversalConcept") -> str | None:  # type: ignore[override]
        if concept == UniversalConcept.DEFINITION:
            return ("\n".join([
                """
                ; Functions and classes (TSX class name uses type_identifier)
                (function_declaration
                    name: (identifier) @name
                ) @definition

                (class_declaration
                    name: (type_identifier) @name
                ) @definition

                ; Exports
                (export_statement) @definition
                """,
                TOP_LEVEL_LEXICAL_CONFIG,
                TOP_LEVEL_VAR_CONFIG,
                # Top-level const/let function/arrow
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

    def is_react_component(self, node: "TSNode | None", source: str) -> bool:
        """Check if a function is a React component.

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
        """Determine if a JSX node should be included as a chunk.

        Extends JavaScript logic with JSX-specific considerations.

        Args:
            node: Tree-sitter node
            source: Source code string

        Returns:
            True if node should be included, False otherwise
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return False

        # Use base JavaScript filtering first
        if not super().should_include_node(node, source):
            return False

        # JSX-specific filtering
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

        # Use base JavaScript cleaning
        return self.clean_comment_text(text)
