"""JSX language mapping for unified parser architecture.

This module provides JSX-specific tree-sitter queries and extraction logic
extending JavaScript functionality for React-specific patterns like JSX elements,
components, hooks, and JSX expressions.
"""

from typing import TYPE_CHECKING

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.javascript import JavaScriptMapping
from chunkhound.parsers.mappings._shared.jsx_extras import JSXExtras
from chunkhound.parsers.universal_engine import UniversalConcept
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


class JSXMapping(JavaScriptMapping, JSXExtras):
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
        # Call BaseMapping.__init__ directly to set the correct language
        # (consistent with TypeScriptMapping and TSXMapping pattern)
        from chunkhound.parsers.mappings.base import BaseMapping
        BaseMapping.__init__(self, Language.JSX)

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

                ; Exported const/let with object/array initializer
                (export_statement
                    (lexical_declaration
                        (variable_declarator
                            name: (identifier) @name
                            value: [(object) (array)]
                        )
                    )
                ) @definition

                ; Exported const/let with function/arrow
                (export_statement
                    (lexical_declaration
                        (variable_declarator
                            name: (identifier) @name
                            value: [(function_expression) (arrow_function)]
                        )
                    )
                ) @definition

                ; Exported var with object/array initializer
                (export_statement
                    (variable_declaration
                        (variable_declarator
                            name: (identifier) @name
                            value: [(object) (array)]
                        )
                    )
                ) @definition

                ; Exported var with function/arrow
                (export_statement
                    (variable_declaration
                        (variable_declarator
                            name: (identifier) @name
                            value: [(function_expression) (arrow_function)]
                        )
                    )
                ) @definition

                ; Exported class expression (const/let)
                (export_statement
                    (lexical_declaration
                        (variable_declarator
                            name: (identifier) @name
                            value: (class)
                        )
                    )
                ) @definition

                ; Exported class expression (var)
                (export_statement
                    (variable_declaration
                        (variable_declarator
                            name: (identifier) @name
                            value: (class)
                        )
                    )
                ) @definition

                ; Exported function declaration
                (export_statement
                    (function_declaration
                        name: (identifier) @name
                    )
                ) @definition

                ; Exported class declaration
                (export_statement
                    (class_declaration
                        name: (type_identifier) @name
                    )
                ) @definition

                ; Exports - generic fallback
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
                (program
                    (variable_declaration
                        (variable_declarator
                            name: (identifier) @name
                            value: (function_expression)
                        ) @definition
                    )
                )
                (program
                    (variable_declaration
                        (variable_declarator
                            name: (identifier) @name
                            value: (arrow_function)
                        ) @definition
                    )
                )
                ; Class expression assigned to const/let at top level
                (program
                    (lexical_declaration
                        (variable_declarator
                            name: (identifier) @name
                            value: (class)
                        )
                    )
                ) @definition
                ; Class expression assigned to var at top level
                (program
                    (variable_declaration
                        (variable_declarator
                            name: (identifier) @name
                            value: (class)
                        )
                    )
                ) @definition
                """,
                COMMONJS_MODULE_EXPORTS,
                COMMONJS_NESTED_EXPORTS,
                COMMONJS_EXPORTS_SHORTHAND,
            ]))
        elif concept == UniversalConcept.COMMENT:
            return """
            (comment) @definition
            """
        return super().get_query_for_concept(concept)

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

        # Call JSX-specific filtering from mixin
        return self.should_include_jsx_node(node, source)

