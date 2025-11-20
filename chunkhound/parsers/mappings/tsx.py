"""TSX language mapping for unified parser architecture.

This module provides TSX-specific tree-sitter queries and extraction logic
extending TypeScript functionality for React-specific patterns with type safety.
Handles JSX elements, React components with types, hooks with generics, and
TSX expressions with type annotations.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from chunkhound.core.types.common import ChunkType, Language
from chunkhound.parsers.mappings._shared.jsx_extras import JSXExtras
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


class TSXMapping(TypeScriptMapping, JSXExtras):
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

        # Call JSX-specific filtering from mixin
        return self.should_include_jsx_node(node, source)

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
