"""Base mapping class for language-specific tree-sitter mappings.

This module provides the base class for language-specific tree-sitter node
mappings. It contains common helper methods and default implementations that
can be overridden by language-specific subclasses.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from chunkhound.core.types.common import ChunkType, Language

if TYPE_CHECKING:
    from tree_sitter import Node as TSNode

    from chunkhound.parsers.universal_engine import UniversalConcept

try:
    from tree_sitter import Node as TSNode

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    TSNode = Any  # type: ignore

# Maximum length for constant values in metadata (prevents bloat)
MAX_CONSTANT_VALUE_LENGTH = 50


class BaseMapping(ABC):
    """Base class for language-specific tree-sitter mappings.

    This class provides common functionality for extracting semantic information
    from tree-sitter AST nodes, including helper methods for node traversal,
    text extraction, and chunk creation patterns.
    """

    def __init__(self, language: Language):
        """Initialize base mapping.

        Args:
            language: The programming language this mapping handles
        """
        self.language = language

    # Abstract methods that subclasses must implement

    @abstractmethod
    def get_function_query(self) -> str:
        """Get tree-sitter query pattern for function definitions.

        Returns:
            Tree-sitter query string for finding function definitions
        """
        pass

    @abstractmethod
    def get_class_query(self) -> str:
        """Get tree-sitter query pattern for class definitions.

        Returns:
            Tree-sitter query string for finding class definitions
        """
        pass

    @abstractmethod
    def get_comment_query(self) -> str:
        """Get tree-sitter query pattern for comments.

        Returns:
            Tree-sitter query string for finding comments
        """
        pass

    @abstractmethod
    def extract_function_name(self, node: TSNode | None, source: str) -> str:
        """Extract function name from a function definition node.

        Args:
            node: Tree-sitter function definition node
            source: Source code string

        Returns:
            Function name or fallback name if extraction fails
        """
        pass

    @abstractmethod
    def extract_class_name(self, node: TSNode | None, source: str) -> str:
        """Extract class name from a class definition node.

        Args:
            node: Tree-sitter class definition node
            source: Source code string

        Returns:
            Class name or fallback name if extraction fails
        """
        pass

    # Common helper methods with default implementations

    def get_node_text(self, node: TSNode | None, source: str) -> str:
        """Extract text content from a tree-sitter node.

        Args:
            node: Tree-sitter node
            source: Source code string

        Returns:
            Text content of the node
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return ""
        source_bytes = source.encode("utf-8")
        return source_bytes[node.start_byte : node.end_byte].decode("utf-8")

    def find_child_by_type(self, node: TSNode | None, node_type: str) -> TSNode | None:
        """Find first child node of specified type.

        Args:
            node: Parent tree-sitter node
            node_type: Target node type to find

        Returns:
            First child node of the specified type, or None if not found
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return None

        for i in range(node.child_count):
            child = node.child(i)
            if child and child.type == node_type:
                return child
        return None

    def find_children_by_type(
        self, node: TSNode | None, node_type: str
    ) -> list[TSNode]:
        """Find all child nodes of specified type.

        Args:
            node: Parent tree-sitter node
            node_type: Target node type to find

        Returns:
            List of child nodes of the specified type
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return []

        children = []
        for i in range(node.child_count):
            child = node.child(i)
            if child and child.type == node_type:
                children.append(child)
        return children

    def walk_tree(self, node: TSNode | None) -> Iterator[TSNode]:
        """Walk through all nodes in the tree depth-first.

        Args:
            node: Root node to start walking from

        Yields:
            Tree-sitter nodes in depth-first order
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return

        yield node
        for i in range(node.child_count):
            child = node.child(i)
            if child:
                yield from self.walk_tree(child)

    def find_nodes_by_type(self, root: TSNode | None, node_type: str) -> list[TSNode]:
        """Find all nodes of specified type in the tree.

        Args:
            root: Root node to search from
            node_type: Target node type to find

        Returns:
            List of all nodes of the specified type
        """
        if not TREE_SITTER_AVAILABLE or root is None:
            return []

        nodes = []
        for node in self.walk_tree(root):
            if node and node.type == node_type:
                nodes.append(node)
        return nodes

    def get_node_line_range(self, node: TSNode | None) -> tuple[int, int]:
        """Get line range for a node (1-based).

        Args:
            node: Tree-sitter node

        Returns:
            Tuple of (start_line, end_line) with 1-based line numbers
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return (1, 1)
        return (node.start_point[0] + 1, node.end_point[0] + 1)

    def get_node_byte_range(self, node: TSNode | None) -> tuple[int, int]:
        """Get byte range for a node.

        Args:
            node: Tree-sitter node

        Returns:
            Tuple of (start_byte, end_byte)
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return (0, 0)
        return (node.start_byte, node.end_byte)

    def create_chunk_dict(
        self,
        node: TSNode | None,
        source: str,
        file_path: Path,
        chunk_type: ChunkType,
        name: str,
        display_name: str | None = None,
        parent: str | None = None,
        **extra_fields: Any,
    ) -> dict[str, Any]:
        """Create a standard chunk dictionary from a tree-sitter node.

        Args:
            node: Tree-sitter node
            source: Source code string
            file_path: Path to source file
            chunk_type: Type of chunk
            name: Chunk name/symbol
            display_name: Display name (defaults to name)
            parent: Parent symbol name
            **extra_fields: Additional fields to include

        Returns:
            Dictionary representing the chunk
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            logger.error(
                "Cannot create chunk: tree-sitter not available or node is None"
            )
            return {}

        code = self.get_node_text(node, source)
        start_line, end_line = self.get_node_line_range(node)
        start_byte, end_byte = self.get_node_byte_range(node)

        chunk = {
            "symbol": name,
            "start_line": start_line,
            "end_line": end_line,
            "code": code,
            "chunk_type": chunk_type.value,
            "language": self.language.value.lower(),
            "path": str(file_path),
            "name": name,
            "display_name": display_name or name,
            "content": code,
            "start_byte": start_byte,
            "end_byte": end_byte,
        }

        if parent:
            chunk["parent"] = parent

        # Add extra fields
        chunk.update(extra_fields)

        return chunk

    def clean_comment_text(self, text: str) -> str:
        """Clean comment text by removing comment markers.

        Args:
            text: Raw comment text

        Returns:
            Cleaned comment text
        """
        cleaned = text.strip()

        # Remove common single-line comment markers
        if cleaned.startswith("//"):
            cleaned = cleaned[2:].strip()
        elif cleaned.startswith("#"):
            cleaned = cleaned[1:].strip()
        elif cleaned.startswith("--"):
            cleaned = cleaned[2:].strip()

        # Remove common multi-line comment markers
        if cleaned.startswith("/*") and cleaned.endswith("*/"):
            cleaned = cleaned[2:-2].strip()
        elif cleaned.startswith("<!--") and cleaned.endswith("-->"):
            cleaned = cleaned[4:-3].strip()

        return cleaned

    def clean_string_literal(self, text: str) -> str:
        """Clean string literal by removing quotes.

        Args:
            text: Raw string literal text

        Returns:
            Cleaned string text
        """
        cleaned = text.strip()

        # Remove triple quotes
        if cleaned.startswith('"""') and cleaned.endswith('"""'):
            cleaned = cleaned[3:-3]
        elif cleaned.startswith("'''") and cleaned.endswith("'''"):
            cleaned = cleaned[3:-3]
        # Remove single quotes
        elif cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]
        elif cleaned.startswith("'") and cleaned.endswith("'"):
            cleaned = cleaned[1:-1]

        return cleaned.strip()

    def get_expression_preview(
        self, expr: str, max_length: int = 20, safe_chars: bool = True
    ) -> str:
        """Get truncated expression for chunk naming.

        This method provides a consistent way to generate safe, readable
        preview strings from expressions across different language mappings.

        Args:
            expr: Expression to truncate
            max_length: Maximum length before truncation
            safe_chars: Replace quotes/spaces with underscores for safe naming

        Returns:
            Safe, truncated expression suitable for chunk naming

        Examples:
            >>> self.get_expression_preview('user.name.first')
            'user_name_first'
            >>> self.get_expression_preview('"hello"', safe_chars=True)
            'hello'
            >>> self.get_expression_preview('very_long_variable_name', max_length=10)
            'very_lo...'
        """
        if not expr:
            return "expr"

        # Clean up the expression if safe_chars is enabled
        if safe_chars:
            # Remove quotes
            expr = expr.replace('"', "").replace("'", "")
            # Replace spaces with underscores
            expr = expr.replace(" ", "_")

        # Truncate if too long
        if len(expr) > max_length:
            expr = expr[:max_length - 3] + "..."

        return expr if expr else "expr"

    def get_fallback_name(self, node: TSNode | None, prefix: str) -> str:
        """Generate a fallback name for a node when extraction fails.

        Args:
            node: Tree-sitter node
            prefix: Prefix for the fallback name

        Returns:
            Fallback name based on line number
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return f"{prefix}_unknown"

        line_num = node.start_point[0] + 1 if hasattr(node, "start_point") else 0
        return f"{prefix}_{line_num}"

    # Optional methods with default implementations that can be overridden

    def get_method_query(self) -> str:
        """Get tree-sitter query pattern for method definitions.

        Default implementation returns empty string. Override in subclasses
        that distinguish between functions and methods.

        Returns:
            Tree-sitter query string for finding method definitions
        """
        return ""

    def get_docstring_query(self) -> str:
        """Get tree-sitter query pattern for docstrings.

        Default implementation returns empty string. Override in subclasses
        that support docstrings.

        Returns:
            Tree-sitter query string for finding docstrings
        """
        return ""

    def extract_method_name(self, node: TSNode | None, source: str) -> str:
        """Extract method name from a method definition node.

        Default implementation delegates to extract_function_name.
        Override in subclasses that need different handling.

        Args:
            node: Tree-sitter method definition node
            source: Source code string

        Returns:
            Method name or fallback name if extraction fails
        """
        return self.extract_function_name(node, source)

    def extract_parameters(self, node: TSNode | None, source: str) -> list[str]:
        """Extract parameter names from a function/method node.

        Default implementation returns empty list. Override in subclasses
        to provide parameter extraction.

        Args:
            node: Tree-sitter function/method definition node
            source: Source code string

        Returns:
            List of parameter names
        """
        return []

    def should_include_node(self, node: TSNode | None, source: str) -> bool:
        """Determine if a node should be included as a chunk.

        Default implementation returns True. Override in subclasses
        to add filtering logic.

        Args:
            node: Tree-sitter node
            source: Source code string

        Returns:
            True if node should be included, False otherwise
        """
        return True

    def _resolve_source_dir(self, source_file: Path, base_dir: Path | None) -> Path:
        """Resolve source directory, anchoring relative paths to base_dir.

        When source_file is relative (e.g., from database), anchor it to base_dir.
        When source_file is absolute, use it directly.

        Args:
            source_file: Source file path (may be relative or absolute)
            base_dir: Project base directory for anchoring relative paths

        Returns:
            The directory containing the source file
        """
        source_dir = source_file.parent
        if not source_file.is_absolute() and base_dir:
            source_dir = base_dir / source_file.parent
        return source_dir

    def resolve_import_path(
        self,
        import_text: str,
        base_dir: Path,
        source_file: Path,
    ) -> Path | None:
        """Resolve import statement to file path.

        Default implementation returns None (no import resolution).
        Override in language-specific subclasses.

        Args:
            import_text: The raw import statement text
            base_dir: Project root directory
            source_file: File containing the import

        Returns:
            Resolved file path or None if external/unresolvable
        """
        return None

    def extract_constants(
        self,
        concept: "UniversalConcept",
        captures: dict[str, "TSNode"],
        content: bytes,
    ) -> list[dict[str, str]] | None:
        """Extract constants from captures.

        Default implementation returns None (no constants).
        Override in language-specific subclasses to extract:
        - UPPER_SNAKE_CASE variables (Python)
        - const declarations (JS/TS, Go, Rust)
        - #define macros (C/C++)
        - static final fields (Java)
        - etc.

        Args:
            concept: The UniversalConcept being extracted
            captures: Tree-sitter query captures
            content: Source file content as bytes

        Returns:
            List of constant dicts with keys: name, value, type (optional)
            Returns None if no constants found or not applicable.
        """
        return None
