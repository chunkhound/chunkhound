"""Code parser module for ChunkHound - clean version using only registry system."""

from pathlib import Path
from typing import Any

from loguru import logger

from chunkhound.core.detection import detect_language
from chunkhound.core.types.common import Language

from .tree_cache import TreeCache, get_default_cache


def is_tree_sitter_node(obj: Any) -> bool:
    """Check if object is a valid TreeSitterNode with required attributes."""
    return (
        obj is not None
        and hasattr(obj, "start_byte")
        and hasattr(obj, "end_byte")
        and hasattr(obj, "id")
    )


class CodeParser:
    """Tree-sitter based code parser using the registry
    system for all language support."""

    def __init__(self, use_cache: bool = True, cache: TreeCache | None = None):
        """Initialize the code parser.

        Args:
            use_cache: Whether to use TreeCache for performance optimization
            cache: Custom TreeCache instance, uses default if None
        """
        # TreeCache integration
        self.use_cache = use_cache
        self.tree_cache = cache or get_default_cache() if use_cache else None
        self._registry = None

    def setup(self) -> None:
        """Set up the parser registry."""
        try:
            from chunkhound.registry import get_registry

            self._registry = get_registry()
            logger.debug("Parser registry initialized successfully")
        except ImportError as e:
            logger.error(f"Registry not available: {e}")
            raise RuntimeError(
                "Registry system is required for parser operation"
            ) from e

    def parse_file(
        self, file_path: Path, source: str | None = None
    ) -> list[dict[str, Any]]:
        """Parse a file and extract semantic chunks using the registry system.

        Args:
            file_path: Path to file to parse
            source: Optional source code string (if None, reads from file)

        Returns:
            List of extracted chunks with metadata

        Raises:
            RuntimeError: If registry is not available or parser not found
        """
        # Determine file type (with content-based detection for ambiguous extensions)
        language = detect_language(file_path)

        # Initialize registry if not already done
        if self._registry is None:
            self.setup()

        # Get parser from registry
        parser = self._registry.get_language_parser(language)

        if not parser:
            raise RuntimeError(
                f"No parser plugin available for language {language}. File: {file_path}"
            )

        # Use the parser interface
        try:
            result = parser.parse_file(file_path, source)
            # Convert ParseResult to list of chunks
            if hasattr(result, "chunks"):
                return result.chunks
            elif isinstance(result, list):
                return result
            else:
                logger.warning(f"Unexpected parser result type: {type(result)}")
                return []
        except Exception as e:
            logger.error(f"Error parsing {file_path} with {language} parser: {e}")
            raise

    def parse_incremental(self, file_path: Path, source_code: str | None = None) -> Any:
        """Parse incrementally using TreeCache if available,
        fallback to direct parsing."""
        if source_code is None:
            # Read file content if not provided
            try:
                with open(file_path, encoding="utf-8") as f:
                    source_code = f.read()
            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {e}")
                return None

        if self.tree_cache:
            # Use cache for incremental parsing
            cached_tree = self.tree_cache.get(file_path)
            if cached_tree:
                return cached_tree

            # Cache miss - parse and cache
            tree = self._parse_tree_directly(file_path, source_code)
            if tree:
                self.tree_cache.put(file_path, tree)
            return tree
        else:
            # Fallback: parse directly without cache
            return self._parse_tree_directly(file_path, source_code)

    def _parse_tree_directly(self, file_path: Path, source_code: str) -> Any:
        """Parse source code directly using tree-sitter without cache."""
        try:
            from tree_sitter_language_pack import get_parser

            language = detect_language(file_path)

            # Simple direct parsing without cache
            if language == Language.PYTHON:
                parser = get_parser("python")
            elif language == Language.JAVA:
                parser = get_parser("java")
            elif language == Language.MARKDOWN:
                parser = get_parser("markdown")
            else:
                logger.warning(f"No direct parser available for {language}")
                return None

            if parser:
                return parser.parse(source_code.encode("utf-8"))
            else:
                return None
        except Exception as e:
            logger.warning(f"Direct parsing failed: {e}")
            return None

    def _get_node_text(self, node: Any, source_code: str) -> str:
        """Extract text content from a tree-sitter node."""
        if node and hasattr(node, "start_byte") and hasattr(node, "end_byte"):
            return source_code[node.start_byte : node.end_byte]
        return ""
