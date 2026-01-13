"""Import context extraction for deep research synthesis.

This module provides utilities for extracting import statements from files
at synthesis time using the existing tree-sitter parser infrastructure.

The ImportContextService uses language-specific parsers to extract IMPORT
concepts, caching results to avoid redundant parsing during synthesis.

Usage:
    service = ImportContextService(parser_factory)

    # Extract imports from a file
    imports = service.get_file_imports(file_path, content)

    # Clear cache when needed
    service.clear_cache()
"""

from pathlib import Path
from typing import Any

from loguru import logger

from chunkhound.parsers.parser_factory import ParserFactory
from chunkhound.parsers.universal_engine import UniversalConcept


class ImportContextService:
    """Extract and cache file imports using existing parsers."""

    def __init__(self, parser_factory: ParserFactory):
        """Initialize import context service.

        Args:
            parser_factory: Factory for creating parsers
        """
        self._parser_factory = parser_factory
        self._import_cache: dict[str, list[str]] = {}

    def get_file_imports(self, file_path: str, content: str) -> list[str]:
        """Extract imports using language-specific parser.

        Delegates to existing tree-sitter parsers via ConceptExtractor.
        Each language mapping already defines IMPORT queries.

        Args:
            file_path: File path (for language detection and caching)
            content: File content to parse

        Returns:
            List of import statement strings (empty if no parser
            available or no imports)

        Examples:
            >>> service = ImportContextService(parser_factory)
            >>> imports = service.get_file_imports(
            ...     "src/main.py",
            ...     "import os\\nfrom pathlib import Path"
            ... )
            >>> imports
            ['import os', 'from pathlib import Path']
        """
        # Check cache first
        if file_path in self._import_cache:
            return self._import_cache[file_path]

        # Create parser for this file
        parser: Any = self._parser_factory.create_parser_for_file(
            Path(file_path)
        )

        # Check if parser has required attributes (engine, extractor)
        if not hasattr(parser, "engine") or not hasattr(parser, "extractor"):
            logger.debug(f"Parser lacks required attributes: {file_path}")
            return []

        try:
            # Parse content to get AST
            content_bytes = content.encode("utf-8")
            tree = parser.engine.parse_to_ast(content)
            if tree is None:
                logger.debug(f"Failed to parse file: {file_path}")
                return []

            # Extract imports using existing concept extraction
            # Uses language-specific tree-sitter queries
            import_chunks = parser.extractor.extract_concept(
                tree.root_node, content_bytes, UniversalConcept.IMPORT
            )

            # Extract content from chunks
            import_lines = [chunk.content for chunk in import_chunks]

            # Cache and return
            self._import_cache[file_path] = import_lines
            logger.debug(
                f"Extracted {len(import_lines)} imports from {file_path}"
            )
            return import_lines

        except Exception as e:
            logger.warning(
                f"Failed to extract imports from {file_path}: {e}"
            )
            return []

    def clear_cache(self) -> None:
        """Clear the import cache.

        Useful for freeing memory after synthesis or when file
        contents may have changed.
        """
        cache_size = len(self._import_cache)
        self._import_cache.clear()
        logger.debug(f"Cleared import cache ({cache_size} entries)")
