"""Import resolution service for deep research synthesis.

This module provides utilities for resolving import statements to their source files
by delegating to language-specific import resolution logic in parser mappings.

The ImportResolverService uses the existing tree-sitter parser infrastructure to
extract imports and then calls language-specific resolvers to map imports to file paths.

Architecture:
- Extract imports using ImportContextService (via tree-sitter parsers)
- Resolve each import using LanguageMapping.resolve_import_path()
- Cache resolved imports to avoid redundant resolution
- Handle external imports gracefully (returns None, skip from results)

Usage:
    service = ImportResolverService(parser_factory)

    # Resolve all imports in a file
    paths = await service.resolve_imports(
        file_path="src/main.py",
        content="import os\\nfrom pathlib import Path",
        base_dir=Path("/project")
    )

    # Clear cache when needed
    service.clear_cache()
"""

from pathlib import Path
from typing import Any

from loguru import logger

from chunkhound.parsers.parser_factory import ParserFactory
from chunkhound.services.research.shared.import_context import ImportContextService


class ImportResolverService:
    """Resolve import statements to source file paths."""

    def __init__(self, parser_factory: ParserFactory):
        """Initialize import resolver service.

        Args:
            parser_factory: Factory for creating parsers
        """
        self._parser_factory = parser_factory
        self._import_context_service = ImportContextService(parser_factory)
        # Cache: (file_path, import_text) -> resolved_path | None
        self._resolution_cache: dict[tuple[str, str], Path | None] = {}

    async def resolve_imports(
        self,
        file_path: str,
        content: str,
        base_dir: Path,
    ) -> list[Path]:
        """Resolve all imports in a file to their source file paths.

        Extracts import statements using ImportContextService, then resolves each
        import to a file path using language-specific mapping logic.

        Args:
            file_path: File path (for language detection and caching)
            content: File content to parse
            base_dir: Project root directory for import resolution

        Returns:
            List of resolved file paths (external/unresolvable imports skipped)

        Examples:
            >>> service = ImportResolverService(parser_factory)
            >>> paths = await service.resolve_imports(
            ...     "src/main.py",
            ...     "import utils\\nfrom pathlib import Path",
            ...     Path("/project")
            ... )
            >>> paths
            [Path('/project/src/utils.py')]  # pathlib is external, skipped
        """
        # Extract imports using ImportContextService
        import_statements = self._import_context_service.get_file_imports(
            file_path, content
        )

        if not import_statements:
            logger.debug(f"No imports found in {file_path}")
            return []

        # Create parser for this file to get language mapping
        parser: Any = self._parser_factory.create_parser_for_file(Path(file_path))

        # Check if parser has extractor with mapping
        if not hasattr(parser, "extractor") or not hasattr(
            parser.extractor, "mapping"
        ):
            logger.debug(f"Parser lacks mapping for import resolution: {file_path}")
            return []

        mapping = parser.extractor.mapping
        source_file = Path(file_path)

        # Resolve each import
        resolved_paths: list[Path] = []
        for import_text in import_statements:
            try:
                # Check cache first
                cache_key = (file_path, import_text)
                if cache_key in self._resolution_cache:
                    cached_path = self._resolution_cache[cache_key]
                    if cached_path is not None:
                        resolved_paths.append(cached_path)
                    continue

                # Call language-specific resolver
                resolved_path = mapping.resolve_import_path(
                    import_text, base_dir, source_file
                )

                # Cache result (including None for external imports)
                self._resolution_cache[cache_key] = resolved_path

                # Add to results if resolved (skip None)
                if resolved_path is not None:
                    resolved_paths.append(resolved_path)
                    logger.debug(
                        f"Resolved import '{import_text}' -> {resolved_path}"
                    )
                else:
                    logger.debug(
                        f"Skipped external/unresolvable import: '{import_text}'"
                    )

            except Exception as e:
                logger.warning(
                    f"Failed to resolve import '{import_text}' in {file_path}: {e}"
                )
                # Cache None to avoid retrying failed resolutions
                self._resolution_cache[(file_path, import_text)] = None
                continue

        logger.debug(
            f"Resolved {len(resolved_paths)}/{len(import_statements)} imports "
            f"in {file_path}"
        )
        return resolved_paths

    def clear_cache(self) -> None:
        """Clear all caches (import extraction and resolution).

        Useful for freeing memory after synthesis or when file contents
        may have changed.
        """
        import_cache_size = len(self._import_context_service._import_cache)
        resolution_cache_size = len(self._resolution_cache)

        self._import_context_service.clear_cache()
        self._resolution_cache.clear()

        logger.debug(
            f"Cleared import resolver caches "
            f"(imports: {import_cache_size}, resolutions: {resolution_cache_size})"
        )
