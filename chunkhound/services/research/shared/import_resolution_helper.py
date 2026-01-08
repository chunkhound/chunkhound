"""Shared helper for import resolution across V2 research phases.

This module provides a shared implementation of import resolution to avoid
code duplication across Phase 1, 1.5, 2, and 3 services.

The helper resolves imports in retrieved chunks and fetches additional chunks
from the import source files, providing cross-file dependency context for
better research quality.
"""

from pathlib import Path
from typing import Any, cast

from loguru import logger


async def resolve_and_fetch_imports(
    chunks: list[dict],
    import_resolver: Any,  # ImportResolverService
    db_services: Any,  # DatabaseServices
    config: Any,  # ResearchConfig
    path_filter: str | None = None,
    default_score: float = 0.3,
) -> list[dict]:
    """Resolve imports and fetch chunks from source files.

    Shared across Phase 1, 1.5, 2, and 3 to avoid code duplication.

    This function:
    1. Extracts unique file paths from input chunks
    2. Resolves imports for each file using ImportResolverService
    3. Fetches chunks from resolved import source files
    4. Assigns default rerank score to import chunks
    5. Returns deduplicated list of import chunks

    Args:
        chunks: Chunks to analyze for imports (must have file_path)
        import_resolver: Import resolver service for resolving imports
        db_services: Database services for file/chunk access
        config: Research config (needs import_resolution_max_files attribute)
        path_filter: Optional path filter for scope limiting (e.g., "src/")
        default_score: Default rerank score for import chunks (0.2-0.3)

    Returns:
        List of chunks from import source files, with rerank_score added

    Example:
        >>> import_chunks = await resolve_and_fetch_imports(
        ...     chunks=phase1_chunks,
        ...     import_resolver=self._import_resolver,
        ...     db_services=self._db_services,
        ...     config=self._config,
        ...     path_filter="src/",
        ...     default_score=0.3,
        ... )
    """
    if not import_resolver:
        return []

    # Get base directory for import resolution
    base_dir = db_services.provider.get_base_directory()

    # Get unique file paths from chunks
    file_paths: set[str] = set()
    for c in chunks:
        fp = c.get("file_path")
        if fp is not None:
            file_paths.add(fp)

    # Resolve imports for each file
    import_files: set[Path] = set()
    for file_path in file_paths:
        try:
            # Read file content
            if Path(file_path).is_absolute():
                path = Path(file_path)
            else:
                path = base_dir / file_path

            if not path.exists():
                logger.debug(f"File not found for import resolution: {file_path}")
                continue

            content_text = path.read_text(encoding="utf-8", errors="ignore")

            # Resolve imports
            resolved = await import_resolver.resolve_imports(
                file_path, content_text, base_dir
            )

            # Add resolved files (limit to max)
            for resolved_path in resolved[: config.import_resolution_max_files]:
                # Apply path filter if set
                if path_filter:
                    try:
                        relative_str = str(resolved_path.relative_to(base_dir))
                    except ValueError:
                        continue  # Path not under base_dir
                    if not relative_str.startswith(path_filter):
                        continue
                import_files.add(resolved_path)

                # Stop if we hit the limit
                if len(import_files) >= config.import_resolution_max_files:
                    break

            if len(import_files) >= config.import_resolution_max_files:
                break

        except Exception as e:
            logger.warning(f"Import resolution failed for {file_path}: {e}")
            continue

    # Fetch chunks from import source files
    import_chunks: list[dict] = []
    for import_file in list(import_files)[: config.import_resolution_max_files]:
        try:
            # Convert to relative path
            relative_path = str(import_file.relative_to(base_dir))

            # Get file record
            file_record_result = db_services.provider.get_file_by_path(
                relative_path, as_model=False
            )

            if not file_record_result:
                logger.debug(f"Import file not indexed: {relative_path}")
                continue
            file_record = cast(dict, file_record_result)

            # Fetch all chunks for this file
            file_chunks_result = db_services.provider.get_chunks_by_file_id(
                file_record["id"], as_model=False
            )

            # Add default score for import chunks
            for chunk_result in file_chunks_result:
                chunk = cast(dict, chunk_result)
                chunk["rerank_score"] = default_score
                import_chunks.append(chunk)

        except Exception as e:
            logger.warning(f"Failed to fetch chunks for import {import_file}: {e}")
            continue

    logger.debug(
        f"Import resolution: fetched {len(import_chunks)} chunks from "
        f"{len(import_files)} import files"
    )

    return import_chunks
