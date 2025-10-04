"""Batch file processor for parallel processing across CPU cores.

# FILE_CONTEXT: Worker function for ProcessPoolExecutor to parse files in parallel
# ROLE: Performs CPU-bound read→parse→chunk pipeline independently per batch
# CRITICAL: Must be picklable (top-level function, serializable arguments)
"""

import os
from dataclasses import dataclass
from pathlib import Path

from chunkhound.core.types.common import FileId, Language
from chunkhound.parsers.parser_factory import create_parser_for_language


@dataclass
class ParsedFileResult:
    """Result from processing a single file in a batch."""

    file_path: Path
    chunks: list[dict]
    language: Language
    file_size: int
    file_mtime: float
    status: str
    error: str | None = None


def process_file_batch(file_paths: list[Path], config_dict: dict) -> list[ParsedFileResult]:
    """Process a batch of files in a worker process.

    This function runs in a separate process via ProcessPoolExecutor.
    Performs the complete read→parse→chunk pipeline for all files in the batch.

    Args:
        file_paths: List of file paths to process in this batch
        config_dict: Serialized configuration dictionary for parser initialization

    Returns:
        List of ParsedFileResult objects with parsed chunks and metadata
    """
    results = []

    for file_path in file_paths:
        try:
            # Get file metadata
            file_stat = os.stat(file_path)

            # Detect language from file extension
            language = Language.from_file_extension(file_path)
            if language == Language.UNKNOWN:
                results.append(
                    ParsedFileResult(
                        file_path=file_path,
                        chunks=[],
                        language=language,
                        file_size=file_stat.st_size,
                        file_mtime=file_stat.st_mtime,
                        status="skipped",
                        error="Unknown file type",
                    )
                )
                continue

            # Skip large config/data files (config files are typically < 20KB)
            if language.is_structured_config_language:
                file_size_kb = file_stat.st_size / 1024
                threshold_kb = config_dict.get("config_file_size_threshold_kb", 20)
                if file_size_kb > threshold_kb:
                    results.append(
                        ParsedFileResult(
                            file_path=file_path,
                            chunks=[],
                            language=language,
                            file_size=file_stat.st_size,
                            file_mtime=file_stat.st_mtime,
                            status="skipped",
                            error="large_config_file",
                        )
                    )
                    continue

            # Create parser for this language
            parser = create_parser_for_language(language)
            if not parser:
                results.append(
                    ParsedFileResult(
                        file_path=file_path,
                        chunks=[],
                        language=language,
                        file_size=file_stat.st_size,
                        file_mtime=file_stat.st_mtime,
                        status="error",
                        error=f"No parser available for {language}",
                    )
                )
                continue

            # Parse file and generate chunks
            # Note: FileId(0) is placeholder - actual ID assigned during storage
            chunks = parser.parse_file(file_path, FileId(0))

            # Convert chunks to dictionaries for ProcessPoolExecutor serialization
            # Using standard Chunk.to_dict() method for consistent serialization
            chunks_data = [chunk.to_dict() for chunk in chunks]

            results.append(
                ParsedFileResult(
                    file_path=file_path,
                    chunks=chunks_data,
                    language=language,
                    file_size=file_stat.st_size,
                    file_mtime=file_stat.st_mtime,
                    status="success",
                )
            )

        except Exception as e:
            # Capture errors but continue processing other files in batch
            results.append(
                ParsedFileResult(
                    file_path=file_path,
                    chunks=[],
                    language=Language.UNKNOWN,
                    file_size=0,
                    file_mtime=0.0,
                    status="error",
                    error=str(e),
                )
            )

    return results
