"""LanguageParser protocol for ChunkHound -
abstract interface for language parsing implementations."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from chunkhound.core.types.common import ChunkType, Language


@dataclass
class ParseConfig:
    """Configuration for language parsers."""

    language: Language
    chunk_types: set[ChunkType]
    max_chunk_size: int = 8000
    min_chunk_size: int = 100
    include_imports: bool = True
    include_comments: bool = False
    include_docstrings: bool = True
    max_depth: int = 10
    use_cache: bool = True


@dataclass
class ParseResult:
    """Result from parsing operation."""

    chunks: list[dict[str, Any]]
    language: Language
    total_chunks: int
    parse_time: float
    errors: list[str]
    warnings: list[str]
    metadata: dict[str, Any]


class LanguageParser(Protocol):
    """Abstract protocol for language-specific parsers.

    Defines the interface that all language parsing implementations must follow.
    This enables pluggable language support (Python, Java, C#, TypeScript, etc.)
    """

    @property
    def language(self) -> Language:
        """Programming language this parser handles."""
        ...

    @property
    def supported_extensions(self) -> set[str]:
        """File extensions supported by this parser (e.g., {'.py', '.pyi'})."""
        ...

    @property
    def supported_chunk_types(self) -> set[ChunkType]:
        """Chunk types this parser can extract."""
        ...

    @property
    def is_initialized(self) -> bool:
        """Check if parser is properly initialized and ready to use."""
        ...

    @property
    def config(self) -> ParseConfig:
        """Parser configuration."""
        ...

    # Core Parsing Operations
    def parse_file(self, file_path: Path) -> list[dict[str, Any]]:
        """Parse a file and extract semantic chunks.

        Args:
            file_path: Path to the file to parse

        Returns:
            List of chunk dictionaries with standardized structure

        Raises:
            ParseError: If parsing fails
            FileNotFoundError: If file doesn't exist
        """
        ...

    def parse_content(
        self, content: str, file_path: Path | None = None
    ) -> list[dict[str, Any]]:
        """Parse content string and extract semantic chunks.

        Args:
            content: Source code content to parse
            file_path: Optional file path for context/error reporting

        Returns:
            List of chunk dictionaries with standardized structure

        Raises:
            ParseError: If parsing fails
        """
        ...

    def parse_with_result(self, file_path: Path) -> ParseResult:
        """Parse a file and return detailed result information.

        Args:
            file_path: Path to the file to parse

        Returns:
            ParseResult with chunks, metadata, and diagnostics

        Raises:
            ParseError: If parsing fails
        """
        ...

    # Incremental Parsing Support
    def supports_incremental_parsing(self) -> bool:
        """Check if this parser supports incremental parsing."""
        ...

    def parse_incremental(
        self, file_path: Path, previous_chunks: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        """Parse file incrementally, reusing previous results when possible.

        Args:
            file_path: Path to the file to parse
            previous_chunks: Previously parsed chunks for comparison

        Returns:
            List of chunk dictionaries (only changed chunks if incremental)

        Raises:
            ParseError: If parsing fails
        """
        ...

    def get_parse_tree(self, content: str) -> Any:
        """Get the raw parse tree for content (for debugging/advanced use).

        Args:
            content: Source code content to parse

        Returns:
            Raw parse tree object (tree-sitter Tree or equivalent)

        Raises:
            ParseError: If parsing fails
        """
        ...

    # Parser Management
    def setup(self) -> None:
        """Initialize the parser (load grammars, prepare resources)."""
        ...

    def cleanup(self) -> None:
        """Cleanup parser resources."""
        ...

    def reset(self) -> None:
        """Reset parser state (clear caches, etc.)."""
        ...

    # Validation and Detection
    def can_parse_file(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file.

        Args:
            file_path: Path to check

        Returns:
            True if this parser can handle the file
        """
        ...

    def detect_language(self, file_path: Path) -> Language | None:
        """Detect the programming language of a file.

        Args:
            file_path: Path to analyze

        Returns:
            Detected language or None if not supported
        """
        ...

    def validate_syntax(self, content: str) -> list[str]:
        """Validate syntax and return list of errors.

        Args:
            content: Source code content to validate

        Returns:
            List of syntax error messages (empty if valid)
        """
        ...

    # Chunk Type Specific Operations
    def extract_functions(self, content: str) -> list[dict[str, Any]]:
        """Extract only function chunks from content."""
        ...

    def extract_classes(self, content: str) -> list[dict[str, Any]]:
        """Extract only class chunks from content."""
        ...

    def extract_imports(self, content: str) -> list[dict[str, Any]]:
        """Extract import/dependency information."""
        ...

    def extract_comments(self, content: str) -> list[dict[str, Any]]:
        """Extract comment chunks (if enabled in config)."""
        ...

    # Metadata and Information
    def get_parser_info(self) -> dict[str, Any]:
        """Get information about this parser implementation."""
        ...

    def get_statistics(self) -> dict[str, Any]:
        """Get parsing statistics (files parsed, chunks extracted, etc.)."""
        ...

    def reset_statistics(self) -> None:
        """Reset parsing statistics."""
        ...

    # Configuration Management
    def update_config(self, **kwargs: Any) -> None:
        """Update parser configuration.

        Args:
            **kwargs: Configuration parameters to update
        """
        ...

    def get_default_config(self) -> ParseConfig:
        """Get default configuration for this parser."""
        ...

    # Cache Integration
    def enable_cache(self, cache_instance: Any = None) -> None:
        """Enable caching for performance optimization."""
        ...

    def disable_cache(self) -> None:
        """Disable caching."""
        ...

    def clear_cache(self) -> None:
        """Clear parser cache."""
        ...

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        ...


class TreeSitterParser(LanguageParser, Protocol):
    """Extended protocol for tree-sitter based parsers."""

    @property
    def tree_sitter_language(self) -> Any:
        """Tree-sitter Language object."""
        ...

    @property
    def tree_sitter_parser(self) -> Any:
        """Tree-sitter Parser object."""
        ...

    def get_node_text(self, node: Any, content: bytes) -> str:
        """Extract text content from a tree-sitter node."""
        ...

    def find_nodes_by_type(self, tree: Any, node_type: str) -> list[Any]:
        """Find all nodes of a specific type in the parse tree."""
        ...

    def walk_tree(self, node: Any, visitor_func: Callable[[Any], None]) -> None:
        """Walk the parse tree with a visitor function."""
        ...


class RegexParser(LanguageParser, Protocol):
    """Extended protocol for regex-based parsers (fallback/simple languages)."""

    @property
    def patterns(self) -> dict[ChunkType, str]:
        """Regex patterns for extracting chunks by type."""
        ...

    def add_pattern(self, chunk_type: ChunkType, pattern: str) -> None:
        """Add a new regex pattern for a chunk type."""
        ...

    def remove_pattern(self, chunk_type: ChunkType) -> None:
        """Remove a regex pattern for a chunk type."""
        ...

    def test_pattern(self, pattern: str, content: str) -> list[dict[str, Any]]:
        """Test a regex pattern against content."""
        ...
