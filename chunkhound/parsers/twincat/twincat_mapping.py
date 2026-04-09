"""TwinCAT mapping for UniversalParser integration.

This module provides the TwinCATMapping class that enables TwinCAT Structured Text
files to be processed through the UniversalParser pipeline, benefiting from
chunk deduplication, cAST algorithm optimization, and comment merging.

Unlike other mappings that use tree-sitter queries, this mapping delegates
to TwinCATParser (Lark-based) and provides an `extract_universal_chunks()`
method that UniversalParser calls when engine=None.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Any

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.base import BaseMapping
from chunkhound.parsers.twincat.twincat_parser import TwinCATParser
from chunkhound.parsers.universal_engine import UniversalChunk

# Import primitive types from parser (single source of truth)
_PRIMITIVE_TYPES = TwinCATParser.PRIMITIVE_TYPES

# IEC 61131-3 Standard Library function blocks to skip
_STDLIB_TYPES = frozenset({
    "TON", "TOF", "TP", "RTC",           # Timers
    "CTU", "CTD", "CTUD",                # Counters
    "R_TRIG", "F_TRIG",                  # Triggers
    "SR", "RS",                          # Flip-flops
})

# File extension for TcPOU files (Function Blocks, Interfaces, Programs, Functions)
_TWINCAT_EXTENSION = ".TcPOU"

@functools.lru_cache(maxsize=1)
def _get_parser() -> TwinCATParser:
    return TwinCATParser()


class TwinCATMapping(BaseMapping):
    """Mapping for TwinCAT Structured Text via Lark parser.

    Unlike other mappings that use tree-sitter queries, this mapping
    uses TwinCATParser (Lark-based) to extract UniversalChunk objects.

    The mapping provides `extract_universal_chunks()` which UniversalParser
    calls when engine=None, enabling TwinCAT files to benefit from the
    full cAST pipeline (deduplication, comment merging, greedy merge).
    """

    def __init__(self) -> None:
        """Initialize TwinCAT mapping."""
        super().__init__(Language.TWINCAT)

    def extract_universal_chunks(
        self,
        content: str,
        file_path: Path | None = None,
    ) -> list[UniversalChunk]:
        """Extract UniversalChunk objects from TcPOU content.

        Called by UniversalParser when engine is None and this method exists.

        Args:
            content: TcPOU XML content string
            file_path: Optional path to the source file

        Returns:
            List of UniversalChunk objects for cAST processing
        """
        parser = _get_parser()
        return parser.extract_universal_chunks(content, file_path)

    def extract_imports(
        self,
        content: str,
    ) -> list[UniversalChunk]:
        """Extract only import chunks from TcPOU content.

        More efficient than extract_universal_chunks() when only
        imports are needed.

        Args:
            content: TcPOU XML content string

        Returns:
            List of UniversalChunk objects representing imports
        """
        parser = _get_parser()
        return parser.extract_import_chunks(content)

    def _extract_symbol_name(self, import_text: str) -> str:
        """Extract symbol name from import text.

        Handles formats like:
        - "FB_Motor" (direct symbol)
        - "gMotor : FB_Motor;" (var declaration)
        - "EXTENDS FB_Base" (inheritance)
        - "type reference: ST_Config" (type ref metadata)
        - "pMotor : POINTER TO FB_Motor;" (pointer)
        - "refMotor : REFERENCE TO FB_Motor;" (reference)
        - "pArray : POINTER TO ARRAY[0..9] OF FB_Motor;" (nested)
        """
        # Extract type part after colon if present (var declaration or type ref)
        if ":" in import_text:
            parts = import_text.split(":")
            type_part = parts[-1].strip().rstrip(";").strip()
        else:
            type_part = import_text.strip()

        upper_part = type_part.upper()

        # Handle EXTENDS/IMPLEMENTS keywords
        for keyword in ["EXTENDS", "IMPLEMENTS"]:
            if keyword in upper_part:
                pos = upper_part.rfind(keyword)
                type_part = type_part[pos + len(keyword) :].strip()
                upper_part = type_part.upper()
                break

        # Handle ARRAY types FIRST - extract element type after last OF
        of_matches = list(re.finditer(r"\bOF\b", upper_part))
        if of_matches:
            of_pos = of_matches[-1].start()  # Last match for nested arrays
            type_part = type_part[of_pos + 2 :].strip()
            upper_part = type_part.upper()

        # THEN strip POINTER TO / REFERENCE TO prefixes (may be nested)
        while True:
            stripped = False
            for keyword in ["POINTER TO", "REFERENCE TO"]:
                if upper_part.startswith(keyword):
                    type_part = type_part[len(keyword) :].strip()
                    upper_part = type_part.upper()
                    stripped = True
                    break
            if not stripped:
                break

        return type_part

    def _find_symbol_file(self, symbol: str, base_dir: Path) -> Path | None:
        """Search for symbol file with case-insensitive matching."""
        symbol_lower = symbol.lower()

        for file_path in base_dir.rglob(f"*{_TWINCAT_EXTENSION}"):
            if file_path.stem.lower() == symbol_lower:
                return file_path
        return None

    def resolve_import_paths(
        self,
        import_text: str,
        base_dir: Path,
        source_file: Path,
    ) -> list[Path]:
        """Resolve TwinCAT import symbol to .TcPOU file paths.

        Unlike Python/JS with `import "file.py"` syntax, TwinCAT uses
        symbol-based imports (e.g., `FB_Motor`). This method maps symbol
        names to `.TcPOU` files.

        Args:
            import_text: Import text (symbol name or declaration)
            base_dir: Base directory to search for files
            source_file: Path to the source file containing the import

        Returns:
            List of paths to resolved .TcPOU files (empty if not found)
        """
        # Extract symbol name from various import formats
        symbol = self._extract_symbol_name(import_text)
        if not symbol:
            return []

        symbol_upper = symbol.upper()

        # Skip primitive types (BOOL, DINT, etc.)
        if symbol_upper in _PRIMITIVE_TYPES:
            return []

        # Skip standard library types (TON, CTU, etc.)
        if symbol_upper in _STDLIB_TYPES:
            return []

        # Search from source file's directory rather than base_dir.
        # While base_dir would provide complete coverage, searching the entire
        # project tree on every import resolution is expensive. We compromise
        # by searching from source_file's directory, which covers most cases
        # since TwinCAT projects typically co-locate related POUs.
        result = self._find_symbol_file(symbol, source_file.parent)
        if result is not None:
            return [result.resolve()]  # Ensure absolute path
        return []

    # Required abstract method implementations (not used for TwinCAT)
    # These are required by BaseMapping but TwinCAT uses Lark instead of tree-sitter

    def get_function_query(self) -> str:
        """Not used - TwinCAT uses Lark parser, not tree-sitter queries."""
        return ""

    def get_class_query(self) -> str:
        """Not used - TwinCAT uses Lark parser, not tree-sitter queries."""
        return ""

    def get_comment_query(self) -> str:
        """Not used - TwinCAT uses Lark parser, not tree-sitter queries."""
        return ""

    def extract_function_name(self, node: Any, source: str) -> str:
        """Not used - TwinCAT uses Lark parser, not tree-sitter nodes."""
        return ""

    def extract_class_name(self, node: Any, source: str) -> str:
        """Not used - TwinCAT uses Lark parser, not tree-sitter nodes."""
        return ""
