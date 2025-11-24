"""Oxc-backed JavaScript/TypeScript parser with CAST optimization.

This module provides the OxcParser class that uses oxc-python for parsing
JavaScript, TypeScript, JSX, and TSX files. Oxc is 10-30x faster than
tree-sitter for JS/TS parsing.

The parser includes the CAST (Contextual AST) algorithm for optimal semantic
chunking, ensuring minified files are handled correctly and all chunks respect
size limits.

ParserFactory uses this as the primary parser for JavaScript-family languages
when oxc-python is available, with automatic fallback to tree-sitter if unavailable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import oxc_python

from chunkhound.core.models.chunk import Chunk
from chunkhound.core.types.common import (
    ByteOffset,
    ChunkType,
    FileId,
    FilePath,
    Language,
    LineNumber,
)
from chunkhound.core.utils import estimate_tokens
from chunkhound.interfaces.language_parser import LanguageParser, ParseResult
from chunkhound.parsers.universal_engine import UniversalChunk, UniversalConcept
from chunkhound.parsers.universal_parser import CASTConfig
from chunkhound.utils.normalization import normalize_content

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetrics:
    """Metrics for measuring chunk quality and size."""

    non_whitespace_chars: int
    total_chars: int
    lines: int
    ast_depth: int

    @classmethod
    def from_content(cls, content: str, ast_depth: int = 0) -> "ChunkMetrics":
        """Calculate metrics from content string."""
        non_ws = len(re.sub(r"\s", "", content))
        total = len(content)
        lines = len(content.split("\n"))
        return cls(non_ws, total, lines, ast_depth)

    def estimated_tokens(self, ratio: float = 3.5) -> int:
        """Estimate token count using character-based ratio."""
        return int(self.non_whitespace_chars / ratio)


def _byte_offset_to_char(source: str, byte_offset: int) -> int:
    """Convert UTF-8 byte offset to character offset.

    Oxc returns byte offsets, but Python string indexing uses character offsets.
    For ASCII, bytes == characters. For Unicode, we need to convert.

    Args:
        source: The source code string
        byte_offset: Byte offset from oxc

    Returns:
        Character offset for Python string slicing
    """
    byte_pos = 0
    for char_pos, char in enumerate(source):
        if byte_pos >= byte_offset:
            return char_pos
        byte_pos += len(char.encode("utf-8"))
    return len(source)  # Clamp to string length


# Node type to ChunkType mapping (kept for metadata purposes)
NODE_TYPE_MAP: dict[str, ChunkType] = {
    "FunctionDeclaration": ChunkType.FUNCTION,
    "ClassDeclaration": ChunkType.CLASS,
    "MethodDefinition": ChunkType.METHOD,
    "ArrowFunctionExpression": ChunkType.FUNCTION,
    "VariableDeclaration": ChunkType.VARIABLE,
    "ImportDeclaration": ChunkType.UNKNOWN,
    "ExportNamedDeclaration": ChunkType.FUNCTION,
    "ExportDefaultDeclaration": ChunkType.FUNCTION,
    "TSTypeAliasDeclaration": ChunkType.TYPE_ALIAS,
    "TSInterfaceDeclaration": ChunkType.INTERFACE,
    "TSEnumDeclaration": ChunkType.ENUM,
    "TSModuleDeclaration": ChunkType.NAMESPACE,
    "ExpressionStatement": ChunkType.VARIABLE,  # For module.exports = ...
}

# Node type to UniversalConcept mapping (for CAST integration)
NODE_TYPE_TO_CONCEPT: dict[str, UniversalConcept] = {
    # Definitions: functions, classes, methods, types
    "FunctionDeclaration": UniversalConcept.DEFINITION,
    "ClassDeclaration": UniversalConcept.DEFINITION,
    "MethodDefinition": UniversalConcept.DEFINITION,
    "ArrowFunctionExpression": UniversalConcept.DEFINITION,
    "VariableDeclaration": UniversalConcept.DEFINITION,
    "ExportNamedDeclaration": UniversalConcept.DEFINITION,
    "ExportDefaultDeclaration": UniversalConcept.DEFINITION,
    "TSTypeAliasDeclaration": UniversalConcept.DEFINITION,
    "TSInterfaceDeclaration": UniversalConcept.DEFINITION,
    "TSEnumDeclaration": UniversalConcept.DEFINITION,
    "TSModuleDeclaration": UniversalConcept.STRUCTURE,

    # Expression statements (module.exports, IIFEs)
    "ExpressionStatement": UniversalConcept.DEFINITION,

    # Imports
    "ImportDeclaration": UniversalConcept.IMPORT,
}

# Node types we want to extract as chunks
CHUNK_NODE_TYPES: set[str] = {
    "FunctionDeclaration",
    "ClassDeclaration",
    "MethodDefinition",
    "ArrowFunctionExpression",
    "VariableDeclaration",
    "ImportDeclaration",
    "ExportNamedDeclaration",
    "ExportDefaultDeclaration",
    "TSTypeAliasDeclaration",
    "TSInterfaceDeclaration",
    "TSEnumDeclaration",
    "TSModuleDeclaration",
    "ExpressionStatement",
}

# Export node types that wrap other declarations - we extract the export as a whole
# and skip the nested declaration to avoid duplication
EXPORT_WRAPPER_TYPES: set[str] = {
    "ExportNamedDeclaration",
    "ExportDefaultDeclaration",
}


class OxcParser(LanguageParser):
    """Oxc-backed JS/TS/JSX/TSX parser with CAST optimization.

    Uses oxc-python for fast parsing of JavaScript-family languages.
    Oxc is 10-30x faster than tree-sitter for these languages.

    Includes CAST (Contextual AST) algorithm for optimal semantic chunking:
    - Minified files are handled correctly (IIFE detection + emergency splitting)
    - All chunks respect size limits (1200 chars, 6000 tokens)
    - Performance remains excellent (10-30x faster than tree-sitter)
    """

    def __init__(self, language: Language, cast_config: CASTConfig | None = None) -> None:
        """Initialize Oxc parser with CAST optimization.

        Args:
            language: The language this parser handles (JS, TS, JSX, TSX)
            cast_config: Configuration for CAST algorithm (uses defaults if not provided)
        """
        self._language = language
        self._allocator = oxc_python.Allocator()
        self.cast_config = cast_config or CASTConfig()

        # Performance counters
        self._count_parsed = 0
        self._count_errors = 0
        self._t_parse = 0.0
        self._t_walk = 0.0
        self._t_build = 0.0

        # Determine supported extensions based on language
        self._extensions = self._get_extensions_for_language(language)

    def _get_extensions_for_language(self, language: Language) -> set[str]:
        """Get file extensions for a language."""
        extension_map = {
            Language.JAVASCRIPT: {".js", ".mjs", ".cjs"},
            Language.TYPESCRIPT: {".ts", ".mts", ".cts"},
            Language.JSX: {".jsx"},
            Language.TSX: {".tsx"},
        }
        return extension_map.get(language, set())

    @property
    def language(self) -> Language:
        return self._language

    @property
    def language_name(self) -> str:
        """Return language name for compatibility with tree-sitter parsers."""
        return self._language.value

    @property
    def supported_extensions(self) -> set[str]:
        return self._extensions

    @property
    def supported_chunk_types(self) -> set[ChunkType]:
        return set(NODE_TYPE_MAP.values())

    @property
    def is_initialized(self) -> bool:
        return True

    def parse_file(self, file_path: Path, file_id: FileId) -> list[Chunk]:
        """Parse a file and extract semantic chunks.

        Args:
            file_path: Path to the file to parse
            file_id: Unique identifier for this file

        Returns:
            List of Chunk objects
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                try:
                    content = file_path.read_text(encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise UnicodeDecodeError(
                    "utf-8", b"", 0, 1, f"Could not decode file {file_path}"
                ) from e

        content = normalize_content(content)
        return self.parse_content(content, file_path, file_id)

    def parse_content(
        self,
        content: str,
        file_path: Path | None = None,
        file_id: FileId | None = None,
    ) -> list[Chunk]:
        """Parse content string and extract optimized semantic chunks.

        This method:
        1. Extracts UniversalChunks using Oxc AST
        2. Applies CAST algorithm for optimization
        3. Converts to final Chunk objects

        Args:
            content: Source code content to parse
            file_path: Optional path to the source file
            file_id: Optional unique identifier for the file

        Returns:
            List of optimized Chunk objects
        """
        if not content.strip():
            return []

        # Step 1: Extract initial UniversalChunks from Oxc AST
        universal_chunks, parse_succeeded = self._parse_content_to_universal(content, file_path, file_id)

        # Step 1.5: FALLBACK - Only if AST parsing actually failed (not just filtered out)
        if not universal_chunks and not parse_succeeded and content.strip():
            logger.warning(
                f"AST parsing failed for {file_path or 'content'}, using text-based fallback"
            )
            universal_chunks = [
                UniversalChunk(
                    concept=UniversalConcept.BLOCK,
                    name="unparseable_content",
                    content=content,
                    start_line=1,
                    end_line=len(content.split("\n")),
                    metadata={"fallback": "text_chunking", "reason": "parse_failure"},
                    language_node_type="text",
                )
            ]

        # Step 2: Apply CAST algorithm for optimization
        optimized_chunks = self._apply_cast_algorithm(universal_chunks, content)

        # Step 3: Convert to final Chunks
        fid = file_id if file_id is not None else FileId(0)
        return self._convert_to_chunks(optimized_chunks, content, file_path, fid)

    def _parse_content_to_universal(
        self,
        content: str,
        file_path: Path | None = None,
        file_id: FileId | None = None,
    ) -> tuple[list[UniversalChunk], bool]:
        """Internal method: Parse content to UniversalChunks.

        This is used internally and by OxcWithCAST wrapper.
        Public methods convert these to dicts for protocol compliance.

        Args:
            content: Source code content to parse
            file_path: Optional file path for metadata
            file_id: Optional file identifier

        Returns:
            Tuple of (list of UniversalChunk objects, parse_succeeded boolean)
        """
        if not content.strip():
            return [], True  # Empty content is considered successful

        # Use provided file_id or default to 0
        fid = file_id if file_id is not None else FileId(0)

        # Detect source type from file extension
        source_type = self._detect_source_type(file_path)

        # Parse with Oxc
        t0 = perf_counter()
        self._allocator.reset()
        result = oxc_python.parse(content, source_type=source_type, allocator=self._allocator)
        self._t_parse += perf_counter() - t0

        if not result.is_valid:
            self._count_errors += 1
            # Log errors but still try to extract what we can
            for error in result.errors:
                # Calculate line from span (no direct line attribute)
                error_line = content[:error.span.start].count('\n') + 1 if error.span.start > 0 else 1
                logger.debug(
                    "Oxc parse error at line %d: %s",
                    error_line,
                    error.message,
                )

        # Walk AST and build chunks
        t0 = perf_counter()
        chunks = self._build_chunks(result, content, file_path, fid)
        self._t_walk += perf_counter() - t0

        self._count_parsed += 1
        return chunks, result.is_valid

    def _universal_to_chunk(
        self, chunk: UniversalChunk, file_id: FileId, file_path: Path | None
    ) -> Chunk:
        """Convert UniversalChunk to Chunk object.

        Args:
            chunk: UniversalChunk to convert
            file_id: File ID for the chunk
            file_path: Optional file path

        Returns:
            Chunk object
        """
        # Get metadata (chunk_type_hint should be there as a ChunkType enum value string)
        metadata = chunk.metadata or {}
        chunk_type_str = metadata.get("chunk_type_hint", "unknown")

        # Convert chunk_type string back to ChunkType enum
        try:
            chunk_type = ChunkType(chunk_type_str)
        except (ValueError, KeyError):
            chunk_type = ChunkType.UNKNOWN

        # Build Chunk object
        return Chunk(
            symbol=chunk.name,
            start_line=LineNumber(chunk.start_line),
            end_line=LineNumber(chunk.end_line),
            code=chunk.content,
            chunk_type=chunk_type,
            file_id=file_id,
            language=self._language,
            file_path=str(file_path) if file_path else None,
            metadata=metadata,
        )

    def _is_extractable_assignment(self, code_text: str) -> bool:
        """Check if an ExpressionStatement is an assignment we should extract.

        Args:
            code_text: The source code text of the expression statement

        Returns:
            True if this is a module.exports, prototype, or static assignment
        """
        # Module.exports assignments (CommonJS)
        if code_text.startswith("module.exports"):
            return True

        # Must be an assignment to continue
        if "=" not in code_text:
            return False

        # Prototype assignments: Constructor.prototype.method = ...
        if ".prototype." in code_text:
            return True

        # Static assignments: Constructor.method = ...
        # (but not variable declarations like const x = ...)
        if not code_text.startswith(("const ", "let ", "var ")):
            left_side = code_text.split("=")[0]
            if "." in left_side:  # Has property access
                return True

        return False

    def _is_iife_pattern(self, code_text: str) -> bool:
        """Check if an ExpressionStatement is an IIFE (Immediately Invoked Function Expression).

        Handles common minification patterns:
        - !function(){...}()
        - (function(){...}())
        - +function(){...}(), -function(){...}(), ~function(){...}()
        - void function(){...}()
        - Async variants

        Args:
            code_text: The source code text of the expression statement

        Returns:
            True if this is an IIFE pattern that should be extracted
        """
        code_text = code_text.strip()

        # Pattern 1: Unary operator IIFEs (!function, +function, -function, ~function)
        if re.match(r'^[!+\-~]\s*(?:async\s+)?function\s*\(', code_text):
            return True

        # Pattern 2: void function IIFE
        if re.match(r'^void\s+(?:async\s+)?function\s*\(', code_text):
            return True

        # Pattern 3: Wrapped IIFE (function(){...}())
        if re.match(r'^\(\s*(?:async\s+)?function\s*\(', code_text):
            return True

        # Pattern 4: Arrow function IIFE (()=>{...}())
        if re.match(r'^[!+\-~]?\s*\(\s*(?:async\s+)?\([^)]*\)\s*=>', code_text):
            return True

        return False

    def _detect_source_type(self, file_path: Path | str | None) -> str:
        """Detect Oxc source type from file extension.

        Args:
            file_path: Optional file path (Path or string)

        Returns:
            Source type string for oxc_python.parse()
        """
        if file_path:
            # Handle both Path and string
            if isinstance(file_path, str):
                file_path = Path(file_path)
            ext = file_path.suffix.lower()
            if ext == ".tsx":
                return "tsx"
            elif ext == ".jsx":
                return "jsx"
            elif ext in {".ts", ".mts", ".cts"}:
                return "typescript"
            elif ext in {".js", ".mjs", ".cjs"}:
                return "module"

        # Default based on language
        if self._language == Language.TSX:
            return "tsx"
        elif self._language == Language.JSX:
            return "jsx"
        elif self._language == Language.TYPESCRIPT:
            return "typescript"
        else:
            return "module"

    def _build_chunks(
        self,
        result: oxc_python.ParseResult,
        source: str,
        file_path: Path | None,
        file_id: FileId,
    ) -> list[UniversalChunk]:
        """Build UniversalChunk objects from Oxc AST for CAST processing.

        Args:
            result: Oxc parse result
            source: Original source code
            file_path: Optional file path
            file_id: File ID for chunks

        Returns:
            List of UniversalChunk objects
        """
        chunks = []
        # Track byte ranges of export wrappers to skip nested declarations
        export_spans: list[tuple[int, int]] = []

        # Build list of JSDoc comments (block comments starting with *)
        # These will be attached to following declarations
        jsdoc_comments: list[tuple[int, int]] = []
        for comment in result.comments:
            if comment.is_block and comment.text.startswith("*"):
                # Comment span includes /** and */, so use as-is
                jsdoc_comments.append((comment.span.start, comment.span.end))

        for node, depth in oxc_python.walk(result.program):
            node_type = node.type

            # Filter for relevant node types
            if node_type not in CHUNK_NODE_TYPES:
                continue

            # Only extract ExpressionStatements that are relevant assignments OR IIFEs
            # (module.exports, prototype methods, static methods/properties, or IIFE patterns)
            if node_type == "ExpressionStatement":
                code_text = node.get_text(source).strip()

                # Accept if it's an extractable assignment (module.exports, prototype, etc.)
                is_assignment = self._is_extractable_assignment(code_text)

                # OR if it's an IIFE pattern (minified code)
                is_iife = self._is_iife_pattern(code_text)

                if not (is_assignment or is_iife):
                    continue  # Skip this ExpressionStatement

            # Convert byte offsets to character offsets for Python string slicing
            # Keep original byte offsets for metadata
            node_byte_start = node.span.start
            node_byte_end = node.span.end
            node_start = _byte_offset_to_char(source, node_byte_start)
            node_end = _byte_offset_to_char(source, node_byte_end)

            # Track whether JSDoc was attached (for line number calculation)
            jsdoc_attached = False

            # Check for preceding JSDoc comment and extend span to include it
            # A JSDoc is "attached" if only whitespace exists between comment end and node start
            for jsdoc_byte_start, jsdoc_byte_end in jsdoc_comments:
                jsdoc_start = _byte_offset_to_char(source, jsdoc_byte_start)
                jsdoc_end = _byte_offset_to_char(source, jsdoc_byte_end)

                if jsdoc_end <= node_start:
                    # Check if only whitespace between comment and node
                    between = source[jsdoc_end:node_start]
                    if between.strip() == "":
                        node_start = jsdoc_start
                        node_byte_start = jsdoc_byte_start
                        jsdoc_attached = True
                        break  # Only attach one JSDoc comment

            # Skip nodes that fall inside an export wrapper we've already extracted
            # This prevents duplicate chunks for `export default class Foo {}`
            if any(
                start < node_start and node_end <= end
                for start, end in export_spans
            ):
                continue

            # Track export wrapper spans (already converted to char offsets)
            if node_type in EXPORT_WRAPPER_TYPES:
                export_spans.append((node_start, node_end))

            # Extract symbol name
            symbol = self._extract_symbol(node, node_type, source)

            # Extract source code (use extended span if JSDoc was attached)
            code = source[node_start:node_end]

            # Get line numbers - recalculate if JSDoc extended the span
            if jsdoc_attached:
                # JSDoc was attached, calculate start line from character offset
                start_line = source[:node_start].count("\n") + 1
                _, end_line = node.get_line_range(source)
            else:
                start_line, end_line = node.get_line_range(source)

            # Build metadata
            metadata = {
                "parser": "oxc",
                "node_type": node_type,
                "depth": depth,
                "chunk_type_hint": self._get_chunk_type_for_node(node, node_type, source).value,
            }

            # Add function-specific metadata
            if hasattr(node, "is_async"):
                metadata["is_async"] = node.is_async
            if hasattr(node, "is_generator"):
                metadata["is_generator"] = node.is_generator

            # Create UniversalChunk for CAST processing
            chunk = UniversalChunk(
                concept=NODE_TYPE_TO_CONCEPT.get(node_type, UniversalConcept.DEFINITION),
                name=symbol,
                content=code,
                start_line=start_line,
                end_line=end_line,
                metadata=metadata,
                language_node_type=node_type,
            )
            chunks.append(chunk)

        return chunks

    def _get_chunk_type_for_node(self, node, node_type: str, source: str) -> ChunkType:
        """Get the appropriate ChunkType for a node.

        For export declarations, determines type from the inner declaration.
        For expression statements, determines type from the assignment value.
        """
        # For ExpressionStatement assignments, check the right side
        if node_type == "ExpressionStatement":
            code_text = node.get_text(source).strip()
            if "=" in code_text:
                right_side = code_text.split("=", 1)[1].strip()
                # Check if right side is a function
                if right_side.startswith(("function", "async function")) or "=>" in right_side:
                    # Prototype assignments are methods, static can be either
                    if ".prototype." in code_text:
                        return ChunkType.METHOD
                    return ChunkType.FUNCTION
                # Otherwise treat as variable/property
                return ChunkType.VARIABLE

        # For export wrappers, check the inner declaration type
        if node_type in EXPORT_WRAPPER_TYPES:
            if hasattr(node, "declaration") and node.declaration:
                inner_type = node.declaration.type
                # Map inner declaration to appropriate chunk type
                if inner_type == "ClassDeclaration":
                    return ChunkType.CLASS
                elif inner_type == "FunctionDeclaration":
                    return ChunkType.FUNCTION
                elif inner_type == "VariableDeclaration":
                    return ChunkType.VARIABLE
                elif inner_type == "TSEnumDeclaration":
                    return ChunkType.ENUM
                elif inner_type == "TSInterfaceDeclaration":
                    return ChunkType.INTERFACE
                elif inner_type == "TSTypeAliasDeclaration":
                    return ChunkType.TYPE_ALIAS
                elif inner_type == "TSModuleDeclaration":
                    return ChunkType.NAMESPACE
                # Handle generic "Declaration" type by parsing text
                elif inner_type == "Declaration":
                    decl_text = node.declaration.get_text(source)
                    if decl_text.startswith("enum ") or decl_text.startswith("const enum "):
                        return ChunkType.ENUM
                    elif decl_text.startswith("interface "):
                        return ChunkType.INTERFACE
                    elif decl_text.startswith("type "):
                        return ChunkType.TYPE_ALIAS
                    elif decl_text.startswith(("namespace ", "module ")):
                        return ChunkType.NAMESPACE
                    elif decl_text.startswith("class "):
                        return ChunkType.CLASS
                    elif decl_text.startswith(("function ", "async function ")):
                        return ChunkType.FUNCTION
                    elif decl_text.startswith(("const ", "let ", "var ")):
                        return ChunkType.VARIABLE

        return NODE_TYPE_MAP.get(node_type, ChunkType.UNKNOWN)

    def _extract_symbol(self, node, node_type: str, source: str) -> str:
        """Extract symbol name from a node.

        Args:
            node: Oxc AST node
            node_type: Node type string
            source: Source code for line number calculation

        Returns:
            Symbol name string
        """
        # Handle ExpressionStatement assignments (module.exports, prototype, static)
        if node_type == "ExpressionStatement":
            code_text = node.get_text(source).strip()
            # Extract symbol from left side of assignment
            if "=" in code_text:
                left_side = code_text.split("=")[0].strip()
                # For prototype assignments: Constructor.prototype.method -> Constructor.prototype.method
                # For static assignments: Constructor.method -> Constructor.method
                # For module.exports: return as-is
                if left_side.startswith("module.exports"):
                    return "module.exports"
                return left_side
            return "expression_statement"

        # Try to get name property
        if hasattr(node, "name") and node.name:
            return node.name

        # For variable declarations, try to get the first declarator's name
        if node_type == "VariableDeclaration":
            if hasattr(node, "declarations") and node.declarations:
                first_decl = node.declarations[0]
                if hasattr(first_decl, "id") and hasattr(first_decl.id, "name"):
                    return first_decl.id.name

        # For exports, try to get the exported declaration's name
        if node_type in {"ExportNamedDeclaration", "ExportDefaultDeclaration"}:
            if hasattr(node, "declaration") and node.declaration:
                decl = node.declaration
                if hasattr(decl, "name") and decl.name:
                    return decl.name
                if hasattr(decl, "id") and hasattr(decl.id, "name"):
                    return decl.id.name
                # Extract name from declaration text for various types
                decl_text = decl.get_text(source)
                # Match: const enum first, then const/let/var, function, class, enum, interface, type, namespace
                match = re.match(
                    r"(?:const\s+enum|const|let|var|function|async\s+function|class|enum|interface|type|namespace|module)\s+(\w+)",
                    decl_text,
                )
                if match:
                    return match.group(1)

        # For namespace/module declarations, extract name from text
        if node_type == "TSModuleDeclaration":
            text = node.get_text(source)
            # Match declare? namespace/module followed by name (dotted allowed)
            match = re.match(r"(?:declare\s+)?(?:namespace|module)\s+([\w.]+)", text)
            if match:
                return match.group(1)

        # Fallback to generic name with line number (calculate from span)
        start_line, _ = node.get_line_range(source)
        return f"{node_type.lower()}_line_{start_line}"

    def parse_with_result(
        self, file_path: Path, file_id: FileId | None = None
    ) -> ParseResult:
        """Parse a file and return detailed result information.

        Args:
            file_path: Path to the file to parse
            file_id: Optional unique identifier for the file

        Returns:
            ParseResult with chunks, metadata, and diagnostics
        """
        import time

        start_time = time.time()

        # Use provided file_id or default to 0
        fid = file_id if file_id is not None else FileId(0)

        try:
            chunk_dicts = self.parse_file(file_path, fid)
            parse_time = time.time() - start_time

            return ParseResult(
                chunks=chunk_dicts,
                language=self._language,
                total_chunks=len(chunk_dicts),
                parse_time=parse_time,
                errors=[],
                warnings=[],
                metadata={
                    "parser_type": "oxc",
                    "file_size": file_path.stat().st_size if file_path.exists() else 0,
                },
            )
        except Exception as e:
            parse_time = time.time() - start_time
            return ParseResult(
                chunks=[],
                language=self._language,
                total_chunks=0,
                parse_time=parse_time,
                errors=[str(e)],
                warnings=[],
                metadata={"parser_type": "oxc", "error": str(e)},
            )

    def supports_incremental_parsing(self) -> bool:
        return False

    def parse_incremental(
        self,
        file_path: Path,
        file_id: FileId,
        previous_chunks: list[dict[str, object]] | None = None,
    ) -> list[Chunk]:
        """Parse a file incrementally (full reparse for now)."""
        return self.parse_file(file_path, file_id)

    def get_parse_tree(self, content: str):
        """Get raw parse tree (returns Oxc Program)."""
        source_type = self._detect_source_type(None)
        self._allocator.reset()
        result = oxc_python.parse(content, source_type=source_type, allocator=self._allocator)
        return result.program

    def setup(self) -> None:
        """Setup parser (no-op for Oxc)."""
        pass

    def cleanup(self) -> None:
        """Log performance summary on cleanup."""
        logger.info(
            "Oxc summary: parsed=%d errors=%d | t_parse=%.2fs t_walk=%.2fs",
            self._count_parsed,
            self._count_errors,
            self._t_parse,
            self._t_walk,
        )

    def reset(self) -> None:
        """Reset parser state."""
        self._allocator.reset()

    def can_parse_file(self, file_path: Path) -> bool:
        """Check if this parser can handle the file."""
        return file_path.suffix.lower() in self._extensions

    def detect_language(self, file_path: Path) -> Language | None:
        """Detect language from file path."""
        ext = file_path.suffix.lower()
        ext_to_lang = {
            ".js": Language.JAVASCRIPT,
            ".mjs": Language.JAVASCRIPT,
            ".cjs": Language.JAVASCRIPT,
            ".jsx": Language.JSX,
            ".ts": Language.TYPESCRIPT,
            ".mts": Language.TYPESCRIPT,
            ".cts": Language.TYPESCRIPT,
            ".tsx": Language.TSX,
        }
        return ext_to_lang.get(ext)

    def _estimate_tokens(self, content: str) -> int:
        """Helper method to estimate tokens using centralized utility."""
        return estimate_tokens(content)

    def _apply_cast_algorithm(
        self, universal_chunks: list[UniversalChunk], content: str
    ) -> list[UniversalChunk]:
        """Apply cAST (Code AST) algorithm for optimal semantic chunking.

        The cAST algorithm uses a split-then-merge recursive approach:
        1. Parse source code into AST (already done)
        2. Apply recursive chunking with top-down traversal
        3. Fit large AST nodes into single chunks when possible
        4. Split nodes that exceed chunk size limit recursively
        5. Greedily merge adjacent sibling nodes to maximize information density
        6. Measure chunk size by non-whitespace characters

        Args:
            universal_chunks: Initial chunks extracted from concepts
            content: Original source content

        Returns:
            List of optimized chunks following cAST principles
        """
        if not universal_chunks:
            return []

        # Deduplicate chunks with identical content before processing
        universal_chunks = self._deduplicate_overlapping_chunks(universal_chunks)

        # Group chunks by concept type for structured processing
        chunks_by_concept: dict[UniversalConcept, list[UniversalChunk]] = {}
        for chunk in universal_chunks:
            if chunk.concept not in chunks_by_concept:
                chunks_by_concept[chunk.concept] = []
            chunks_by_concept[chunk.concept].append(chunk)

        optimized_chunks = []

        # Process each concept type with appropriate chunking strategy
        for concept, concept_chunks in chunks_by_concept.items():
            if concept == UniversalConcept.DEFINITION:
                # Definitions (functions, classes) should remain intact when possible
                optimized_chunks.extend(
                    self._chunk_definitions(concept_chunks, content)
                )
            elif concept == UniversalConcept.BLOCK:
                # Blocks can be merged more aggressively
                optimized_chunks.extend(self._chunk_blocks(concept_chunks, content))
            elif concept == UniversalConcept.COMMENT:
                # Comments can be merged with nearby code
                optimized_chunks.extend(self._chunk_comments(concept_chunks, content))
            else:
                # Other concepts use default chunking
                optimized_chunks.extend(self._chunk_generic(concept_chunks, content))

        # Final pass: merge adjacent chunks that are below threshold
        if self.cast_config.greedy_merge:
            optimized_chunks = self._greedy_merge_pass(optimized_chunks, content)

        return optimized_chunks

    def _deduplicate_overlapping_chunks(
        self, chunks: list[UniversalChunk]
    ) -> list[UniversalChunk]:
        """Remove duplicate and overlapping chunks.

        When tree-sitter extraction creates multiple chunks for the same or overlapping
        source code, keep only the most semantically specific chunk.

        Args:
            chunks: List of chunks potentially containing duplicates

        Returns:
            Deduplicated list with only the most specific chunk for each unique content
        """
        if not chunks:
            return []

        # Group chunks by normalized content (exact match required)
        content_groups: dict[str, list[UniversalChunk]] = {}

        for chunk in chunks:
            # Normalize content for comparison (strip whitespace)
            normalized_content = chunk.content.strip()

            # Skip empty chunks
            if not normalized_content:
                continue

            # Group by exact content match
            if normalized_content not in content_groups:
                content_groups[normalized_content] = []
            content_groups[normalized_content].append(chunk)

        # For each content group with exact matches, keep only the most specific
        result = []
        for chunk_group in content_groups.values():
            if len(chunk_group) == 1:
                # No duplicates - keep the single chunk
                result.append(chunk_group[0])
            else:
                # Multiple chunks with identical content - select most specific
                def name_quality(chunk: UniversalChunk) -> int:
                    """Return quality score for chunk name (higher = better)."""
                    name = chunk.name
                    # Fallback names
                    if name.startswith("definition_line_"):
                        return 0
                    if name.startswith("comment_line_"):
                        return 0
                    if name.startswith("unnamed_"):
                        return 0
                    if name.startswith("export_default"):
                        return 1
                    if name.startswith("module_exports"):
                        return 1
                    # Prefer names that appear in the content (actual identifiers)
                    if name in chunk.content:
                        return 2
                    return 1

                best_chunk = max(
                    chunk_group,
                    key=lambda c: (
                        name_quality(c),  # Prefer better names
                        self._get_chunk_specificity(c),
                        -(c.end_line - c.start_line),  # Smaller spans preferred
                    ),
                )
                result.append(best_chunk)

        # Second pass: Remove BLOCK chunks whose content is a substring
        # of DEFINITION/STRUCTURE chunks AND have overlapping line ranges
        final_result = []
        for i, chunk_i in enumerate(result):
            is_substring = False
            normalized_i = chunk_i.content.strip()

            # Only apply substring deduplication to BLOCK chunks
            if chunk_i.concept != UniversalConcept.BLOCK:
                final_result.append(chunk_i)
                continue

            for j, chunk_j in enumerate(result):
                if i == j:
                    continue

                normalized_j = chunk_j.content.strip()

                # Only deduplicate BLOCKs contained in DEFINITION or STRUCTURE chunks
                if chunk_j.concept not in (
                    UniversalConcept.DEFINITION,
                    UniversalConcept.STRUCTURE,
                ):
                    continue

                # Check if chunks have overlapping line ranges
                lines_overlap = not (
                    chunk_i.end_line < chunk_j.start_line
                    or chunk_i.start_line > chunk_j.end_line
                )

                # Check if chunk_i's content is substring of chunk_j
                if (
                    normalized_i in normalized_j
                    and len(normalized_i) < len(normalized_j)
                    and lines_overlap
                ):
                    # BLOCK chunk is contained within DEFINITION/STRUCTURE chunk
                    is_substring = True
                    break

            if not is_substring:
                final_result.append(chunk_i)

        # Third pass: Remove nested DEFINITION chunks
        # Sort by line position and size for efficient nested detection
        sorted_chunks = sorted(
            final_result,
            key=lambda c: (c.start_line, -(c.end_line - c.start_line))
        )

        # NESTED DEFINITION DEDUPLICATION
        # Remove smaller DEFINITION chunks that are fully contained within larger DEFINITION chunks
        # This handles nested functions: outer function contains inner functions
        # Optimized O(n) algorithm using a stack to track active containers
        #
        # IMPORTANT: Skip this for now - tests expect nested functions to be extracted separately
        # TODO: Make this configurable or refine the logic to handle specific patterns
        keep_chunks = sorted_chunks  # Disable nested deduplication for now

        # Original nested deduplication logic (commented out):
        # keep_chunks = []
        # definition_stack = []  # Stack of (chunk, index) tuples for active outer DEFINITIONs
        #
        # for chunk in sorted_chunks:
        #     # Clean up stack: remove DEFINITIONs that have ended
        #     while definition_stack and definition_stack[-1][0].end_line < chunk.start_line:
        #         definition_stack.pop()
        #
        #     if chunk.concept == UniversalConcept.DEFINITION:
        #         # Check if this chunk is nested inside any chunk in the stack
        #         is_nested = False
        #         for outer_chunk, _ in definition_stack:
        #             if (
        #                 chunk.start_line >= outer_chunk.start_line
        #                 and chunk.end_line <= outer_chunk.end_line
        #                 and len(chunk.content) < len(outer_chunk.content)
        #             ):
        #                 # This chunk is nested inside outer_chunk
        #                 logger.debug(
        #                     f"Nested DEFINITION dedup: Removing '{chunk.name}' "
        #                     f"(lines {chunk.start_line}-{chunk.end_line}, {len(chunk.content)} chars) "
        #                     f"contained in '{outer_chunk.name}' "
        #                     f"(lines {outer_chunk.start_line}-{outer_chunk.end_line}, {len(outer_chunk.content)} chars)"
        #                 )
        #                 is_nested = True
        #                 break
        #
        #         if not is_nested:
        #             # Not nested - keep this chunk and add to stack
        #             keep_chunks.append(chunk)
        #             definition_stack.append((chunk, len(keep_chunks) - 1))
        #     else:
        #         # Not a DEFINITION - always keep
        #         keep_chunks.append(chunk)

        # Fourth pass: Remove chunks with overlapping line ranges
        # This handles cases where splitting creates chunks that overlap with nested extractions
        # Sort by start line, then by size (larger first) to process containers before contained
        sorted_by_lines = sorted(
            keep_chunks,
            key=lambda c: (c.start_line, -(c.end_line - c.start_line))
        )

        final_chunks = []

        for chunk in sorted_by_lines:
            # Check if this chunk is fully contained within any already-kept chunk
            should_remove = False

            for kept_chunk in final_chunks:
                # Check if chunk is fully contained within kept_chunk
                if (
                    chunk.start_line >= kept_chunk.start_line
                    and chunk.end_line <= kept_chunk.end_line
                    and (chunk.start_line != kept_chunk.start_line or chunk.end_line != kept_chunk.end_line)  # Not exact same range
                ):
                    # By default, remove contained chunks (they're duplicates)
                    # EXCEPT for specific intentional nestings:
                    chunk_node_type = chunk.metadata.get("node_type", "")
                    kept_node_type = kept_chunk.metadata.get("node_type", "")

                    # Whitelist of intentional nested extractions:
                    # 1. Methods within classes
                    if chunk_node_type == "MethodDefinition" and kept_node_type == "ClassDeclaration":
                        continue  # Keep the method

                    # 2. Nested classes within classes (inner classes)
                    if chunk_node_type == "ClassDeclaration" and kept_node_type == "ClassDeclaration":
                        continue  # Keep the inner class

                    # 3. Function/class definitions within TS modules/namespaces
                    if kept_node_type == "TSModuleDeclaration":
                        if chunk_node_type in ("FunctionDeclaration", "ClassDeclaration", "TSInterfaceDeclaration"):
                            continue  # Keep exports from module

                    # For all other cases, remove the contained chunk (it's a duplicate)
                    should_remove = True
                    break

            if not should_remove:
                final_chunks.append(chunk)

        # Sort back to original line order
        final_chunks.sort(key=lambda c: (c.start_line, c.end_line))

        return final_chunks

    def _chunk_definitions(
        self, chunks: list[UniversalChunk], content: str
    ) -> list[UniversalChunk]:
        """Apply cAST chunking to definition chunks (functions, classes, etc.).

        Definitions remain intact as complete semantic units.
        Only split if they exceed the maximum chunk size significantly.
        """
        result = []

        for chunk in chunks:
            # Always validate and split if needed
            split_chunks = self._validate_and_split_chunk(chunk, content)
            result.extend(split_chunks)

        return result

    def _chunk_blocks(
        self, chunks: list[UniversalChunk], content: str
    ) -> list[UniversalChunk]:
        """Apply cAST chunking to block chunks.

        Blocks are more flexible and can be merged aggressively with siblings.
        """
        if not chunks:
            return []

        # Sort chunks by line position
        sorted_chunks = sorted(chunks, key=lambda c: c.start_line)
        result = []
        current_group = [sorted_chunks[0]]

        for chunk in sorted_chunks[1:]:
            # Check if we can merge with current group
            if self._can_merge_chunks(current_group, chunk, content):
                current_group.append(chunk)
            else:
                # Finalize current group and start new one
                merged = self._merge_chunk_group(current_group, content)
                result.extend(merged)
                current_group = [chunk]

        # Don't forget the last group
        if current_group:
            merged = self._merge_chunk_group(current_group, content)
            result.extend(merged)

        # Final validation: ensure all chunks meet size constraints
        validated_result = []
        for chunk in result:
            validated_result.extend(self._validate_and_split_chunk(chunk, content))

        return validated_result

    def _chunk_comments(
        self, chunks: list[UniversalChunk], content: str
    ) -> list[UniversalChunk]:
        """Apply cAST chunking to comment chunks.

        Comments are merged conservatively - only consecutive comments (gap <= 1)
        are merged together.
        """
        if not chunks:
            return []

        # Sort comments by line position
        sorted_chunks = sorted(chunks, key=lambda c: c.start_line)
        result = []
        current_group = [sorted_chunks[0]]

        for chunk in sorted_chunks[1:]:
            # Only merge if comments are consecutive or adjacent (gap <= 1)
            last_chunk = current_group[-1]
            line_gap = chunk.start_line - last_chunk.end_line

            if line_gap <= 1:
                # Comments are consecutive - can merge
                current_group.append(chunk)
            else:
                # Gap is too large - finalize current group and start new one
                merged = self._merge_chunk_group(current_group, content)
                result.extend(merged)
                current_group = [chunk]

        # Don't forget the last group
        if current_group:
            merged = self._merge_chunk_group(current_group, content)
            result.extend(merged)

        # Final validation: ensure all chunks meet size constraints
        validated_result = []
        for chunk in result:
            validated_result.extend(self._validate_and_split_chunk(chunk, content))

        return validated_result

    def _chunk_generic(
        self, chunks: list[UniversalChunk], content: str
    ) -> list[UniversalChunk]:
        """Apply generic cAST chunking to other chunk types."""
        return self._chunk_blocks(chunks, content)  # Use block strategy as default

    def _validate_and_split_chunk(
        self, chunk: UniversalChunk, content: str
    ) -> list[UniversalChunk]:
        """Validate chunk size and split if necessary."""
        metrics = ChunkMetrics.from_content(chunk.content)
        estimated_tokens = self._estimate_tokens(chunk.content)

        if (
            metrics.non_whitespace_chars <= self.cast_config.max_chunk_size
            and estimated_tokens <= self.cast_config.safe_token_limit
        ):
            # Chunk fits within both limits
            return [chunk]
        else:
            # Too large, apply recursive splitting
            return self._recursive_split_chunk(chunk, content)

    def _analyze_lines(self, lines: list[str]) -> tuple[bool, bool]:
        """Analyze line length statistics to choose optimal splitting strategy.

        Returns:
            (has_very_long_lines, is_regular_code)
        """
        if not lines:
            return False, False

        lengths = [len(line) for line in lines]
        max_length = max(lengths)
        avg_length = sum(lengths) / len(lengths)

        # 20% of chunk size threshold for detecting minified/concatenated code
        long_line_threshold = self.cast_config.max_chunk_size * 0.2
        has_very_long_lines = max_length > long_line_threshold

        # Regular code heuristics
        is_regular_code = len(lines) > 10 and max_length < 200 and avg_length < 100.0

        return has_very_long_lines, is_regular_code

    def _recursive_split_chunk(
        self, chunk: UniversalChunk, content: str
    ) -> list[UniversalChunk]:
        """Smart content-aware splitting that chooses the optimal strategy."""
        # First: Check if we even need to split
        metrics = ChunkMetrics.from_content(chunk.content)
        estimated_tokens = self._estimate_tokens(chunk.content)

        if (
            metrics.non_whitespace_chars <= self.cast_config.max_chunk_size
            and estimated_tokens <= self.cast_config.safe_token_limit
        ):
            return [chunk]  # No splitting needed

        # Second: Analyze the content structure
        lines = chunk.content.split("\n")
        has_very_long_lines, is_regular_code = self._analyze_lines(lines)

        # Third: Choose splitting strategy based on content analysis
        if len(lines) <= 2 or has_very_long_lines:
            # Case 1: Single/few lines OR any line is very long
            # Use character-based emergency splitting
            return self._emergency_split_code(chunk, content)

        elif is_regular_code:
            # Case 2: Many short lines (normal code)
            # Use simple line-based splitting
            return self._split_by_lines_simple(chunk, lines)

        else:
            # Case 3: Mixed content - try line-based with emergency fallback
            return self._split_by_lines_with_fallback(chunk, lines, content)

    def _split_by_lines_simple(
        self, chunk: UniversalChunk, lines: list[str]
    ) -> list[UniversalChunk]:
        """Split chunk by lines for regular code with short lines."""
        if len(lines) <= 2:
            return [chunk]

        mid_point = len(lines) // 2

        # Create two sub-chunks
        chunk1_content = "\n".join(lines[:mid_point])
        chunk2_content = "\n".join(lines[mid_point:])

        # Simple line distribution based on content split
        chunk1_lines = len(lines[:mid_point])
        chunk1_end_line = chunk.start_line + chunk1_lines - 1
        chunk2_start_line = chunk1_end_line + 1

        # Ensure valid bounds
        chunk1_end_line = max(chunk.start_line, min(chunk1_end_line, chunk.end_line))
        chunk2_start_line = max(
            chunk.start_line, min(chunk2_start_line, chunk.end_line)
        )

        chunk1 = UniversalChunk(
            concept=chunk.concept,
            name=f"{chunk.name}_part1",
            content=chunk1_content,
            start_line=chunk.start_line,
            end_line=chunk1_end_line,
            metadata=chunk.metadata.copy(),
            language_node_type=chunk.language_node_type,
        )

        chunk2 = UniversalChunk(
            concept=chunk.concept,
            name=f"{chunk.name}_part2",
            content=chunk2_content,
            start_line=chunk2_start_line,
            end_line=chunk.end_line,
            metadata=chunk.metadata.copy(),
            language_node_type=chunk.language_node_type,
        )

        # Recursively check if sub-chunks still need splitting
        result = []
        for sub_chunk in [chunk1, chunk2]:
            sub_metrics = ChunkMetrics.from_content(sub_chunk.content)
            sub_tokens = self._estimate_tokens(sub_chunk.content)

            if (
                sub_metrics.non_whitespace_chars > self.cast_config.max_chunk_size
                or sub_tokens > self.cast_config.safe_token_limit
            ):
                result.extend(self._recursive_split_chunk(sub_chunk, sub_chunk.content))
            else:
                result.append(sub_chunk)

        return result

    def _split_by_lines_with_fallback(
        self, chunk: UniversalChunk, lines: list[str], content: str
    ) -> list[UniversalChunk]:
        """Split by lines but fall back to emergency split if needed."""
        # Try line-based splitting first
        line_split_result = self._split_by_lines_simple(chunk, lines)

        # Check if any chunks still exceed limits
        validated_result = []
        for sub_chunk in line_split_result:
            sub_metrics = ChunkMetrics.from_content(sub_chunk.content)
            sub_tokens = self._estimate_tokens(sub_chunk.content)

            # If still over limit, use emergency split
            if (
                sub_metrics.non_whitespace_chars > self.cast_config.max_chunk_size
                or sub_tokens > self.cast_config.safe_token_limit
            ):
                validated_result.extend(
                    self._emergency_split_code(sub_chunk, sub_chunk.content)
                )
            else:
                validated_result.append(sub_chunk)

        return validated_result

    def _emergency_split_code(
        self, chunk: UniversalChunk, content: str
    ) -> list[UniversalChunk]:
        """Smart code splitting for minified/large single-line files."""
        # Use the stricter limit: character limit or token-based limit
        estimated_tokens = self._estimate_tokens(chunk.content)
        if estimated_tokens > 0:
            # Calculate actual chars-to-token ratio for this content
            actual_ratio = len(chunk.content) / estimated_tokens
            max_chars_from_tokens = int(
                self.cast_config.safe_token_limit * actual_ratio * 0.8
            )
        else:
            # Fallback to conservative estimation
            max_chars_from_tokens = int(self.cast_config.safe_token_limit * 3.5 * 0.8)
        max_chars = min(self.cast_config.max_chunk_size, max_chars_from_tokens)

        metrics = ChunkMetrics.from_content(chunk.content)
        if (
            metrics.non_whitespace_chars <= self.cast_config.max_chunk_size
            and len(chunk.content) <= max_chars_from_tokens
        ):
            return [chunk]

        # Smart split points for code (in order of preference)
        split_chars = [";", "}", "{", ",", " "]

        chunks = []
        remaining = chunk.content
        part_num = 1
        total_content_length = len(chunk.content)
        current_pos = 0  # Track position in original content for line number calculation

        while remaining:
            remaining_metrics = ChunkMetrics.from_content(remaining)
            if (
                remaining_metrics.non_whitespace_chars
                <= self.cast_config.max_chunk_size
            ):
                chunks.append(
                    self._create_split_chunk(
                        chunk, remaining, part_num, current_pos, total_content_length
                    )
                )
                break

            # Find best split point within size limit
            best_split = 0
            for split_char in split_chars:
                # Search within character limit
                search_end = min(max_chars, len(remaining))
                pos = remaining.rfind(split_char, 0, search_end)

                if pos > best_split:
                    # Check if this split point gives us valid chunk size
                    test_content = remaining[: pos + 1]
                    test_metrics = ChunkMetrics.from_content(test_content)
                    if (
                        test_metrics.non_whitespace_chars
                        <= self.cast_config.max_chunk_size
                    ):
                        best_split = pos + 1  # Include the split character
                        break

            # If no good split found, force split at character limit
            if best_split == 0:
                best_split = max_chars

            chunks.append(
                self._create_split_chunk(
                    chunk,
                    remaining[:best_split],
                    part_num,
                    current_pos,
                    total_content_length,
                )
            )
            remaining = remaining[best_split:]
            current_pos += best_split  # Update position tracker for next chunk's line calculation
            part_num += 1

        return chunks

    def _create_split_chunk(
        self,
        original: UniversalChunk,
        content: str,
        part_num: int,
        content_start_pos: int = 0,
        total_content_length: int = 0,
    ) -> UniversalChunk:
        """Create a split chunk from emergency splitting with proportional lines."""

        # Simple proportional line calculation based on content position
        original_line_span = original.end_line - original.start_line + 1

        if total_content_length > 0 and content_start_pos >= 0:
            # Calculate proportional position and length
            position_ratio = content_start_pos / total_content_length
            content_ratio = len(content) / total_content_length

            # Distribute lines proportionally
            line_offset = int(position_ratio * original_line_span)
            line_span = max(1, int(content_ratio * original_line_span))

            start_line = original.start_line + line_offset
            end_line = min(original.end_line, start_line + line_span - 1)

            # Ensure valid bounds
            start_line = min(start_line, original.end_line)
            end_line = max(end_line, start_line)
        else:
            # Fallback to original bounds
            start_line = original.start_line
            end_line = original.end_line

        return UniversalChunk(
            concept=original.concept,
            name=f"{original.name}_part{part_num}",
            content=content,
            start_line=start_line,
            end_line=end_line,
            metadata=original.metadata.copy(),
            language_node_type=original.language_node_type,
        )

    def _can_merge_chunks(
        self,
        current_group: list[UniversalChunk],
        candidate: UniversalChunk,
        content: str,
    ) -> bool:
        """Check if a chunk can be merged with the current group."""
        if not current_group:
            return True

        # Calculate combined size
        total_content = (
            "\n".join(chunk.content for chunk in current_group)
            + "\n"
            + candidate.content
        )
        metrics = ChunkMetrics.from_content(total_content)

        # Check BOTH character and token constraints
        estimated_tokens = self._estimate_tokens(total_content)
        safe_token_limit = 6000

        if (
            metrics.non_whitespace_chars
            > self.cast_config.max_chunk_size * self.cast_config.merge_threshold
            or estimated_tokens > safe_token_limit * self.cast_config.merge_threshold
        ):
            return False

        # Check line proximity (chunks should be close to each other)
        last_chunk = current_group[-1]
        line_gap = candidate.start_line - last_chunk.end_line

        if line_gap > 5:  # Allow small gaps for related code
            return False

        # Check concept compatibility
        if last_chunk.concept != candidate.concept:
            # Only merge compatible concepts
            compatible_pairs = {
                (UniversalConcept.COMMENT, UniversalConcept.DEFINITION),
                (UniversalConcept.DEFINITION, UniversalConcept.COMMENT),
                (UniversalConcept.BLOCK, UniversalConcept.COMMENT),
                (UniversalConcept.COMMENT, UniversalConcept.BLOCK),
                (UniversalConcept.DEFINITION, UniversalConcept.STRUCTURE),
                (UniversalConcept.STRUCTURE, UniversalConcept.DEFINITION),
            }
            if (last_chunk.concept, candidate.concept) not in compatible_pairs:
                return False

        return True

    def _merge_chunk_group(
        self, group: list[UniversalChunk], content: str
    ) -> list[UniversalChunk]:
        """Merge a group of chunks into optimized chunks."""
        if len(group) <= 1:
            return group

        # Sort by line position
        sorted_group = sorted(group, key=lambda c: c.start_line)

        # Simple merge: combine content without duplication
        combined_content = sorted_group[0].content
        for chunk in sorted_group[1:]:
            # Only add content if not already included (prevent duplication)
            if chunk.content.strip() not in combined_content:
                combined_content += "\n" + chunk.content

        metrics = ChunkMetrics.from_content(combined_content)
        estimated_tokens = self._estimate_tokens(combined_content)

        # If combined chunk is too large, return original chunks
        if (
            metrics.non_whitespace_chars > self.cast_config.max_chunk_size
            or estimated_tokens > self.cast_config.safe_token_limit
        ):
            return group

        # Create merged chunk
        first_chunk = sorted_group[0]
        last_chunk = sorted_group[-1]

        # Combine names
        unique_names = list(dict.fromkeys(chunk.name for chunk in sorted_group))
        merged_name = (
            "_".join(unique_names) if len(unique_names) > 1 else unique_names[0]
        )

        # Combine metadata
        merged_metadata = first_chunk.metadata.copy()
        merged_metadata["merged_from"] = [chunk.name for chunk in sorted_group]
        merged_metadata["chunk_count"] = len(sorted_group)

        merged_chunk = UniversalChunk(
            concept=first_chunk.concept,  # Use primary concept
            name=merged_name,
            content=combined_content,
            start_line=first_chunk.start_line,
            end_line=last_chunk.end_line,
            metadata=merged_metadata,
            language_node_type=first_chunk.language_node_type,
        )

        return [merged_chunk]

    def _greedy_merge_pass(
        self, chunks: list[UniversalChunk], content: str
    ) -> list[UniversalChunk]:
        """Final greedy merge pass to maximize information density."""
        if len(chunks) <= 1:
            return chunks

        # Sort chunks by line position
        sorted_chunks = sorted(chunks, key=lambda c: c.start_line)
        result = []
        current_chunk = sorted_chunks[0]

        for next_chunk in sorted_chunks[1:]:
            # Check concept compatibility before merging
            if current_chunk.concept != next_chunk.concept:
                # Define compatible concept pairs
                compatible_pairs = {
                    (UniversalConcept.COMMENT, UniversalConcept.DEFINITION),
                    (UniversalConcept.DEFINITION, UniversalConcept.COMMENT),
                    (UniversalConcept.BLOCK, UniversalConcept.COMMENT),
                    (UniversalConcept.COMMENT, UniversalConcept.BLOCK),
                    (UniversalConcept.DEFINITION, UniversalConcept.STRUCTURE),
                    (UniversalConcept.STRUCTURE, UniversalConcept.DEFINITION),
                }

                # If concepts are not compatible, don't merge
                if (current_chunk.concept, next_chunk.concept) not in compatible_pairs:
                    result.append(current_chunk)
                    current_chunk = next_chunk
                    continue

            # Don't merge if next_chunk is nested inside current_chunk
            is_nested = (
                next_chunk.start_line > current_chunk.start_line
                and next_chunk.end_line <= current_chunk.end_line
            )
            if is_nested:
                result.append(current_chunk)
                current_chunk = next_chunk
                continue

            # Don't merge if either chunk explicitly prevents merging
            current_prevents_merge = current_chunk.metadata.get("prevent_merge_across_concepts", False)
            next_prevents_merge = next_chunk.metadata.get("prevent_merge_across_concepts", False)
            if current_prevents_merge or next_prevents_merge:
                result.append(current_chunk)
                current_chunk = next_chunk
                continue

            # Simple merge logic: only if content is different and fits size limit
            if next_chunk.content.strip() not in current_chunk.content:
                combined_content = current_chunk.content + "\n" + next_chunk.content
            else:
                combined_content = current_chunk.content  # Skip duplicate content

            metrics = ChunkMetrics.from_content(combined_content)
            estimated_tokens = self._estimate_tokens(combined_content)

            # Check for semantic incompatibility within same concept type
            semantic_mismatch = False
            if (
                current_chunk.concept
                == next_chunk.concept
                == UniversalConcept.DEFINITION
            ):
                # Check if both chunks have 'kind' metadata
                current_kind = current_chunk.metadata.get("kind")
                next_kind = next_chunk.metadata.get("kind")

                # Don't merge if kinds are different
                if current_kind and next_kind and current_kind != next_kind:
                    semantic_mismatch = True

                # Don't merge definitions with different node types
                # This prevents merging interfaces with arrow functions, classes with variables, etc.
                current_node_type = current_chunk.metadata.get("node_type", "")
                next_node_type = next_chunk.metadata.get("node_type", "")
                if current_node_type and next_node_type and current_node_type != next_node_type:
                    semantic_mismatch = True

                # Don't merge top-level definitions with each other
                # Note: Oxc uses PascalCase for node types
                top_level_types = {
                    "ClassDeclaration",
                    "FunctionDeclaration",
                    "MethodDefinition",  # Methods within classes should not be merged
                    "TSInterfaceDeclaration",
                    "TSTypeAliasDeclaration",
                    "TSEnumDeclaration",
                    "TSModuleDeclaration",
                    "VariableDeclaration",
                    "ExportNamedDeclaration",
                    "ExportDefaultDeclaration",
                    "ExpressionStatement",  # Includes prototype assignments, module.exports, IIFEs
                }
                if (
                    current_node_type in top_level_types
                    and next_node_type in top_level_types
                ):
                    semantic_mismatch = True

                # Prevent merging when there's a gap between top-level definitions
                line_gap = next_chunk.start_line - current_chunk.end_line
                if line_gap > 1:
                    semantic_mismatch = True

            # Determine maximum allowed gap based on chunk types
            max_gap = 5  # Default: allow reasonable gaps for related code
            if current_chunk.concept != next_chunk.concept:
                # Cross-concept merge - check if either is COMMENT
                if (
                    current_chunk.concept == UniversalConcept.COMMENT
                    or next_chunk.concept == UniversalConcept.COMMENT
                ):
                    max_gap = 1  # Strict: only merge immediately adjacent comments/code

            # Simple merge condition: fits in size limit and close proximity
            can_merge = (
                not semantic_mismatch
                and metrics.non_whitespace_chars <= self.cast_config.max_chunk_size
                and estimated_tokens <= self.cast_config.safe_token_limit
                and next_chunk.start_line - current_chunk.end_line <= max_gap
            )

            if can_merge:
                # When merging chunks with different concepts, prefer the more specific one
                if current_chunk.concept != next_chunk.concept:
                    # Determine which chunk is more specific
                    current_spec = self._get_chunk_specificity(current_chunk)
                    next_spec = self._get_chunk_specificity(next_chunk)

                    if next_spec > current_spec:
                        # Next chunk is more specific - use its name and concept
                        merged_concept = next_chunk.concept
                        merged_name = next_chunk.name
                        merged_metadata = next_chunk.metadata.copy()
                        merged_language_node_type = next_chunk.language_node_type
                    else:
                        # Current chunk is more specific - keep its attributes
                        merged_concept = current_chunk.concept
                        merged_name = current_chunk.name
                        merged_metadata = current_chunk.metadata.copy()
                        merged_language_node_type = current_chunk.language_node_type
                else:
                    # Same concept - keep current chunk's attributes
                    merged_concept = current_chunk.concept
                    merged_name = current_chunk.name
                    merged_metadata = current_chunk.metadata.copy()
                    merged_language_node_type = current_chunk.language_node_type

                # Simple merge without complex metadata
                current_chunk = UniversalChunk(
                    concept=merged_concept,
                    name=merged_name,
                    content=combined_content,
                    start_line=current_chunk.start_line,
                    end_line=next_chunk.end_line,
                    metadata=merged_metadata,
                    language_node_type=merged_language_node_type,
                )
            else:
                # Cannot merge, finalize current chunk
                result.append(current_chunk)
                current_chunk = next_chunk

        # Don't forget the last chunk
        result.append(current_chunk)

        return result

    def _get_chunk_specificity(self, chunk: UniversalChunk) -> int:
        """Get specificity ranking for a chunk's concept.

        Returns:
            Specificity score (higher = more specific)
        """
        specificity = {
            UniversalConcept.DEFINITION: 4,
            UniversalConcept.IMPORT: 3,
            UniversalConcept.COMMENT: 2,
            UniversalConcept.BLOCK: 1,
            UniversalConcept.STRUCTURE: 0,
        }
        return specificity.get(chunk.concept, -1)

    def _convert_to_chunks(
        self,
        universal_chunks: list[UniversalChunk],
        content: str,
        file_path: Path | None,
        file_id: FileId,
    ) -> list[Chunk]:
        """Convert UniversalChunks to standard Chunk format.

        Args:
            universal_chunks: Optimized chunks from CAST
            content: Original source content
            file_path: Optional file path
            file_id: File ID for chunk association

        Returns:
            List of standard Chunk objects
        """
        from chunkhound.core.types.common import ByteOffset, FilePath, LineNumber

        chunks = []

        for uc in universal_chunks:
            # Map UniversalConcept to ChunkType using metadata hint
            chunk_type = self._map_concept_to_chunk_type(uc.concept, uc.metadata)

            # Calculate byte offsets
            start_byte = None
            end_byte = None
            if content:
                lines_before = content.split("\n")[: uc.start_line - 1]
                start_byte = ByteOffset(
                    sum(len(line) + 1 for line in lines_before)
                )
                end_byte = ByteOffset(start_byte + len(uc.content.encode("utf-8")))

            chunk = Chunk(
                symbol=uc.name,
                start_line=LineNumber(uc.start_line),
                end_line=LineNumber(uc.end_line),
                code=uc.content,
                chunk_type=chunk_type,
                file_id=file_id,
                language=self._language,
                file_path=FilePath(str(file_path)) if file_path else None,
                start_byte=start_byte,
                end_byte=end_byte,
                metadata=uc.metadata,
            )
            chunks.append(chunk)

        return chunks

    def _map_concept_to_chunk_type(
        self, concept: UniversalConcept, metadata: dict[str, Any]
    ) -> ChunkType:
        """Map UniversalConcept to ChunkType using metadata hints.

        Args:
            concept: Universal concept from extraction
            metadata: Metadata with chunk_type_hint

        Returns:
            Appropriate ChunkType for the concept
        """
        # Check for explicit chunk_type_hint from OxcParser
        chunk_type_hint = metadata.get("chunk_type_hint", "").lower()
        if chunk_type_hint:
            # Try to parse as ChunkType enum value
            try:
                return ChunkType(chunk_type_hint)
            except ValueError:
                pass  # Fall through to concept-based mapping

        # Fallback to concept-based mapping
        if concept == UniversalConcept.DEFINITION:
            node_type = metadata.get("node_type", "").lower()
            if "function" in node_type:
                return ChunkType.FUNCTION
            elif "class" in node_type:
                return ChunkType.CLASS
            elif "method" in node_type:
                return ChunkType.METHOD
            elif "enum" in node_type:
                return ChunkType.ENUM
            elif "interface" in node_type:
                return ChunkType.INTERFACE
            elif "type" in node_type:
                return ChunkType.TYPE_ALIAS
            else:
                return ChunkType.FUNCTION
        elif concept == UniversalConcept.BLOCK:
            return ChunkType.BLOCK
        elif concept == UniversalConcept.COMMENT:
            return ChunkType.COMMENT
        elif concept == UniversalConcept.IMPORT:
            return ChunkType.UNKNOWN
        elif concept == UniversalConcept.STRUCTURE:
            return ChunkType.NAMESPACE
        else:
            return ChunkType.UNKNOWN
    def validate_syntax(self, content: str) -> list[str]:
        """Validate syntax and return errors."""
        source_type = self._detect_source_type(None)
        self._allocator.reset()
        result = oxc_python.parse(content, source_type=source_type, allocator=self._allocator)
        errors = []
        for e in result.errors:
            line = content[:e.span.start].count('\n') + 1 if e.span.start > 0 else 1
            errors.append(f"Line {line}: {e.message}")
        return errors
