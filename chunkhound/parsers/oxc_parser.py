"""Oxc-backed JavaScript/TypeScript parser.

This module provides the OxcParser class that uses oxc-python for parsing
JavaScript, TypeScript, JSX, and TSX files. Oxc is 10-30x faster than
tree-sitter for JS/TS parsing.

ParserFactory uses this as the primary parser for JavaScript-family languages
when oxc-python is available, with automatic fallback to tree-sitter if unavailable.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from time import perf_counter

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
from chunkhound.interfaces.language_parser import LanguageParser, ParseResult
from chunkhound.utils.normalization import normalize_content

logger = logging.getLogger(__name__)


# Node type to ChunkType mapping
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
    """Oxc-backed JS/TS/JSX/TSX parser.

    Uses oxc-python for fast parsing of JavaScript-family languages.
    Oxc is 10-30x faster than tree-sitter for these languages.
    """

    def __init__(self, language: Language) -> None:
        """Initialize Oxc parser.

        Args:
            language: The language this parser handles (JS, TS, JSX, TSX)
        """
        self._language = language
        self._allocator = oxc_python.Allocator()

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
            file_id: Database file ID for chunk association

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
        """Parse content string and extract semantic chunks.

        Args:
            content: Source code content to parse
            file_path: Optional file path for metadata
            file_id: Optional file ID for chunk association

        Returns:
            List of Chunk objects
        """
        if not content.strip():
            return []

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
        chunks = self._build_chunks(result, content, file_path, file_id or FileId(0))
        self._t_walk += perf_counter() - t0

        self._count_parsed += 1
        return chunks

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
    ) -> list[Chunk]:
        """Build ChunkHound chunks from Oxc AST.

        Args:
            result: Oxc parse result
            source: Original source code
            file_path: Optional file path
            file_id: File ID for chunks

        Returns:
            List of Chunk objects
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

            # Only extract ExpressionStatements that are relevant assignments
            # (module.exports, prototype methods, static methods/properties)
            if node_type == "ExpressionStatement":
                code_text = node.get_text(source).strip()
                if not self._is_extractable_assignment(code_text):
                    continue

            node_start = node.span.start
            node_end = node.span.end

            # Check for preceding JSDoc comment and extend span to include it
            # A JSDoc is "attached" if only whitespace exists between comment end and node start
            for jsdoc_start, jsdoc_end in jsdoc_comments:
                if jsdoc_end <= node_start:
                    # Check if only whitespace between comment and node
                    between = source[jsdoc_end:node_start]
                    if between.strip() == "":
                        node_start = jsdoc_start
                        break  # Only attach one JSDoc comment

            # Skip nodes that fall inside an export wrapper we've already extracted
            # This prevents duplicate chunks for `export default class Foo {}`
            if any(
                start < node_start and node_end <= end
                for start, end in export_spans
            ):
                continue

            # Track export wrapper spans
            if node_type in EXPORT_WRAPPER_TYPES:
                export_spans.append((node_start, node_end))

            # Get chunk type - for exports, determine from inner declaration
            chunk_type = self._get_chunk_type_for_node(node, node_type, source)

            # Extract symbol name
            symbol = self._extract_symbol(node, node_type, source)

            # Extract source code (use extended span if JSDoc was attached)
            code = source[node_start:node_end]

            # Get line numbers - recalculate if JSDoc extended the span
            if node_start < node.span.start:
                # JSDoc was attached, calculate start line from byte offset
                start_line = source[:node_start].count("\n") + 1
                _, end_line = node.get_line_range(source)
            else:
                start_line, end_line = node.get_line_range(source)

            # Calculate byte offsets
            start_byte = ByteOffset(node_start)
            end_byte = ByteOffset(node_end)

            # Build metadata
            metadata = {
                "parser": "oxc",
                "node_type": node_type,
                "depth": depth,
            }

            # Add function-specific metadata
            if hasattr(node, "is_async"):
                metadata["is_async"] = node.is_async
            if hasattr(node, "is_generator"):
                metadata["is_generator"] = node.is_generator

            chunk = Chunk(
                symbol=symbol,
                start_line=LineNumber(start_line),
                end_line=LineNumber(end_line),
                code=code,
                chunk_type=chunk_type,
                file_id=file_id,
                language=self._language,
                file_path=FilePath(str(file_path)) if file_path else None,
                start_byte=start_byte,
                end_byte=end_byte,
                metadata=metadata,
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

    def parse_with_result(self, file_path: Path, file_id: FileId) -> ParseResult:
        """Parse a file and return detailed result information.

        Args:
            file_path: Path to the file to parse
            file_id: Database file ID for chunk association

        Returns:
            ParseResult with chunks, metadata, and diagnostics
        """
        import time

        start_time = time.time()

        try:
            chunks = self.parse_file(file_path, file_id)
            parse_time = time.time() - start_time

            chunk_dicts = [chunk.to_dict() for chunk in chunks]

            return ParseResult(
                chunks=chunk_dicts,
                language=self._language,
                total_chunks=len(chunks),
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
        self, file_path: Path, previous_chunks: list[dict[str, object]] | None = None
    ) -> list[Chunk]:
        return self.parse_file(file_path, FileId(0))

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
