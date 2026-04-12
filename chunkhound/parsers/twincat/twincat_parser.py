"""TwinCAT Structured Text parser for ChunkHound.

Architecture: Custom Orchestration (like Svelte/Vue)
- Uses Lark for parsing (not tree-sitter)
- Directly processes lark.Tree and lark.Token objects (no AST transformation)
- Handles multi-section XML files (declaration + implementation)
- Adjusts line numbers from CDATA-relative to XML-absolute
"""

import re
from pathlib import Path
from typing import Any

from lark import Lark, Token, Tree
from lark.exceptions import LarkError
from loguru import logger

from chunkhound.core.types.common import ChunkType
from chunkhound.parsers.twincat.xml_extractor import (
    ActionContent,
    MethodContent,
    POUContent,
    PropertyContent,
    SourceLocation,
    TcPOUExtractor,
)
from chunkhound.parsers.universal_engine import UniversalChunk, UniversalConcept

# Regex patterns for comment extraction
# Block comments: (* ... *)
BLOCK_COMMENT_RE = re.compile(r"\(\*[\s\S]*?\*\)")
# Line comments: // ...
LINE_COMMENT_RE = re.compile(r"//[^\n]*")

# Map VAR block keywords to semantic variable classes
VAR_BLOCK_MAP = {
    "VAR_INPUT": "input",
    "VAR_OUTPUT": "output",
    "VAR_IN_OUT": "in_out",
    "VAR_GLOBAL": "global",
    "VAR_EXTERNAL": "external",
    "VAR_TEMP": "temp",
    "VAR_STAT": "static",
    "VAR": "local",
}


class TwinCATParser:
    """Parser for TwinCAT TcPOU files.

    Directly processes Lark parse trees (Tree/Token objects) without
    AST transformation to extract semantic chunks.
    """

    def __init__(self) -> None:
        self._grammar_dir = Path(__file__).parent
        self._decl_parser: Lark | None = None
        self._impl_parser: Lark | None = None
        self._extractor = TcPOUExtractor()
        self._parse_errors: list[str] = []

    @property
    def parse_errors(self) -> list[str]:
        """Errors from the most recent parse operation. Cleared on each parse."""
        return self._parse_errors

    @property
    def decl_parser(self) -> Lark:
        """Lazy-load declaration parser."""
        if self._decl_parser is None:
            grammar_path = self._grammar_dir / "declarations.lark"
            self._decl_parser = Lark.open(
                str(grammar_path),
                parser="lalr",
                lexer="contextual",
                propagate_positions=True,
            )
        return self._decl_parser

    @property
    def impl_parser(self) -> Lark:
        """Lazy-load implementation parser for Structured Text code blocks.

        Used to parse METHOD, ACTION, and PROPERTY implementation bodies
        when extracting detailed chunk information from TcPOU files.
        """
        if self._impl_parser is None:
            grammar_path = self._grammar_dir / "implementation.lark"
            self._impl_parser = Lark.open(
                str(grammar_path),
                parser="lalr",
                lexer="contextual",
                propagate_positions=True,
            )
        return self._impl_parser

    # =========================================================================
    # UniversalChunk Extraction (for UniversalParser integration)
    # =========================================================================

    def extract_universal_chunks(
        self,
        content: str,
        file_path: Path | None = None,
    ) -> list[UniversalChunk]:
        """Extract UniversalChunk objects from TcPOU content.

        This method produces UniversalChunk objects that can flow through
        the UniversalParser's cAST pipeline for deduplication, comment
        merging, and greedy merge optimization.

        Args:
            content: TcPOU XML content string
            file_path: Optional path to the source file

        Returns:
            List of UniversalChunk objects
        """
        pou_content = self._extractor.extract_string(content)
        return self._process_pou_content_to_universal(pou_content, file_path, content)

    def _process_pou_content_to_universal(
        self,
        content: POUContent,
        file_path: Path | None,
        raw_content: str,
    ) -> list[UniversalChunk]:
        """Process extracted POU content into UniversalChunk objects."""
        self._parse_errors = []  # Clear errors at start of each parse
        chunks: list[UniversalChunk] = []

        # 1. Create POU declaration and implementation chunks
        pou_chunks = self._create_pou_universal_chunks(content, file_path)
        chunks.extend(pou_chunks)

        # 2. Parse declaration section → extract variable chunks
        if content.declaration and content.declaration.strip():
            try:
                decl_tree = self.decl_parser.parse(content.declaration)
                var_chunks = self._extract_var_universal_chunks_from_tree(
                    decl_tree, content, file_path
                )
                chunks.extend(var_chunks)
            except LarkError as e:
                error_msg = f"Declaration parse error in {content.name}: {e}"
                logger.error(error_msg)
                self._parse_errors.append(error_msg)

        if content.implementation and content.implementation.strip():
            block_chunks = self._extract_block_universal_chunks_from_implementation(
                content.implementation,
                content.implementation_location,
                content.name,
                content.pou_type.upper(),
                file_path,
            )
            chunks.extend(block_chunks)

        # 3. Extract comments from declaration and implementation
        decl_base_line = (
            content.declaration_location.line if content.declaration_location else 1
        )
        impl_base_line = (
            content.implementation_location.line
            if content.implementation_location
            else decl_base_line
        )

        if content.declaration and content.declaration.strip():
            comment_chunks = self._extract_comment_universal_chunks(
                content.declaration,
                file_path,
                content.name,
                content.pou_type.upper(),
                decl_base_line,
            )
            chunks.extend(comment_chunks)

        if content.implementation and content.implementation.strip():
            comment_chunks = self._extract_comment_universal_chunks(
                content.implementation,
                file_path,
                content.name,
                content.pou_type.upper(),
                impl_base_line,
            )
            chunks.extend(comment_chunks)

        # 4. Parse actions → create action chunks
        for action in content.actions:
            action_chunks = self._extract_action_universal_chunks(
                action, content, file_path
            )
            chunks.extend(action_chunks)

        # 5. Parse methods → create method chunks
        for method in content.methods:
            method_chunks = self._extract_method_universal_chunks(
                method, content, file_path
            )
            chunks.extend(method_chunks)

        # 6. Parse properties → create property chunks
        for prop in content.properties:
            property_chunks = self._extract_property_universal_chunks(
                prop, content, file_path
            )
            chunks.extend(property_chunks)

        # 7. Extract imports (VAR_EXTERNAL, EXTENDS, IMPLEMENTS, type references)
        import_chunks = self._extract_import_universal_chunks_from_pou(content)
        chunks.extend(import_chunks)

        # Validate chunk positions against raw file content
        self._validate_chunk_positions(chunks, raw_content)

        # Stable sort: by start line, then larger spans first (containers first).
        # The -end_line trick: when chunks share a start line, larger spans
        # (more negative -end_line) sort first, placing containers before children.
        return sorted(chunks, key=lambda c: (c.start_line, -c.end_line))


    def _map_chunk_type_to_concept(self, chunk_type: ChunkType) -> UniversalConcept:
        """Map TwinCAT ChunkType to UniversalConcept.

        Mapping:
        - PROGRAM, FUNCTION_BLOCK, FUNCTION → DEFINITION
        - METHOD, ACTION, PROPERTY → DEFINITION
        - VARIABLE, FIELD → DEFINITION
        - BLOCK (control flow) → BLOCK
        - COMMENT → COMMENT
        """
        if chunk_type in (
            ChunkType.PROGRAM,
            ChunkType.FUNCTION_BLOCK,
            ChunkType.FUNCTION,
            ChunkType.METHOD,
            ChunkType.ACTION,
            ChunkType.PROPERTY,
            ChunkType.VARIABLE,
            ChunkType.FIELD,
        ):
            return UniversalConcept.DEFINITION
        elif chunk_type == ChunkType.BLOCK:
            return UniversalConcept.BLOCK
        elif chunk_type == ChunkType.COMMENT:
            return UniversalConcept.COMMENT
        else:
            return UniversalConcept.DEFINITION  # Default for unknown types

    def _create_universal_chunk(
        self,
        chunk_type: ChunkType,
        name: str,
        content: str,
        start_line: int,
        end_line: int,
        metadata: dict[str, Any],
        language_node_type: str,
    ) -> UniversalChunk:
        """Create UniversalChunk from TwinCAT extraction data.

        Args:
            chunk_type: The TwinCAT ChunkType
            name: Symbol name for the chunk
            content: Code content
            start_line: Starting line number (1-based)
            end_line: Ending line number (1-based)
            metadata: Additional metadata dict
            language_node_type: Original node type (using "lark_{rule}" format)

        Returns:
            UniversalChunk instance
        """
        # Store original ChunkType name for accurate reverse mapping
        enriched_metadata = {
            **metadata,
            "chunk_type_hint": chunk_type.name.lower(),
        }

        return UniversalChunk(
            concept=self._map_chunk_type_to_concept(chunk_type),
            name=name,
            content=content,
            start_line=start_line,
            end_line=end_line,
            metadata=enriched_metadata,
            language_node_type=language_node_type,
        )

    def _create_cdata_chunk(
        self,
        content: str | None,
        location: SourceLocation | None,
        name_suffix: str,
        base_name: str,
        chunk_type: ChunkType,
        base_metadata: dict[str, Any],
        language_node_type: str,
    ) -> UniversalChunk | None:
        """Create UniversalChunk for a single CDATA section.

        Args:
            content: The CDATA content (declaration or implementation)
            location: Source location of the CDATA section
            name_suffix: Section type ("declaration", "implementation", "get", "set")
            base_name: Base FQN name (e.g., "FB_Test" or "FB_Test.Method1")
            chunk_type: The ChunkType for this chunk
            base_metadata: Base metadata dict to extend
            language_node_type: Original node type for the chunk

        Returns:
            UniversalChunk if content is non-empty, None otherwise
        """
        if not content or not content.strip():
            return None

        # Calculate line numbers
        start_line = location.line if location else 1
        end_line = start_line + content.count("\n")

        # Build FQN with section suffix
        fqn = f"{base_name}.{name_suffix}"

        # Extend metadata with section info
        metadata = {
            **base_metadata,
            "section": name_suffix,
        }

        return self._create_universal_chunk(
            chunk_type=chunk_type,
            name=fqn,
            content=content,
            start_line=start_line,
            end_line=end_line,
            metadata=metadata,
            language_node_type=language_node_type,
        )

    def _create_pou_universal_chunks(
        self,
        content: POUContent,
        file_path: Path | None,
    ) -> list[UniversalChunk]:
        """Create UniversalChunks for POU declaration and implementation sections."""
        chunks: list[UniversalChunk] = []

        # Map POU type to ChunkType
        pou_type = content.pou_type.upper()
        if pou_type == "PROGRAM":
            chunk_type = ChunkType.PROGRAM
        elif pou_type == "FUNCTION_BLOCK":
            chunk_type = ChunkType.FUNCTION_BLOCK
        elif pou_type == "FUNCTION":
            chunk_type = ChunkType.FUNCTION
        else:
            chunk_type = ChunkType.BLOCK

        base_metadata = {
            "kind": pou_type.lower(),
            "pou_type": pou_type,
            "pou_name": content.name,
            "pou_id": content.id,
        }

        # Create declaration chunk
        decl_chunk = self._create_cdata_chunk(
            content=content.declaration,
            location=content.declaration_location,
            name_suffix="declaration",
            base_name=content.name,
            chunk_type=chunk_type,
            base_metadata=base_metadata,
            language_node_type="lark_pou_declaration",
        )
        if decl_chunk:
            chunks.append(decl_chunk)

        # Create implementation chunk
        impl_chunk = self._create_cdata_chunk(
            content=content.implementation,
            location=content.implementation_location,
            name_suffix="implementation",
            base_name=content.name,
            chunk_type=chunk_type,
            base_metadata=base_metadata,
            language_node_type="lark_pou_implementation",
        )
        if impl_chunk:
            chunks.append(impl_chunk)

        return chunks

    def _extract_var_universal_chunks_from_tree(
        self,
        tree: Tree,
        content: POUContent,
        file_path: Path | None,
        declaration_location: SourceLocation | None = None,
        action_name: str | None = None,
        method_name: str | None = None,
    ) -> list[UniversalChunk]:
        """Walk Lark tree and extract variable UniversalChunks from VAR blocks."""
        chunks: list[UniversalChunk] = []

        # Use provided location or fall back to POU declaration location
        location = declaration_location or content.declaration_location

        # Find all var_block nodes
        var_blocks = list(tree.find_data("var_block"))

        for var_block in var_blocks:
            # Extract var class from var_block_start
            var_class = "local"  # default
            retain = False
            persistent = False
            constant = False

            for child in var_block.children:
                if isinstance(child, Tree):
                    if child.data == "var_block_start":
                        for token in child.children:
                            if isinstance(token, Token):
                                var_class = VAR_BLOCK_MAP.get(
                                    token.type, token.value.lower()
                                )
                                break
                    elif child.data == "var_qualifier":
                        for token in child.children:
                            if isinstance(token, Token):
                                if token.type == "RETAIN":
                                    retain = True
                                elif token.type == "PERSISTENT":
                                    persistent = True
                                elif token.type == "CONSTANT":
                                    constant = True
                    elif child.data == "var_declaration":
                        var_chunks = self._extract_var_decl_universal_chunk(
                            child,
                            content,
                            file_path,
                            var_class,
                            retain,
                            persistent,
                            constant,
                            location,
                            action_name,
                            method_name,
                        )
                        chunks.extend(var_chunks)

        return chunks

    def _extract_var_decl_universal_chunk(
        self,
        var_decl: Tree,
        content: POUContent,
        file_path: Path | None,
        var_class: str,
        retain: bool,
        persistent: bool,
        constant: bool,
        declaration_location: SourceLocation | None = None,
        action_name: str | None = None,
        method_name: str | None = None,
    ) -> list[UniversalChunk]:
        """Extract UniversalChunk(s) from a var_declaration node."""
        chunks: list[UniversalChunk] = []

        # Collect variable names (IDENTIFIERs before the colon)
        var_names: list[str] = []
        data_type: str | None = None
        hw_address: str | None = None

        # Get line number from node metadata
        line = var_decl.meta.line if hasattr(var_decl, "meta") and var_decl.meta else 1

        # Use provided location or fall back to POU declaration location
        location = declaration_location or content.declaration_location

        # Adjust line number to XML position
        adjusted_line = self._adjust_line_number(line, location)

        for child in var_decl.children:
            if isinstance(child, Token):
                if child.type == "IDENTIFIER" and data_type is None:
                    var_names.append(str(child))
            elif isinstance(child, Tree):
                if child.data == "hw_location":
                    hw_addr_token = self._get_token_value(child, "HW_ADDRESS")
                    if hw_addr_token:
                        hw_address = hw_addr_token
                elif child.data == "type_spec":
                    data_type = self._extract_type_spec(child)

        # Reconstruct the declaration code
        code = ", ".join(var_names)
        if hw_address:
            code += f" AT {hw_address}"
        code += f" : {data_type or 'UNKNOWN'};"

        # Determine ChunkType and kind based on variable scope
        if var_class in ("global", "external"):
            chunk_type = ChunkType.VARIABLE
            kind = "variable"
        else:
            chunk_type = ChunkType.FIELD
            kind = "field"

        # Build metadata
        metadata: dict[str, Any] = {
            "kind": kind,
            "pou_type": content.pou_type,
            "pou_name": content.name,
            "var_class": var_class,
            "data_type": data_type,
            "hw_address": hw_address,
            "retain": retain,
            "persistent": persistent,
            "constant": constant,
        }
        if action_name:
            metadata["action_name"] = action_name
        if method_name:
            metadata["method_name"] = method_name

        # Create a chunk for each variable name
        for var_name in var_names:
            fqn = self._build_fqn(content.name, var_name, method_name, action_name)
            chunk = self._create_universal_chunk(
                chunk_type=chunk_type,
                name=fqn,
                content=code,
                start_line=adjusted_line,
                end_line=adjusted_line,
                metadata=metadata.copy(),
                language_node_type="lark_var_declaration",
            )
            chunks.append(chunk)

        return chunks

    def _extract_action_universal_chunks(
        self,
        action: ActionContent,
        content: POUContent,
        file_path: Path | None,
    ) -> list[UniversalChunk]:
        """Create UniversalChunks for action declaration and implementation sections."""
        chunks: list[UniversalChunk] = []

        # Base metadata for action chunks
        base_metadata = {
            "kind": "action",
            "pou_type": content.pou_type,
            "pou_name": content.name,
            "action_id": action.id,
        }

        base_name = f"{content.name}.{action.name}"

        # Create declaration chunk (only if declaration exists)
        if action.declaration and action.declaration.strip():
            decl_chunk = self._create_cdata_chunk(
                content=action.declaration,
                location=action.declaration_location,
                name_suffix="declaration",
                base_name=base_name,
                chunk_type=ChunkType.ACTION,
                base_metadata=base_metadata,
                language_node_type="lark_action_declaration",
            )
            if decl_chunk:
                chunks.append(decl_chunk)

        # Create implementation chunk
        impl_chunk = self._create_cdata_chunk(
            content=action.implementation,
            location=action.implementation_location,
            name_suffix="implementation",
            base_name=base_name,
            chunk_type=ChunkType.ACTION,
            base_metadata=base_metadata,
            language_node_type="lark_action_implementation",
        )
        if impl_chunk:
            chunks.append(impl_chunk)

        # Skip further processing if no chunks created
        if not chunks:
            return chunks

        # Parse action declaration for variables
        if action.declaration and action.declaration.strip():
            try:
                decl_tree = self.decl_parser.parse(action.declaration)
                var_chunks = self._extract_var_universal_chunks_from_tree(
                    decl_tree,
                    content,
                    file_path,
                    declaration_location=action.declaration_location,
                    action_name=action.name,
                )
                chunks.extend(var_chunks)
            except LarkError as e:
                error_msg = f"Action '{action.name}' declaration parse error: {e}"
                logger.error(error_msg)
                self._parse_errors.append(error_msg)

        # Parse action implementation for control flow blocks
        if action.implementation and action.implementation.strip():
            block_chunks = self._extract_block_universal_chunks_from_implementation(
                action.implementation,
                action.implementation_location,
                content.name,
                content.pou_type.upper(),
                file_path,
                action_name=action.name,
            )
            chunks.extend(block_chunks)

        # Extract comments
        if action.declaration and action.declaration.strip():
            decl_base = (
                action.declaration_location.line
                if action.declaration_location
                else 1
            )
            comment_chunks = self._extract_comment_universal_chunks(
                action.declaration,
                file_path,
                content.name,
                content.pou_type.upper(),
                decl_base,
                action_name=action.name,
            )
            chunks.extend(comment_chunks)

        if action.implementation and action.implementation.strip():
            impl_base = (
                action.implementation_location.line
                if action.implementation_location
                else 1
            )
            comment_chunks = self._extract_comment_universal_chunks(
                action.implementation,
                file_path,
                content.name,
                content.pou_type.upper(),
                impl_base,
                action_name=action.name,
            )
            chunks.extend(comment_chunks)

        return chunks

    def _extract_method_universal_chunks(
        self,
        method: MethodContent,
        content: POUContent,
        file_path: Path | None,
    ) -> list[UniversalChunk]:
        """Create UniversalChunks for method declaration and implementation sections."""
        chunks: list[UniversalChunk] = []

        # Base metadata for method chunks
        base_metadata = {
            "kind": "method",
            "pou_type": content.pou_type,
            "pou_name": content.name,
            "method_id": method.id,
        }

        base_name = f"{content.name}.{method.name}"

        # Create declaration chunk
        decl_chunk = self._create_cdata_chunk(
            content=method.declaration,
            location=method.declaration_location,
            name_suffix="declaration",
            base_name=base_name,
            chunk_type=ChunkType.METHOD,
            base_metadata=base_metadata,
            language_node_type="lark_method_declaration",
        )
        if decl_chunk:
            chunks.append(decl_chunk)

        # Create implementation chunk
        impl_chunk = self._create_cdata_chunk(
            content=method.implementation,
            location=method.implementation_location,
            name_suffix="implementation",
            base_name=base_name,
            chunk_type=ChunkType.METHOD,
            base_metadata=base_metadata,
            language_node_type="lark_method_implementation",
        )
        if impl_chunk:
            chunks.append(impl_chunk)

        # Skip further processing if no chunks created
        if not chunks:
            return chunks

        # Parse method declaration for variables
        if method.declaration and method.declaration.strip():
            try:
                decl_tree = self.decl_parser.parse(method.declaration)
                var_chunks = self._extract_var_universal_chunks_from_tree(
                    decl_tree,
                    content,
                    file_path,
                    declaration_location=method.declaration_location,
                    method_name=method.name,
                )
                chunks.extend(var_chunks)
            except LarkError as e:
                error_msg = f"Method '{method.name}' declaration parse error: {e}"
                logger.error(error_msg)
                self._parse_errors.append(error_msg)

        # Parse method implementation for control flow blocks
        if method.implementation and method.implementation.strip():
            block_chunks = self._extract_block_universal_chunks_from_implementation(
                method.implementation,
                method.implementation_location,
                content.name,
                content.pou_type.upper(),
                file_path,
                method_name=method.name,
            )
            chunks.extend(block_chunks)

        # Extract comments
        if method.declaration and method.declaration.strip():
            decl_base = (
                method.declaration_location.line
                if method.declaration_location
                else 1
            )
            comment_chunks = self._extract_comment_universal_chunks(
                method.declaration,
                file_path,
                content.name,
                content.pou_type.upper(),
                decl_base,
                method_name=method.name,
            )
            chunks.extend(comment_chunks)

        if method.implementation and method.implementation.strip():
            impl_base = (
                method.implementation_location.line
                if method.implementation_location
                else 1
            )
            comment_chunks = self._extract_comment_universal_chunks(
                method.implementation,
                file_path,
                content.name,
                content.pou_type.upper(),
                impl_base,
                method_name=method.name,
            )
            chunks.extend(comment_chunks)

        return chunks

    def _extract_property_universal_chunks(
        self,
        prop: PropertyContent,
        content: POUContent,
        file_path: Path | None,
    ) -> list[UniversalChunk]:
        """Create UniversalChunks for a property's declaration and accessor sections."""
        chunks: list[UniversalChunk] = []

        # Base metadata for property chunks
        base_metadata = {
            "kind": "property",
            "pou_type": content.pou_type,
            "pou_name": content.name,
            "property_id": prop.id,
        }

        base_name = f"{content.name}.{prop.name}"

        # Create declaration chunk
        decl_chunk = self._create_cdata_chunk(
            content=prop.declaration,
            location=prop.declaration_location,
            name_suffix="declaration",
            base_name=base_name,
            chunk_type=ChunkType.PROPERTY,
            base_metadata=base_metadata,
            language_node_type="lark_property_declaration",
        )
        if decl_chunk:
            chunks.append(decl_chunk)

        # Create get accessor chunk
        if prop.get and prop.get.implementation and prop.get.implementation.strip():
            get_chunk = self._create_cdata_chunk(
                content=prop.get.implementation,
                location=prop.get.implementation_location,
                name_suffix="get",
                base_name=base_name,
                chunk_type=ChunkType.PROPERTY,
                base_metadata=base_metadata,
                language_node_type="lark_property_get",
            )
            if get_chunk:
                chunks.append(get_chunk)

        # Create set accessor chunk
        if prop.set and prop.set.implementation and prop.set.implementation.strip():
            set_chunk = self._create_cdata_chunk(
                content=prop.set.implementation,
                location=prop.set.implementation_location,
                name_suffix="set",
                base_name=base_name,
                chunk_type=ChunkType.PROPERTY,
                base_metadata=base_metadata,
                language_node_type="lark_property_set",
            )
            if set_chunk:
                chunks.append(set_chunk)

        return chunks

    def _extract_block_universal_chunks_from_implementation(
        self,
        implementation: str,
        implementation_location: SourceLocation | None,
        pou_name: str,
        pou_type: str,
        file_path: Path | None,
        action_name: str | None = None,
        method_name: str | None = None,
    ) -> list[UniversalChunk]:
        """Extract control flow blocks as UniversalChunks."""
        chunks: list[UniversalChunk] = []

        try:
            tree = self.impl_parser.parse(implementation)
        except LarkError as e:
            if method_name:
                context = f"method '{method_name}'"
            elif action_name:
                context = f"action '{action_name}'"
            else:
                context = f"FUNCTION '{pou_name}'"
            error_msg = f"Implementation parse error in {context}: {e}"
            logger.error(error_msg)
            self._parse_errors.append(error_msg)
            return chunks

        # Find all control flow statement nodes
        statement_nodes = self._find_statement_nodes(tree)

        for node in statement_nodes:
            chunk = self._create_block_universal_chunk(
                node,
                implementation,
                implementation_location,
                pou_name,
                pou_type,
                file_path,
                action_name,
                method_name,
            )
            if chunk:
                chunks.append(chunk)

        return chunks

    def _create_block_universal_chunk(
        self,
        node: Tree,
        implementation: str,
        implementation_location: SourceLocation | None,
        pou_name: str,
        pou_type: str,
        file_path: Path | None,
        action_name: str | None,
        method_name: str | None = None,
    ) -> UniversalChunk | None:
        """Create UniversalChunk for a control flow block."""
        # Get line numbers from node metadata
        if not hasattr(node, "meta") or node.meta is None:
            return None

        start_line = node.meta.line
        end_line = node.meta.end_line or start_line

        # Reconstruct code from implementation using line numbers
        code = self._reconstruct_code_from_lines(implementation, start_line, end_line)

        # Adjust line numbers to XML-absolute
        adjusted_start = self._adjust_line_number(start_line, implementation_location)
        adjusted_end = self._adjust_line_number(end_line, implementation_location)

        # Determine kind from statement type
        kind = self._STATEMENT_KIND_MAP.get(node.data, "block")

        # Build FQN
        symbol = self._build_fqn(
            pou_name, f"{kind}_{adjusted_start}", method_name, action_name
        )

        # Build metadata
        metadata: dict[str, Any] = {
            "kind": kind,
            "pou_type": pou_type,
            "pou_name": pou_name,
        }
        if action_name:
            metadata["action_name"] = action_name
        if method_name:
            metadata["method_name"] = method_name

        return self._create_universal_chunk(
            chunk_type=ChunkType.BLOCK,
            name=symbol,
            content=code,
            start_line=adjusted_start,
            end_line=adjusted_end,
            metadata=metadata,
            language_node_type=f"lark_{node.data}",
        )

    def _extract_comment_universal_chunks(
        self,
        source: str,
        file_path: Path | None,
        pou_name: str,
        pou_type: str,
        base_line: int,
        method_name: str | None = None,
        action_name: str | None = None,
    ) -> list[UniversalChunk]:
        """Extract comments as UniversalChunks."""
        chunks: list[UniversalChunk] = []

        # Block comments: (* ... *)
        for match in BLOCK_COMMENT_RE.finditer(source):
            line = source[: match.start()].count("\n") + base_line
            chunk = self._create_comment_universal_chunk(
                content=match.group(),
                line=line,
                file_path=file_path,
                pou_name=pou_name,
                pou_type=pou_type,
                comment_type="block",
                method_name=method_name,
                action_name=action_name,
            )
            chunks.append(chunk)

        # Line comments: // ...
        for match in LINE_COMMENT_RE.finditer(source):
            line = source[: match.start()].count("\n") + base_line
            chunk = self._create_comment_universal_chunk(
                content=match.group(),
                line=line,
                file_path=file_path,
                pou_name=pou_name,
                pou_type=pou_type,
                comment_type="line",
                method_name=method_name,
                action_name=action_name,
            )
            chunks.append(chunk)

        return chunks

    def _create_comment_universal_chunk(
        self,
        content: str,
        line: int,
        file_path: Path | None,
        pou_name: str,
        pou_type: str,
        comment_type: str,
        method_name: str | None = None,
        action_name: str | None = None,
    ) -> UniversalChunk:
        """Create a comment UniversalChunk."""
        # Build FQN
        element_name = f"comment_line_{line}"
        fqn = self._build_fqn(pou_name, element_name, method_name, action_name)

        # Calculate end line for multi-line block comments
        end_line = line + content.count("\n")

        # Clean comment text (strip markers)
        cleaned_text = self._clean_st_comment(content)

        # Build metadata
        metadata: dict[str, Any] = {
            "kind": "comment",
            "comment_type": comment_type,
            "pou_name": pou_name,
            "pou_type": pou_type,
            "cleaned_text": cleaned_text,
        }
        if method_name:
            metadata["method_name"] = method_name
        if action_name:
            metadata["action_name"] = action_name

        return self._create_universal_chunk(
            chunk_type=ChunkType.COMMENT,
            name=fqn,
            content=content,
            start_line=line,
            end_line=end_line,
            metadata=metadata,
            language_node_type="lark_comment",
        )

    def _extract_type_spec(self, type_spec: Tree) -> str:
        """Extract type specification as a string."""
        parts: list[str] = []

        for child in type_spec.children:
            if isinstance(child, Token):
                parts.append(str(child))
            elif isinstance(child, Tree):
                if child.data == "primitive_type":
                    # Get the token from primitive_type
                    for token in child.children:
                        if isinstance(token, Token):
                            parts.append(str(token))
                elif child.data == "string_type_with_size":
                    # STRING(80) or WSTRING[100]
                    parts.append(self._extract_string_type(child))
                elif child.data == "array_type":
                    parts.append(self._extract_array_type(child))
                elif child.data == "pointer_type":
                    parts.append(f"POINTER TO {self._extract_type_spec(child)}")
                elif child.data == "reference_type":
                    parts.append(f"REFERENCE TO {self._extract_type_spec(child)}")
                elif child.data == "user_type":
                    # User-defined type is just an IDENTIFIER
                    for token in child.children:
                        if isinstance(token, Token) and token.type == "IDENTIFIER":
                            parts.append(str(token))
                elif child.data == "type_spec":
                    # Nested type spec (for POINTER TO, REFERENCE TO)
                    parts.append(self._extract_type_spec(child))

        return " ".join(parts) if parts else "UNKNOWN"

    def _extract_string_type(self, string_type: Tree) -> str:
        """Extract STRING(n) or WSTRING[n] type."""
        type_name = "STRING"
        size: str | None = None

        for child in string_type.children:
            if isinstance(child, Token):
                if child.type == "STRING_TYPE":
                    type_name = "STRING"
                elif child.type == "WSTRING":
                    type_name = "WSTRING"
                elif child.type == "INTEGER":
                    size = str(child)

        if size:
            return f"{type_name}({size})"
        return type_name

    def _extract_array_type(self, array_type: Tree) -> str:
        """Extract ARRAY[...] OF type."""
        ranges: list[str] = []
        element_type = "UNKNOWN"

        for child in array_type.children:
            if isinstance(child, Tree):
                if child.data == "array_range":
                    ranges.append(self._extract_array_range(child))
                elif child.data == "type_spec":
                    element_type = self._extract_type_spec(child)

        return f"ARRAY[{', '.join(ranges)}] OF {element_type}"

    def _extract_array_range(self, array_range: Tree) -> str:
        """Extract array range like 0..9 or 1..MAX_SIZE."""
        bounds: list[str] = []

        for child in array_range.children:
            if isinstance(child, Tree):
                if child.data == "array_bound":
                    bounds.append(self._extract_array_bound(child))
                elif child.data == "integer_value":
                    bounds.append(self._extract_integer_value(child))
            elif isinstance(child, Token):
                if child.type == "IDENTIFIER":
                    bounds.append(str(child))

        return "..".join(bounds)

    def _extract_array_bound(self, bound: Tree) -> str:
        """Extract a single array bound."""
        for child in bound.children:
            if isinstance(child, Token):
                if child.type == "IDENTIFIER":
                    return str(child)
            elif isinstance(child, Tree):
                if child.data == "integer_value":
                    return self._extract_integer_value(child)

        return "0"

    def _extract_integer_value(self, int_val: Tree) -> str:
        """Extract integer value (may include sign)."""
        parts: list[str] = []

        for child in int_val.children:
            if isinstance(child, Token):
                parts.append(str(child))

        return "".join(parts)

    def _adjust_line_number(
        self, line: int, location: SourceLocation | None
    ) -> int:
        """Adjust line number from CDATA-relative to XML-absolute."""
        if location is None:
            return line
        return line + (location.line - 1)

    def _validate_chunk_positions(
        self,
        chunks: list[UniversalChunk],
        raw_content: str,
    ) -> None:
        """Validate that chunk line numbers match the source file.

        For each chunk, verifies that:
        - The first line of chunk content is a substring of file's start_line
        - The last line of chunk content is a substring of file's end_line

        Logs warnings for mismatches but does not modify or filter chunks.

        Args:
            chunks: List of UniversalChunk objects to validate
            raw_content: The raw TcPOU XML file content
        """
        file_lines = raw_content.splitlines()

        for chunk in chunks:
            # Skip validation for synthesized content that won't match source
            skip_types = ("lark_var_declaration", "lark_type_reference")
            if chunk.language_node_type in skip_types:
                continue

            chunk_lines = chunk.content.splitlines()
            if not chunk_lines:
                continue

            # Get first and last content lines (stripped for comparison)
            first_content_line = chunk_lines[0].strip()
            last_content_line = chunk_lines[-1].strip()

            # Convert 1-based line numbers to 0-based array indices
            # start_line=1 means first line of file, which is index 0
            start_idx = chunk.start_line - 1
            end_idx = chunk.end_line - 1

            # Bounds check
            chunk_info = f"{chunk.name} ({chunk.concept.value})"
            if start_idx < 0 or start_idx >= len(file_lines):
                logger.warning(
                    f"Chunk {chunk_info} start_line {chunk.start_line} "
                    f"out of bounds (file has {len(file_lines)} lines)"
                )
                continue
            if end_idx < 0 or end_idx >= len(file_lines):
                logger.warning(
                    f"Chunk {chunk_info} end_line {chunk.end_line} "
                    f"out of bounds (file has {len(file_lines)} lines)"
                )
                continue

            file_start_line = file_lines[start_idx]
            file_end_line = file_lines[end_idx]

            # Validate first line
            if first_content_line and first_content_line not in file_start_line:
                logger.warning(
                    f"Chunk {chunk_info} first line mismatch at {chunk.start_line}:\n"
                    f"  Expected: {first_content_line!r}\n"
                    f"  File: {file_start_line!r}"
                )

            # Validate last line (with off-by-one tolerance)
            if last_content_line:
                found_at_end = last_content_line in file_end_line
                found_at_prev = False
                if not found_at_end and end_idx - 1 >= 0:
                    found_at_prev = last_content_line in file_lines[end_idx - 1]

                if not found_at_end and not found_at_prev:
                    msg = (
                        f"Chunk {chunk_info} last line mismatch at {chunk.end_line}:\n"
                        f"  Expected: {last_content_line!r}"
                    )
                    if end_idx - 1 >= 0:
                        prev_line = file_lines[end_idx - 1]
                        msg += f"\n  Line {chunk.end_line - 1}: {prev_line!r}"
                    msg += f"\n  Line {chunk.end_line}: {file_end_line!r}"

                    logger.warning(msg)

    @staticmethod
    def _build_fqn(
        pou_name: str,
        element_name: str,
        method_name: str | None = None,
        action_name: str | None = None,
    ) -> str:
        """Build a fully qualified name for a chunk symbol.

        FQN hierarchy: POUName[.MethodName|.ActionName].ElementName
        """
        if method_name:
            return f"{pou_name}.{method_name}.{element_name}"
        elif action_name:
            return f"{pou_name}.{action_name}.{element_name}"
        return f"{pou_name}.{element_name}"

    def _get_token_value(self, tree: Tree, token_type: str) -> str | None:
        """Find first token of given type in tree's children."""
        for child in tree.children:
            if isinstance(child, Token) and child.type == token_type:
                return str(child)
        return None

    # =========================================================================
    # Implementation Block Extraction (Tier 4 - Control Flow Blocks)
    # =========================================================================

    # Statement type to metadata kind mapping
    _STATEMENT_KIND_MAP = {
        "if_stmt": "if_block",
        "case_stmt": "case_block",
        "for_stmt": "for_loop",
        "while_stmt": "while_loop",
        "repeat_stmt": "repeat_loop",
    }

    def _find_statement_nodes(self, tree: Tree) -> list[Tree]:
        """Recursively find all control flow statement nodes in parse tree.

        Finds: if_stmt, case_stmt, for_stmt, while_stmt, repeat_stmt
        """
        results: list[Tree] = []

        if isinstance(tree, Tree):
            if tree.data in self._STATEMENT_KIND_MAP:
                results.append(tree)
            # Recurse into children to find nested statements
            for child in tree.children:
                if isinstance(child, Tree):
                    results.extend(self._find_statement_nodes(child))

        return results

    def _reconstruct_code_from_lines(
        self, source: str, start_line: int, end_line: int
    ) -> str:
        """Extract code substring using line numbers.

        Args:
            source: Full source code string
            start_line: 1-based start line number
            end_line: 1-based end line number (inclusive)

        Returns:
            Code substring spanning the specified lines
        """
        lines = source.splitlines()

        # Convert to 0-based indices
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)

        return "\n".join(lines[start_idx:end_idx])

    def _clean_st_comment(self, text: str) -> str:
        """Strip ST comment markers (* *) and //.

        Args:
            text: Raw comment text with markers

        Returns:
            Cleaned comment text without markers
        """
        cleaned = text.strip()
        if cleaned.startswith("(*") and cleaned.endswith("*)"):
            cleaned = cleaned[2:-2].strip()
        elif cleaned.startswith("//"):
            cleaned = cleaned[2:].strip()
        return cleaned

    # =========================================================================
    # Import Extraction (Tier 5 - Dependencies)
    # =========================================================================

    # IEC 61131-3 primitive types to skip when extracting type references
    PRIMITIVE_TYPES = frozenset({
        "BOOL", "BYTE", "WORD", "DWORD", "LWORD",
        "SINT", "USINT", "INT", "UINT", "DINT", "UDINT", "LINT", "ULINT",
        "REAL", "LREAL", "TIME", "LTIME", "DATE", "LDATE",
        "TIME_OF_DAY", "TOD", "LTOD", "DATE_AND_TIME", "DT", "LDT",
        "STRING", "WSTRING",
        "ANY", "ANY_INT", "ANY_REAL", "ANY_NUM", "ANY_BIT", "ANY_STRING", "ANY_DATE",
    })

    def extract_import_chunks(
        self,
        content: str,
    ) -> list[UniversalChunk]:
        """Extract import-like constructs as UniversalChunks.

        Public method for efficient import-only extraction.

        Extracts:
        - VAR_EXTERNAL references to global variables
        - EXTENDS inheritance clauses
        - IMPLEMENTS interface implementations
        - User-defined type references in variable declarations

        Args:
            content: TcPOU XML content string

        Returns:
            List of UniversalChunk objects representing imports
        """
        pou_content = self._extractor.extract_string(content)
        return self._extract_import_universal_chunks_from_pou(pou_content)

    def _extract_import_universal_chunks_from_pou(
        self,
        content: POUContent,
    ) -> list[UniversalChunk]:
        """Extract import-like constructs as UniversalChunks.

        Extracts:
        - VAR_EXTERNAL references to global variables
        - EXTENDS inheritance clauses
        - IMPLEMENTS interface implementations
        - User-defined type references in variable declarations
        """
        chunks: list[UniversalChunk] = []

        # Parse declaration section
        if not content.declaration or not content.declaration.strip():
            return chunks

        try:
            decl_tree = self.decl_parser.parse(content.declaration)
        except LarkError:
            # Parse errors already logged in main extraction
            return chunks

        # 1. Extract VAR_EXTERNAL imports
        var_external_chunks = self._extract_var_external_imports(decl_tree, content)
        chunks.extend(var_external_chunks)

        # 2. Extract EXTENDS/IMPLEMENTS from pou_header
        inheritance_chunks = self._extract_inheritance_imports(decl_tree, content)
        chunks.extend(inheritance_chunks)

        # 3. Extract user-defined type references
        type_ref_chunks = self._extract_type_reference_imports(decl_tree, content)
        chunks.extend(type_ref_chunks)

        return chunks

    def _extract_var_external_imports(
        self,
        tree: Tree,
        content: POUContent,
    ) -> list[UniversalChunk]:
        """Extract VAR_EXTERNAL declarations as import chunks."""
        chunks: list[UniversalChunk] = []

        # Find all var_block nodes
        var_blocks = list(tree.find_data("var_block"))

        for var_block in var_blocks:
            # Check if this is a VAR_EXTERNAL block
            is_external = False
            for child in var_block.children:
                if isinstance(child, Tree) and child.data == "var_block_start":
                    for token in child.children:
                        if isinstance(token, Token) and token.type == "VAR_EXTERNAL":
                            is_external = True
                            break
                    break

            if not is_external:
                continue

            # Extract variable declarations from this VAR_EXTERNAL block
            for child in var_block.children:
                if isinstance(child, Tree) and child.data == "var_declaration":
                    chunk = self._create_var_external_import_chunk(child, content)
                    if chunk:
                        chunks.append(chunk)

        return chunks

    def _create_var_external_import_chunk(
        self,
        var_decl: Tree,
        content: POUContent,
    ) -> UniversalChunk | None:
        """Create import chunk for a VAR_EXTERNAL declaration."""
        # Extract variable name and type
        var_names: list[str] = []
        data_type: str | None = None

        for child in var_decl.children:
            if isinstance(child, Token) and child.type == "IDENTIFIER":
                if data_type is None:  # Names come before the type
                    var_names.append(str(child))
            elif isinstance(child, Tree) and child.data == "type_spec":
                data_type = self._extract_type_spec(child)

        if not var_names:
            return None

        # Use first variable name for the chunk
        var_name = var_names[0]

        # Get line number
        line = var_decl.meta.line if hasattr(var_decl, "meta") and var_decl.meta else 1
        adjusted_line = self._adjust_line_number(line, content.declaration_location)

        # Build FQN
        fqn = f"{content.name}.{var_name}"

        # Reconstruct declaration code
        code = f"{', '.join(var_names)} : {data_type or 'UNKNOWN'};"

        metadata: dict[str, Any] = {
            "kind": "import",
            "import_type": "var_external",
            "var_name": var_name,
            "data_type": data_type,
            "var_class": "external",
            "pou_name": content.name,
            "pou_type": content.pou_type.upper(),
        }

        return UniversalChunk(
            concept=UniversalConcept.IMPORT,
            name=fqn,
            content=code,
            start_line=adjusted_line,
            end_line=adjusted_line,
            metadata=metadata,
            language_node_type="lark_var_external",
        )

    def _extract_inheritance_imports(
        self,
        tree: Tree,
        content: POUContent,
    ) -> list[UniversalChunk]:
        """Extract EXTENDS and IMPLEMENTS clauses as import chunks."""
        chunks: list[UniversalChunk] = []

        # Find pou_header node
        pou_headers = list(tree.find_data("pou_header"))
        if not pou_headers:
            return chunks

        pou_header = pou_headers[0]

        # Extract extends_clause
        extends_clauses = list(pou_header.find_data("extends_clause"))
        for extends_clause in extends_clauses:
            chunk = self._create_extends_import_chunk(extends_clause, content)
            if chunk:
                chunks.append(chunk)

        # Extract implements_clause
        implements_clauses = list(pou_header.find_data("implements_clause"))
        for implements_clause in implements_clauses:
            impl_chunks = self._create_implements_import_chunks(
                implements_clause, content
            )
            chunks.extend(impl_chunks)

        return chunks

    def _create_extends_import_chunk(
        self,
        extends_clause: Tree,
        content: POUContent,
    ) -> UniversalChunk | None:
        """Create import chunk for EXTENDS clause."""
        # Extract base type identifier
        base_type: str | None = None
        for child in extends_clause.children:
            if isinstance(child, Token) and child.type == "IDENTIFIER":
                base_type = str(child)
                break

        if not base_type:
            return None

        # Get line number
        line = (
            extends_clause.meta.line
            if hasattr(extends_clause, "meta") and extends_clause.meta
            else 1
        )
        adjusted_line = self._adjust_line_number(line, content.declaration_location)

        # Build FQN
        fqn = f"{content.name}:extends:{base_type}"

        metadata: dict[str, Any] = {
            "kind": "import",
            "import_type": "extends",
            "base_type": base_type,
            "target_type": content.name,
            "pou_name": content.name,
            "pou_type": content.pou_type.upper(),
        }

        return UniversalChunk(
            concept=UniversalConcept.IMPORT,
            name=fqn,
            content=f"EXTENDS {base_type}",
            start_line=adjusted_line,
            end_line=adjusted_line,
            metadata=metadata,
            language_node_type="lark_extends",
        )

    def _create_implements_import_chunks(
        self,
        implements_clause: Tree,
        content: POUContent,
    ) -> list[UniversalChunk]:
        """Create import chunks for IMPLEMENTS clause (multiple interfaces)."""
        chunks: list[UniversalChunk] = []

        # Get line number
        line = (
            implements_clause.meta.line
            if hasattr(implements_clause, "meta") and implements_clause.meta
            else 1
        )
        adjusted_line = self._adjust_line_number(line, content.declaration_location)

        # Extract all interface identifiers
        for child in implements_clause.children:
            if isinstance(child, Token) and child.type == "IDENTIFIER":
                interface_name = str(child)

                # Build FQN
                fqn = f"{content.name}:implements:{interface_name}"

                metadata: dict[str, Any] = {
                    "kind": "import",
                    "import_type": "implements",
                    "interface_name": interface_name,
                    "implementing_type": content.name,
                    "pou_name": content.name,
                    "pou_type": content.pou_type.upper(),
                }

                chunk = UniversalChunk(
                    concept=UniversalConcept.IMPORT,
                    name=fqn,
                    content=f"IMPLEMENTS {interface_name}",
                    start_line=adjusted_line,
                    end_line=adjusted_line,
                    metadata=metadata,
                    language_node_type="lark_implements",
                )
                chunks.append(chunk)

        return chunks

    def _extract_type_reference_imports(
        self,
        tree: Tree,
        content: POUContent,
    ) -> list[UniversalChunk]:
        """Extract user-defined type references as import chunks.

        Scans all variable declarations for non-primitive type usage:
        - Direct: fbMotor : FB_Motor
        - Arrays: ARRAY[...] OF FB_Type
        - Pointers: POINTER TO FB_Type
        - References: REFERENCE TO FB_Type
        """
        chunks: list[UniversalChunk] = []
        seen_types: set[str] = set()  # Deduplicate type references

        # Find all var_declaration nodes
        var_decls = list(tree.find_data("var_declaration"))

        for var_decl in var_decls:
            # Get variable name(s) for context
            var_names: list[str] = []
            for child in var_decl.children:
                if isinstance(child, Token) and child.type == "IDENTIFIER":
                    var_names.append(str(child))
                elif isinstance(child, Tree) and child.data == "type_spec":
                    break  # Stop at type_spec

            # Find type_spec and extract user-defined types
            for child in var_decl.children:
                if isinstance(child, Tree) and child.data == "type_spec":
                    user_types = self._extract_user_types_from_type_spec(child)
                    for user_type in user_types:
                        # Skip if already seen
                        if user_type in seen_types:
                            continue
                        seen_types.add(user_type)

                        # Create import chunk
                        chunk = self._create_type_reference_import_chunk(
                            var_decl,
                            user_type,
                            var_names,
                            content,
                        )
                        if chunk:
                            chunks.append(chunk)

        return chunks

    def _extract_user_types_from_type_spec(self, type_spec: Tree) -> list[str]:
        """Recursively extract user-defined type names from type_spec."""
        user_types: list[str] = []

        for child in type_spec.children:
            if isinstance(child, Tree):
                if child.data == "user_type":
                    # Direct user type
                    for token in child.children:
                        if isinstance(token, Token) and token.type == "IDENTIFIER":
                            type_name = str(token).upper()
                            if type_name not in self.PRIMITIVE_TYPES:
                                user_types.append(str(token))
                elif child.data in ("array_type", "pointer_type", "reference_type"):
                    # Recurse into nested type_spec
                    nested_type_specs = list(child.find_data("type_spec"))
                    for nested in nested_type_specs:
                        user_types.extend(
                            self._extract_user_types_from_type_spec(nested)
                        )
                elif child.data == "type_spec":
                    # Direct nested type_spec
                    user_types.extend(self._extract_user_types_from_type_spec(child))

        return user_types

    def _create_type_reference_import_chunk(
        self,
        var_decl: Tree,
        referenced_type: str,
        var_names: list[str],
        content: POUContent,
    ) -> UniversalChunk | None:
        """Create import chunk for a user-defined type reference."""
        # Get line number
        line = var_decl.meta.line if hasattr(var_decl, "meta") and var_decl.meta else 1
        adjusted_line = self._adjust_line_number(line, content.declaration_location)

        # Build FQN
        fqn = f"{content.name}:type_ref:{referenced_type}"

        # Determine usage context
        var_name = var_names[0] if var_names else "unknown"

        metadata: dict[str, Any] = {
            "kind": "import",
            "import_type": "type_reference",
            "referenced_type": referenced_type,
            "var_name": var_name,
            "usage_context": "declaration",
            "pou_name": content.name,
            "pou_type": content.pou_type.upper(),
        }

        return UniversalChunk(
            concept=UniversalConcept.IMPORT,
            name=fqn,
            content=f"type reference: {referenced_type}",
            start_line=adjusted_line,
            end_line=adjusted_line,
            metadata=metadata,
            language_node_type="lark_type_reference",
        )
