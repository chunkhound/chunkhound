"""Specialized parser for Svelte Single File Components.

Svelte SFCs require special handling because they contain multiple language
sections (template, script, style) that need to be parsed separately.
"""

from pathlib import Path

from chunkhound.core.models.chunk import Chunk
from chunkhound.core.types.common import (
    ChunkType,
    FileId,
    FilePath,
    Language,
    LineNumber,
)
from chunkhound.parsers.mappings.svelte import SvelteMapping
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.parsers.universal_parser import CASTConfig


class SvelteParser:
    """Parser for Svelte Single File Components.

    Architecture: Custom Orchestration (Non-Standard)
    ================================================

    SvelteParser uses custom orchestration instead of UniversalParser because:

    1. Multi-Language Sections:
       - <script>: JavaScript/TypeScript (uses TypeScript tree-sitter grammar)
       - <template>: Svelte template syntax (extracted as text chunks)
       - <style>: CSS/SCSS (extracted as text chunks)

    2. Section Extraction Required:
       - Must split file BEFORE parsing (regex-based)
       - Script sections need TypeScript parser
       - Template and style are searchable text blocks

    3. Line Number Adjustments:
       - Chunks must show position in original file, not section position
       - Script at line 2 in file, line 1 in section → report line 2

    Standard UniversalParser Pattern (Other Languages):
    ---------------------------------------------------
    ParserFactory → UniversalParser(TreeSitterEngine, Mapping) → Chunks

    Svelte Pattern:
    --------------
    ParserFactory → SvelteParser → [
        UniversalParser(TypeScript) for script,
        Text chunks for template/style
    ] → Chunks

    Why Not UniversalParser:
    - Assumes single language per file
    - Cannot split multi-section files
    - Svelte template syntax not fully parsed initially (text-based approach)

    Note: Similar to VueParser but simpler (no cross-references in phase 1)
    """

    def __init__(self, cast_config: CASTConfig | None = None):
        """Initialize Svelte parser.

        Args:
            cast_config: Configuration for cAST chunking algorithm
        """
        self.svelte_mapping = SvelteMapping()
        self.cast_config = cast_config or CASTConfig()

        # Create TypeScript parser for script sections
        self.ts_parser = create_parser_for_language(Language.TYPESCRIPT, cast_config)

    def parse_file(self, file_path: Path, file_id: FileId) -> list[Chunk]:
        """Parse a Svelte SFC file.

        Args:
            file_path: Path to .svelte file
            file_id: Database file ID

        Returns:
            List of chunks from all sections
        """
        content = file_path.read_text(encoding="utf-8")
        return self.parse_content(content, file_path, file_id)

    def parse_content(
        self,
        content: str,
        file_path: Path | None = None,
        file_id: FileId | None = None,
    ) -> list[Chunk]:
        """Parse Svelte SFC content.

        Args:
            content: Full Svelte SFC source
            file_path: Optional file path for metadata
            file_id: Optional file ID for chunks

        Returns:
            List of chunks from all sections
        """
        chunks: list[Chunk] = []

        # Extract sections using regex
        sections = self.svelte_mapping.extract_sections(content)

        # Parse script sections with TypeScript parser
        for attrs, script_content, start_line in sections["script"]:
            # Parse script content as TypeScript/JavaScript
            parsed_chunks = self.ts_parser.parse_content(
                script_content, file_path, file_id
            )

            # Create new chunks with adjusted line numbers and Svelte-specific metadata
            for chunk in parsed_chunks:
                # Create updated metadata
                updated_metadata = (
                    chunk.metadata.copy() if chunk.metadata is not None else {}
                )
                updated_metadata["svelte_section"] = "script"
                updated_metadata["is_svelte_sfc"] = True

                # Add script language if detected
                if attrs:
                    # Simple lang extraction
                    import re

                    lang_match = re.search(
                        r'lang\s*=\s*["\']?(\w+)["\']?', attrs, re.IGNORECASE
                    )
                    if lang_match:
                        updated_metadata["svelte_script_lang"] = lang_match.group(
                            1
                        ).lower()

                # Create new chunk with adjusted line numbers and metadata
                # Chunks are frozen dataclasses, so we need to create a new one
                from dataclasses import replace

                adjusted_chunk = replace(
                    chunk,
                    start_line=LineNumber(chunk.start_line + start_line - 1),
                    end_line=LineNumber(chunk.end_line + start_line - 1),
                    language=Language.SVELTE,  # Override to SVELTE from TYPESCRIPT
                    metadata=updated_metadata,
                )

                chunks.append(adjusted_chunk)

        # Create chunks for template sections (as text blocks)
        for _, template_content, start_line in sections["template"]:
            if template_content.strip():
                # Calculate end line: start_line is already positioned correctly
                # by extract_sections(), and we need to count newlines in the content
                # The end line is the last line that contains content
                newline_count = template_content.count("\n")
                # If content ends with newline, the last line is before that newline
                if template_content.endswith("\n"):
                    end_line = start_line + newline_count - 1
                else:
                    end_line = start_line + newline_count

                # Ensure end_line is at least start_line
                end_line = max(start_line, end_line)

                template_chunk = Chunk(
                    symbol="svelte_template",
                    start_line=LineNumber(start_line),
                    end_line=LineNumber(end_line),
                    code=template_content,
                    chunk_type=ChunkType.BLOCK,
                    file_id=file_id or FileId(0),
                    language=Language.SVELTE,
                    file_path=FilePath(str(file_path)) if file_path else None,
                    metadata={"svelte_section": "template", "is_svelte_sfc": True},
                )
                chunks.append(template_chunk)

        # Create chunks for style sections (as text blocks)
        for _, style_content, start_line in sections["style"]:
            if style_content.strip():
                # Calculate end line consistently with template logic
                newline_count = style_content.count("\n")
                if style_content.endswith("\n"):
                    end_line = start_line + newline_count - 1
                else:
                    end_line = start_line + newline_count

                # Ensure end_line is at least start_line
                end_line = max(start_line, end_line)

                style_chunk = Chunk(
                    symbol="svelte_style",
                    start_line=LineNumber(start_line),
                    end_line=LineNumber(end_line),
                    code=style_content,
                    chunk_type=ChunkType.BLOCK,
                    file_id=file_id or FileId(0),
                    language=Language.SVELTE,
                    file_path=FilePath(str(file_path)) if file_path else None,
                    metadata={"svelte_section": "style", "is_svelte_sfc": True},
                )
                chunks.append(style_chunk)

        return chunks
