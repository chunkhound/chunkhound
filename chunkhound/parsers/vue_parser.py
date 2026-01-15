"""Specialized parser for Vue Single File Components.

Vue SFCs require special handling because they contain multiple language
sections (template, script, style) that need to be parsed separately.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from chunkhound.core.models.chunk import Chunk
from chunkhound.core.types.common import (
    FileId,
    Language,
    LineNumber,
)
from chunkhound.parsers.chunk_splitter import (
    CASTConfig,
    ChunkSplitter,
    universal_to_chunk,
)
from chunkhound.parsers.mappings.vue import VueMapping
from chunkhound.parsers.mappings.vue_template import VueTemplateMapping
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.parsers.universal_engine import UniversalChunk, UniversalConcept
from chunkhound.parsers.universal_parser import UniversalParser
from chunkhound.parsers.vue_cross_ref import add_cross_references

if TYPE_CHECKING:
    pass


class VueParser:
    """Parser for Vue Single File Components.

    Architecture: Custom Orchestration (Non-Standard)
    ================================================

    VueParser uses custom orchestration instead of UniversalParser because:

    1. Multi-Language Sections:
       - <script>: JavaScript/TypeScript (uses TypeScript tree-sitter grammar)
       - <template>: Vue template syntax (uses Vue tree-sitter grammar)
       - <style>: CSS/SCSS (extracted as text chunks)

    2. Section Extraction Required:
       - Must split file BEFORE parsing (regex or tree-sitter-vue)
       - Each section needs different parser instance

    3. Cross-Reference Post-Processing:
       - Template variables link to script symbols
       - Example: {{ userName }} → const userName = ref('')
       - Requires both sections parsed before linking

    4. Line Number Adjustments:
       - Chunks must show position in original file, not section position
       - Template at line 50 in file, line 1 in section → report line 50

    Standard UniversalParser Pattern (Other Languages):
    ---------------------------------------------------
    ParserFactory → UniversalParser(TreeSitterEngine, Mapping) → Chunks

    Vue Pattern:
    -----------
    ParserFactory → VueParser → [
        UniversalParser(TypeScript) for script,
        UniversalParser(VueTemplate) for template,
        add_cross_references(script_chunks, template_chunks)
    ] → Chunks

    Why Not UniversalParser:
    - Assumes single language per file
    - Cannot split multi-section files
    - No cross-language reference linking

    See: vue_cross_ref.py for reference linking implementation
    """

    def __init__(self, cast_config: CASTConfig | None = None):
        """Initialize Vue parser.

        Args:
            cast_config: Configuration for cAST chunking algorithm
        """
        self.vue_mapping = VueMapping()
        self.vue_template_mapping = VueTemplateMapping()
        self.cast_config = cast_config or CASTConfig()
        self.chunk_splitter = ChunkSplitter(self.cast_config)

        # Create TypeScript parser for script sections
        self.ts_parser = create_parser_for_language(Language.TYPESCRIPT, cast_config)

        # Create template parser using tree-sitter-vue
        self.template_parser = self._create_template_parser()

    def _create_template_parser(self) -> UniversalParser | None:
        """Create a UniversalParser for Vue template content.

        Returns:
            UniversalParser instance or None if tree-sitter-vue is not available
        """
        try:
            from tree_sitter_language_pack import get_language

            from chunkhound.parsers.universal_engine import TreeSitterEngine

            # Get Vue tree-sitter language
            vue_lang = get_language("vue")

            # Create TreeSitterEngine for Vue templates
            engine = TreeSitterEngine("vue", vue_lang)

            # Create UniversalParser with VueTemplateMapping
            return UniversalParser(engine, self.vue_template_mapping, self.cast_config)

        except ImportError:
            # tree-sitter-language-pack not available
            return None
        except Exception:
            # Any other error initializing template parser
            return None

    def parse_file(self, file_path: Path, file_id: FileId) -> list[Chunk]:
        """Parse a Vue SFC file.

        Args:
            file_path: Path to .vue file
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
        """Parse Vue SFC content.

        Args:
            content: Full Vue SFC source
            file_path: Optional file path for metadata
            file_id: Optional file ID for chunks

        Returns:
            List of chunks from all sections
        """
        chunks: list[Chunk] = []
        script_chunks: list[Chunk] = []
        template_chunks: list[Chunk] = []
        full_script_content: str | None = None  # Store for cross-reference analysis

        # Extract sections using tree-sitter (falls back to regex if not available)
        sections = self.vue_mapping.extract_sections_ts(content)

        # Parse script sections with TypeScript parser
        for attrs, script_content, start_line in sections["script"]:
            # Store script content for cross-reference analysis
            full_script_content = script_content
            is_setup = self.vue_mapping.is_script_setup(attrs)
            script_lang = self.vue_mapping.get_script_lang(attrs)

            # Detect Vue macros and composables in script content
            vue_macros = self.vue_mapping.detect_vue_macros(script_content)
            vue_composables = self.vue_mapping.detect_composables(script_content)

            # Parse script content as TypeScript/JavaScript
            parsed_chunks = self.ts_parser.parse_content(
                script_content, file_path, file_id
            )

            # Create new chunks with adjusted line numbers and Vue-specific metadata
            for chunk in parsed_chunks:
                # Create updated metadata
                updated_metadata = (
                    chunk.metadata.copy() if chunk.metadata is not None else {}
                )
                updated_metadata["vue_section"] = "script"
                updated_metadata["vue_script_setup"] = is_setup
                updated_metadata["vue_script_lang"] = script_lang
                updated_metadata["is_vue_sfc"] = True

                # Add macros and composables if detected
                if vue_macros:
                    updated_metadata["vue_macros"] = vue_macros
                if vue_composables:
                    updated_metadata["vue_composables"] = vue_composables

                # Create new chunk with adjusted line numbers and metadata
                # Chunks are frozen dataclasses, so we need to create a new one
                from dataclasses import replace

                adjusted_chunk = replace(
                    chunk,
                    start_line=LineNumber(chunk.start_line + start_line),
                    end_line=LineNumber(chunk.end_line + start_line),
                    language=Language.VUE,  # Override to VUE from TYPESCRIPT
                    metadata=updated_metadata,
                )

                script_chunks.append(adjusted_chunk)

        # Parse template sections with VueTemplateMapping
        for attrs, template_content, start_line in sections["template"]:
            if template_content.strip():
                # Try to parse template directives using tree-sitter
                parsed_template_chunks = self._parse_template_content(
                    template_content, start_line, file_path, file_id
                )

                if parsed_template_chunks:
                    # Successfully parsed template directives
                    template_chunks.extend(parsed_template_chunks)
                else:
                    # Fallback: create simple text block for template
                    end_line = start_line + template_content.count("\n")

                    uchunk = UniversalChunk(
                        concept=UniversalConcept.BLOCK,
                        name="vue_template",
                        content=template_content,
                        start_line=start_line,
                        end_line=end_line,
                        metadata={"vue_section": "template", "is_vue_sfc": True},
                        language_node_type="vue_template",
                    )

                    # Split if needed and convert to Chunks
                    for uc in self.chunk_splitter.validate_and_split(uchunk):
                        chunk = universal_to_chunk(
                            uc, file_path, file_id, Language.VUE
                        )
                        template_chunks.append(chunk)

        # Create chunks for style sections (optional, as text blocks)
        for attrs, style_content, start_line in sections["style"]:
            if style_content.strip():
                end_line = start_line + style_content.count("\n")
                is_scoped = "scoped" in attrs.lower()

                uchunk = UniversalChunk(
                    concept=UniversalConcept.BLOCK,
                    name="vue_style",
                    content=style_content,
                    start_line=start_line,
                    end_line=end_line,
                    metadata={
                        "vue_section": "style",
                        "vue_style_scoped": is_scoped,
                        "is_vue_sfc": True,
                    },
                    language_node_type="vue_style",
                )

                # Split if needed and convert to Chunks
                for uc in self.chunk_splitter.validate_and_split(uchunk):
                    chunks.append(
                        universal_to_chunk(uc, file_path, file_id, Language.VUE)
                    )

        # Perform cross-reference analysis (Phase 2.3)
        # Link template references to script symbols
        if script_chunks and template_chunks:
            symbol_table, updated_template_chunks = add_cross_references(
                script_chunks, template_chunks, full_script_content
            )
            # Add script chunks first (for ordering)
            chunks.extend(script_chunks)
            # Add updated template chunks with cross-references
            chunks.extend(updated_template_chunks)
        else:
            # No cross-references to add, just add all chunks
            chunks.extend(script_chunks)
            chunks.extend(template_chunks)

        return chunks

    def _parse_template_content(
        self,
        template_content: str,
        start_line: int,
        file_path: Path | None,
        file_id: FileId | None,
    ) -> list[Chunk]:
        """Parse template content to extract directives and component usage.

        Args:
            template_content: Template section content
            start_line: Starting line number of the template section
            file_path: Optional file path
            file_id: Optional file ID

        Returns:
            List of chunks extracted from template, or empty list if parsing fails
        """
        if not self.template_parser:
            # Template parser not available, return empty list
            return []

        try:
            # Wrap template content in <template> tags for tree-sitter-vue parsing
            # This is necessary because tree-sitter-vue expects full Vue SFC structure
            wrapped_content = f"<template>\n{template_content}\n</template>"

            # Parse wrapped template content with VueTemplateMapping
            template_chunks = self.template_parser.parse_content(
                wrapped_content, file_path, file_id
            )

            # Adjust line numbers to account for:
            # 1. The <template> wrapper line (subtract 1)
            # 2. Template section position in file (add start_line)
            from dataclasses import replace

            adjusted_chunks = []
            for chunk in template_chunks:
                adjusted_chunk = replace(
                    chunk,
                    start_line=LineNumber(chunk.start_line + start_line - 2),
                    end_line=LineNumber(chunk.end_line + start_line - 2),
                )
                adjusted_chunks.append(adjusted_chunk)

            return adjusted_chunks

        except Exception:
            # Parsing failed, return empty list to trigger fallback
            return []
