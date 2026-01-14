"""PDF language mapping for unified parser architecture.

This module provides PDF text extraction and parsing logic
for the universal concept system. It handles PDF pages, paragraphs,
and text blocks using PyMuPDF for text extraction and follows the
same chunking patterns as TextMapping.
"""

import re
from pathlib import Path
from typing import Any

from tree_sitter import Node

from chunkhound.core.models.chunk import Chunk
from chunkhound.core.types.common import (
    ChunkType,
    FileId,
    FilePath,
    Language,
    LineNumber,
)
from chunkhound.parsers.mappings.base import BaseMapping
from chunkhound.parsers.universal_engine import UniversalConcept

try:
    import fitz  # type: ignore  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


class PDFMapping(BaseMapping):
    """PDF-specific mapping for universal concepts using text extraction."""

    def __init__(self) -> None:
        """Initialize PDF mapping."""
        super().__init__(Language.PDF)

    def parse_pdf_content(
        self, content_bytes: bytes, file_path: Path | None, file_id: FileId | None
    ) -> list[Chunk]:
        """Parse PDF content and extract semantic chunks.

        This method handles the complete PDF parsing workflow:
        1. Extract text from PDF using PyMuPDF
        2. Create chunks organized by pages and paragraphs
        3. Apply proper naming and metadata

        Args:
            content_bytes: PDF file content as bytes
            file_path: Optional file path for metadata
            file_id: Optional file ID for chunk association

        Returns:
            List of Chunk objects with optimal boundaries
        """
        if not PYMUPDF_AVAILABLE:
            # Return a single error chunk when PyMuPDF is not available
            return [
                Chunk(
                    symbol="pdf_unavailable",
                    start_line=LineNumber(1),
                    end_line=LineNumber(1),
                    code="PDF parsing not available (PyMuPDF not installed)",
                    chunk_type=ChunkType.UNKNOWN,
                    file_id=file_id or FileId(0),
                    language=Language.PDF,
                    file_path=FilePath(str(file_path)) if file_path else None,
                )
            ]

        try:
            # Extract text from PDF using PyMuPDF
            doc = fitz.open(stream=content_bytes, filetype="pdf")
            chunks = []
            current_line = 1

            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                page_text = page.get_text()

                # Clean up the extracted text
                page_text = self._clean_pdf_text(page_text)

                if not page_text.strip():
                    continue  # Skip empty pages

                # Split page into paragraphs
                paragraphs = self._split_into_paragraphs(page_text)
                page_start_line = current_line

                for para_idx, paragraph in enumerate(paragraphs):
                    if not paragraph.strip():
                        continue

                    # Only create chunk if it has meaningful content
                    if len(paragraph.strip()) >= 50:  # Minimum content threshold
                        para_lines = len(paragraph.split("\n"))

                        chunk = Chunk(
                            symbol=f"page_{page_num + 1}_paragraph_{para_idx + 1}",
                            start_line=LineNumber(current_line),
                            end_line=LineNumber(current_line + para_lines - 1),
                            code=paragraph,
                            chunk_type=ChunkType.PARAGRAPH,
                            file_id=file_id or FileId(0),
                            language=Language.PDF,
                            file_path=FilePath(str(file_path)) if file_path else None,
                        )
                        chunks.append(chunk)
                        current_line += para_lines
                    else:
                        # Still increment line counter for small paragraphs
                        current_line += len(paragraph.split("\n"))

                # If no paragraphs were extracted, create a single page chunk
                if not any(
                    c.symbol.startswith(f"page_{page_num + 1}_") for c in chunks
                ):
                    page_lines = len(page_text.split("\n"))
                    chunk = Chunk(
                        symbol=f"page_{page_num + 1}_content",
                        start_line=LineNumber(page_start_line),
                        end_line=LineNumber(current_line + page_lines - 1),
                        code=page_text,
                        chunk_type=ChunkType.PARAGRAPH,
                        file_id=file_id or FileId(0),
                        language=Language.PDF,
                        file_path=FilePath(str(file_path)) if file_path else None,
                    )
                    chunks.append(chunk)
                    current_line += page_lines

            doc.close()
            return chunks

        except Exception as e:
            # Return a single error chunk on parsing failure
            return [
                Chunk(
                    symbol="pdf_parse_error",
                    start_line=LineNumber(1),
                    end_line=LineNumber(1),
                    code=f"Error parsing PDF: {str(e)}",
                    chunk_type=ChunkType.UNKNOWN,
                    file_id=file_id or FileId(0),
                    language=Language.PDF,
                    file_path=FilePath(str(file_path)) if file_path else None,
                )
            ]

    # BaseMapping required methods
    def get_function_query(self) -> str:
        """Get tree-sitter query pattern for function definitions.

        Not applicable to PDF files.
        """
        return ""

    def get_class_query(self) -> str:
        """Get tree-sitter query pattern for class definitions.

        Not applicable to PDF files.
        """
        return ""

    def get_comment_query(self) -> str:
        """Get tree-sitter query pattern for comments (not applicable to PDF)."""
        return ""

    def extract_function_name(self, node: Node | None, source: str) -> str:
        """Extract function name from a function definition node.

        Not applicable to PDF files.
        """
        return ""

    def extract_class_name(self, node: Node | None, source: str) -> str:
        """Extract class name from a class definition node (not applicable to PDF)."""
        return ""

    # LanguageMapping protocol methods
    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        """Get tree-sitter query for universal concept in PDF.

        Since PDF doesn't use tree-sitter, we return None for all concepts
        and handle parsing through text extraction and analysis.
        """
        # PDF doesn't use tree-sitter queries - we'll parse using text extraction
        return None

    def extract_name(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract name from captures for this concept."""

        # Extract text from PDF content
        pdf_text, page_info = self._extract_pdf_text(content)

        if concept == UniversalConcept.DEFINITION:
            # Look for headings or prominent lines across pages
            for page_num, page_text in page_info.items():
                lines = page_text.split("\n")
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue

                    # Check for heading patterns
                    if line.startswith("#"):
                        # Markdown-style heading
                        heading = line.lstrip("#").strip()
                        if heading:
                            return f"page_{page_num}_heading_{heading[:30]}"
                    elif len(line) > 5 and (line.isupper() or line.endswith(":")):
                        # All caps or ending with colon might be a section header
                        header = line.rstrip(":").strip()
                        return f"page_{page_num}_section_{header[:30]}"
                    elif i < len(lines) - 1 and lines[i + 1].strip() in [
                        "=" * len(line),
                        "-" * len(line),
                    ]:
                        # Underlined heading
                        return f"page_{page_num}_title_{line[:30]}"
                    elif line.startswith(("•", "-", "·")):
                        # List item
                        item_text = line.lstrip("•-· ").strip()
                        if item_text:
                            return f"page_{page_num}_list_item_{item_text[:20]}"
                    elif re.match(r"^\d+\.", line):
                        # Numbered list
                        item_text = re.sub(r"^\d+\.\s*", "", line).strip()
                        if item_text:
                            return f"page_{page_num}_numbered_item_{item_text[:20]}"

                    # If it's the first non-empty line on the page, use it
                    if i == 0 or (
                        i > 0 and not any(lines[j].strip() for j in range(i))
                    ):
                        return f"page_{page_num}_first_line_{line[:30]}"

            return "pdf_definition"

        elif concept == UniversalConcept.BLOCK:
            # Identify text blocks (paragraphs, pages)
            total_paragraphs = sum(
                len(self._split_into_paragraphs(page_text))
                for page_text in page_info.values()
            )
            total_pages = len(page_info)

            if total_pages > 1:
                return f"pdf_document_{total_pages}_pages_{total_paragraphs}_paragraphs"
            else:
                return f"single_page_{total_paragraphs}_paragraphs"

        elif concept == UniversalConcept.COMMENT:
            # Look for annotations, footnotes, or metadata
            for page_num, page_text in page_info.items():
                lines = page_text.split("\n")
                for line in lines:
                    line = line.strip()
                    if line.startswith(("NOTE:", "WARNING:", "IMPORTANT:", "CAUTION:")):
                        annotation = line.split(":")[0].lower()
                        return f"page_{page_num}_annotation_{annotation}"
                    elif re.match(r"^\[\d+\]", line) or re.match(r"^\d+\)", line):
                        # Footnote or reference
                        return f"page_{page_num}_footnote"
                    elif line.startswith("---") or line.startswith("==="):
                        # Divider lines
                        return f"page_{page_num}_divider"

            return "pdf_comment"

        elif concept == UniversalConcept.IMPORT:
            # Look for references to other documents, URLs, or cross-references
            for page_num, page_text in page_info.items():
                lines = page_text.split("\n")
                for line in lines:
                    line = line.strip().lower()
                    if any(
                        keyword in line
                        for keyword in [
                            "see page",
                            "see section",
                            "refer to",
                            "reference",
                        ]
                    ):
                        return f"page_{page_num}_cross_reference"
                    elif line.startswith(("http://", "https://", "www.")):
                        return f"page_{page_num}_url_reference"
                    elif re.search(
                        r"\b[\w.-]+\.(pdf|doc|docx|txt)\b", line, re.IGNORECASE
                    ):
                        return f"page_{page_num}_document_reference"

            return "pdf_import"

        elif concept == UniversalConcept.STRUCTURE:
            return f"pdf_document_{len(page_info)}_pages"

        # This should never be reached since all enum cases are handled above
        raise ValueError(f"Unexpected concept: {concept}")

    def extract_content(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract content from captures for this concept."""

        # Extract text from PDF content
        pdf_text, page_info = self._extract_pdf_text(content)

        if concept == UniversalConcept.DEFINITION:
            # Extract the first meaningful section or paragraph from first page
            if page_info:
                first_page_text = list(page_info.values())[0]
                paragraphs = self._split_into_paragraphs(first_page_text)
                if paragraphs:
                    return paragraphs[0]
                else:
                    lines = first_page_text.split("\n")
                    # Return first few non-empty lines
                    meaningful_lines = [line for line in lines[:5] if line.strip()]
                    return (
                        "\n".join(meaningful_lines)
                        if meaningful_lines
                        else first_page_text[:200]
                    )
            return pdf_text[:200] if pdf_text else ""

        elif concept == UniversalConcept.BLOCK:
            # Return content organized by pages and paragraphs
            if len(page_info) > 1:
                # Multi-page document - return summary with page structure
                summary_parts = []
                for page_num, page_text in page_info.items():
                    paragraphs = self._split_into_paragraphs(page_text)
                    summary_parts.append(
                        f"Page {page_num}: {len(paragraphs)} paragraphs"
                    )
                    if paragraphs:
                        # Include first paragraph preview
                        preview = (
                            paragraphs[0][:100] + "..."
                            if len(paragraphs[0]) > 100
                            else paragraphs[0]
                        )
                        summary_parts.append(f"Preview: {preview}")
                return "\n".join(summary_parts)
            else:
                return pdf_text

        else:
            # Return the entire PDF content for other concepts
            return pdf_text

    def extract_metadata(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> dict[str, Any]:
        """Extract PDF-specific metadata."""

        pdf_text, page_info = self._extract_pdf_text(content)
        metadata: dict[str, Any] = {}

        # Basic PDF statistics
        total_pages = len(page_info)
        total_words = len(pdf_text.split()) if pdf_text else 0
        total_chars = len(pdf_text) if pdf_text else 0

        metadata["page_count"] = total_pages
        metadata["word_count"] = total_words
        metadata["char_count"] = total_chars

        # Per-page analysis
        page_stats = {}
        for page_num, page_text in page_info.items():
            page_words = len(page_text.split())
            page_lines = len([line for line in page_text.split("\n") if line.strip()])
            page_stats[f"page_{page_num}"] = {
                "word_count": page_words,
                "line_count": page_lines,
                "char_count": len(page_text),
            }
        metadata["page_statistics"] = page_stats

        if concept == UniversalConcept.DEFINITION:
            # Analyze document structure
            all_headings = []
            for page_num, page_text in page_info.items():
                headings = self._extract_headings(page_text, page_num)
                all_headings.extend(headings)

            if all_headings:
                metadata["headings"] = all_headings[:10]  # Limit to first 10
                metadata["has_structure"] = True

            # Look for tables of contents or indices
            if any(
                "table of contents" in page_text.lower()
                or "contents" in page_text.lower()
                for page_text in page_info.values()
            ):
                metadata["has_table_of_contents"] = True
                metadata["document_type"] = "structured"
            elif len(all_headings) > 3:
                metadata["document_type"] = "hierarchical"
            else:
                metadata["document_type"] = "plain"

        elif concept == UniversalConcept.BLOCK:
            # Analyze paragraph structure across pages
            all_paragraphs = []
            for page_text in page_info.values():
                paragraphs = self._split_into_paragraphs(page_text)
                all_paragraphs.extend(paragraphs)

            metadata["total_paragraphs"] = len(all_paragraphs)

            if all_paragraphs:
                avg_paragraph_length = sum(
                    len(p.split()) for p in all_paragraphs
                ) / len(all_paragraphs)
                metadata["avg_paragraph_words"] = int(avg_paragraph_length)

                # Analyze paragraph distribution across pages
                pages_with_single_paragraph = sum(
                    1
                    for page_text in page_info.values()
                    if len(self._split_into_paragraphs(page_text)) == 1
                )
                pages_with_many_paragraphs = sum(
                    1
                    for page_text in page_info.values()
                    if len(self._split_into_paragraphs(page_text)) > 5
                )

                if pages_with_single_paragraph > total_pages * 0.7:
                    metadata["layout_style"] = "single_paragraph_pages"
                elif pages_with_many_paragraphs > total_pages * 0.5:
                    metadata["layout_style"] = "multi_paragraph_pages"
                else:
                    metadata["layout_style"] = "mixed_layout"

        elif concept == UniversalConcept.COMMENT:
            # Look for annotations, footnotes, headers/footers
            footnotes = 0
            annotations = []
            for page_num, page_text in page_info.items():
                footnotes += len(re.findall(r"^\[\d+\]", page_text, re.MULTILINE))
                page_annotations = self._extract_annotations(page_text)
                annotations.extend(page_annotations)

            if footnotes > 0:
                metadata["footnote_count"] = footnotes
            if annotations:
                metadata["annotation_count"] = len(annotations)
                metadata["annotation_types"] = list(set(annotations))

        elif concept == UniversalConcept.IMPORT:
            # Extract URLs and document references
            all_urls = []
            all_references = []
            for page_text in page_info.values():
                urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', page_text)
                all_urls.extend(urls)

                file_refs = re.findall(
                    r"\b[\w.-]+\.(pdf|doc|docx|txt|html)\b", page_text, re.IGNORECASE
                )
                all_references.extend(file_refs)

            if all_urls:
                metadata["url_count"] = len(all_urls)
                metadata["unique_urls"] = len(set(all_urls))

            if all_references:
                metadata["document_references"] = len(all_references)
                metadata["reference_types"] = list(
                    set(ext.split(".")[-1].lower() for ext in all_references)
                )

        elif concept == UniversalConcept.STRUCTURE:
            # Overall document analysis
            metadata["document_format"] = "pdf"

            # Estimate document complexity based on structure
            total_headings = sum(
                len(self._extract_headings(page_text, page_num))
                for page_num, page_text in page_info.items()
            )

            if total_pages > 50:
                metadata["document_size"] = "large"
            elif total_pages > 10:
                metadata["document_size"] = "medium"
            else:
                metadata["document_size"] = "small"

            if total_headings > total_pages * 2:
                metadata["structure_complexity"] = "high"
            elif total_headings > total_pages * 0.5:
                metadata["structure_complexity"] = "medium"
            else:
                metadata["structure_complexity"] = "low"

            # Reading time estimate (200 words per minute)
            if total_words > 0:
                estimated_minutes = max(1, total_words // 200)
                metadata["estimated_reading_time_minutes"] = estimated_minutes

        return metadata

    def _extract_pdf_text(self, content: bytes) -> tuple[str, dict[int, str]]:
        """Extract text from PDF content using PyMuPDF.

        Args:
            content: PDF file content as bytes

        Returns:
            Tuple of (full_text, page_info_dict)
            where page_info_dict maps page_number -> page_text
        """
        if not PYMUPDF_AVAILABLE:
            return "PDF parsing not available (PyMuPDF not installed)", {}

        try:
            doc = fitz.open(stream=content, filetype="pdf")
            page_info = {}
            all_text = []

            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                page_text = page.get_text()

                # Clean up the extracted text
                page_text = self._clean_pdf_text(page_text)

                if page_text.strip():  # Only include non-empty pages
                    page_info[page_num + 1] = page_text  # 1-based page numbers
                    all_text.append(f"\n--- Page {page_num + 1} ---\n{page_text}")

            doc.close()
            return "\n".join(all_text), page_info

        except Exception as e:
            return f"Error extracting PDF text: {str(e)}", {}

    def _clean_pdf_text(self, text: str) -> str:
        """Clean up extracted PDF text."""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        # Remove form feed characters and other control characters
        text = re.sub(r"[\f\v\r]+", "\n", text)

        # Fix common PDF extraction issues
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)  # Add space between words
        text = re.sub(r"(\w)(\d)", r"\1 \2", text)  # Space between word and number
        text = re.sub(
            r"(\d)([A-Za-z])", r"\1 \2", text
        )  # Space between number and word

        return text.strip()

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs using blank line separation."""
        paragraphs = []
        current_paragraph = []

        for line in text.split("\n"):
            if line.strip():
                current_paragraph.append(line)
            else:
                if current_paragraph:
                    paragraphs.append("\n".join(current_paragraph))
                    current_paragraph = []

        # Add the last paragraph if it exists
        if current_paragraph:
            paragraphs.append("\n".join(current_paragraph))

        return paragraphs

    def _extract_headings(self, text: str, page_num: int) -> list[str]:
        """Extract potential headings from text."""
        headings = []
        lines = text.split("\n")

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Markdown-style headings
            if line.startswith("#"):
                heading = line.lstrip("#").strip()
                if heading:
                    headings.append(f"Page {page_num}: {heading}")

            # Underlined headings
            elif i < len(lines) - 1 and lines[i + 1].strip() in [
                "=" * len(line),
                "-" * len(line),
            ]:
                headings.append(f"Page {page_num}: {line}")

            # All caps lines (potential headings)
            elif len(line) > 3 and line.isupper() and not line.endswith("."):
                headings.append(f"Page {page_num}: {line}")

            # Lines ending with colon (section headers)
            elif line.endswith(":") and len(line.split()) < 8:
                headings.append(f"Page {page_num}: {line[:-1]}")

        return headings

    def _extract_annotations(self, text: str) -> list[str]:
        """Extract annotation types from text."""
        annotations = []

        for match in re.finditer(
            r"\b(NOTE|IMPORTANT|WARNING|CAUTION|ATTENTION):", text, re.IGNORECASE
        ):
            annotations.append(match.group(1).upper())

        return annotations

    def resolve_import_path(
        self,
        import_text: str,
        base_dir: Path,
        source_file: Path
    ) -> Path | None:
        """Data formats don't have imports."""
        return None
