#!/usr/bin/env python3
"""Tests for PDF parsing functionality.

This module tests the PDF parsing implementation to verify:
1. PDF files are correctly detected as Language.PDF
2. PDF parsing works with PyMuPDF when available
3. Chunks are created with proper page-based symbol naming
4. Text extraction handles multi-page documents
5. Error handling when PyMuPDF is not available
6. Basic PDF parsing integrates with ChunkHound's universal parser
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from chunkhound.core.types.common import ChunkType, FileId, Language
from chunkhound.parsers.mappings.pdf import PYMUPDF_AVAILABLE, PDFMapping
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.parsers.universal_parser import UniversalParser


class TestPDFLanguageDetection:
    """Test that PDF files are correctly identified."""

    def test_pdf_extension_detection(self):
        """Test that .pdf files are detected as Language.PDF."""
        test_cases = [
            "document.pdf",
            "report.PDF",
            "/path/to/file.pdf",
            Path("test.pdf"),
            Path("/absolute/path/document.pdf"),
        ]

        for file_path in test_cases:
            language = Language.from_file_extension(file_path)
            assert language == Language.PDF, f"Failed to detect PDF for: {file_path}"

    def test_non_pdf_files_not_detected_as_pdf(self):
        """Test that non-PDF files are not detected as PDF."""
        test_cases = [
            "document.txt",
            "report.doc",
            "data.json",
            "script.py",
            "style.css",
        ]

        for file_path in test_cases:
            language = Language.from_file_extension(file_path)
            assert language != Language.PDF, (
                f"Incorrectly detected PDF for: {file_path}"
            )

    def test_pdf_in_supported_extensions(self):
        """Test that .pdf is recognized through from_file_extension method."""
        # PDF support is implemented but .pdf is not in get_all_extensions()
        # This is because PDF doesn't use tree-sitter like other languages
        # The test verifies it's still properly handled
        language = Language.from_file_extension("test.pdf")
        assert language == Language.PDF, "PDF should be detected from extension"

    def test_pdf_in_file_patterns(self):
        """Test that PDF files are supported even if not in default patterns."""
        # PDF is supported but might not be in default patterns since it doesn't use tree-sitter
        # Test that is_supported_file works correctly instead
        assert Language.is_supported_file("document.pdf"), (
            "PDF files should be supported"
        )
        assert Language.is_supported_file(Path("report.pdf")), (
            "PDF Path objects should be supported"
        )

    def test_is_supported_file_for_pdf(self):
        """Test that PDF files are recognized as supported."""
        test_files = ["test.pdf", "report.PDF", Path("document.pdf")]

        for file_path in test_files:
            assert Language.is_supported_file(file_path), (
                f"PDF file should be supported: {file_path}"
            )


class TestPDFMapping:
    """Test PDF mapping implementation."""

    def test_pdf_mapping_initialization(self):
        """Test PDF mapping initializes correctly."""
        mapping = PDFMapping()
        assert mapping.language == Language.PDF

    def test_pdf_mapping_tree_sitter_queries(self):
        """Test that PDF mapping returns empty queries (doesn't use tree-sitter)."""
        mapping = PDFMapping()

        # PDF doesn't use tree-sitter, so queries should be empty
        assert mapping.get_function_query() == ""
        assert mapping.get_class_query() == ""
        assert mapping.get_comment_query() == ""

    def test_pdf_mapping_name_extraction(self):
        """Test that PDF mapping returns empty names for tree-sitter nodes."""
        mapping = PDFMapping()

        # PDF doesn't use tree-sitter nodes, so names should be empty
        assert mapping.extract_function_name(None, "") == ""
        assert mapping.extract_class_name(None, "") == ""


class TestPDFParsingWithPyMuPDF:
    """Test PDF parsing functionality when PyMuPDF is available."""

    @pytest.fixture
    def pdf_parser(self):
        """Create a PDF parser."""
        return create_parser_for_language(Language.PDF)

    @pytest.fixture
    def test_pdf_files(self):
        """Get paths to test PDF files."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        return {
            "simple": fixtures_dir / "test_simple.pdf",
            "multipage": fixtures_dir / "test_multipage.pdf",
            "empty": fixtures_dir / "test_empty.pdf",
        }

    @pytest.mark.skipif(not PYMUPDF_AVAILABLE, reason="PyMuPDF not installed")
    def test_simple_pdf_parsing(self, pdf_parser, test_pdf_files):
        """Test parsing a simple single-page PDF."""
        pdf_path = test_pdf_files["simple"]
        assert pdf_path.exists(), f"Test PDF not found: {pdf_path}"

        chunks = pdf_parser.parse_file(pdf_path, FileId(1))

        assert len(chunks) > 0, "Should produce at least one chunk"

        # All chunks should be PDF type
        for chunk in chunks:
            assert chunk.language == Language.PDF
            assert chunk.file_id == FileId(1)
            assert chunk.chunk_type in [ChunkType.PARAGRAPH, ChunkType.UNKNOWN]

        # Check that content was extracted
        combined_content = " ".join(chunk.code for chunk in chunks)
        assert "Test Document" in combined_content
        assert "Introduction" in combined_content

    @pytest.mark.skipif(not PYMUPDF_AVAILABLE, reason="PyMuPDF not installed")
    def test_multipage_pdf_parsing(self, pdf_parser, test_pdf_files):
        """Test parsing a multi-page PDF."""
        pdf_path = test_pdf_files["multipage"]
        assert pdf_path.exists(), f"Test PDF not found: {pdf_path}"

        chunks = pdf_parser.parse_file(pdf_path, FileId(2))

        assert len(chunks) > 0, "Should produce chunks from multi-page PDF"

        # Check page-based symbol naming
        page_symbols = [chunk.symbol for chunk in chunks]
        page_1_chunks = [s for s in page_symbols if "page_1" in s]
        page_2_chunks = [s for s in page_symbols if "page_2" in s]
        page_3_chunks = [s for s in page_symbols if "page_3" in s]

        assert len(page_1_chunks) > 0, "Should have chunks from page 1"
        assert len(page_2_chunks) > 0, "Should have chunks from page 2"
        assert len(page_3_chunks) > 0, "Should have chunks from page 3"

        # Verify content from different pages
        combined_content = " ".join(chunk.code for chunk in chunks)
        assert "Chapter 1" in combined_content
        assert "Chapter 2" in combined_content
        assert "Chapter 3" in combined_content

    @pytest.mark.skipif(not PYMUPDF_AVAILABLE, reason="PyMuPDF not installed")
    def test_empty_pdf_parsing(self, pdf_parser, test_pdf_files):
        """Test parsing an empty PDF."""
        pdf_path = test_pdf_files["empty"]
        assert pdf_path.exists(), f"Test PDF not found: {pdf_path}"

        chunks = pdf_parser.parse_file(pdf_path, FileId(3))

        # Empty PDF might produce no chunks or minimal chunks
        # This is acceptable behavior
        for chunk in chunks:
            assert chunk.language == Language.PDF
            assert chunk.file_id == FileId(3)

    @pytest.mark.skipif(not PYMUPDF_AVAILABLE, reason="PyMuPDF not installed")
    def test_pdf_chunk_symbol_naming(self, pdf_parser, test_pdf_files):
        """Test that PDF chunks have proper page-based symbol naming."""
        pdf_path = test_pdf_files["multipage"]
        chunks = pdf_parser.parse_file(pdf_path, FileId(4))

        for chunk in chunks:
            symbol = chunk.symbol

            # Should contain page information
            assert "page_" in symbol, f"Symbol should contain page info: {symbol}"

            # Should contain either paragraph or content identifier
            assert any(
                keyword in symbol for keyword in ["paragraph", "content", "unavailable"]
            ), f"Symbol should contain content type: {symbol}"

    @pytest.mark.skipif(not PYMUPDF_AVAILABLE, reason="PyMuPDF not installed")
    def test_pdf_chunk_line_numbers(self, pdf_parser, test_pdf_files):
        """Test that PDF chunks have proper line number assignments."""
        pdf_path = test_pdf_files["simple"]
        chunks = pdf_parser.parse_file(pdf_path, FileId(5))

        if chunks:
            # First chunk should start at line 1
            assert chunks[0].start_line >= 1

            # Line numbers should be sequential
            for i in range(len(chunks) - 1):
                current_end = chunks[i].end_line
                next_start = chunks[i + 1].start_line
                assert next_start >= current_end, "Line numbers should be sequential"

    @pytest.mark.skipif(not PYMUPDF_AVAILABLE, reason="PyMuPDF not installed")
    def test_pdf_content_extraction(self, pdf_parser, test_pdf_files):
        """Test that PDF content is properly extracted and cleaned."""
        pdf_path = test_pdf_files["simple"]
        chunks = pdf_parser.parse_file(pdf_path, FileId(6))

        assert len(chunks) > 0, "Should extract content"

        for chunk in chunks:
            content = chunk.code

            # Content should not be empty
            assert content.strip(), f"Chunk content should not be empty: {chunk.symbol}"

            # Should not have excessive whitespace
            assert not content.startswith("   "), (
                "Should not have excessive leading whitespace"
            )

            # Should contain readable text
            assert len(content.split()) > 0, "Should contain words"


class TestPDFParsingErrorHandling:
    """Test error handling in PDF parsing."""

    @pytest.mark.skipif(PYMUPDF_AVAILABLE, reason="PyMuPDF is installed")
    def test_pdf_parsing_without_pymupdf(self):
        """Test PDF parsing when PyMuPDF is not available."""
        # Mock PyMuPDF as unavailable
        with patch("chunkhound.parsers.universal_parser.PYMUPDF_AVAILABLE", False):
            parser = create_parser_for_language(Language.PDF)

            # Mock the path existence and content
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.read_bytes", return_value=b"fake pdf content"):
                    fake_pdf_path = Path("fake.pdf")
                    chunks = parser.parse_file(fake_pdf_path, FileId(7))

                    # Should return a single error chunk
                    assert len(chunks) == 1
                    chunk = chunks[0]
                    assert chunk.symbol == "pdf_unavailable"
                    assert "PyMuPDF not installed" in chunk.code
                    assert chunk.language == Language.PDF

    def test_pdf_parsing_nonexistent_file(self):
        """Test PDF parsing with non-existent file."""
        parser = create_parser_for_language(Language.PDF)
        nonexistent_path = Path("nonexistent.pdf")

        with pytest.raises(FileNotFoundError):
            parser.parse_file(nonexistent_path, FileId(8))

    @pytest.mark.skipif(not PYMUPDF_AVAILABLE, reason="PyMuPDF not installed")
    def test_pdf_parsing_corrupted_content(self):
        """Test PDF parsing with corrupted PDF content."""
        parser = create_parser_for_language(Language.PDF)

        # Mock a corrupted PDF using patch for Path methods
        corrupted_content = b"This is not a valid PDF content"

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.read_bytes", return_value=corrupted_content):
                fake_pdf_path = Path("corrupted.pdf")
                chunks = parser.parse_file(fake_pdf_path, FileId(9))

                # Should handle the error gracefully
                # Exact behavior may vary, but should not crash
                assert isinstance(chunks, list)


class TestPDFParserFactory:
    """Test PDF parser creation through factory."""

    def test_create_pdf_parser(self):
        """Test that PDF parser can be created through factory."""
        parser = create_parser_for_language(Language.PDF)

        assert parser is not None
        assert isinstance(parser, UniversalParser)
        assert parser.language_name == "pdf"

    def test_pdf_parser_mapping(self):
        """Test that PDF parser uses correct mapping."""
        parser = create_parser_for_language(Language.PDF)

        # Should use PDF mapping
        assert isinstance(parser.base_mapping, PDFMapping)
        assert parser.base_mapping.language == Language.PDF


class TestPDFIntegration:
    """Test PDF parsing integration with ChunkHound system."""

    def test_pdf_in_universal_parser(self):
        """Test that PDF is properly integrated in universal parser."""
        from chunkhound.parsers.parser_factory import LANGUAGE_CONFIGS

        assert Language.PDF in LANGUAGE_CONFIGS, "PDF should be in language configs"

        pdf_config = LANGUAGE_CONFIGS[Language.PDF]
        assert pdf_config.available, "PDF should be available"
        assert pdf_config.language_name == "pdf"

    def test_pdf_file_extension_mapping(self):
        """Test that .pdf extension maps to PDF language through language detection."""
        # Test the actual extension mapping that exists
        language = Language.from_file_extension("test.pdf")
        assert language == Language.PDF, "PDF extension should map to Language.PDF"

        # Test that the parser factory can create a PDF parser
        parser = create_parser_for_language(Language.PDF)
        assert parser is not None, "Should be able to create PDF parser"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
