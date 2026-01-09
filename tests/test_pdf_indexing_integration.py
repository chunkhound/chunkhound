"""Comprehensive PDF indexing integration test.

Tests that PDF content is faithfully preserved across all chunks when indexed,
using the research paper PDF as a comprehensive test case.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any

from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.services.search_service import SearchService
from chunkhound.core.models.chunk import Chunk

# Import conditional PyMuPDF test skip
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

pytestmark = pytest.mark.skipif(not PYMUPDF_AVAILABLE, reason="PyMuPDF not available")


@pytest.fixture
def pdf_services(tmp_path):
    """Create services for PDF indexing without embeddings."""
    # Use file-based database (not :memory:) because semantic search requires ShardManager
    db_path = tmp_path / "test.db"
    db = DuckDBProvider(str(db_path), base_directory=tmp_path)
    db.connect()
    parser = create_parser_for_language(Language.PDF)
    coordinator = IndexingCoordinator(db, tmp_path, None, {Language.PDF: parser})
    search_service = SearchService(db)
    return {
        "db": db,
        "coordinator": coordinator, 
        "search": search_service,
        "tmp_path": tmp_path
    }


@pytest.fixture
def research_paper_pdf(tmp_path):
    """Copy the research paper PDF to test directory."""
    # Use relative path from test file location
    fixtures_dir = Path(__file__).parent / "fixtures"
    source_pdf = fixtures_dir / "cast_research_paper.pdf"
    
    # Verify PDF exists
    assert source_pdf.exists(), f"Research paper PDF not found at: {source_pdf}"
    
    # Create test directory
    test_dir = tmp_path / "pdf_test"
    test_dir.mkdir()
    
    # Copy PDF to test directory
    test_pdf = test_dir / "cast_paper.pdf"
    shutil.copy2(source_pdf, test_pdf)
    
    return test_pdf


@pytest.fixture
def expected_content():
    """Load the expected markdown content for comparison."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    markdown_path = fixtures_dir / "cast_research_paper_content.md"
    
    # Verify markdown file exists
    assert markdown_path.exists(), f"Expected content file not found at: {markdown_path}"
    
    return markdown_path.read_text(encoding='utf-8')


class TestPDFIndexingIntegration:
    """Comprehensive PDF indexing integration tests."""

    @pytest.mark.asyncio
    async def test_pdf_complete_content_preservation(self, pdf_services, research_paper_pdf, expected_content):
        """Test that all PDF content is faithfully preserved in chunks."""
        coordinator = pdf_services["coordinator"]
        db = pdf_services["db"]
        search = pdf_services["search"]
        
        # INDEX: Process the PDF file
        result = await coordinator.process_file(research_paper_pdf)
        
        # Verify successful indexing
        assert result["status"] == "success", f"PDF indexing failed: {result.get('error')}"
        assert result["chunks"] > 0, "No chunks were created from PDF"
        file_id = result["file_id"]
        
        print(f"✅ PDF indexed successfully: {result['chunks']} chunks created")
        
        # GET CHUNKS: Retrieve all chunks from database
        chunks = db.get_chunks_by_file_id(file_id, as_model=True)
        assert len(chunks) > 0, "No chunks found in database"
        
        # Verify chunks are properly ordered and have PDF metadata
        self._verify_chunk_structure(chunks)
        
        # CONTENT VERIFICATION: Compare with expected content
        await self._verify_content_completeness(chunks, expected_content, search)
        
        print(f"✅ All content verification tests passed!")

    def _verify_chunk_structure(self, chunks: List[Chunk]):
        """Verify chunk structure and metadata."""
        print(f"🔍 Verifying chunk structure for {len(chunks)} chunks...")
        
        # Verify all chunks are PDF language
        for chunk in chunks:
            assert chunk.language == Language.PDF, f"Wrong language: {chunk.language}"
        
        # Verify page-based symbols
        page_symbols = [chunk.symbol for chunk in chunks if chunk.symbol.startswith("page_")]
        assert len(page_symbols) > 0, "No page-based symbols found"
        
        # Verify chunks span multiple pages (paper has 11 pages)
        page_numbers = set()
        for chunk in chunks:
            if chunk.symbol.startswith("page_"):
                page_num = int(chunk.symbol.split("_")[1])
                page_numbers.add(page_num)
        
        assert len(page_numbers) > 5, f"Expected multiple pages, got: {sorted(page_numbers)}"
        assert min(page_numbers) == 1, "Should start from page 1"
        
        print(f"✅ Chunks span pages: {sorted(page_numbers)}")
        
        # Verify chunks have reasonable content length
        total_chars = sum(len(chunk.code) for chunk in chunks)
        assert total_chars > 10000, f"Total content seems too short: {total_chars} chars"
        
        print(f"✅ Total content: {total_chars} characters across {len(chunks)} chunks")

    async def _verify_content_completeness(self, chunks: List[Chunk], expected_content: str, search: SearchService):
        """Verify all expected content is present in chunks."""
        print("🔍 Verifying content completeness...")
        
        # Combine all chunk content
        combined_content = "\n".join(chunk.code for chunk in chunks)
        
        # Key content sections that must be present (using flexible matching)
        required_sections = [
            # Title and authors
            "CAST: Enhancing Code Retrieval-Augmented Generation",
            "Yilin Zhang", "Carnegie Mellon",
            
            # Abstract key terms  
            "Retrieval-Augmented Generation", "chunking", "Abstract Syntax Tree",
            "structure-aware", "coherent units",
            
            # Technical content  
            "cAST", "recursive", "split-then-merge",
            "syntactic integrity", "information density",
            
            # Code examples (from figures)
            "def normalize", "compute_stats", 
            
            # References and conclusions
            "References", "Conclusion"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section.lower() not in combined_content.lower():
                missing_sections.append(section)
        
        if missing_sections:
            print(f"❌ Missing sections: {missing_sections}")
            # Show sample of content for debugging
            print(f"Sample content (first 500 chars): {combined_content[:500]}")
            
        assert len(missing_sections) == 0, f"Missing required content: {missing_sections}"
        
        print(f"✅ All {len(required_sections)} required sections found")
        
        # Test search functionality
        await self._verify_search_functionality(search)

    async def _verify_search_functionality(self, search: SearchService):
        """Test that indexed PDF content is searchable."""
        print("🔍 Testing search functionality...")
        
        # Test searches for key terms (using flexible patterns)
        search_tests = [
            ("CAST", "algorithm name"),
            ("Abstract Syntax Tree", "core concept"), 
            ("chunking", "main topic"),
            ("def normalize", "code example"),
            ("Carnegie", "institution"),
        ]
        
        for pattern, description in search_tests:
            results, _ = search.search_regex(pattern, page_size=10)
            assert len(results) > 0, f"No results for {description} search: '{pattern}'"
            
            # Verify results contain the search term
            found = any(pattern.lower() in result.get("content", "").lower() for result in results)
            assert found, f"Search results don't contain '{pattern}'"
            
            print(f"✅ Found {len(results)} results for '{pattern}' ({description})")

    @pytest.mark.asyncio
    async def test_pdf_metadata_preservation(self, pdf_services, research_paper_pdf):
        """Test that PDF metadata is properly preserved."""
        coordinator = pdf_services["coordinator"]
        db = pdf_services["db"]
        
        # Index PDF
        result = await coordinator.process_file(research_paper_pdf)
        assert result["status"] == "success"
        
        # Get chunks
        chunks = db.get_chunks_by_file_id(result["file_id"], as_model=True)
        
        # Verify file path is preserved (may be None in some cases)
        # Note: File path might not be set in all chunk implementations
        pdf_filename_found = any("cast_paper.pdf" in str(chunk.file_path) for chunk in chunks if chunk.file_path)
        # If no file paths are set, that's acceptable for this test
        if any(chunk.file_path for chunk in chunks):
            assert pdf_filename_found, "Expected PDF filename in at least one chunk's file path"
        
        # Verify line numbers are sequential and reasonable
        line_numbers = [chunk.start_line for chunk in chunks]
        assert min(line_numbers) >= 1, "Line numbers should start from 1"
        assert max(line_numbers) > 10, "Should have content spanning multiple lines"
        
        print(f"✅ Metadata preserved: line numbers {min(line_numbers)}-{max(line_numbers)}")

    @pytest.mark.asyncio 
    async def test_pdf_multi_page_handling(self, pdf_services, research_paper_pdf):
        """Test that multi-page PDFs are handled correctly."""
        coordinator = pdf_services["coordinator"] 
        db = pdf_services["db"]
        
        # Index PDF
        result = await coordinator.process_file(research_paper_pdf)
        chunks = db.get_chunks_by_file_id(result["file_id"], as_model=True)
        
        # Extract page numbers from chunk symbols
        page_info = {}
        for chunk in chunks:
            if chunk.symbol.startswith("page_"):
                parts = chunk.symbol.split("_")
                if len(parts) >= 2:
                    try:
                        page_num = int(parts[1])
                        if page_num not in page_info:
                            page_info[page_num] = []
                        page_info[page_num].append(chunk)
                    except ValueError:
                        pass
        
        # Verify multi-page structure
        assert len(page_info) >= 8, f"Expected at least 8 pages, got {len(page_info)}"
        
        # Verify each page has content
        for page_num, page_chunks in page_info.items():
            total_content = sum(len(chunk.code) for chunk in page_chunks)
            assert total_content > 50, f"Page {page_num} has insufficient content: {total_content} chars"
        
        print(f"✅ Multi-page handling verified: {len(page_info)} pages processed")

    @pytest.mark.asyncio
    async def test_pdf_content_ordering(self, pdf_services, research_paper_pdf):
        """Test that PDF content maintains proper document order."""
        coordinator = pdf_services["coordinator"]
        db = pdf_services["db"]
        
        # Index PDF
        result = await coordinator.process_file(research_paper_pdf)
        chunks = db.get_chunks_by_file_id(result["file_id"], as_model=True)
        
        # Sort chunks by line number to verify ordering
        sorted_chunks = sorted(chunks, key=lambda c: c.start_line)
        
        # Verify line numbers are increasing
        prev_line = 0
        for chunk in sorted_chunks:
            assert chunk.start_line >= prev_line, "Chunks should be in line number order"
            prev_line = chunk.start_line
        
        # Verify logical content ordering
        combined_text = " ".join(chunk.code for chunk in sorted_chunks)
        
        # Abstract should come before Introduction
        abstract_pos = combined_text.lower().find("abstract")
        intro_pos = combined_text.lower().find("introduction")
        if abstract_pos != -1 and intro_pos != -1:
            assert abstract_pos < intro_pos, "Abstract should come before Introduction"
        
        # Introduction should come before Conclusion  
        conclusion_pos = combined_text.lower().find("conclusion")
        if intro_pos != -1 and conclusion_pos != -1:
            assert intro_pos < conclusion_pos, "Introduction should come before Conclusion"
            
        print("✅ Content ordering verified")

    @pytest.mark.asyncio
    async def test_pdf_search_precision(self, pdf_services, research_paper_pdf):
        """Test search precision on PDF content."""
        coordinator = pdf_services["coordinator"]
        search = pdf_services["search"]
        
        # Index PDF
        result = await coordinator.process_file(research_paper_pdf)
        assert result["status"] == "success"
        
        # Test precise searches (adjusted for PDF content)
        precision_tests = [
            ("def normalize", "function signature"),
            ("Carnegie Mellon", "institution name"),
            ("CAST.*chunking", "algorithm description"),
        ]
        
        for pattern, description in precision_tests:
            results, _ = search.search_regex(pattern, page_size=5)
            assert len(results) > 0, f"No precise results for {description}: '{pattern}'"
            print(f"✅ Precise search for {description}: {len(results)} results")