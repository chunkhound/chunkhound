"""Unit tests for chunk ID generation using content-based hashing."""

import pytest
from pathlib import Path

from chunkhound.core.models.chunk import Chunk
from chunkhound.core.types.common import ChunkType, Language
from chunkhound.providers.database.lancedb_provider import LanceDBProvider
from chunkhound.utils.chunk_hashing import generate_chunk_id


class TestChunkHashing:
    """Tests for deterministic chunk ID generation."""

    def test_chunk_id_deterministic(self):
        """Verify same file_id and content produces same ID."""
        code = "def foo(): pass"
        file_id = 123

        id1 = generate_chunk_id(file_id, code)
        id2 = generate_chunk_id(file_id, code)

        assert id1 == id2, "Same file_id + content should produce identical IDs"

    def test_chunk_id_per_file_unique(self):
        """Verify same content in different files gets different IDs."""
        code = "def foo(): pass"
        file_id_1 = 1
        file_id_2 = 2

        id1 = generate_chunk_id(file_id_1, code)
        id2 = generate_chunk_id(file_id_2, code)

        assert id1 != id2, "Same content in different files should have different IDs"

    def test_chunk_id_content_sensitivity(self):
        """Verify different content produces different IDs."""
        file_id = 123

        id1 = generate_chunk_id(file_id, "def foo(): pass")
        id2 = generate_chunk_id(file_id, "def bar(): pass")

        assert id1 != id2, "Different content should produce different IDs"

    def test_chunk_id_normalization_line_endings(self):
        """Verify CRLF vs LF line endings don't change ID (normalized)."""
        file_id = 123

        # Unix line endings (LF)
        code_lf = "def foo():\n    pass"
        # Windows line endings (CRLF)
        code_crlf = "def foo():\r\n    pass"

        id_lf = generate_chunk_id(file_id, code_lf)
        id_crlf = generate_chunk_id(file_id, code_crlf)

        assert id_lf == id_crlf, "Line ending differences should be normalized"

    def test_chunk_id_normalization_trailing_whitespace(self):
        """Verify trailing whitespace doesn't change ID (normalized)."""
        file_id = 123

        code_no_trailing = "def foo():\n    pass"
        code_with_trailing = "def foo():\n    pass   \n"

        id1 = generate_chunk_id(file_id, code_no_trailing)
        id2 = generate_chunk_id(file_id, code_with_trailing)

        assert id1 == id2, "Trailing whitespace should be normalized"

    def test_chunk_id_returns_integer(self):
        """Verify function returns an integer (for database storage)."""
        chunk_id = generate_chunk_id(123, "def foo(): pass")

        assert isinstance(chunk_id, int), "Chunk ID should be an integer"

    def test_chunk_id_64bit_range(self):
        """Verify chunk ID fits in 64-bit signed integer range."""
        chunk_id = generate_chunk_id(123, "def foo(): pass")

        # Python's int is unbounded, but we want to ensure it fits in int64
        # int64 range: -2^63 to 2^63-1
        assert -(2**63) <= chunk_id <= (2**63 - 1), "Chunk ID should fit in 64-bit signed integer"

    def test_chunk_id_empty_content(self):
        """Verify empty content is handled correctly."""
        file_id = 123

        id1 = generate_chunk_id(file_id, "")
        id2 = generate_chunk_id(file_id, "")

        assert id1 == id2, "Empty content should produce consistent IDs"
        assert isinstance(id1, int), "Empty content should still produce valid integer ID"

    def test_chunk_id_large_content(self):
        """Verify large content is handled efficiently."""
        file_id = 123
        # Large code chunk (10KB)
        large_code = "def function_{}(): pass\n" * 500

        chunk_id = generate_chunk_id(file_id, large_code)

        assert isinstance(chunk_id, int), "Large content should produce valid integer ID"

    def test_chunk_id_unicode_content(self):
        """Verify Unicode content is handled correctly."""
        file_id = 123

        # Code with Unicode characters
        code = "# Comment with émojis 🎉\ndef foo():\n    return 'héllo'"

        id1 = generate_chunk_id(file_id, code)
        id2 = generate_chunk_id(file_id, code)

        assert id1 == id2, "Unicode content should produce consistent IDs"

    def test_chunk_id_collision_resistance(self):
        """Verify different chunks produce different IDs (no obvious collisions)."""
        file_id = 123
        chunk_ids = set()

        # Generate IDs for 1000 different code samples
        for i in range(1000):
            code = f"def function_{i}(): pass"
            chunk_id = generate_chunk_id(file_id, code)
            chunk_ids.add(chunk_id)

        # All IDs should be unique
        assert len(chunk_ids) == 1000, "No collisions expected for 1000 different chunks"

    def test_chunk_id_different_concepts_different_ids(self):
        """Verify same content with different concepts produces different IDs."""
        file_id = 123
        code = "def foo(): pass"

        id_definition = generate_chunk_id(file_id, code, concept="DEFINITION")
        id_block = generate_chunk_id(file_id, code, concept="BLOCK")

        assert id_definition != id_block, \
            "Same content with different concepts should have different IDs"

    def test_chunk_id_concept_optional(self):
        """Verify concept parameter is optional (backward compatibility)."""
        file_id = 123
        code = "def foo(): pass"

        id_without = generate_chunk_id(file_id, code)
        id_with_none = generate_chunk_id(file_id, code, concept=None)

        assert id_without == id_with_none, \
            "Calling without concept should be same as concept=None"

    def test_chunk_id_deterministic_with_concept(self):
        """Verify same inputs produce same ID (determinism with concept)."""
        file_id = 123
        code = "def bar(): return 42"

        id1 = generate_chunk_id(file_id, code, concept="DEFINITION")
        id2 = generate_chunk_id(file_id, code, concept="DEFINITION")

        assert id1 == id2, \
            "Same file_id + content + concept should produce identical IDs"


class TestLanceDBProviderChunkIDGeneration:
    """Tests for LanceDBProvider._generate_chunk_id_safe() method."""

    @pytest.fixture
    def provider(self, tmp_path):
        """Create a LanceDBProvider instance for testing."""
        db_path = tmp_path / "test.lancedb"
        return LanceDBProvider(
            db_path=db_path,
            base_directory=tmp_path,
            embedding_manager=None,
            config=None,
        )

    def test_generate_chunk_id_safe_with_existing_id(self, provider):
        """Verify method returns existing chunk.id if present."""
        chunk = Chunk(
            symbol="test_func",
            start_line=1,
            end_line=3,
            code="test code",
            chunk_type=ChunkType.FUNCTION,
            file_id=1,
            language=Language.PYTHON,
            id=999,
        )

        result_id = provider._generate_chunk_id_safe(chunk)
        assert result_id == 999, "Should return existing chunk.id"

    def test_generate_chunk_id_safe_generates_when_missing(self, provider):
        """Verify method generates ID when chunk.id is None."""
        chunk = Chunk(
            symbol="test_func",
            start_line=1,
            end_line=3,
            code="test code",
            chunk_type=ChunkType.FUNCTION,
            file_id=1,
            language=Language.PYTHON,
            id=None,
        )

        result_id = provider._generate_chunk_id_safe(chunk)
        assert isinstance(result_id, int), "Should generate integer ID"
        assert result_id > 0, "Generated ID should be positive"

    def test_generate_chunk_id_safe_deterministic(self, provider):
        """Verify same chunk generates same ID consistently."""
        chunk1 = Chunk(
            symbol="foo",
            start_line=1,
            end_line=1,
            code="def foo(): pass",
            chunk_type=ChunkType.FUNCTION,
            file_id=1,
            language=Language.PYTHON,
            id=None,
        )
        chunk2 = Chunk(
            symbol="foo",
            start_line=1,
            end_line=1,
            code="def foo(): pass",
            chunk_type=ChunkType.FUNCTION,
            file_id=1,
            language=Language.PYTHON,
            id=None,
        )

        id1 = provider._generate_chunk_id_safe(chunk1)
        id2 = provider._generate_chunk_id_safe(chunk2)

        assert id1 == id2, "Same chunk content should produce same ID"

    def test_generate_chunk_id_safe_different_content(self, provider):
        """Verify different content produces different IDs."""
        chunk1 = Chunk(
            symbol="foo",
            start_line=1,
            end_line=1,
            code="def foo(): pass",
            chunk_type=ChunkType.FUNCTION,
            file_id=1,
            language=Language.PYTHON,
            id=None,
        )
        chunk2 = Chunk(
            symbol="bar",
            start_line=2,
            end_line=2,
            code="def bar(): pass",
            chunk_type=ChunkType.FUNCTION,
            file_id=1,
            language=Language.PYTHON,
            id=None,
        )

        id1 = provider._generate_chunk_id_safe(chunk1)
        id2 = provider._generate_chunk_id_safe(chunk2)

        assert id1 != id2, "Different content should produce different IDs"

    def test_generate_chunk_id_safe_different_chunk_type(self, provider):
        """Verify same content with different ChunkType produces different IDs."""
        chunk1 = Chunk(
            symbol="test",
            start_line=1,
            end_line=1,
            code="test",
            chunk_type=ChunkType.FUNCTION,
            file_id=1,
            language=Language.PYTHON,
            id=None,
        )
        chunk2 = Chunk(
            symbol="test",
            start_line=1,
            end_line=1,
            code="test",
            chunk_type=ChunkType.CLASS,
            file_id=1,
            language=Language.PYTHON,
            id=None,
        )

        id1 = provider._generate_chunk_id_safe(chunk1)
        id2 = provider._generate_chunk_id_safe(chunk2)

        assert id1 != id2, "Different ChunkType should produce different IDs"
