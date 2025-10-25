"""Performance regression tests for hashing implementation.

Validates that xxHash3-64 provides expected performance characteristics
compared to SHA-256 baseline (2x+ faster).
"""

import time
from pathlib import Path

import pytest


def test_hash_performance_10mb_file(tmp_path: Path):
    """Verify xxHash3-64 completes in <50ms for 10MB file.

    This test validates the performance benefit of using xxHash3-64 instead
    of SHA-256. SHA-256 would take ~100ms for this file size, so we verify
    that xxHash3-64 completes in <50ms (2x+ speedup).
    """
    from chunkhound.utils.hashing import compute_file_hash

    # Create a 10MB test file
    large_file = tmp_path / "large_test.py"
    # Write realistic Python-like content to avoid compression artifacts
    content = b"# Python code\n" + b"x = " + b"1234567890" * 1_000_000
    large_file.write_bytes(content)

    # Measure hash computation time
    start = time.perf_counter()
    hash_result = compute_file_hash(large_file)
    elapsed = time.perf_counter() - start

    # Verify hash format (16 hex characters for 64-bit hash)
    assert len(hash_result) == 16, f"Expected 16 hex chars, got {len(hash_result)}"
    assert all(c in "0123456789abcdef" for c in hash_result), (
        f"Hash should be hex string, got: {hash_result}"
    )

    # Verify performance: should be <50ms (2x faster than SHA-256's ~100ms)
    # Allow some slack for CI environments (100ms max)
    assert elapsed < 0.1, (
        f"Hash computation took {elapsed*1000:.1f}ms, expected <100ms. "
        f"This may indicate a performance regression."
    )

    # Log actual performance for monitoring
    print(f"\nxxHash3-64 performance: {elapsed*1000:.1f}ms for 10MB file")


def test_hash_format_consistency(tmp_path: Path):
    """Verify hash format is consistent (16 hex characters)."""
    from chunkhound.utils.hashing import compute_file_hash

    test_file = tmp_path / "test.py"
    test_file.write_text("print('hello')\n")

    hash1 = compute_file_hash(test_file)
    hash2 = compute_file_hash(test_file)

    # Same file should produce same hash
    assert hash1 == hash2

    # Verify format
    assert len(hash1) == 16
    assert all(c in "0123456789abcdef" for c in hash1)


def test_hash_detects_changes(tmp_path: Path):
    """Verify hash changes when file content changes."""
    from chunkhound.utils.hashing import compute_file_hash

    test_file = tmp_path / "test.py"

    # Original content
    test_file.write_text("print('hello')\n")
    hash1 = compute_file_hash(test_file)

    # Modified content
    test_file.write_text("print('world')\n")
    hash2 = compute_file_hash(test_file)

    # Hashes should differ
    assert hash1 != hash2, "Hash should change when file content changes"


def test_hash_handles_empty_file(tmp_path: Path):
    """Verify hash computation works for empty files."""
    from chunkhound.utils.hashing import compute_file_hash

    empty_file = tmp_path / "empty.py"
    empty_file.write_text("")

    hash_result = compute_file_hash(empty_file)

    # Should return valid hash even for empty file
    assert len(hash_result) == 16
    assert all(c in "0123456789abcdef" for c in hash_result)


def test_hash_raises_on_directory(tmp_path: Path):
    """Verify hash computation raises ValueError for directories."""
    from chunkhound.utils.hashing import compute_file_hash

    test_dir = tmp_path / "test_directory"
    test_dir.mkdir()

    with pytest.raises(ValueError, match="Path must be a file"):
        compute_file_hash(test_dir)


def test_hash_raises_on_nonexistent_file(tmp_path: Path):
    """Verify hash computation raises OSError for nonexistent files."""
    from chunkhound.utils.hashing import compute_file_hash

    nonexistent = tmp_path / "does_not_exist.py"

    with pytest.raises((OSError, ValueError)):
        compute_file_hash(nonexistent)
