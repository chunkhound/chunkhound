#!/usr/bin/env python3
"""Quick performance test for Vue parser Phase 2."""

import time
from pathlib import Path

from chunkhound.core.types.common import FileId
from chunkhound.parsers.vue_parser import VueParser


def test_performance():
    """Test parsing performance with different file sizes."""
    parser = VueParser()

    # Test fixtures
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "vue"

    # Test small file
    small_file = fixtures_dir / "basic_setup.vue"
    if small_file.exists():
        start = time.time()
        chunks = parser.parse_file(small_file, FileId(1))
        small_time = (time.time() - start) * 1000
        print(
            f"Small file ({small_file.name}): {small_time:.1f}ms ({len(chunks)} chunks)"
        )

    # Test medium file
    medium_file = fixtures_dir / "template_directives.vue"
    if medium_file.exists():
        start = time.time()
        chunks = parser.parse_file(medium_file, FileId(2))
        medium_time = (time.time() - start) * 1000
        print(
            f"Medium file ({medium_file.name}): {medium_time:.1f}ms ({len(chunks)} chunks)"
        )

    # Test large file
    large_file = fixtures_dir / "form_component.vue"
    if large_file.exists():
        start = time.time()
        chunks = parser.parse_file(large_file, FileId(3))
        large_time = (time.time() - start) * 1000
        print(
            f"Large file ({large_file.name}): {large_time:.1f}ms ({len(chunks)} chunks)"
        )

    # Test cross-reference file
    cross_ref_file = fixtures_dir / "cross_reference.vue"
    if cross_ref_file.exists():
        start = time.time()
        chunks = parser.parse_file(cross_ref_file, FileId(4))
        cross_ref_time = (time.time() - start) * 1000
        print(
            f"Cross-ref file ({cross_ref_file.name}): {cross_ref_time:.1f}ms ({len(chunks)} chunks)"
        )

    print("\nPerformance targets:")
    print(
        "  Small files: < 50ms ✓"
        if small_time < 50
        else f"  Small files: < 50ms ✗ (got {small_time:.1f}ms)"
    )
    print(
        "  Large files: < 200ms ✓"
        if large_time < 200
        else f"  Large files: < 200ms ✗ (got {large_time:.1f}ms)"
    )


if __name__ == "__main__":
    test_performance()
