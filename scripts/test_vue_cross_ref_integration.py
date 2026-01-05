#!/usr/bin/env python3
"""Manual integration test for Vue cross-reference analysis."""

from pathlib import Path

from chunkhound.core.types.common import FileId
from chunkhound.parsers.vue_parser import VueParser


def main():
    """Test Vue cross-reference analysis on fixtures."""
    parser = VueParser()

    # Test cross-reference fixture
    fixture_path = (
        Path(__file__).parent.parent
        / "tests"
        / "fixtures"
        / "vue"
        / "cross_reference.vue"
    )

    if not fixture_path.exists():
        print(f"Fixture not found: {fixture_path}")
        return

    print(f"Parsing: {fixture_path}")
    chunks = parser.parse_file(fixture_path, FileId(1))

    # Separate script and template chunks
    script_chunks = [
        c for c in chunks if c.metadata and c.metadata.get("vue_section") == "script"
    ]
    template_chunks = [
        c for c in chunks if c.metadata and c.metadata.get("vue_section") == "template"
    ]

    print(f"\nFound {len(script_chunks)} script chunks")
    print(f"Found {len(template_chunks)} template chunks")

    # Check for cross-references
    chunks_with_refs = [
        c for c in template_chunks if c.metadata and "script_references" in c.metadata
    ]

    print(f"\nTemplate chunks with cross-references: {len(chunks_with_refs)}")

    if chunks_with_refs:
        print("\nSample cross-references:")
        for i, chunk in enumerate(chunks_with_refs[:5]):  # Show first 5
            refs = chunk.metadata.get("script_references", [])
            print(f"  {i + 1}. {chunk.symbol}: {refs}")

    # Check for undefined references
    chunks_with_undefined = [
        c
        for c in template_chunks
        if c.metadata and "undefined_references" in c.metadata
    ]

    if chunks_with_undefined:
        print(
            f"\nTemplate chunks with undefined references: {len(chunks_with_undefined)}"
        )
        for chunk in chunks_with_undefined[:3]:  # Show first 3
            undefined = chunk.metadata.get("undefined_references", [])
            print(f"  {chunk.symbol}: {undefined}")

    # Collect all unique references
    all_refs = set()
    for chunk in chunks_with_refs:
        if "script_references" in chunk.metadata:
            all_refs.update(chunk.metadata["script_references"])

    print(f"\nTotal unique script references found: {len(all_refs)}")
    print("References:", sorted(all_refs)[:20])  # Show first 20

    # Test basic setup fixture
    print("\n" + "=" * 60)
    basic_fixture = (
        Path(__file__).parent.parent / "tests" / "fixtures" / "vue" / "basic_setup.vue"
    )
    if basic_fixture.exists():
        print(f"\nParsing: {basic_fixture}")
        chunks = parser.parse_file(basic_fixture, FileId(2))

        template_chunks = [
            c
            for c in chunks
            if c.metadata and c.metadata.get("vue_section") == "template"
        ]
        chunks_with_refs = [
            c
            for c in template_chunks
            if c.metadata and "script_references" in c.metadata
        ]

        print(f"Template chunks with references: {len(chunks_with_refs)}")
        if chunks_with_refs:
            for chunk in chunks_with_refs[:3]:
                refs = chunk.metadata.get("script_references", [])
                print(f"  {chunk.symbol}: {refs}")

    print("\nIntegration test complete!")


if __name__ == "__main__":
    main()
