#!/usr/bin/env python3
"""Test PHP parser Phase 3 implementation."""

from pathlib import Path

from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import ParserFactory

# Comprehensive test PHP code
test_php = """<?php
namespace App\\Controllers;

use App\\Models\\User;

/**
 * User controller class.
 */
class UserController {
    /**
     * Get user by ID.
     */
    public function getUserById($id) {
        if ($id > 0) {
            return $id;
        }
        return null;
    }
}

/**
 * Helper function for sanitizing text.
 */
function sanitizeText($text) {
    return htmlspecialchars($text);
}

// Line comment
interface UserRepository {
    public function find($id);
}

trait Timestampable {
    public function updateTimestamp() {
        $this->updated_at = time();
    }
}
"""

print("=== Testing PHP Parser Phase 3 ===\n")

try:
    factory = ParserFactory()
    parser = factory.create_parser(Language.PHP)

    print("✅ PHP parser created successfully")

    # Debug: Check if mapping has the protocol methods
    print(f"\nMapping type: {type(parser.mapping).__name__}")
    print(
        f"Has get_query_for_concept: {hasattr(parser.mapping, 'get_query_for_concept')}"
    )
    print(f"Base mapping type: {type(parser.base_mapping).__name__}")

    # Check which queries are compiled
    from chunkhound.parsers.universal_engine import UniversalConcept

    print("\nCompiled queries:")
    for concept in UniversalConcept:
        has_query = concept in parser.extractor._compiled_queries
        print(f"  {concept.value}: {'✅' if has_query else '❌'}")
        if has_query:
            query_str = parser.mapping.get_query_for_concept(concept)
            if query_str:
                print(f"    Query length: {len(query_str)} chars")

    # Parse the test code
    print("\n=== Parsing ===")

    # Add debug to see what gets extracted
    ast_tree = parser.engine.parse_to_ast(test_php)
    content_bytes = test_php.encode("utf-8")

    print("Extracting concepts...")
    for concept in UniversalConcept:
        concept_chunks = parser.extractor.extract_concept(
            ast_tree.root_node, content_bytes, concept
        )
        print(f"  {concept.value}: {len(concept_chunks)} chunks")
        for chunk in concept_chunks[:5]:  # Show first 5
            content_preview = (
                chunk.content[:50].replace("\n", "\\n")
                if len(chunk.content) > 50
                else chunk.content.replace("\n", "\\n")
            )
            print(
                f"    - {chunk.name} (lines {chunk.start_line}-{chunk.end_line}): {content_preview}..."
            )

    print("\nNow parsing with full pipeline...")
    # Temporarily disable greedy_merge to see what happens
    from chunkhound.parsers.universal_parser import CASTConfig

    test_config = CASTConfig(greedy_merge=False)
    parser_no_merge = factory.create_parser(Language.PHP, test_config)
    chunks_no_merge = parser_no_merge.parse_content(
        test_php, Path("test.php"), file_id=1
    )
    print(f"Without greedy merge: {len(chunks_no_merge)} chunks")
    for chunk in chunks_no_merge[:5]:
        print(
            f"  - {chunk.symbol} ({chunk.chunk_type.value}, lines {chunk.start_line}-{chunk.end_line})"
        )
        # Print first 100 chars of content
        preview = (
            chunk.code[:100].replace("\n", "\\n")
            if len(chunk.code) > 100
            else chunk.code.replace("\n", "\\n")
        )
        print(f"    Content: {preview}...")

    print("\nWith greedy merge:")
    chunks = parser.parse_content(test_php, Path("test.php"), file_id=1)

    print(f"\n✅ Parsed successfully - found {len(chunks)} chunks\n")

    # Group chunks by type
    chunks_by_type = {}
    for chunk in chunks:
        chunk_type = chunk.chunk_type.value
        if chunk_type not in chunks_by_type:
            chunks_by_type[chunk_type] = []
        chunks_by_type[chunk_type].append(chunk)

    # Display chunks by type
    for chunk_type, type_chunks in sorted(chunks_by_type.items()):
        print(f"\n{chunk_type.upper()} chunks ({len(type_chunks)}):")
        for chunk in type_chunks:
            print(f"  • {chunk.symbol} (lines {chunk.start_line}-{chunk.end_line})")
            if chunk.metadata:
                print(f"    Metadata: {chunk.metadata}")

    # Verify expected chunks
    print("\n=== Verification ===")

    symbols = [c.symbol for c in chunks]

    checks = [
        ("UserController" in str(symbols), "UserController class"),
        ("getUserById" in str(symbols), "getUserById method"),
        ("sanitizeText" in str(symbols), "sanitizeText function"),
        ("UserRepository" in str(symbols), "UserRepository interface"),
        ("Timestampable" in str(symbols), "Timestampable trait"),
        ("namespace" in str(symbols).lower(), "namespace declaration"),
        (
            "use" in str(symbols).lower()
            or len([c for c in chunks if "use" in c.code.lower()]) > 0,
            "use statement",
        ),
        (
            len([c for c in chunks if c.chunk_type.value == "comment"]) > 0,
            "comments/docblocks",
        ),
    ]

    for passed, check_name in checks:
        status = "✅" if passed else "⚠️ "
        print(f"{status} {check_name}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback

    traceback.print_exc()
