#!/usr/bin/env python3
"""Test PHP helper methods for metadata extraction."""

from pathlib import Path

from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import ParserFactory

test_php = """<?php

function getUser(int $id, ?string $name = null): ?User {
    return null;
}


function simpleFunction(string $message): void {
    echo $message;
}


abstract class BaseService {
    private static $instance;
}


final class FinalClass {
    // Empty final class for testing
}


interface ServiceInterface {
    public function execute(): mixed;
}


trait Loggable {
    private function log(string $message): void {
        echo $message;
    }
}
"""

print("=== Testing PHP Helper Methods ===\n")

try:
    factory = ParserFactory()
    parser = factory.create_parser(Language.PHP)
    chunks = parser.parse_content(test_php, Path("test.php"), file_id=1)

    print(f"Found {len(chunks)} chunks\n")

    # Debug: Show all chunks
    print("=== All Chunks ===")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk.chunk_type.value}: {chunk.symbol}")
        if chunk.metadata:
            print(f"   Metadata: {chunk.metadata}")
    print()

    # Look for chunks with detailed metadata
    for chunk in chunks:
        if chunk.metadata and chunk.metadata.get("kind") in (
            "class",
            "method",
            "function",
            "interface",
            "trait",
        ):
            print(f"\n{chunk.chunk_type.value.upper()}: {chunk.symbol}")
            print(f"  Kind: {chunk.metadata.get('kind')}")

            # Visibility
            if "visibility" in chunk.metadata:
                print(f"  Visibility: {chunk.metadata['visibility']}")

            # Modifiers
            if chunk.metadata.get("is_static"):
                print("  Static: Yes")
            if chunk.metadata.get("is_abstract"):
                print("  Abstract: Yes")
            if chunk.metadata.get("is_final"):
                print("  Final: Yes")

            # Parameters
            if "parameters" in chunk.metadata:
                params = chunk.metadata["parameters"]
                print(f"  Parameters ({len(params)}):")
                for param in params:
                    param_str = param.get("name", "?")
                    if "type" in param:
                        param_str = f"{param['type']} {param_str}"
                    print(f"    - {param_str}")

            # Return type
            if "return_type" in chunk.metadata:
                print(f"  Return type: {chunk.metadata['return_type']}")

    print("\n=== Verification ===")

    # Check specific features
    checks = [
        (
            any("visibility" in c.metadata for c in chunks if c.metadata),
            "Visibility modifiers",
        ),
        (
            any(c.metadata.get("is_static") for c in chunks if c.metadata),
            "Static modifier",
        ),
        (
            any(c.metadata.get("is_abstract") for c in chunks if c.metadata),
            "Abstract modifier",
        ),
        (
            any(c.metadata.get("is_final") for c in chunks if c.metadata),
            "Final modifier",
        ),
        (
            any("parameters" in c.metadata for c in chunks if c.metadata),
            "Parameter extraction",
        ),
        (
            any("return_type" in c.metadata for c in chunks if c.metadata),
            "Return type extraction",
        ),
    ]

    for passed, check_name in checks:
        status = "✅" if passed else "⚠️ "
        print(f"{status} {check_name}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback

    traceback.print_exc()
