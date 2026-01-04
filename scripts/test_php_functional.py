"""Functional test for PHP parser - end-to-end verification."""

from pathlib import Path

from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import ParserFactory

# Test 1: Parser availability
factory = ParserFactory()
assert factory.is_language_available(Language.PHP), "PHP not available"
print("✅ Test 1: PHP parser is available")

# Test 2: Extension detection
assert factory.detect_language(Path("test.php")) == Language.PHP, (
    "Extension detection failed"
)
print("✅ Test 2: Extension detection works")

# Test 3: Parser creation
parser = factory.create_parser(Language.PHP)
assert parser is not None, "Parser creation failed"
print("✅ Test 3: Parser creation successful")

# Test 4: Basic parsing
code = """<?php
namespace App;
class User {
    public function getName(): string {
        return "test";
    }
}
"""
chunks = parser.parse_content(code, Path("test.php"), file_id=1)
assert len(chunks) > 0, "No chunks extracted"
print(f"✅ Test 4: Basic parsing successful ({len(chunks)} chunks)")

# Test 5: Metadata extraction
has_metadata = any(c.metadata for c in chunks)
assert has_metadata, "No metadata extracted"
print("✅ Test 5: Metadata extraction works")

# Additional verification
print("\n=== Extracted Chunks ===")
for i, chunk in enumerate(chunks, 1):
    print(f"\nChunk {i}:")
    print(f"  Type: {chunk.chunk_type}")
    print(f"  Symbol: {chunk.symbol}")
    print(f"  Metadata: {chunk.metadata}")
    print(f"  Code preview: {chunk.code[:100]}...")

print("\n✅ All functional tests passed!")
