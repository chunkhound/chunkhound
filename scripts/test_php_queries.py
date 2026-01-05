#!/usr/bin/env python3
"""Test PHP parser queries work correctly."""

from pathlib import Path

import tree_sitter_php as ts_php
from tree_sitter import Language as TSLanguage
from tree_sitter import Parser

from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import ParserFactory

# Test PHP code
test_php = """<?php
namespace App\\Controllers;

class UserController {
    public function getUserById($id) {
        return $id;
    }
}

function helperFunction($param) {
    return $param;
}
"""

print("=== Testing PHP Parser ===\n")

# First, test the raw queries
print("Step 1: Testing raw tree-sitter queries\n")
try:
    parser = Parser()
    lang_result = ts_php.language_php()
    if not isinstance(lang_result, TSLanguage):
        lang_result = TSLanguage(lang_result)
    parser.language = lang_result

    tree = parser.parse(test_php.encode("utf-8"))

    # Test function query
    from tree_sitter import Query, QueryCursor

    function_query = Query(
        lang_result,
        """
        (function_definition
            name: (name) @function_name
        ) @function_def
    """,
    )

    class_query = Query(
        lang_result,
        """
        (class_declaration
            name: (name) @class_name
        ) @class_def
    """,
    )

    method_query = Query(
        lang_result,
        """
        (method_declaration
            name: (name) @method_name
        ) @method_def
    """,
    )

    func_cursor = QueryCursor(function_query)
    class_cursor = QueryCursor(class_query)
    method_cursor = QueryCursor(method_query)

    func_matches = list(func_cursor.matches(tree.root_node))
    class_matches = list(class_cursor.matches(tree.root_node))
    method_matches = list(method_cursor.matches(tree.root_node))

    print(f"  Found {len(func_matches)} functions")
    print(f"  Found {len(class_matches)} classes")
    print(f"  Found {len(method_matches)} methods")

    for pattern_idx, captures_dict in func_matches:
        if "function_def" in captures_dict:
            nodes = captures_dict["function_def"]
            if nodes:
                node = nodes[0]
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                print(f"    Function at lines {start_line}-{end_line}")

    for pattern_idx, captures_dict in class_matches:
        if "class_def" in captures_dict:
            nodes = captures_dict["class_def"]
            if nodes:
                node = nodes[0]
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                print(f"    Class at lines {start_line}-{end_line}")

    for pattern_idx, captures_dict in method_matches:
        if "method_def" in captures_dict:
            nodes = captures_dict["method_def"]
            if nodes:
                node = nodes[0]
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                print(f"    Method at lines {start_line}-{end_line}")

except Exception as e:
    print(f"  ❌ Raw query test failed: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
print("\nStep 2: Testing UniversalParser\n")

try:
    factory = ParserFactory()
    parser = factory.create_parser(Language.PHP)

    print("✅ PHP parser created successfully")

    # Parse the test code
    chunks = parser.parse_content(test_php, Path("test.php"), file_id=1)

    print(f"\n✅ Parsed successfully - found {len(chunks)} chunks:")

    for i, chunk in enumerate(chunks, 1):
        print(f"\n  Chunk {i}:")
        print(f"    Type: {chunk.chunk_type}")
        print(f"    Symbol: {chunk.symbol}")
        print(f"    Lines: {chunk.start_line}-{chunk.end_line}")
        print(f"    Code length: {len(chunk.code)} chars")
        print(f"    Code:\n{chunk.code}")

    # Check we found expected chunks
    symbols = [c.symbol for c in chunks]

    if any("UserController" in s for s in symbols):
        print("\n✅ Found UserController class")
    else:
        print("\n⚠️  Did not find UserController class")

    if any("getUserById" in s for s in symbols):
        print("✅ Found getUserById method")
    else:
        print("⚠️  Did not find getUserById method")

    if any("helperFunction" in s for s in symbols):
        print("✅ Found helperFunction")
    else:
        print("⚠️  Did not find helperFunction")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback

    traceback.print_exc()
