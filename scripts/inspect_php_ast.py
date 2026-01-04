#!/usr/bin/env python3
"""Inspect PHP AST structure to verify node type names."""

import tree_sitter_php as ts_php
from tree_sitter import Language, Parser

# Initialize parser
parser = Parser()
# Get language capsule and wrap it if needed
lang_result = ts_php.language_php()
# If it's not already a Language object, wrap it
if not isinstance(lang_result, Language):
    lang_result = Language(lang_result)
parser.language = lang_result

# Test PHP code samples
test_cases = [
    # Test 1: Function
    (
        b"""<?php
function myFunction($param1, $param2) {
    return $param1 + $param2;
}
""",
        "function",
    ),
    # Test 2: Class
    (
        b"""<?php
class MyClass {
    public function myMethod() {
        return true;
    }
}
""",
        "class",
    ),
    # Test 3: Interface
    (
        b"""<?php
interface MyInterface {
    public function myMethod();
}
""",
        "interface",
    ),
    # Test 4: Trait
    (
        b"""<?php
trait MyTrait {
    public function myMethod() {}
}
""",
        "trait",
    ),
    # Test 5: Namespace
    (
        b"""<?php
namespace App\\Controllers;

use App\\Models\\User;

class MyClass {}
""",
        "namespace",
    ),
    # Test 6: Comment
    (
        b"""<?php
// Line comment
/* Block comment */
/** PHPDoc comment */
""",
        "comment",
    ),
]

print("=== PHP AST Node Type Inspection ===\n")

for code, label in test_cases:
    print(f"\n--- Testing: {label} ---")
    print(f"Code:\n{code.decode('utf-8')}")

    tree = parser.parse(code)

    # Walk the tree and find interesting node types
    def walk_and_print(node, depth=0, max_depth=4):
        if depth > max_depth:
            return

        indent = "  " * depth
        # Print node type and text if it's a named node
        if node.is_named:
            text = code[node.start_byte : node.end_byte].decode(
                "utf-8", errors="ignore"
            )
            # Truncate long text
            text_preview = (
                text[:40].replace("\n", "\\n")
                if len(text) > 40
                else text.replace("\n", "\\n")
            )
            print(
                f"{indent}{node.type} [{node.start_point[0]}:{node.start_point[1]} - {node.end_point[0]}:{node.end_point[1]}] = '{text_preview}'"
            )

        for child in node.children:
            walk_and_print(child, depth + 1, max_depth)

    print("\nTree structure (depth 4):")
    walk_and_print(tree.root_node)
    print("\n" + "=" * 60)

print("\n=== Key Findings ===")
print("Look for these patterns in the output above:")
print(
    "- Function nodes: What are they called? (function_definition, function_declaration, method_declaration?)"
)
print("- Class nodes: class_declaration?")
print("- Interface nodes: interface_declaration?")
print("- Trait nodes: trait_declaration?")
print("- Namespace nodes: namespace_definition?")
print("- Comment nodes: comment?")
print("- Name nodes: How are names captured? (name, identifier, etc.)")
