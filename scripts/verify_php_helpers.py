#!/usr/bin/env python3
"""Directly verify PHP helper methods work at the AST level."""

import tree_sitter
import tree_sitter_php

from chunkhound.parsers.mappings.php import PHPMapping

# Create parser
parser = tree_sitter.Parser(tree_sitter.Language(tree_sitter_php.language_php()))

# Test code with various PHP features
php_code = """<?php
class MyClass {
    public function publicMethod(): void {
        echo "public";
    }

    private static function privateStaticMethod(int $count): string {
        return "";
    }

    protected function protectedMethod(string $msg, ?int $id = null): mixed {
        return null;
    }
}

abstract class AbstractClass {
    abstract protected function process(): void;
}

final class FinalClass {
    final public static function finalStaticMethod(): self {
        return new self();
    }
}
"""

# Parse
tree = parser.parse(bytes(php_code, "utf8"))

# Create mapping
mapping = PHPMapping()

print("=== Direct AST-Level Verification ===\n")

# Find all method declarations using tree-sitter Query
query = tree_sitter.Query(
    parser.language,
    """
(method_declaration
    name: (name) @name
) @method
""",
)

captures = query.matches(tree.root_node)

print(f"Found {len(captures)} methods\n")

for match in captures:
    method_node = None
    name_node = None

    for capture_id, node in match[1]:
        capture_name = query.capture_names[capture_id]
        if capture_name == "method":
            method_node = node
        elif capture_name == "name":
            name_node = node

    if method_node and name_node:
        method_name = php_code[name_node.start_byte : name_node.end_byte]

        print(f"Method: {method_name}")
        print(f"  Node type: {method_node.type}")

        # Test visibility extraction
        visibility = mapping._extract_visibility(method_node, php_code)
        print(f"  ✅ Visibility: {visibility}")

        # Test static detection
        is_static = mapping._is_static(method_node, php_code)
        if is_static:
            print("  ✅ Static: Yes")

        # Test parameter extraction
        params = mapping._extract_parameters(method_node, php_code)
        if params:
            print(f"  ✅ Parameters ({len(params)}):")
            for p in params:
                type_hint = p.get("type", "no-type")
                name = p.get("name", "no-name")
                print(f"      - {type_hint} {name}")

        # Test return type extraction
        return_type = mapping._extract_return_type(method_node, php_code)
        if return_type:
            print(f"  ✅ Return type: {return_type}")

        print()

# Test class-level modifiers
query = tree_sitter.Query(
    parser.language,
    """
(class_declaration
    name: (name) @name
) @class
""",
)

captures = query.matches(tree.root_node)

print(f"\nFound {len(captures)} classes\n")

for match in captures:
    class_node = None
    name_node = None

    for capture_id, node in match[1]:
        capture_name = query.capture_names[capture_id]
        if capture_name == "class":
            class_node = node
        elif capture_name == "name":
            name_node = node

    if class_node and name_node:
        class_name = php_code[name_node.start_byte : name_node.end_byte]

        print(f"Class: {class_name}")

        # Test abstract detection
        is_abstract = mapping._is_abstract(class_node, php_code)
        if is_abstract:
            print("  ✅ Abstract: Yes")

        # Test final detection
        is_final = mapping._is_final(class_node, php_code)
        if is_final:
            print("  ✅ Final: Yes")

        print()

print("=" * 60)
print("Verification complete!")
print("=" * 60)
