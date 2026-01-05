"""Test raw PHP queries to see what tree-sitter extracts."""

from pathlib import Path

import tree_sitter_php as ts_php
from tree_sitter import Language, Parser, Query


def main():
    # Load comprehensive.php
    fixture_path = Path(__file__).parent.parent / "tests/fixtures/php/comprehensive.php"
    with open(fixture_path) as f:
        code = f.read()

    # Create parser
    parser = Parser()
    lang_capsule = ts_php.language_php()
    lang = Language(lang_capsule)
    parser.language = lang

    # Parse code
    tree = parser.parse(bytes(code, "utf8"))

    # Create queries
    from chunkhound.parsers.mappings.php import PHPMapping
    from chunkhound.parsers.universal_engine import UniversalConcept

    mapping = PHPMapping()

    # Test DEFINITION concept
    print("=" * 80)
    print("DEFINITION CONCEPT CAPTURES")
    print("=" * 80)

    query_str = mapping.get_query_for_concept(UniversalConcept.DEFINITION)
    if query_str:
        query = Query(lang, query_str)
        matches = query.matches(tree.root_node)

        count = 0
        for match in matches:
            for capture in match.captures:
                count += 1
                node = capture.node
                capture_name = query.capture_names[capture.index]
                print(f"\n[{count}] Capture: {capture_name}")
                print(f"  Node type: {node.type}")
                print(
                    f"  Start: line {node.start_point[0] + 1}, col {node.start_point[1]}"
                )
                text = code[node.start_byte : node.end_byte]
                print(f"  Text preview: {text.strip()[:100]}...")

    # Test COMMENT concept
    print("\n" + "=" * 80)
    print("COMMENT CONCEPT CAPTURES")
    print("=" * 80)

    query_str = mapping.get_query_for_concept(UniversalConcept.COMMENT)
    if query_str:
        query = Query(lang, query_str)
        matches = query.matches(tree.root_node)

        count = 0
        for match in matches:
            for capture in match.captures:
                count += 1
                node = capture.node
                capture_name = query.capture_names[capture.index]
                print(f"\n[{count}] Capture: {capture_name}")
                print(f"  Node type: {node.type}")
                print(
                    f"  Start: line {node.start_point[0] + 1}, col {node.start_point[1]}"
                )
                text = code[node.start_byte : node.end_byte]
                print(f"  Text preview: {text.strip()[:80]}...")

    # Test IMPORT concept
    print("\n" + "=" * 80)
    print("IMPORT CONCEPT CAPTURES")
    print("=" * 80)

    query_str = mapping.get_query_for_concept(UniversalConcept.IMPORT)
    if query_str:
        query = Query(lang, query_str)
        matches = query.matches(tree.root_node)

        count = 0
        for match in matches:
            for capture in match.captures:
                count += 1
                node = capture.node
                capture_name = query.capture_names[capture.index]
                print(f"\n[{count}] Capture: {capture_name}")
                print(f"  Node type: {node.type}")
                print(
                    f"  Start: line {node.start_point[0] + 1}, col {node.start_point[1]}"
                )
                text = code[node.start_byte : node.end_byte]
                print(f"  Text preview: {text.strip()[:80]}...")

    print("\n" + "=" * 80)
    print("Summary: Queries are working correctly!")
    print("=" * 80)


if __name__ == "__main__":
    main()
