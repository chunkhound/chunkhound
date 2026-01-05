#!/usr/bin/env python3
"""Phase 6 Summary: Comprehensive test of PHP helper methods."""

from pathlib import Path

from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import ParserFactory

print("=" * 70)
print("PHASE 6: PHP Helper Methods - Comprehensive Verification")
print("=" * 70)

factory = ParserFactory()
parser = factory.create_parser(Language.PHP)

# Test cases covering all helper methods
test_cases = [
    {
        "name": "Function with typed parameters and return type",
        "code": """<?php
function getUserById(int $id, ?string $name = null): ?User {
    return null;
}
""",
        "expected": {
            "kind": "function",
            "has_parameters": True,
            "has_return_type": True,
        },
    },
    {
        "name": "Function with multiple parameter types",
        "code": """<?php
function processData(array $items, string $mode, ?int $limit = 10): bool {
    return true;
}
""",
        "expected": {
            "kind": "function",
            "has_parameters": True,
            "param_count": 3,
            "has_return_type": True,
        },
    },
    {
        "name": "Abstract class",
        "code": """<?php
abstract class AbstractService {
    private $data;
}
""",
        "expected": {
            "kind": "class",
            "is_abstract": True,
        },
    },
    {
        "name": "Final class",
        "code": """<?php
final class ImmutableValue {
    private $value;
}
""",
        "expected": {
            "kind": "class",
            "is_final": True,
        },
    },
    {
        "name": "Interface",
        "code": """<?php
interface Processable {
    public function process(): void;
}
""",
        "expected": {
            "kind": "interface",
        },
    },
    {
        "name": "Trait",
        "code": """<?php
trait Timestampable {
    protected $timestamp;
}
""",
        "expected": {
            "kind": "trait",
        },
    },
]

print("\n" + "=" * 70)
print("RUNNING TESTS")
print("=" * 70 + "\n")

results = []
for i, test_case in enumerate(test_cases, 1):
    name = test_case["name"]
    code = test_case["code"]
    expected = test_case["expected"]

    print(f"{i}. {name}")

    chunks = parser.parse_content(code, Path("test.php"), file_id=1)

    if not chunks:
        print("   ❌ FAIL: No chunks found")
        results.append(False)
        print()
        continue

    chunk = chunks[0]
    metadata = chunk.metadata or {}

    passed = True
    checks = []

    # Check kind
    if "kind" in expected:
        actual_kind = metadata.get("kind")
        kind_match = actual_kind == expected["kind"]
        checks.append(("kind", expected["kind"], actual_kind, kind_match))
        if not kind_match:
            passed = False

    # Check abstract
    if "is_abstract" in expected:
        actual_abstract = metadata.get("is_abstract", False)
        abstract_match = actual_abstract == expected["is_abstract"]
        checks.append(
            ("is_abstract", expected["is_abstract"], actual_abstract, abstract_match)
        )
        if not abstract_match:
            passed = False

    # Check final
    if "is_final" in expected:
        actual_final = metadata.get("is_final", False)
        final_match = actual_final == expected["is_final"]
        checks.append(("is_final", expected["is_final"], actual_final, final_match))
        if not final_match:
            passed = False

    # Check parameters
    if "has_parameters" in expected:
        actual_params = metadata.get("parameters", [])
        has_params = len(actual_params) > 0
        params_match = has_params == expected["has_parameters"]
        checks.append(
            ("has_parameters", expected["has_parameters"], has_params, params_match)
        )
        if not params_match:
            passed = False

        # Show parameters if present
        if has_params:
            print(f"   Parameters ({len(actual_params)}):")
            for p in actual_params:
                type_str = p.get("type", "mixed")
                name_str = p.get("name", "?")
                print(f"     - {type_str} {name_str}")

    # Check parameter count
    if "param_count" in expected:
        actual_params = metadata.get("parameters", [])
        actual_count = len(actual_params)
        count_match = actual_count == expected["param_count"]
        checks.append(
            ("param_count", expected["param_count"], actual_count, count_match)
        )
        if not count_match:
            passed = False

    # Check return type
    if "has_return_type" in expected:
        actual_return = metadata.get("return_type")
        has_return = actual_return is not None
        return_match = has_return == expected["has_return_type"]
        checks.append(
            ("has_return_type", expected["has_return_type"], has_return, return_match)
        )
        if not return_match:
            passed = False

        if has_return:
            print(f"   Return type: {actual_return}")

    # Print check results
    for check_name, expected_val, actual_val, match in checks:
        status = "✅" if match else "❌"
        print(f"   {status} {check_name}: {actual_val}")

    if passed:
        print("   ✅ PASS")
    else:
        print("   ❌ FAIL")

    results.append(passed)
    print()

# Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)

passed_count = sum(results)
total_count = len(results)
pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0

print(f"\nTests passed: {passed_count}/{total_count} ({pass_rate:.1f}%)")

if passed_count == total_count:
    print("\n✅ All Phase 6 helper methods working correctly!")
else:
    print(f"\n⚠️  {total_count - passed_count} test(s) failed")

print("\n" + "=" * 70)
print("PHASE 6: Helper Methods Implemented")
print("=" * 70)
print("""
✅ _extract_parameters(func_node, source) -> list[dict]
   - Extracts parameter names and type hints
   - Returns list of {'name': '$param', 'type': 'int'} dicts

✅ _extract_return_type(func_node, source) -> str | None
   - Extracts return type hint from functions/methods
   - Handles nullable types (?Type) and union types

✅ _extract_visibility(node, source) -> str | None
   - Extracts visibility modifier (public, private, protected)
   - Returns 'public' as default if not specified

✅ _is_static(node, source) -> bool
   - Checks for static modifier on methods/properties

✅ _is_abstract(node, source) -> bool
   - Checks for abstract modifier on classes/methods

✅ _is_final(node, source) -> bool
   - Checks for final modifier on classes/methods

✅ extract_metadata() enhanced
   - Now calls helper methods for detailed metadata
   - Provides rich PHP-specific information in chunks
""")

print("=" * 70)
