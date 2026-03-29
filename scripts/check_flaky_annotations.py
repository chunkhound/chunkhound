#!/usr/bin/env python3
"""
Post-test flaky annotation gate (D5).

Parses pytest JUnit XML output and checks that any persistently failing tests
(failed after retry) have a '# flaky: reason' annotation in the source.

Exit 0: all failures are annotated as flaky (acknowledged debt)
Exit 1: unannotated failure — PR is blocked until annotation is added
Exit 2: usage error or missing input file
"""
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def find_failed_tests(junit_xml: Path) -> list[tuple[str, str, int]]:
    """Return list of (file_path, test_name, line_no) for failed/errored tests."""
    if not junit_xml.exists():
        print(f"ERROR: {junit_xml} not found — pytest may not have produced output.", file=sys.stderr)
        sys.exit(2)

    tree = ET.parse(junit_xml)
    root = tree.getroot()

    failed = []
    for tc in root.iter("testcase"):
        if tc.find("failure") is not None or tc.find("error") is not None:
            file_attr = tc.get("file", "")
            name = tc.get("name", "")
            line = int(tc.get("line", "0")) if tc.get("line", "").strip() else 0

            # pytest-asyncio class-based tests may omit the file attribute.
            # Fall back to deriving the path from classname (e.g.
            # "tests.test_foo.TestBar" → "tests/test_foo.py").
            if not file_attr:
                classname = tc.get("classname", "")
                parts = classname.split(".")
                module_parts = []
                for part in parts:
                    if part and part[0].isupper():
                        break
                    module_parts.append(part)
                if module_parts:
                    file_attr = "/".join(module_parts) + ".py"

            failed.append((file_attr, name, line))
    return failed


def has_flaky_annotation(source_file: Path, test_name: str, line_no: int) -> bool:
    """
    Return True if the test function has a '# flaky:' comment nearby.

    Searches:
    - The def line and up to 5 lines before it (covers decorators and inline comments)
    - Falls back to the line number reported by pytest if the def cannot be located
    """
    if not source_file.is_file():
        return False

    lines = source_file.read_text(encoding="utf-8").splitlines()

    # Strip parametrize suffix: test_foo[param1-param2] -> test_foo
    # Use r"\[.*$" (not r"\[.*\]$") to correctly handle nested brackets
    # and edge cases like test_foo[a]-extra by stripping from the first "[" onward.
    base_name = re.sub(r"\[.*$", "", test_name)

    def_line_idx: int | None = None
    for i, line in enumerate(lines):
        if re.match(rf"\s*(?:async\s+)?def\s+{re.escape(base_name)}\s*[:(]", line):
            def_line_idx = i
            break

    if def_line_idx is None and line_no > 0:
        def_line_idx = line_no - 1

    if def_line_idx is None:
        return False

    search_start = max(0, def_line_idx - 5)
    snippet = "\n".join(lines[search_start : def_line_idx + 1])
    return bool(re.search(r"#\s*flaky\s*:", snippet, re.IGNORECASE))


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: check_flaky_annotations.py <junit-xml-path>", file=sys.stderr)
        sys.exit(2)

    junit_xml = Path(sys.argv[1])
    failed = find_failed_tests(junit_xml)

    if not failed:
        print("No failed tests in XML — nothing to check.")
        sys.exit(0)

    unannotated = []
    for file_path, test_name, line_no in failed:
        if not has_flaky_annotation(Path(file_path), test_name, line_no):
            unannotated.append((file_path, test_name, line_no))

    if not unannotated:
        print(f"All {len(failed)} failing test(s) carry '# flaky: reason' annotations.")
        print("Acknowledged flaky tests — PR is unblocked.")
        sys.exit(0)

    print("FLAKY ANNOTATION REQUIRED", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(
        "The following test(s) failed on both CI attempts but lack a '# flaky: reason' annotation.\n"
        "Add the annotation to unblock the PR and register the test as known flaky debt.\n",
        file=sys.stderr,
    )
    for file_path, test_name, line_no in unannotated:
        loc = f"{file_path}:{line_no}" if line_no else file_path
        print(f"  MISSING annotation: {loc}::{test_name}", file=sys.stderr)

    print("\nExample fix:", file=sys.stderr)
    print("  def test_something():  # flaky: timing-sensitive, tracked in issue #123", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
