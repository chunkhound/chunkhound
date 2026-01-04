#!/usr/bin/env python3
"""
Test runner for config unification tests.

This script runs the TDD tests for config system unification
and shows which tests are currently failing (expected) and
which ones are passing (good baseline).
"""

import subprocess
from pathlib import Path


def run_test_file(test_file):
    """Run a specific test file and return results."""
    print(f"\n{'=' * 60}")
    print(f"Running {test_file}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(
            ["python", "-m", "pytest", test_file, "-v"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=30,
        )

        print(f"Exit code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("Test timed out!")
        return False
    except Exception as e:
        print(f"Error running test: {e}")
        return False


def main():
    """Run all config unification tests."""
    test_files = [
        "tests/test_config_unification.py",
        "tests/test_mcp_server_config_patterns.py",
        "tests/test_config_integration.py",
    ]

    results = {}

    print("Running TDD tests for config system unification...")
    print("These tests are EXPECTED to fail until the refactor is complete.")
    print("This establishes the baseline for what needs to be implemented.")

    for test_file in test_files:
        results[test_file] = run_test_file(test_file)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    passing = sum(1 for passed in results.values() if passed)
    failing = len(results) - passing

    print(f"Passing tests: {passing}")
    print(f"Failing tests: {failing}")

    for test_file, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL (expected)"
        print(f"  {status} {test_file}")

    if failing > 0:
        print(f"\n🔨 {failing} test files are failing - this is expected!")
        print(
            "These tests define the target behavior for the config system unification."
        )
        print("Run the refactor to make these tests pass.")

    if passing > 0:
        print(f"\n✅ {passing} test files are passing - good baseline!")
        print("These tests should continue to pass after the refactor.")


if __name__ == "__main__":
    main()
