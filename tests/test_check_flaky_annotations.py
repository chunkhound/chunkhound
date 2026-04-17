"""
Tests for scripts/check_flaky_annotations.py

Covers: happy path (all annotated), blocking path (unannotated failure),
        and the XML-missing edge case.
"""
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from scripts.check_flaky_annotations import find_failed_tests, has_flaky_annotation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_junit(tmp_path: Path, testcases: list[dict]) -> Path:
    """Write a minimal JUnit XML file and return its path."""
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    for tc in testcases:
        el = ET.SubElement(suite, "testcase", **{k: v for k, v in tc.items() if k != "_outcome"})
        outcome = tc.get("_outcome", "pass")
        if outcome == "failure":
            ET.SubElement(el, "failure", message="assert False")
        elif outcome == "error":
            ET.SubElement(el, "error", message="setup error")
    xml_path = tmp_path / "results.xml"
    ET.ElementTree(root).write(xml_path)
    return xml_path


# ---------------------------------------------------------------------------
# find_failed_tests
# ---------------------------------------------------------------------------

class TestFindFailedTests:
    def test_happy_path_no_failures(self, tmp_path):
        xml_path = _write_junit(tmp_path, [
            {"classname": "tests.test_foo", "name": "test_ok", "file": "tests/test_foo.py", "line": "1"},
        ])
        assert find_failed_tests(xml_path) == []

    def test_returns_failed_and_errored(self, tmp_path):
        xml_path = _write_junit(tmp_path, [
            {"classname": "tests.test_foo", "name": "test_fail", "file": "tests/test_foo.py", "line": "10", "_outcome": "failure"},
            {"classname": "tests.test_foo", "name": "test_err",  "file": "tests/test_foo.py", "line": "20", "_outcome": "error"},
            {"classname": "tests.test_foo", "name": "test_pass", "file": "tests/test_foo.py", "line": "30"},
        ])
        failed = find_failed_tests(xml_path)
        assert len(failed) == 2
        names = {t[1] for t in failed}
        assert names == {"test_fail", "test_err"}

    def test_falls_back_to_classname_when_file_missing(self, tmp_path):
        xml_path = _write_junit(tmp_path, [
            {"classname": "tests.test_bar.TestSuite", "name": "test_x", "line": "5", "_outcome": "failure"},
        ])
        failed = find_failed_tests(xml_path)
        assert len(failed) == 1
        file_path, name, line = failed[0]
        assert file_path == "tests/test_bar.py"
        assert name == "test_x"

    def test_missing_xml_exits_2(self, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            find_failed_tests(tmp_path / "nonexistent.xml")
        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# has_flaky_annotation
# ---------------------------------------------------------------------------

class TestHasFlakyAnnotation:
    def test_annotated_inline(self, tmp_path):
        src = tmp_path / "test_foo.py"
        src.write_text(textwrap.dedent("""\
            def test_something():  # flaky: timing-sensitive, tracked in issue #123
                pass
        """))
        assert has_flaky_annotation(src, "test_something", 1) is True

    def test_annotated_on_preceding_line(self, tmp_path):
        src = tmp_path / "test_foo.py"
        src.write_text(textwrap.dedent("""\
            # flaky: known race condition, tracked in issue #42
            def test_something():
                pass
        """))
        assert has_flaky_annotation(src, "test_something", 2) is True

    def test_not_annotated(self, tmp_path):
        src = tmp_path / "test_foo.py"
        src.write_text(textwrap.dedent("""\
            def test_something():
                pass
        """))
        assert has_flaky_annotation(src, "test_something", 1) is False

    def test_async_def(self, tmp_path):
        src = tmp_path / "test_foo.py"
        src.write_text(textwrap.dedent("""\
            async def test_async():  # flaky: event loop timing
                pass
        """))
        assert has_flaky_annotation(src, "test_async", 1) is True

    def test_parametrized_suffix_stripped(self, tmp_path):
        src = tmp_path / "test_foo.py"
        src.write_text(textwrap.dedent("""\
            def test_param():  # flaky: parametrize race
                pass
        """))
        assert has_flaky_annotation(src, "test_param[case1-val2]", 1) is True

    def test_missing_source_file_returns_false(self, tmp_path):
        assert has_flaky_annotation(tmp_path / "ghost.py", "test_x", 1) is False

    def test_case_insensitive_flaky_keyword(self, tmp_path):
        src = tmp_path / "test_foo.py"
        src.write_text("def test_upper():  # FLAKY: reason\n    pass\n")
        assert has_flaky_annotation(src, "test_upper", 1) is True
