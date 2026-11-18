"""QA contracts for realtime indexing and the public MCP search tool.

The tests convert the manual QA checklist into deterministic checks of what a
user sees: content becomes searchable, edits and deletions are reflected, every
supported language is indexed, concurrent writes do not break search, and the
markdown pagination contract holds. They never inspect queues, chunk counts, or
provider internals.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from chunkhound.core.types.common import Language
from tests.helpers.realtime_harness import RealtimeHarness, realtime_harness
from tests.helpers.search_contract import (
    results_are_empty,
    results_contain,
    search_tool,
    wait_for_search,
)
from tests.utils.windows_compat import get_fs_event_timeout

# =============================================================================
# Test Configuration Constants
# =============================================================================

CONCURRENT_SETUP_MARKER = "concurrent_qa_test"
SEARCH_ITERATION_DELAY_SECONDS = 0.2  # Paces the concurrent search scenario
FILE_OPERATION_DELAY_SECONDS = 0.3  # Paces the concurrent write scenario
NUM_BASE_CONCURRENT_FILES = 3
NUM_CONCURRENT_SEARCHES = 10
NUM_RAPID_MODIFICATIONS = 5
AVG_SEARCH_TIME_LIMIT_SECONDS = 2.0
MAX_SEARCH_TIME_LIMIT_SECONDS = 5.0

NUM_PAGINATION_TEST_FILES = 15
DEFAULT_PAGE_SIZE = 10
MAX_PAGINATION_PAGES = 100
PAGINATION_PATTERN = "pagination_test_common_pattern"

MIN_MAJOR_LANGUAGES_REQUIRED = 3
MIN_TOTAL_LANGUAGES_REQUIRED = 3
BASE_OVERHEAD_SECONDS = 60  # Fixture setup, initial scan, assertions
# Per-language budget tracks the platform fs-event settle timeout (45s Windows CI).
LANGUAGE_COVERAGE_TIMEOUT_SECONDS = BASE_OVERHEAD_SECONDS + len(
    [lang for lang in Language if lang != Language.UNKNOWN]
) * get_fs_event_timeout()

EXISTING_FILE_CONTENT = """def existing_function():
    '''This is an existing function for QA testing'''
    return "existing_content"

class ExistingClass:
    def existing_method(self):
        return "existing_method_result"
"""

NEW_FILE_CONTENT = """def newly_added_function():
    '''This function was just added for QA validation'''
    return "newly_added_content_unique_string"

class NewlyAddedClass:
    def new_method(self):
        return "new_method_qa_test"
"""

ADDED_EDIT_CONTENT = (
    EXISTING_FILE_CONTENT
    + """

def added_during_edit():
    '''This function was added during file edit'''
    return "added_content_edit_qa"
"""
)

MODIFIED_EDIT_CONTENT = """def existing_function():
    '''This function was MODIFIED during edit'''
    return "MODIFIED_existing_content"

def added_during_edit():
    '''This function was added during file edit'''
    return "added_content_edit_qa"

# Note: ExistingClass was DELETED
"""

# =============================================================================
# Language Coverage Data
# =============================================================================

LANGUAGE_CONTENT_TEMPLATES: dict[Language, str | None] = {
    Language.PYTHON: 'def qa_test_function():\n    """Python QA test"""\n    return "python_qa_unique"',  # noqa: E501
    Language.JAVASCRIPT: 'function qaTestFunction() {\n    // JavaScript QA test\n    return "javascript_qa_unique";\n}',  # noqa: E501
    Language.TYPESCRIPT: 'function qaTestFunction(): string {\n    // TypeScript QA test\n    return "typescript_qa_unique";\n}',  # noqa: E501
    Language.TSX: "function QAComponent(): JSX.Element {\n    // TSX QA test\n    return <div>tsx_qa_unique</div>;\n}",  # noqa: E501
    Language.JSX: "function QAComponent() {\n    // JSX QA test\n    return <div>jsx_qa_unique</div>;\n}",  # noqa: E501
    Language.JAVA: 'public class QATest {\n    // Java QA test\n    public String test() { return "java_qa_unique"; }\n}',  # noqa: E501
    Language.CSHARP: 'public class QATest {\n    // C# QA test\n    public string Test() { return "csharp_qa_unique"; }\n}',  # noqa: E501
    Language.GO: 'package main\n\n// Go QA test\nfunc qaTestFunction() string {\n    return "go_qa_unique"\n}',  # noqa: E501
    Language.RUST: 'fn qa_test_function() -> &\'static str {\n    // Rust QA test\n    "rust_qa_unique"\n}',  # noqa: E501
    Language.C: '#include <stdio.h>\n\n// C QA test\nchar* qa_test_function() {\n    return "c_qa_unique";\n}',  # noqa: E501
    Language.CPP: '#include <string>\n\n// C++ QA test\nstd::string qaTestFunction() {\n    return "cpp_qa_unique";\n}',  # noqa: E501
    Language.METAL: '// Metal QA test (MSL is C++14)\nconst char* qaTestFunction() {\n    return "metal_qa_unique";\n}',  # noqa: E501
    Language.BASH: '#!/bin/bash\n# Bash QA test\nqa_test_function() {\n    echo "bash_qa_unique"\n}',  # noqa: E501
    Language.MARKDOWN: "# QA Test\n\nThis is a **markdown QA test** with `markdown_qa_unique` content.",  # noqa: E501
    Language.JSON: '{\n    "qa_test": true,\n    "content": "json_qa_unique",\n    "type": "qa_validation"\n}',  # noqa: E501
    Language.YAML: 'qa_test: true\ncontent: "yaml_qa_unique"\ntype: qa_validation',
    Language.TOML: '[qa_test]\ncontent = "toml_qa_unique"\ntype = "qa_validation"',
    Language.TEXT: "Plain text QA test file.\nContains: text_qa_unique\nFor validation purposes.",  # noqa: E501
    Language.VUE: """<template>
  <div class="qa-test">
    <h1>{{ message }}</h1>
  </div>
</template>

<script setup>
function qaTestFunction() {
  return "vue_qa_unique";
}
</script>

<style scoped>
.qa-test {
  color: blue;
}
</style>""",
    Language.SVELTE: """<script lang="ts">
  function qaTestFunction() {
    return "svelte_qa_unique";
  }
</script>

<main>
  <h1>Svelte QA Test</h1>
</main>

<style>
  h1 {
    color: blue;
  }
</style>""",
    Language.GROOVY: 'def qaTestFunction() {\n    // Groovy QA test\n    return "groovy_qa_unique"\n}',  # noqa: E501
    Language.KOTLIN: 'fun qaTestFunction(): String {\n    // Kotlin QA test\n    return "kotlin_qa_unique"\n}',  # noqa: E501
    Language.MAKEFILE: '.PHONY: qa_test\n# Makefile QA test\nqa_test:\n\t@echo "makefile_qa_unique"',  # noqa: E501
    Language.MATLAB: '% MATLAB QA test\nfunction result = qa_test_function()\n    result = "matlab_qa_unique";\nend',  # noqa: E501
    Language.LUA: '-- Lua QA test\nfunction qa_test_function()\n    return "lua_qa_unique"\nend',  # noqa: E501
    Language.HASKELL: '-- Haskell QA test\nqaTestFunction :: String\nqaTestFunction = "haskell_qa_unique"',  # noqa: E501
    Language.HCL: '# HCL QA test\nvariable "qa_test" {\n  default = "hcl_qa_unique"\n}',
    Language.DART: '// Dart QA test\nString qaTestFunction() {\n  return "dart_qa_unique";\n}',  # noqa: E501
    Language.OBJC: '// Objective-C QA test\n@implementation QATest\n- (NSString *)qaTestMethod {\n    return @"objc_qa_unique";\n}\n@end',  # noqa: E501
    Language.PHP: '<?php\n// PHP QA test\nfunction qa_test_function() {\n    return "php_qa_unique";\n}',  # noqa: E501
    Language.SWIFT: '// Swift QA test\nfunc qaTestFunction() -> String {\n    return "swift_qa_unique"\n}',  # noqa: E501
    Language.POWERSHELL: '# PowerShell QA test\nfunction Get-QaTest {\n    return "powershell_qa_unique"\n}',  # noqa: E501
    Language.ZIG: '// Zig QA test\nfn qa_test_function() []const u8 {\n    return "zig_qa_unique";\n}',  # noqa: E501
    Language.PDF: None,  # PDF is binary, skip content template
    Language.SQL: "-- SQL QA test\nCREATE TABLE qa_test (\n    id INTEGER PRIMARY KEY,\n    content TEXT DEFAULT 'sql_qa_unique'\n);",  # noqa: E501
    Language.ELIXIR: 'defmodule QATest do\n  # Elixir QA test\n  def test, do: "elixir_qa_unique"\nend',  # noqa: E501
    Language.HTML: "<!DOCTYPE html>\n<html>\n<body>\n  <!-- HTML QA test -->\n  <p>html_qa_unique</p>\n</body>\n</html>",  # noqa: E501
    Language.CSS: '/* CSS QA test */\n.qa-test {\n  content: "css_qa_unique";\n  color: blue;\n}',  # noqa: E501
    Language.SCSS: '/* SCSS QA test */\n$color: blue;\n.qa-test {\n  content: "scss_qa_unique";\n  color: $color;\n}',  # noqa: E501
    Language.JINJA: '<!DOCTYPE html>\n<html>\n<body>\n  {# Jinja QA test #}\n  <p>{{ "jinja_qa_unique" }}</p>\n</body>\n</html>',  # noqa: E501
    Language.TWINCAT: '<?xml version="1.0" encoding="utf-8"?>\n<TcPlcObject Version="1.1.0.1">\n  <POU Name="QA_TEST" Id="{00000000-0000-0000-0000-000000000001}">\n    <Declaration><![CDATA[PROGRAM QA_TEST\nVAR\n  bFlag : BOOL := TRUE;\nEND_VAR]]></Declaration>\n    <Implementation>\n      <ST><![CDATA[IF bFlag THEN\n  (* twincat_qa_unique *)\nEND_IF]]></ST>\n    </Implementation>\n  </POU>\n</TcPlcObject>',  # noqa: E501
}

LANGUAGE_EXTENSIONS: dict[Language, str] = {
    Language.PYTHON: ".py",
    Language.JAVASCRIPT: ".js",
    Language.TYPESCRIPT: ".ts",
    Language.TSX: ".tsx",
    Language.JSX: ".jsx",
    Language.JAVA: ".java",
    Language.CSHARP: ".cs",
    Language.GO: ".go",
    Language.RUST: ".rs",
    Language.C: ".c",
    Language.CPP: ".cpp",
    Language.METAL: ".metal",
    Language.BASH: ".sh",
    Language.MARKDOWN: ".md",
    Language.JSON: ".json",
    Language.YAML: ".yaml",
    Language.TOML: ".toml",
    Language.TEXT: ".txt",
    Language.GROOVY: ".groovy",
    Language.KOTLIN: ".kt",
    Language.MAKEFILE: ".mk",
    Language.MATLAB: ".m",
    Language.VUE: ".vue",
    Language.SVELTE: ".svelte",
    Language.LUA: ".lua",
    Language.HASKELL: ".hs",
    Language.HCL: ".tf",
    Language.DART: ".dart",
    # OBJC shares .m with MATLAB; content-based detection disambiguates them.
    Language.OBJC: ".m",
    Language.PHP: ".php",
    Language.SWIFT: ".swift",
    Language.POWERSHELL: ".ps1",
    Language.ZIG: ".zig",
    Language.PDF: ".pdf",
    Language.SQL: ".sql",
    Language.ELIXIR: ".ex",
    Language.HTML: ".html",
    Language.CSS: ".css",
    Language.SCSS: ".scss",
    Language.JINJA: ".jinja",
    Language.TWINCAT: ".TcPOU",
}

MAJOR_LANGUAGES = ("python", "javascript", "typescript", "java", "go")


# =============================================================================
# Public Search Contract Helpers
# =============================================================================

_RESULT_HEADING_RE = re.compile(r"^## `[^`]+`.*$", re.MULTILINE)
_PAGE_FOOTER_RE = re.compile(r"results (\d+)–(\d+) of (\d+)")
_NEXT_OFFSET_RE = re.compile(r"next_offset=(\d+)")


@dataclass(frozen=True, slots=True)
class SearchPage:
    """The public pagination facts rendered by the search tool."""

    headings: tuple[str, ...]
    start: int
    end: int
    total: int
    next_offset: int | None


def parse_search_page(output: str) -> SearchPage:
    """Parse one markdown page; fail loudly when the footer contract breaks."""
    footer = _PAGE_FOOTER_RE.search(output)
    if footer is None:
        raise AssertionError(
            f"search output is missing a pagination footer: {output[-200:]!r}"
        )
    start, end, total = (int(value) for value in footer.groups())
    next_match = _NEXT_OFFSET_RE.search(output)
    return SearchPage(
        headings=tuple(_RESULT_HEADING_RE.findall(output)),
        start=start,
        end=end,
        total=total,
        next_offset=int(next_match.group(1)) if next_match else None,
    )


async def _expect_searchable(services, query: str, marker: str) -> None:
    """Wait until the public search tool renders ``marker``."""
    await wait_for_search(
        services, query, lambda output: results_contain(output, marker)
    )


async def _expect_absent(services, query: str) -> None:
    """Wait until the public search tool reports no matches."""
    await wait_for_search(services, query, results_are_empty)


def _validate_page(page: SearchPage, offset: int, page_size: int) -> None:
    """Check the public page/offset contract for one rendered page."""
    assert page.start == offset + 1, (
        f"page at offset {offset} reports results starting at {page.start}"
    )
    assert len(page.headings) == page.end - page.start + 1, (
        f"page at offset {offset} reports {page.end - page.start + 1} results "
        f"but rendered {len(page.headings)}"
    )
    assert len(page.headings) <= page_size, (
        f"page at offset {offset} returned {len(page.headings)} results "
        f"for page_size={page_size}"
    )
    if page.next_offset is not None:
        assert page.next_offset > offset, (
            f"page at offset {offset} advertises non-advancing "
            f"next_offset={page.next_offset}"
        )


def _assert_language_fixture_coverage() -> None:
    """Fail explicitly when a supported language lacks test data."""
    testable = {
        language
        for language in Language
        if language not in (Language.UNKNOWN, Language.GIT_DIFF)
    }
    uncovered = testable - (set(LANGUAGE_CONTENT_TEMPLATES) & set(LANGUAGE_EXTENSIONS))
    assert not uncovered, (
        "Language(s) missing from test coverage: "
        f"{sorted(language.value for language in uncovered)}. "
        "Add a content template and extension mapping for each language."
    )


async def _collect_all_pages(services, query: str, page_size: int) -> list[SearchPage]:
    """Walk the public pagination links until the tool reports no more pages."""
    pages: list[SearchPage] = []
    offset = 0
    while len(pages) < MAX_PAGINATION_PAGES:
        output = await search_tool(services, query, page_size=page_size, offset=offset)
        if results_are_empty(output):
            break
        page = parse_search_page(output)
        _validate_page(page, offset, page_size)
        pages.append(page)
        if page.next_offset is None:
            break
        offset = page.next_offset
    return pages


def _assert_pagination_contract(pages: list[SearchPage]) -> int:
    """Verify total stability, completeness, and cross-page uniqueness."""
    assert len(pages) >= 2, f"expected multiple pages, got {len(pages)}"

    total = pages[0].total
    assert pages[-1].next_offset is None, (
        f"pagination stopped after {len(pages)} pages while more results remain"
    )

    headings: list[str] = []
    for index, page in enumerate(pages):
        assert page.total == total, (
            f"page {index + 1} reports total={page.total}, first page reports {total}"
        )
        headings.extend(page.headings)

    assert len(headings) == total, (
        f"pagination exposed {len(headings)} results but reports total={total}"
    )
    assert len(headings) == len(set(headings)), "pagination repeated a result"
    return total


# =============================================================================
# Scenario Helpers
# =============================================================================


def _create_language_files(watch_dir: Path) -> list[tuple[Language, str]]:
    """Write one fixture per language that has textual content."""
    created: list[tuple[Language, str]] = []
    for language, content in LANGUAGE_CONTENT_TEMPLATES.items():
        if content is None:
            continue
        marker = f"{language.value}_qa_unique"
        path = watch_dir / f"qa_test_{language.value}{LANGUAGE_EXTENSIONS[language]}"
        path.write_text(content)
        created.append((language, marker))
    return created


async def _searchable_languages(
    services, created: list[tuple[Language, str]]
) -> tuple[set[str], list[str]]:
    """Wait for each marker concurrently; report searchable and failed languages."""
    outcomes = await asyncio.gather(
        *(
            wait_for_search(
                services,
                marker,
                lambda output, expected=marker: results_contain(output, expected),
            )
            for _, marker in created
        ),
        return_exceptions=True,
    )
    pairs = zip(created, outcomes, strict=True)
    failed: list[str] = []
    successful: set[str] = set()
    for (language, _), outcome in pairs:
        if isinstance(outcome, BaseException):
            failed.append(f"{language.value} ({outcome})")
        else:
            successful.add(language.value)
    return successful, failed


def _create_concurrent_base_files(watch_dir: Path) -> list[tuple[Path, str]]:
    """Create the files that concurrent searches observe while writes happen."""
    base_files: list[tuple[Path, str]] = []
    for index in range(NUM_BASE_CONCURRENT_FILES):
        path = watch_dir / f"concurrent_test_{index}.py"
        marker = f"{CONCURRENT_SETUP_MARKER}_{index}"
        path.write_text(f'def concurrent_function_{index}():\n    return "{marker}"\n')
        base_files.append((path, marker))
    return base_files


def _rapid_file_content(index: int) -> str:
    return f'def rapid_function_{index}():\n    return "rapid_qa_test_{index}"\n'


async def _search_while_modifying(services) -> list[float]:
    """Measure search latency while another task rewrites files."""
    durations: list[float] = []
    for _ in range(NUM_CONCURRENT_SEARCHES):
        started = time.perf_counter()
        await search_tool(services, CONCURRENT_SETUP_MARKER, page_size=50)
        durations.append(time.perf_counter() - started)
        await asyncio.sleep(SEARCH_ITERATION_DELAY_SECONDS)
    return durations


async def _rapidly_modify_files(
    watch_dir: Path, base_files: list[tuple[Path, str]]
) -> int:
    """Create and rewrite files concurrently with search; return operation count."""
    operations = 0
    for index in range(NUM_RAPID_MODIFICATIONS):
        content = _rapid_file_content(index)
        (watch_dir / f"rapid_modify_{index}.py").write_text(content)
        operations += 1
        if index < len(base_files):
            base_path, _ = base_files[index]
            base_path.write_text(content + f"\n# Modified at iteration {index}\n")
            operations += 1
        await asyncio.sleep(FILE_OPERATION_DELAY_SECONDS)
    return operations


def _pagination_functions(file_index: int) -> str:
    return "\n\n".join(
        f"def {PAGINATION_PATTERN}_function_{file_index}_{position}():\n"
        f'    """Fixture {position} for pagination file {file_index}."""\n'
        f'    values = ["{PAGINATION_PATTERN}_value_{file_index}_{position}_"'
        f" + str(item) for item in range(8)]\n"
        f'    return "{PAGINATION_PATTERN}_result_{file_index}_{position}"'
        f" + str(values)\n"
        for position in range(2)
    )


def _pagination_processor(file_index: int) -> str:
    return (
        f"class {PAGINATION_PATTERN}_processor_{file_index}:\n"
        f'    """Processor for pagination file {file_index}."""\n\n'
        f"    def process(self, items):\n"
        f'        return ["{PAGINATION_PATTERN}_processed_{file_index}_"'
        f" + str(item) for item in items]\n"
    )


def _pagination_file_content(file_index: int) -> str:
    """Deterministic module so each file yields several search chunks."""
    return (
        f'"""{PAGINATION_PATTERN} module {file_index}."""\n\n'
        f"{PAGINATION_PATTERN}_CONSTANT_{file_index} = "
        f'"{PAGINATION_PATTERN}_constant_{file_index}"\n\n'
        f"{_pagination_functions(file_index)}\n\n"
        f"{_pagination_processor(file_index)}\n"
    )


# =============================================================================
# Tests
# =============================================================================


class TestQADeterministic:
    """Deterministic QA contracts for realtime indexing plus public search."""

    @pytest.fixture
    async def qa_setup(self, tmp_path: Path) -> AsyncIterator[RealtimeHarness]:
        """Run real services against a private workspace with QA include rules."""
        async with realtime_harness(
            tmp_path,
            include=("*",),
            exclude=("*.log", "__pycache__/"),
        ) as harness:
            yield harness

    async def test_file_lifecycle_search_validation(self, qa_setup: RealtimeHarness):
        """Created, added, modified, and deleted content follows search visibility."""
        services = qa_setup.services
        watch_dir = qa_setup.watch_dir

        existing_file = watch_dir / "existing_test.py"
        existing_file.write_text(EXISTING_FILE_CONTENT)
        await _expect_searchable(services, "existing_function", "existing_function")

        new_file = watch_dir / "new_added_file.py"
        new_file.write_text(NEW_FILE_CONTENT)
        await _expect_searchable(
            services,
            "newly_added_content_unique_string",
            "newly_added_content_unique_string",
        )

        existing_file.write_text(ADDED_EDIT_CONTENT)
        await _expect_searchable(
            services, "added_content_edit_qa", "added_content_edit_qa"
        )

        existing_file.write_text(MODIFIED_EDIT_CONTENT)
        await _expect_searchable(
            services, "MODIFIED_existing_content", "MODIFIED_existing_content"
        )
        await _expect_absent(services, r"class ExistingClass:")

        new_file.unlink()
        await _expect_absent(services, "newly_added_content_unique_string")

    @pytest.mark.timeout(LANGUAGE_COVERAGE_TIMEOUT_SECONDS)
    async def test_language_coverage_comprehensive(self, qa_setup: RealtimeHarness):
        """Every supported language with a text fixture becomes searchable."""
        _assert_language_fixture_coverage()
        created = _create_language_files(qa_setup.watch_dir)
        successful, failed = await _searchable_languages(qa_setup.services, created)
        working_major = sorted(successful & set(MAJOR_LANGUAGES))

        assert len(working_major) >= MIN_MAJOR_LANGUAGES_REQUIRED, (
            f"Expected at least {MIN_MAJOR_LANGUAGES_REQUIRED} major languages "
            f"searchable, got {working_major}. Failures: {failed}"
        )
        assert len(successful) >= MIN_TOTAL_LANGUAGES_REQUIRED, (
            f"Expected at least {MIN_TOTAL_LANGUAGES_REQUIRED} languages searchable, "
            f"got {sorted(successful)}. Failures: {failed}"
        )

    async def test_concurrent_operations_and_timing(self, qa_setup: RealtimeHarness):
        """Searches stay fast and usable while files are created and rewritten."""
        services = qa_setup.services
        base_files = _create_concurrent_base_files(qa_setup.watch_dir)

        await wait_for_search(
            services,
            CONCURRENT_SETUP_MARKER,
            lambda output: all(marker in output for _, marker in base_files),
            page_size=50,
        )

        search_durations, modification_count = await asyncio.gather(
            _search_while_modifying(services),
            _rapidly_modify_files(qa_setup.watch_dir, base_files),
        )

        assert len(search_durations) == NUM_CONCURRENT_SEARCHES
        expected_operations = NUM_RAPID_MODIFICATIONS + min(
            NUM_RAPID_MODIFICATIONS, NUM_BASE_CONCURRENT_FILES
        )
        assert modification_count == expected_operations

        average = sum(search_durations) / len(search_durations)
        slowest = max(search_durations)
        assert average < AVG_SEARCH_TIME_LIMIT_SECONDS, (
            f"Average search during concurrent writes was {average:.3f}s"
        )
        assert slowest < MAX_SEARCH_TIME_LIMIT_SECONDS, (
            f"Slowest search during concurrent writes was {slowest:.3f}s"
        )

        await wait_for_search(
            services,
            "rapid_qa_test",
            lambda output: all(
                f"rapid_qa_test_{index}" in output
                for index in range(NUM_RAPID_MODIFICATIONS)
            ),
            page_size=50,
        )

    async def test_pagination_contract(self, qa_setup: RealtimeHarness):
        """Pagination exposes every match once, with accurate offsets and bounds."""
        services = qa_setup.services

        await _expect_absent(services, "non_existing_unique_pattern_qa_test_12345")

        single_file = qa_setup.watch_dir / "single_result_test.py"
        single_marker = "single_unique_result_qa_test"
        single_file.write_text(
            f'def single_unique_function():\n    return "{single_marker}"\n'
        )
        single_output = await wait_for_search(
            services,
            single_marker,
            lambda output: results_contain(output, single_marker),
        )
        assert len(parse_search_page(single_output).headings) == 1

        for index in range(NUM_PAGINATION_TEST_FILES):
            path = qa_setup.watch_dir / f"pagination_test_{index:03d}.py"
            path.write_text(_pagination_file_content(index))

        await asyncio.gather(
            *(
                _expect_searchable(
                    services,
                    f"{PAGINATION_PATTERN}_constant_{index}",
                    f"{PAGINATION_PATTERN}_constant_{index}",
                )
                for index in range(NUM_PAGINATION_TEST_FILES)
            )
        )

        pages = await _collect_all_pages(
            services, PAGINATION_PATTERN, DEFAULT_PAGE_SIZE
        )
        total = _assert_pagination_contract(pages)

        beyond = await search_tool(
            services, PAGINATION_PATTERN, page_size=10, offset=total + 100
        )
        assert results_are_empty(beyond), (
            f"offset {total + 100} beyond total {total} should be empty"
        )

        large_page = parse_search_page(
            await search_tool(services, PAGINATION_PATTERN, page_size=100)
        )
        assert 0 < len(large_page.headings) <= min(100, total), (
            f"large page returned {len(large_page.headings)} results for total={total}"
        )
        assert large_page.total == total
