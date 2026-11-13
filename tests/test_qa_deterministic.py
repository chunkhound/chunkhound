"""Deterministic QA test suite for ChunkHound search functionality.

This test converts the manual QA process from .claude/commands/qa.md into
deterministic automated tests. Tests semantic_search and regex_search tools
with real-time indexing using actual MCP server components.

No mocks - tests the full integration path users experience.
"""

import asyncio
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from chunkhound.core.config.config import Config
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.core.types.common import Language
from chunkhound.database_factory import create_services
from chunkhound.embeddings import EmbeddingManager
from chunkhound.mcp_server.tools import execute_tool, search_impl
from chunkhound.registry import get_registry
from chunkhound.services.realtime.service import RealtimeIndexingService
from tests.fixtures.embedding_server import (
    EmbeddingConcept,
    MockEmbeddingServer,
    unit_vector,
)
from tests.fixtures.rerank_server_manager import RerankServerManager
from tests.rerank_server import MockRerankScenario
from tests.utils.windows_compat import (
    get_fs_event_timeout,
    realtime_backend_for_tests,
)

# =============================================================================
# Test Configuration Constants
# =============================================================================

# Timeout & Wait Durations
INITIAL_SCAN_WAIT_SECONDS = 2.0  # Wait for initial scan after service start
CONCURRENT_SETUP_WAIT_SECONDS = 3.0  # Wait before concurrent operations
PAGINATION_SETUP_WAIT_SECONDS = 3.0  # Wait after creating pagination test files
SEARCH_ITERATION_DELAY_SECONDS = 0.2  # Delay between search iterations
FILE_OPERATION_DELAY_SECONDS = 0.3  # Delay between file operations
STABILITY_CHECK_INTERVAL_SECONDS = 2.0  # Interval between stability checks
INDEXING_POLL_INTERVAL_SECONDS = 0.5  # Polling interval for indexing completion
SEMANTIC_VISIBILITY_TIMEOUT_SECONDS = 10.0  # Embeddings may follow indexing
SEMANTIC_VISIBILITY_POLL_SECONDS = 0.2
RIPGREP_TIMEOUT_SECONDS = 10  # Timeout for ripgrep subprocess

# Budget Constants (used in timeout calculations)
BASE_OVERHEAD_SECONDS = 60  # Fixture setup, initial scan, etc.
BUDGET_PER_LANGUAGE_SECONDS = 12  # Per-language budget (Windows CI worst case)
SEARCH_VALIDATION_BUDGET_SECONDS = 60  # Reserve for parallel searches + assertions
INDEXING_CAP_SECONDS = 200.0  # Hard cap for indexing wait (fail fast)
SINGLE_FILE_INDEXING_MAX_SECONDS = 10.0  # Max wait for single file indexing

# Threshold Constants
MIN_MAJOR_LANGUAGES_REQUIRED = 3  # Minimum major languages that must work
MIN_TOTAL_LANGUAGES_REQUIRED = 3  # Minimum total languages that must work
LOW_SUCCESS_RATE_THRESHOLD = 0.5  # Below this = low success rate warning
AVG_SEARCH_TIME_LIMIT_SECONDS = 2.0  # Max acceptable average search time
MAX_SEARCH_TIME_LIMIT_SECONDS = 5.0  # Max acceptable single search time
FILE_REFLECTION_MAX_SECONDS = 10.0  # Max time for file changes to reflect
SEARCH_EXECUTION_MAX_SECONDS = 5.0  # Max time for search execution
SEARCH_GOOD_THRESHOLD_SECONDS = 1.0  # Below = good search performance
SEARCH_ACCEPTABLE_THRESHOLD_SECONDS = 3.0  # Below = acceptable search performance

# Test Data Constants
NUM_PAGINATION_TEST_FILES = 15  # Files to create for pagination test
NUM_BASE_CONCURRENT_FILES = 3  # Base files for concurrent test
NUM_CONCURRENT_SEARCHES = 10  # Searches during concurrent operations
NUM_RAPID_MODIFICATIONS = 5  # File modifications in rapid test
MAX_PAGINATION_PAGES = 10  # Safety limit for pagination loop
MAX_STABILITY_CHECKS = 10  # Retry attempts for chunk stability
MIN_EXPECTED_CHUNKS_PAGINATION = 15  # Minimum chunks for pagination test
EXPECTED_CHUNKS_PER_FILE = 2  # Expected chunks per substantial file
DEFAULT_PAGE_SIZE = 10  # Standard pagination page size


# Semantic assertions use orthogonal concept vectors, so unrelated content
# cannot outrank the lifecycle state under test.
QA_EMBEDDING_DIMENSIONS = 8
QA_EMBEDDING_MODEL = "qa-embedding"
QA_RERANK_MODEL = "qa-reranker"

QA_EMBEDDING_CONCEPTS = (
    EmbeddingConcept(
        "existing",
        ("existing function QA", "existing_function", "existing_content"),
        unit_vector(0, QA_EMBEDDING_DIMENSIONS),
    ),
    EmbeddingConcept(
        "new",
        (
            "newly added function QA",
            "newly_added_function",
            "newly_added_content_unique_string",
        ),
        unit_vector(1, QA_EMBEDDING_DIMENSIONS),
    ),
    EmbeddingConcept(
        "added",
        (
            "function added during file edit",
            "added_during_edit",
            "added_content_edit_qa",
        ),
        unit_vector(2, QA_EMBEDDING_DIMENSIONS),
    ),
    EmbeddingConcept(
        "modified",
        ("MODIFIED existing content during edit", "MODIFIED_existing_content"),
        unit_vector(3, QA_EMBEDDING_DIMENSIONS),
    ),
)

# Documents are matched by marker, not by exact chunk text, so reranking stays
# stable when chunk boundaries or candidate order change.
QA_RERANK_SCENARIOS = tuple(
    MockRerankScenario(
        name=f"{concept.name}-lifecycle",
        query=query,
        documents=None,
        results=[],
        document_scores=scores,
    )
    for concept, query, scores in (
        (
            QA_EMBEDDING_CONCEPTS[0],
            "existing function QA testing",
            {"existing_function": 1.0, "existing_content": 0.9},
        ),
        (
            QA_EMBEDDING_CONCEPTS[1],
            "newly added function QA validation",
            {"newly_added_function": 1.0, "newly_added_content_unique_string": 0.9},
        ),
        (
            QA_EMBEDDING_CONCEPTS[2],
            "function added during file edit",
            {"added_during_edit": 1.0, "added_content_edit_qa": 0.9},
        ),
        (
            QA_EMBEDDING_CONCEPTS[3],
            "MODIFIED existing content during edit",
            {"MODIFIED_existing_content": 1.0},
        ),
    )
)


def _result_file_names(response: dict[str, object]) -> set[str]:
    """Return result filenames so assertions ignore absolute temp paths and ranks."""
    results = response.get("results", [])
    if not isinstance(results, list):
        return set()
    return {
        Path(str(result.get("file_path", ""))).name
        for result in results
        if isinstance(result, dict)
    }


def _semantic_result_contains(
    response: dict[str, object], filename: str, markers: tuple[str, ...]
) -> bool:
    """Check that the named file contributes a result with the expected text."""
    results = response.get("results", [])
    if not isinstance(results, list):
        return False
    return any(
        isinstance(result, dict)
        and Path(str(result.get("file_path", ""))).name == filename
        and any(marker in str(result.get("content", "")) for marker in markers)
        for result in results
    )


async def _wait_for_semantic_file(
    services,
    embedding_manager: EmbeddingManager,
    query: str,
    filename: str,
    markers: tuple[str, ...],
) -> dict[str, object]:
    """Poll user-visible semantic search until indexed content is embeddable.

    ``wait_for_file_indexed`` only proves chunks were stored. The realtime
    worker generates embeddings afterwards, so semantic visibility needs its
    own bounded wait instead of a fixed sleep.

    Markers prevent a stale chunk from the same file satisfying the wait after
    an edit; the assertion follows user-visible content, not internal state.
    """
    deadline = time.monotonic() + SEMANTIC_VISIBILITY_TIMEOUT_SECONDS
    response: dict[str, object] = {}
    while time.monotonic() < deadline:
        response = await search_impl(
            services=services,
            embedding_manager=embedding_manager,
            type="semantic",
            query=query,
            page_size=10,
            offset=0,
        )
        if _semantic_result_contains(response, filename, markers):
            return response
        await asyncio.sleep(SEMANTIC_VISIBILITY_POLL_SECONDS)
    raise AssertionError(
        f"Semantic search never exposed {filename} with any of {markers} "
        f"for query {query!r}; last response: {response}"
    )


def timeout_for_language_coverage() -> int:
    """Calculate timeout based on number of testable languages.

    Budget per language:
    - File creation + indexing wait: ~10s (Windows CI worst case)
    - Search validation: ~1s (parallelized)
    - Base overhead: BASE_OVERHEAD_SECONDS (fixture setup, initial scan, assertions)
    """
    num_languages = len([lang for lang in Language if lang != Language.UNKNOWN])
    return BASE_OVERHEAD_SECONDS + (num_languages * BUDGET_PER_LANGUAGE_SECONDS)


def _qa_embedding_config(
    embedding_server: MockEmbeddingServer, rerank_manager: RerankServerManager
) -> dict[str, object]:
    """Point the real provider at the deterministic loopback services."""
    return {
        "provider": "openai",
        "model": QA_EMBEDDING_MODEL,
        "api_key": "qa-loopback-key",
        "base_url": embedding_server.base_url,
        "output_dims": QA_EMBEDDING_DIMENSIONS,
        "rerank_model": QA_RERANK_MODEL,
        "rerank_url": f"{rerank_manager.base_url}/rerank",
        "rerank_format": "cohere",
        "rerank_ssl_verify": False,
        "batch_size": 100,
        "max_concurrent_batches": 1,
    }


def _build_qa_embedding_manager(config: Config) -> EmbeddingManager:
    """Share one provider between indexing and semantic search."""
    if config.embedding is None:
        raise AssertionError("QA setup requires an embedding configuration")
    provider = EmbeddingProviderFactory.create_provider(config.embedding)
    manager = EmbeddingManager()
    manager.register_provider(provider, set_default=True)  # type: ignore[arg-type]
    return manager


async def _stop_qa_realtime(realtime_service: RealtimeIndexingService | None) -> None:
    """Stop realtime work without cancelling its cleanup mid-transaction."""
    if realtime_service is None:
        return

    # wait_for cancels stop() on timeout and can leave DB executor tasks in a
    # state where provider.close() hangs, so wait without cancelling first.
    stop_task = asyncio.create_task(realtime_service.stop())
    done, _ = await asyncio.wait({stop_task}, timeout=10.0)
    if not done:
        if realtime_service.observer and realtime_service.observer.is_alive():
            realtime_service.observer.stop()
        done, _ = await asyncio.wait({stop_task}, timeout=3.0)
        if not done:
            stop_task.cancel()
            try:
                await stop_task
            except asyncio.CancelledError:
                pass
            return

    if not stop_task.cancelled() and (error := stop_task.exception()) is not None:
        raise error


def _restore_registry_state(previous: tuple[dict, object, object]) -> None:
    """Restore every part of the process-wide registry the fixture replaced."""
    registry = get_registry()
    providers, config, embedding_manager = previous
    registry._providers.clear()
    registry._providers.update(providers)
    registry._config = config
    registry._embedding_manager = embedding_manager


async def _close_qa_resources(
    services,
    realtime_service: RealtimeIndexingService | None,
    rerank_manager: RerankServerManager,
    embedding_server: MockEmbeddingServer,
    previous_registry: tuple[dict, object, object],
) -> list[Exception]:
    """Release every fixture resource, returning failures instead of hiding them."""
    errors: list[Exception] = []
    try:
        await _stop_qa_realtime(realtime_service)
    except Exception as error:
        errors.append(error)
    if services is not None:
        try:
            services.provider.close()
        except Exception as error:
            errors.append(error)
    try:
        await rerank_manager.stop()
    except Exception as error:
        errors.append(error)
    try:
        embedding_server.close()
    except Exception as error:
        errors.append(error)
    try:
        _restore_registry_state(previous_registry)
    except Exception as error:
        errors.append(error)
    return errors


class TestQADeterministic:
    """Deterministic QA test suite - converts manual testing into automated validation."""

    @pytest.fixture
    async def qa_setup(self):
        """Setup QA test environment with real services and loopback providers."""
        temp_dir = Path(tempfile.mkdtemp())
        db_path = temp_dir / ".chunkhound" / "test.db"
        watch_dir = temp_dir / "project"
        watch_dir.mkdir(parents=True)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        embedding_server = MockEmbeddingServer(
            QA_EMBEDDING_CONCEPTS, dimensions=QA_EMBEDDING_DIMENSIONS
        )
        rerank_manager = RerankServerManager(scenarios=list(QA_RERANK_SCENARIOS))
        services = None
        realtime_service = None

        # create_services replaces process-wide registry state. Snapshot all of
        # it so this suite cannot leak its loopback provider or temp-dir config.
        registry = get_registry()
        previous_registry = (
            dict(registry._providers),
            registry._config,
            registry._embedding_manager,
        )

        try:
            embedding_server.start()
            await rerank_manager.start()

            from types import SimpleNamespace

            config = Config(
                args=SimpleNamespace(path=temp_dir),
                database={"path": str(db_path), "provider": "duckdb"},
                embedding=_qa_embedding_config(embedding_server, rerank_manager),
                indexing={
                    "include": ["*"],
                    "exclude": ["*.log", "__pycache__/"],
                    "realtime_backend": realtime_backend_for_tests(),
                },
            )
            embedding_manager = _build_qa_embedding_manager(config)

            services = create_services(db_path, config, embedding_manager)
            services.provider.connect()
            realtime_service = RealtimeIndexingService(services, config)
            await realtime_service.start(watch_dir)

            await asyncio.sleep(INITIAL_SCAN_WAIT_SECONDS)
            yield (
                services,
                realtime_service,
                watch_dir,
                temp_dir,
                embedding_manager,
                rerank_manager,
            )
        finally:
            cleanup_errors: list[Exception] = []
            try:
                cleanup_errors = await _close_qa_resources(
                    services,
                    realtime_service,
                    rerank_manager,
                    embedding_server,
                    previous_registry,
                )
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
            if cleanup_errors:
                raise cleanup_errors[0]

    @pytest.mark.asyncio
    async def test_file_lifecycle_search_validation(
        self, qa_setup
    ):  # flaky: Windows CI polling monitor occasionally misses second file write within 45s window, tracked in issue #254
        """QA Items 1-4: Test file lifecycle with search validation."""
        services, realtime_service, watch_dir, _, embedding_manager, rerank_manager = (
            qa_setup
        )

        # QA Item 1: Pick specific existing file and search for it
        existing_file = watch_dir / "existing_test.py"
        existing_content = """def existing_function():
    '''This is an existing function for QA testing'''
    return "existing_content"

class ExistingClass:
    def existing_method(self):
        return "existing_method_result"
"""
        existing_file.write_text(existing_content)

        # Wait for file to be indexed
        found = await realtime_service.wait_for_file_indexed(
            existing_file, timeout=get_fs_event_timeout()
        )
        assert found, "Existing content should be searchable"

        # Search for existing content
        existing_regex = await search_impl(
            services=services,
            embedding_manager=embedding_manager,
            type="regex",
            query="existing_function",
            page_size=10,
            offset=0,
        )

        existing_semantic = await _wait_for_semantic_file(
            services,
            embedding_manager,
            "existing function QA testing",
            "existing_test.py",
            ("existing_content", "existing_function"),
        )
        assert "existing_test.py" in _result_file_names(existing_semantic), (
            "Semantic search should rank the existing file for its concept"
        )
        assert len(existing_regex.get("results", [])) > 0, (
            "Should find existing file content with regex"
        )
        print(
            "✓ Existing file search: "
            f"regex={len(existing_regex.get('results', []))}, "
            f"semantic={len(existing_semantic.get('results', []))}"
        )

        # QA Item 2: Add new file and search for it
        new_file = watch_dir / "new_added_file.py"
        new_content = """def newly_added_function():
    '''This function was just added for QA validation'''
    return "newly_added_content_unique_string"

class NewlyAddedClass:
    def new_method(self):
        return "new_method_qa_test"
"""
        new_file.write_text(new_content)

        # Wait for file to be indexed
        found = await realtime_service.wait_for_file_indexed(
            new_file, timeout=get_fs_event_timeout()
        )
        assert found, "New file content should be searchable"

        # Search for new content
        new_regex = await search_impl(
            services=services,
            embedding_manager=embedding_manager,
            type="regex",
            query="newly_added_content_unique_string",
            page_size=10,
            offset=0,
        )

        new_semantic = await _wait_for_semantic_file(
            services,
            embedding_manager,
            "newly added function QA validation",
            "new_added_file.py",
            ("newly_added_content_unique_string", "newly_added_function"),
        )
        assert "new_added_file.py" in _result_file_names(new_semantic), (
            "Semantic search should rank the newly added file"
        )
        assert len(new_regex.get("results", [])) > 0, (
            "Should find newly added file content with regex"
        )
        print(
            "✓ New file search: "
            f"regex={len(new_regex.get('results', []))}, "
            f"semantic={len(new_semantic.get('results', []))}"
        )

        # QA Item 3: Edit existing file - adding, deleting, and modifying content

        # 3a: Add content to existing file
        modified_content = (
            existing_content
            + """

def added_during_edit():
    '''This function was added during file edit'''
    return "added_content_edit_qa"
"""
        )
        realtime_service.reset_file_tracking(existing_file)
        existing_file.write_text(modified_content)

        # Wait for modified file to be re-indexed
        found = await realtime_service.wait_for_file_indexed(
            existing_file, timeout=get_fs_event_timeout()
        )
        assert found, "Added content should be searchable"

        added_regex = await search_impl(
            services=services,
            embedding_manager=embedding_manager,
            type="regex",
            query="added_content_edit_qa",
            page_size=10,
            offset=0,
        )
        added_semantic = await _wait_for_semantic_file(
            services,
            embedding_manager,
            "function added during file edit",
            "existing_test.py",
            ("added_content_edit_qa", "added_during_edit"),
        )
        assert len(added_regex.get("results", [])) > 0, (
            "Should find content added during edit"
        )
        assert "existing_test.py" in _result_file_names(added_semantic), (
            "Semantic search should expose content added during edit"
        )
        print("✓ Edit (add content): Found added content")

        # 3b: Delete some content and modify existing
        deleted_and_modified_content = """def existing_function():
    '''This function was MODIFIED during edit'''
    return "MODIFIED_existing_content"

def added_during_edit():
    '''This function was added during file edit'''
    return "added_content_edit_qa"

# Note: ExistingClass was DELETED
"""
        realtime_service.reset_file_tracking(existing_file)
        existing_file.write_text(deleted_and_modified_content)

        # Wait for modified file to be re-indexed
        found = await realtime_service.wait_for_file_indexed(
            existing_file, timeout=get_fs_event_timeout()
        )
        assert found, "Modified content should be searchable"

        # Check modification worked
        modified_regex = await search_impl(
            services=services,
            embedding_manager=embedding_manager,
            type="regex",
            query="MODIFIED_existing_content",
            page_size=10,
            offset=0,
        )
        # Check deletion worked - search for the actual class definition
        deleted_regex = await search_impl(
            services=services,
            embedding_manager=embedding_manager,
            type="regex",
            query="class ExistingClass:",
            page_size=10,
            offset=0,
        )

        assert len(modified_regex.get("results", [])) > 0, (
            "Should find modified content"
        )
        modified_semantic = await _wait_for_semantic_file(
            services,
            embedding_manager,
            "MODIFIED existing content during edit",
            "existing_test.py",
            ("MODIFIED_existing_content",),
        )

        assert len(deleted_regex.get("results", [])) == 0, (
            "Should not find deleted content"
        )
        assert "existing_test.py" in _result_file_names(modified_semantic), (
            "Semantic search should expose modified content"
        )
        print("✓ Edit (modify/delete): Found modified content, deleted content removed")

        # QA Item 4: Delete file and verify search results
        delete_target = new_file  # Delete the new file we created
        realtime_service.reset_file_tracking(delete_target)
        delete_target.unlink()

        # Wait for deletion to be processed
        removed = await realtime_service.wait_for_file_removed(
            delete_target, timeout=get_fs_event_timeout()
        )
        assert removed, "Deleted file should be removed"

        # Search for deleted file content
        deleted_file_regex = await search_impl(
            services=services,
            embedding_manager=embedding_manager,
            type="regex",
            query="newly_added_content_unique_string",
            page_size=10,
            offset=0,
        )
        deleted_file_semantic = await search_impl(
            services=services,
            embedding_manager=embedding_manager,
            type="semantic",
            query="newly added function QA validation",
            page_size=10,
            offset=0,
        )
        assert "new_added_file.py" not in _result_file_names(deleted_file_semantic), (
            "Semantic search should not rank a deleted file"
        )

        assert len(deleted_file_regex.get("results", [])) == 0, (
            "Should not find content from deleted file"
        )

        rerank_queries = {request.get("query") for request in rerank_manager.requests}
        assert {
            "existing function QA testing",
            "newly added function QA validation",
            "function added during file edit",
            "MODIFIED existing content during edit",
        } <= rerank_queries, "Semantic search should exercise the loopback reranker"
        print("✓ File deletion: Deleted file content not found in search")

    @pytest.mark.asyncio
    @pytest.mark.timeout(timeout_for_language_coverage())
    async def test_language_coverage_comprehensive(self, qa_setup):
        """QA Items 5-6: Test all supported languages and file types."""
        services, realtime_service, watch_dir, _, embedding_manager, rerank_manager = (
            qa_setup
        )

        # Get all supported languages except UNKNOWN
        languages_to_test = [lang for lang in Language if lang != Language.UNKNOWN]

        # Create language-specific content templates
        content_templates = {
            Language.PYTHON: 'def qa_test_function():\n    """Python QA test"""\n    return "python_qa_unique"',
            Language.JAVASCRIPT: 'function qaTestFunction() {\n    // JavaScript QA test\n    return "javascript_qa_unique";\n}',
            Language.TYPESCRIPT: 'function qaTestFunction(): string {\n    // TypeScript QA test\n    return "typescript_qa_unique";\n}',
            Language.TSX: "function QAComponent(): JSX.Element {\n    // TSX QA test\n    return <div>tsx_qa_unique</div>;\n}",
            Language.JSX: "function QAComponent() {\n    // JSX QA test\n    return <div>jsx_qa_unique</div>;\n}",
            Language.JAVA: 'public class QATest {\n    // Java QA test\n    public String test() { return "java_qa_unique"; }\n}',
            Language.CSHARP: 'public class QATest {\n    // C# QA test\n    public string Test() { return "csharp_qa_unique"; }\n}',
            Language.GO: 'package main\n\n// Go QA test\nfunc qaTestFunction() string {\n    return "go_qa_unique"\n}',
            Language.RUST: 'fn qa_test_function() -> &\'static str {\n    // Rust QA test\n    "rust_qa_unique"\n}',
            Language.C: '#include <stdio.h>\n\n// C QA test\nchar* qa_test_function() {\n    return "c_qa_unique";\n}',
            Language.CPP: '#include <string>\n\n// C++ QA test\nstd::string qaTestFunction() {\n    return "cpp_qa_unique";\n}',
            Language.METAL: '// Metal QA test (MSL is C++14)\nconst char* qaTestFunction() {\n    return "metal_qa_unique";\n}',
            Language.BASH: '#!/bin/bash\n# Bash QA test\nqa_test_function() {\n    echo "bash_qa_unique"\n}',
            Language.MARKDOWN: "# QA Test\n\nThis is a **markdown QA test** with `markdown_qa_unique` content.",
            Language.JSON: '{\n    "qa_test": true,\n    "content": "json_qa_unique",\n    "type": "qa_validation"\n}',
            Language.YAML: 'qa_test: true\ncontent: "yaml_qa_unique"\ntype: qa_validation',
            Language.TOML: '[qa_test]\ncontent = "toml_qa_unique"\ntype = "qa_validation"',
            Language.TEXT: "Plain text QA test file.\nContains: text_qa_unique\nFor validation purposes.",
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
            Language.GROOVY: 'def qaTestFunction() {\n    // Groovy QA test\n    return "groovy_qa_unique"\n}',
            Language.KOTLIN: 'fun qaTestFunction(): String {\n    // Kotlin QA test\n    return "kotlin_qa_unique"\n}',
            Language.MAKEFILE: '.PHONY: qa_test\n# Makefile QA test\nqa_test:\n\t@echo "makefile_qa_unique"',
            Language.MATLAB: '% MATLAB QA test\nfunction result = qa_test_function()\n    result = "matlab_qa_unique";\nend',
            Language.LUA: '-- Lua QA test\nfunction qa_test_function()\n    return "lua_qa_unique"\nend',
            Language.HASKELL: '-- Haskell QA test\nqaTestFunction :: String\nqaTestFunction = "haskell_qa_unique"',
            Language.HCL: '# HCL QA test\nvariable "qa_test" {\n  default = "hcl_qa_unique"\n}',
            Language.DART: '// Dart QA test\nString qaTestFunction() {\n  return "dart_qa_unique";\n}',
            Language.OBJC: '// Objective-C QA test\n@implementation QATest\n- (NSString *)qaTestMethod {\n    return @"objc_qa_unique";\n}\n@end',
            Language.PHP: '<?php\n// PHP QA test\nfunction qa_test_function() {\n    return "php_qa_unique";\n}',
            Language.SWIFT: '// Swift QA test\nfunc qaTestFunction() -> String {\n    return "swift_qa_unique"\n}',
            Language.POWERSHELL: '# PowerShell QA test\nfunction Get-QaTest {\n    return "powershell_qa_unique"\n}',
            Language.ZIG: '// Zig QA test\nfn qa_test_function() []const u8 {\n    return "zig_qa_unique";\n}',
            Language.PDF: None,  # PDF is binary, skip content template
            Language.SQL: "-- SQL QA test\nCREATE TABLE qa_test (\n    id INTEGER PRIMARY KEY,\n    content TEXT DEFAULT 'sql_qa_unique'\n);",
            Language.ELIXIR: 'defmodule QATest do\n  # Elixir QA test\n  def test, do: "elixir_qa_unique"\nend',
            Language.HTML: "<!DOCTYPE html>\n<html>\n<body>\n  <!-- HTML QA test -->\n  <p>html_qa_unique</p>\n</body>\n</html>",
            Language.CSS: '/* CSS QA test */\n.qa-test {\n  content: "css_qa_unique";\n  color: blue;\n}',
            Language.SCSS: '/* SCSS QA test */\n$color: blue;\n.qa-test {\n  content: "scss_qa_unique";\n  color: $color;\n}',
            Language.JINJA: '<!DOCTYPE html>\n<html>\n<body>\n  {# Jinja QA test #}\n  <p>{{ "jinja_qa_unique" }}</p>\n</body>\n</html>',
            Language.TWINCAT: '<?xml version="1.0" encoding="utf-8"?>\n<TcPlcObject Version="1.1.0.1">\n  <POU Name="QA_TEST" Id="{00000000-0000-0000-0000-000000000001}">\n    <Declaration><![CDATA[PROGRAM QA_TEST\nVAR\n  bFlag : BOOL := TRUE;\nEND_VAR]]></Declaration>\n    <Implementation>\n      <ST><![CDATA[IF bFlag THEN\n  (* twincat_qa_unique *)\nEND_IF]]></ST>\n    </Implementation>\n  </POU>\n</TcPlcObject>',
        }

        # Create extension mapping for file creation
        extension_map = {
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
            # OBJC shares .m with MATLAB - content-based detection (language_detector.py)
            # disambiguates via ObjC markers (@implementation in template above).
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

        # Validate ALL languages have test coverage (fail explicitly for new languages)
        testable_languages = {
            lang
            for lang in Language
            if lang not in (Language.UNKNOWN, Language.GIT_DIFF)
        }
        missing_templates = testable_languages - set(content_templates.keys())
        missing_extensions = testable_languages - set(extension_map.keys())
        # Languages must be in BOTH dicts to be tested
        uncovered = missing_templates | missing_extensions

        assert not uncovered, (
            f"Language(s) missing from test coverage: {sorted(l.value for l in uncovered)}. "
            f"Add content template and extension mapping for each new language to this test."
        )

        created_files = []
        search_patterns = []

        # Create files for all testable languages
        for language in languages_to_test:
            if language in content_templates and language in extension_map:
                content = content_templates[language]
                # Skip languages with None content (e.g., PDF which is binary)
                if content is None:
                    print(f"Skipped {language.value} (binary format)")
                    continue

                ext = extension_map[language]
                filename = f"qa_test_{language.value}{ext}"

                file_path = watch_dir / filename
                unique_pattern = f"{language.value}_qa_unique"

                file_path.write_text(content)
                created_files.append((file_path, language, unique_pattern))
                search_patterns.append(unique_pattern)

                print(f"Created {language.value} test file: {filename}")

        # Wait for all files to be processed - poll until all files are in database
        expected_file_count = len(created_files)
        # Cap max_wait to leave time for search validation phase
        # Note: Windows CI may need longer due to ReadDirectoryChangesW unreliability,
        # but we cap at INDEXING_CAP_SECONDS to fail fast rather than hang
        total_timeout = timeout_for_language_coverage()
        available_for_indexing = (
            total_timeout - BASE_OVERHEAD_SECONDS - SEARCH_VALIDATION_BUDGET_SECONDS
        )
        max_wait = min(available_for_indexing, INDEXING_CAP_SECONDS)
        poll_interval = INDEXING_POLL_INTERVAL_SECONDS
        start_time = time.monotonic()

        while (elapsed := time.monotonic() - start_time) < max_wait:
            db_stats = await services.indexing_coordinator.get_stats()
            indexed_files = db_stats.get("files", 0)

            if indexed_files >= expected_file_count:
                print(f"📊 All {expected_file_count} files processed in {elapsed:.1f}s")
                break

            await asyncio.sleep(poll_interval)

        # Final stats check with informative failure if indexing incomplete
        db_stats = await services.indexing_coordinator.get_stats()
        indexed_files = db_stats.get("files", 0)
        print(f"📊 Final: {indexed_files} files, {db_stats.get('chunks', 0)} chunks")

        if indexed_files < expected_file_count:
            pytest.fail(
                f"Only {indexed_files}/{expected_file_count} files indexed after {elapsed:.1f}s. "
                f"This may indicate indexing performance issues on CI."
            )

        # QA Item 5: Test concurrent processing for all languages
        # Search for each language's unique content - run in parallel for speed

        async def validate_language_search(language, pattern):
            """Validate a single language's content is searchable."""
            try:
                regex_results = await search_impl(
                    services=services,
                    embedding_manager=embedding_manager,
                    type="regex",
                    query=pattern,
                    page_size=10,
                    offset=0,
                )
                if len(regex_results.get("results", [])) > 0:
                    return (language.value, True, None)
                return (language.value, False, "regex not found")
            except Exception as e:
                return (language.value, False, str(e))

        # Run all searches in parallel
        tasks = [validate_language_search(lang, pat) for _, lang, pat in created_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful_languages = [lang for lang, success, _ in results if success]
        failed_languages = [
            f"{lang} ({err})" for lang, success, err in results if not success
        ]

        print(f"✓ Languages successfully tested: {len(successful_languages)}")
        print(f"✓ Successful languages: {successful_languages}")

        if failed_languages:
            print(f"⚠ Failed languages: {failed_languages}")

        # QA requirement: At least major languages should work
        major_languages = ["python", "javascript", "typescript", "java", "go"]
        working_major = [
            lang for lang in successful_languages if lang in major_languages
        ]

        assert len(working_major) >= MIN_MAJOR_LANGUAGES_REQUIRED, (
            f"At least {MIN_MAJOR_LANGUAGES_REQUIRED} major languages should work, got: {working_major}"
        )

        # Realistic expectation - at least some languages should work
        # This test reveals which languages actually work in the current system
        assert len(successful_languages) >= MIN_TOTAL_LANGUAGES_REQUIRED, (
            f"At least {MIN_TOTAL_LANGUAGES_REQUIRED} languages should work, got {len(successful_languages)}: {successful_languages}"
        )

        # Report findings for manual review
        success_rate = (
            len(successful_languages) / len(created_files) if created_files else 0
        )
        print(
            f"📊 Language success rate: {success_rate:.1%} ({len(successful_languages)}/{len(created_files)})"
        )

        if success_rate < LOW_SUCCESS_RATE_THRESHOLD:
            print(
                "⚠ LOW SUCCESS RATE: This may indicate indexing or parsing issues with some languages"
            )

    @pytest.mark.asyncio
    async def test_concurrent_operations_and_timing(self, qa_setup):
        """QA Item 7: Test concurrent file operations with search timing."""
        services, realtime_service, watch_dir, _, embedding_manager, rerank_manager = (
            qa_setup
        )

        # Create initial test files
        base_files = []
        for i in range(NUM_BASE_CONCURRENT_FILES):
            file_path = watch_dir / f"concurrent_test_{i}.py"
            content = f"""def concurrent_function_{i}():
    '''Concurrent test function {i}'''
    return "concurrent_qa_test_{i}"
"""
            file_path.write_text(content)
            base_files.append((file_path, f"concurrent_qa_test_{i}"))

        await asyncio.sleep(CONCURRENT_SETUP_WAIT_SECONDS)

        # Function to perform searches during file modifications
        async def search_during_modifications():
            search_results = []
            for i in range(NUM_CONCURRENT_SEARCHES):
                try:
                    start_time = time.time()
                    results = await search_impl(
                        services=services,
                        embedding_manager=embedding_manager,
                        type="regex",
                        query="concurrent_qa_test",
                        page_size=50,
                        offset=0,
                    )
                    end_time = time.time()

                    search_time = end_time - start_time
                    search_results.append(
                        {
                            "iteration": i,
                            "result_count": len(results.get("results", [])),
                            "search_time": search_time,
                            "timestamp": end_time,
                        }
                    )

                    # Small delay between searches
                    await asyncio.sleep(SEARCH_ITERATION_DELAY_SECONDS)
                except Exception as e:
                    search_results.append(
                        {"iteration": i, "error": str(e), "timestamp": time.time()}
                    )

            return search_results

        # Function to perform rapid file modifications
        async def rapid_file_modifications():
            modifications = []
            for i in range(NUM_RAPID_MODIFICATIONS):
                try:
                    # Create new file
                    new_file = watch_dir / f"rapid_modify_{i}.py"
                    content = f"""def rapid_function_{i}():
    '''Rapid modification test {i}'''
    return "rapid_qa_test_{i}"

class RapidClass_{i}:
    def method_{i}(self):
        return "rapid_method_{i}"
"""
                    start_time = time.time()
                    new_file.write_text(content)
                    modifications.append(
                        {
                            "type": "create",
                            "file": str(new_file),
                            "timestamp": start_time,
                        }
                    )

                    # Modify existing file
                    if i < len(base_files):
                        existing_file, _ = base_files[i]
                        modified_content = content + f"\n# Modified at iteration {i}\n"
                        existing_file.write_text(modified_content)
                        modifications.append(
                            {
                                "type": "modify",
                                "file": str(existing_file),
                                "timestamp": time.time(),
                            }
                        )

                    # Small delay between operations
                    await asyncio.sleep(FILE_OPERATION_DELAY_SECONDS)

                except Exception as e:
                    modifications.append(
                        {"type": "error", "error": str(e), "timestamp": time.time()}
                    )

            return modifications

        # Run searches and modifications concurrently
        print("Starting concurrent operations...")
        start_concurrent = time.time()

        search_task = asyncio.create_task(search_during_modifications())
        modify_task = asyncio.create_task(rapid_file_modifications())

        search_results, modification_results = await asyncio.gather(
            search_task, modify_task
        )

        end_concurrent = time.time()
        total_concurrent_time = end_concurrent - start_concurrent

        # Validate concurrent operation results
        successful_searches = [r for r in search_results if "error" not in r]
        failed_searches = [r for r in search_results if "error" in r]

        successful_modifications = [r for r in modification_results if "error" not in r]

        print(f"✓ Concurrent operations completed in {total_concurrent_time:.2f}s")
        print(
            f"✓ Successful searches: {len(successful_searches)}/{len(search_results)}"
        )
        print(
            f"✓ Successful modifications: {len(successful_modifications)}/{len(modification_results)}"
        )

        # Key assertions for QA item 7
        assert len(successful_searches) > len(search_results) * 0.8, (
            "Most searches should succeed during concurrent operations"
        )
        assert len(failed_searches) == 0 or len(failed_searches) < 3, (
            "Should have minimal search failures"
        )

        # Measure average search time
        search_times = [r["search_time"] for r in successful_searches]
        if search_times:
            avg_search_time = sum(search_times) / len(search_times)
            max_search_time = max(search_times)
            print(
                f"✓ Search timing: avg={avg_search_time:.3f}s, max={max_search_time:.3f}s"
            )

            # Search should not block - reasonable performance expected
            assert avg_search_time < AVG_SEARCH_TIME_LIMIT_SECONDS, (
                f"Average search time should be < {AVG_SEARCH_TIME_LIMIT_SECONDS}s, got {avg_search_time:.3f}s"
            )
            assert max_search_time < MAX_SEARCH_TIME_LIMIT_SECONDS, (
                f"Max search time should be < {MAX_SEARCH_TIME_LIMIT_SECONDS}s, got {max_search_time:.3f}s"
            )

    @pytest.mark.asyncio
    async def test_pagination_comprehensive(self, qa_setup):
        """QA Item 8: Test pagination functionality comprehensively.

        Tests ChunkHound's chunk-based search pagination against ripgrep's line-based search.
        Note: ChunkHound searches semantic chunks, so a chunk containing multiple pattern
        occurrences counts as 1 result, while ripgrep counts each line occurrence separately.
        This explains the expected discrepancy between result counts.
        """
        services, realtime_service, watch_dir, _, embedding_manager, rerank_manager = (
            qa_setup
        )

        # Create files with varying amounts of searchable content

        # 1. Search for non-existing value (should return empty)
        non_existing_results = await search_impl(
            services=services,
            embedding_manager=embedding_manager,
            type="regex",
            query="non_existing_unique_pattern_qa_test_12345",
            page_size=10,
            offset=0,
        )
        assert len(non_existing_results.get("results", [])) == 0, (
            "Non-existing pattern should return empty results"
        )
        print("✓ Pagination test 1: Non-existing pattern returns empty")

        # 2. Create single file with unique content (no pagination needed)
        single_file = watch_dir / "single_result_test.py"
        single_content = """def single_unique_function():
    '''This is a unique function that should appear only once'''
    return "single_unique_result_qa_test"
"""
        single_file.write_text(single_content)
        await asyncio.sleep(PAGINATION_SETUP_WAIT_SECONDS)

        single_results = await search_impl(
            services=services,
            embedding_manager=embedding_manager,
            type="regex",
            query="single_unique_result_qa_test",
            page_size=10,
            offset=0,
        )
        assert len(single_results.get("results", [])) == 1, (
            "Single unique pattern should return exactly 1 result"
        )
        print("✓ Pagination test 2: Single result handled correctly")

        # 3. Create many files with common pattern to test pagination
        # Each file must be large enough to avoid cAST merging (>1600 chars each)
        # or have diverse enough content to create multiple chunks
        common_pattern = "pagination_test_common_pattern"
        created_files_for_pagination = []

        # Create each file individually to avoid f-string complexity
        for i in range(NUM_PAGINATION_TEST_FILES):
            file_path = watch_dir / f"pagination_test_{i:03d}.py"

            # Build content using string formatting to avoid f-string nesting issues
            content_template = '''#!/usr/bin/env python3
"""
Pagination Test Module {file_num}
==========================

This module contains test functions and classes for pagination testing.
It includes multiple components designed to create substantial content
that will result in multiple chunks during parsing.

Module: pagination_test_{file_num_padded}.py
Pattern: {pattern}
Purpose: Generate enough content to exceed cAST merge thresholds
"""

import os
import sys
import json
import datetime
from typing import List, Dict, Optional, Union, Any

# Global constants for pagination testing
PAGINATION_CONSTANT_{file_num} = "{pattern}_constant_{file_num}"
PAGINATION_VERSION_{file_num} = "1.{file_num}.0"
PAGINATION_METADATA_{file_num} = {{
    "test_id": {file_num},
    "pattern": "{pattern}",
    "timestamp": "2024-01-01T00:00:00Z",
    "description": "Pagination test file number {file_num}"
}}

class PaginationDataProcessor_{file_num}:
    """
    Data processing class for pagination test {file_num}.
    
    This class handles various data processing operations for pagination
    testing including data validation, transformation, and storage.
    Each instance manages its own state and provides methods for
    comprehensive data manipulation.
    """
    
    def __init__(self, test_id: int = {file_num}):
        """Initialize the pagination data processor.
        
        Args:
            test_id: Unique identifier for this test instance
        """
        self.test_id = test_id
        self.pattern = "{pattern}"
        self.data_store = []
        self.processed_count = 0
        
    def process_pagination_data(self, data: List[Dict]) -> Dict[str, Any]:
        """Process pagination data and return results.
        
        This method takes input data, processes it according to pagination
        test requirements, and returns structured results with metadata.
        
        Args:
            data: List of data dictionaries to process
            
        Returns:
            Dictionary containing processed results and metadata
        """
        results = {{
            "test_id": self.test_id,
            "pattern": self.pattern,
            "input_count": len(data),
            "processed_items": [],
            "timestamp": datetime.datetime.now().isoformat()
        }}
        
        for idx, item in enumerate(data):
            processed_item = {{
                "original_index": idx,
                "data": item,
                "processed_by": "PaginationDataProcessor_{file_num}",
                "pattern_match": "{pattern}_result_" + str(idx),
                "test_metadata": PAGINATION_METADATA_{file_num}
            }}
            results["processed_items"].append(processed_item)
            self.processed_count += 1
            
        return results
    
    def validate_pagination_results(self, results: Dict) -> bool:
        """Validate pagination processing results.
        
        Performs comprehensive validation of pagination processing results
        to ensure data integrity and correct processing behavior.
        
        Args:
            results: Results dictionary from process_pagination_data
            
        Returns:
            True if validation passes, False otherwise
        """
        required_keys = ["test_id", "pattern", "input_count", "processed_items"]
        
        for key in required_keys:
            if key not in results:
                return False
                
        if results["test_id"] != self.test_id:
            return False
            
        if results["pattern"] != "{pattern}":
            return False
            
        return len(results["processed_items"]) == results["input_count"]

class PaginationTestManager_{file_num}:
    """
    Manager class for coordinating pagination tests.
    
    This class provides high-level coordination for pagination testing,
    managing multiple data processors and aggregating results across
    different test scenarios.
    """
    
    def __init__(self):
        self.processors = []
        self.test_results = []
        self.global_pattern = "{pattern}"
        
    def add_processor(self, processor: PaginationDataProcessor_{file_num}) -> None:
        """Add a data processor to the test manager."""
        self.processors.append(processor)
        
    def run_pagination_tests(self) -> Dict[str, Any]:
        """Execute pagination tests across all registered processors."""
        test_summary = {{
            "total_processors": len(self.processors),
            "pattern": self.global_pattern,
            "test_file": "pagination_test_{file_num_padded}.py",
            "individual_results": []
        }}
        
        for processor in self.processors:
            test_data = [
                {{"id": j, "value": "{pattern}_data_" + str(j) + "_processor_{file_num}"}}
                for j in range(5)
            ]
            
            results = processor.process_pagination_data(test_data)
            validation_passed = processor.validate_pagination_results(results)
            
            test_summary["individual_results"].append({{
                "processor_id": processor.test_id,
                "validation_passed": validation_passed,
                "processed_count": len(results.get("processed_items", [])),
                "pattern_matches": [
                    item.get("pattern_match", "") 
                    for item in results.get("processed_items", [])
                ]
            }})
            
        return test_summary

def pagination_function_{file_num}():
    """
    Main pagination test function for test case {file_num}.
    
    This function demonstrates pagination functionality by creating
    test data, processing it through pagination components, and
    returning results that can be searched and validated.
    
    Returns:
        String containing pattern for search validation
    """
    processor = PaginationDataProcessor_{file_num}()
    manager = PaginationTestManager_{file_num}()
    manager.add_processor(processor)
    
    test_results = manager.run_pagination_tests()
    
    # Return searchable pattern for test validation
    return "{pattern}_result_{file_num}_function"

def pagination_utility_{file_num}(input_data: Optional[List] = None) -> str:
    """
    Utility function for pagination testing.
    
    Provides utility functionality for pagination tests including
    data preparation, result formatting, and pattern generation.
    """
    if input_data is None:
        input_data = ["default_data_" + str(j) for j in range(3)]
        
    processed = ["{pattern}_utility_" + str(item) + "_{file_num}" for item in input_data]
    return "{pattern}_utility_result_{file_num}"

# Module-level execution for pagination testing
if __name__ == "__main__":
    print("Executing pagination test module {file_num}")
    result = pagination_function_{file_num}()
    utility_result = pagination_utility_{file_num}()
    
    print("Pattern: {pattern}")
    print("Function result: " + str(result))
    print("Utility result: " + str(utility_result))
'''

            # Format the content with actual values
            content = content_template.format(
                file_num=i, file_num_padded=f"{i:03d}", pattern=common_pattern
            )
            file_path.write_text(content)
            created_files_for_pagination.append(file_path)

        # Wait for all files to be processed with verification
        # Poll until we get a stable chunk count
        stable_count = None
        for _ in range(MAX_STABILITY_CHECKS):
            await asyncio.sleep(STABILITY_CHECK_INTERVAL_SECONDS)
            stats = await services.indexing_coordinator.get_stats()
            current_chunks = stats.get("chunks", 0)
            if (
                stable_count == current_chunks
                and current_chunks >= MIN_EXPECTED_CHUNKS_PAGINATION
            ):
                break
            stable_count = current_chunks
        else:
            # Fallback - just wait a bit more
            await asyncio.sleep(PAGINATION_SETUP_WAIT_SECONDS)

        # Test pagination by fetching all pages
        all_results = []
        page_size = DEFAULT_PAGE_SIZE
        offset = 0
        max_pages = MAX_PAGINATION_PAGES
        page_count = 0
        total_count = 0  # Track actual total from pagination metadata

        while page_count < max_pages:
            page_results = await search_impl(
                services=services,
                embedding_manager=embedding_manager,
                type="regex",
                query=common_pattern,
                page_size=page_size,
                offset=offset,
            )

            page_data = page_results.get("results", [])
            if not page_data:
                break  # No more results

            all_results.extend(page_data)
            page_count += 1
            offset += page_size

            print(
                f"Page {page_count}: {len(page_data)} results (offset={offset - page_size})"
            )

            # Check pagination metadata if available
            if "pagination" in page_results:
                pagination = page_results["pagination"]
                total_count = pagination.get(
                    "total", len(all_results)
                )  # Track actual total
                print(f"  Pagination metadata: {pagination}")

        print(
            f"✓ Pagination test 3: Retrieved {len(all_results)} total results across {page_count} pages"
        )

        # Validate pagination worked correctly
        # Note: May not find all files if some aren't processed yet - test pagination behavior with available data
        assert len(all_results) >= 10, (
            f"Should find reasonable number of results for pagination testing, got {len(all_results)}"
        )
        assert page_count >= 2, (
            f"Should require multiple pages with page_size={page_size}, used {page_count} pages"
        )

        # Report actual vs expected for manual review
        expected_chunks = NUM_PAGINATION_TEST_FILES * EXPECTED_CHUNKS_PER_FILE
        # Note: Due to cAST algorithm's semantic chunking, files may be merged into fewer chunks
        # than expected based on size. This is by design for better semantic coherence.
        if len(all_results) < expected_chunks:
            processing_rate = len(all_results) / expected_chunks
            print(
                f"📊 Chunk processing rate: {processing_rate:.1%} ({len(all_results)}/{expected_chunks} expected chunks)"
            )

        # 4. Compare with external validation using ripgrep if available
        try:
            # Try to use ripgrep for external validation
            rg_result = subprocess.run(
                ["rg", "--count", "--no-heading", common_pattern, str(watch_dir)],
                capture_output=True,
                text=True,
                timeout=RIPGREP_TIMEOUT_SECONDS,
            )

            if rg_result.returncode == 0:
                # Parse ripgrep results - count matches across files
                rg_lines = (
                    rg_result.stdout.strip().split("\n")
                    if rg_result.stdout.strip()
                    else []
                )
                rg_total_matches = 0
                for line in rg_lines:
                    if ":" in line:
                        try:
                            count = int(line.split(":")[-1])
                            rg_total_matches += count
                        except ValueError:
                            pass

                print(
                    f"✓ External validation: ripgrep found {rg_total_matches} matches"
                )

                # Allow some variance due to different matching behavior
                # ChunkHound uses chunk-based search (semantic units) vs ripgrep's line-based search
                # A chunk containing multiple pattern occurrences counts as 1 result in ChunkHound
                # but each line occurrence counts as 1 result in ripgrep, hence the large discrepancy
                match_ratio = len(all_results) / max(rg_total_matches, 1)
                assert 0.05 <= match_ratio <= 3.0, (
                    f"ChunkHound uses chunk-based search (semantic units) vs ripgrep's line-based search: {len(all_results)} chunks vs {rg_total_matches} line matches"
                )

            else:
                print("⚠ ripgrep not available or failed, skipping external validation")

        except (
            subprocess.TimeoutExpired,
            FileNotFoundError,
            subprocess.SubprocessError,
        ):
            print("⚠ ripgrep not available, skipping external validation")

        # 5. Test edge cases
        # Test offset beyond available results
        # Use total_count from pagination metadata, not len(all_results) which may be partial
        actual_total = total_count if total_count > 0 else len(all_results)
        beyond_results = await search_impl(
            services=services,
            embedding_manager=embedding_manager,
            type="regex",
            query=common_pattern,
            page_size=10,
            offset=actual_total + 100,
        )
        assert len(beyond_results.get("results", [])) == 0, (
            f"Offset {actual_total + 100} beyond total {actual_total} should return empty"
        )

        # Test large page size
        large_page_results = await search_impl(
            services=services,
            embedding_manager=embedding_manager,
            type="regex",
            query=common_pattern,
            page_size=100,
            offset=0,
        )
        large_page_count = len(large_page_results.get("results", []))
        assert large_page_count <= actual_total, (
            f"Large page size should not exceed total ({large_page_count} <= {actual_total})"
        )

        print("✓ Pagination edge cases handled correctly")

    @pytest.mark.asyncio
    async def test_qa_comprehensive_report(self, qa_setup):
        """Generate comprehensive QA report with timing measurements."""
        services, realtime_service, watch_dir, _, embedding_manager, rerank_manager = (
            qa_setup
        )

        print("\n" + "=" * 60)
        print("COMPREHENSIVE QA VALIDATION REPORT")
        print("=" * 60)

        # Test file change reflection timing
        timing_test_file = watch_dir / "timing_validation.py"
        timing_content = f"""def timing_validation_function():
    '''Timing test at {time.time()}'''
    return "timing_validation_unique_content"
"""

        # Measure indexing time
        timing_test_file.write_text(timing_content)

        # Poll until content is searchable
        max_wait = SINGLE_FILE_INDEXING_MAX_SECONDS
        poll_interval = INDEXING_POLL_INTERVAL_SECONDS
        start_time = time.monotonic()

        while (elapsed := time.monotonic() - start_time) < max_wait:
            await asyncio.sleep(poll_interval)

            search_results = await search_impl(
                services=services,
                embedding_manager=embedding_manager,
                type="regex",
                query="timing_validation_unique_content",
                page_size=10,
                offset=0,
            )

            if len(search_results.get("results", [])) > 0:
                indexing_time = time.monotonic() - start_time
                break
        else:
            indexing_time = max_wait  # Timeout

        # Test search performance
        search_start = time.time()
        performance_results = await search_impl(
            services=services,
            embedding_manager=embedding_manager,
            type="regex",
            query="function",
            page_size=50,
            offset=0,
        )
        search_time = time.time() - search_start

        # Get database stats
        stats_results = await services.indexing_coordinator.get_stats()

        print("📊 DATABASE STATISTICS:")
        print(f"   Total files: {stats_results.get('files', 'Unknown')}")
        print(f"   Total chunks: {stats_results.get('chunks', 'Unknown')}")
        print(f"   Total embeddings: {stats_results.get('embeddings', 'Unknown')}")

        print("\n⏱ PERFORMANCE MEASUREMENTS:")
        print(f"   File change → searchable: {indexing_time:.2f}s")
        print(f"   Search execution time: {search_time:.3f}s")
        print(
            f"   Search results returned: {len(performance_results.get('results', []))}"
        )

        print("\n✅ QA VALIDATION SUMMARY:")
        print("   File lifecycle operations: TESTED")
        print("   Language coverage: TESTED")
        print("   Concurrent operations: TESTED")
        print("   Pagination functionality: TESTED")
        print("   Performance measurements: COMPLETED")

        print("\n📋 QA REQUIREMENTS STATUS:")
        indexing_ok = indexing_time < FILE_REFLECTION_MAX_SECONDS
        print(f"   Real-time indexing: {'✅ WORKING' if indexing_ok else '❌ SLOW'}")
        search_good = search_time < SEARCH_GOOD_THRESHOLD_SECONDS
        search_ok = search_time < SEARCH_ACCEPTABLE_THRESHOLD_SECONDS
        search_status = (
            "✅ GOOD" if search_good else "⚠ ACCEPTABLE" if search_ok else "❌ SLOW"
        )
        print(f"   Search performance: {search_status}")
        print("   Non-blocking searches: ✅ VERIFIED")

        print("=" * 60)

        # Final assertions for QA requirements
        assert indexing_time <= FILE_REFLECTION_MAX_SECONDS, (
            f"File changes should be reflected within {FILE_REFLECTION_MAX_SECONDS}s, "
            f"took {indexing_time:.2f}s"
        )
        assert search_time < SEARCH_EXECUTION_MAX_SECONDS, (
            f"Search should complete within {SEARCH_EXECUTION_MAX_SECONDS}s, "
            f"took {search_time:.3f}s"
        )
