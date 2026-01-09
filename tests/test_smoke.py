#!/usr/bin/env python3
"""
Smoke tests to catch basic import and startup failures.

These tests are designed to catch crashes that occur during:
1. Module import time (like type annotation syntax errors)
2. CLI command initialization
3. Basic server startup

They run quickly and should be part of every test run.
"""

import subprocess
import importlib
import pkgutil
import sys
import os
import asyncio
import pytest
from pathlib import Path

# Import Windows-safe subprocess utilities
from tests.utils.windows_subprocess import (
    create_subprocess_exec_safe,
    get_safe_subprocess_env,
    terminate_async_process_tree,
)
from tests.utils.windows_compat import windows_safe_tempdir
from tests.utils import SubprocessJsonRpcClient

# Add parent directory to path to import chunkhound
sys.path.insert(0, str(Path(__file__).parent.parent))
import chunkhound


class TestModuleImports:
    """Test that all modules can be imported without errors."""

    def test_all_modules_import(self):
        """Test that all chunkhound modules can be imported."""
        failed_imports = []

        # Walk through all chunkhound modules
        for _, module_name, _ in pkgutil.walk_packages(
            chunkhound.__path__, prefix="chunkhound."
        ):
            try:
                importlib.import_module(module_name)
            except Exception as e:
                failed_imports.append((module_name, str(e)))

        if failed_imports:
            error_msg = "Failed to import modules:\n"
            for module, error in failed_imports:
                error_msg += f"  - {module}: {error}\n"
            pytest.fail(error_msg)

    def test_critical_imports(self):
        """Test critical modules that have caused issues before."""
        critical_modules = [
            "chunkhound.mcp_server.stdio",
            "chunkhound.api.cli.main",
            "chunkhound.database",
            "chunkhound.embeddings",
        ]

        for module_name in critical_modules:
            try:
                importlib.import_module(module_name)
            except Exception as e:
                pytest.fail(f"Failed to import {module_name}: {e}")


class TestCLICommands:
    """Test that CLI commands at least show help without crashing."""

    @pytest.mark.parametrize(
        "command",
        [
            ["chunkhound", "--help"],
            ["chunkhound", "--version"],
            ["chunkhound", "index", "--help"],
            ["chunkhound", "search", "--help"],
            ["chunkhound", "research", "--help"],
            ["chunkhound", "map", "--help"],
            ["chunkhound", "mcp", "--help"],
            ["chunkhound", "calibrate", "--help"],
        ],
    )
    def test_cli_help_commands(self, command):
        """Test that CLI help commands work without crashing."""
        result = subprocess.run(
            ["uv", "run"] + command, capture_output=True, text=True, timeout=5
        )

        # Help commands should exit with 0
        assert result.returncode == 0, (
            f"Command {' '.join(command)} failed with code {result.returncode}\n"
            f"stderr: {result.stderr}"
        )

        # Should have some output
        assert result.stdout or result.stderr, (
            f"Command {' '.join(command)} produced no output"
        )

class TestServerStartup:
    """Test that servers can at least start without immediate crashes."""





class TestParserLoading:
    """Test that all parsers can be loaded and created."""

    def test_all_parsers_load(self):
        """Test that all supported language parsers can be created and initialized."""
        from chunkhound.core.types.common import FileId, Language
        from chunkhound.parsers.parser_factory import get_parser_factory
        from chunkhound.parsers.universal_engine import SetupError

        # Minimal valid code samples for smoke testing
        language_samples = {
            Language.PYTHON: "def hello(): pass",
            Language.JAVA: "class Test { }",
            Language.CSHARP: "class Test { }",
            Language.TYPESCRIPT: "const x = 1;",
            Language.JAVASCRIPT: "const x = 1;",
            Language.TSX: "const x = <div>hello</div>;",
            Language.JSX: "const x = <div>hello</div>;",
            Language.GROOVY: "def hello() { }",
            Language.KOTLIN: "fun hello() { }",
            Language.GO: "package main\nfunc main() { }",
            Language.RUST: "fn main() { }",
            Language.BASH: "echo hello",
            Language.MAKEFILE: "all:\n\techo hello",
            Language.C: "int main() { return 0; }",
            Language.CPP: "int main() { return 0; }",
            Language.MATLAB: "function result = hello()\nresult = 1;\nend",
            Language.MARKDOWN: "# Hello\nWorld",
            Language.JSON: '{"hello": "world"}',
            Language.YAML: "hello: world",
            Language.TOML: "hello = 'world'",
            Language.TEXT: "hello world",
            Language.PDF: "hello world",
            Language.SWIFT: "struct Point { let x: Int; let y: Int }",
        }

        factory = get_parser_factory()
        failed_parsers = []
        setup_errors = []

        # Test all languages except UNKNOWN (not a real parser)
        for language in Language:
            if language == Language.UNKNOWN:
                continue

            try:
                parser = factory.create_parser(language)
                assert parser is not None, f"Parser for {language.value} was None"

                # Actually test parsing to trigger tree-sitter Language initialization
                sample_code = language_samples.get(language, "")
                if sample_code:
                    chunks = parser.parse_content(sample_code, f"test.{language.value}", FileId(1))
                    assert isinstance(chunks, list), f"Parser for {language.value} didn't return a list"

            except SetupError as e:
                # SetupError indicates critical issues like version incompatibility
                setup_errors.append((language.value, str(e)))
            except Exception as e:
                failed_parsers.append((language.value, str(e)))

        # SetupErrors should cause immediate test failure (critical issues)
        if setup_errors:
            error_msg = "CRITICAL: Parser setup failures (version incompatibility or missing dependencies):\n"
            for language, error in setup_errors:
                error_msg += f"  - {language}: {error}\n"
            pytest.fail(error_msg)

        # Other failures are also important but less critical
        if failed_parsers:
            error_msg = "Failed to initialize parsers:\n"
            for language, error in failed_parsers:
                error_msg += f"  - {language}: {error}\n"
            pytest.fail(error_msg)


class TestTypeAnnotations:
    """Test for specific type annotation patterns that have caused issues."""

    def test_no_invalid_forward_reference_unions(self):
        """Check for problematic forward reference union patterns."""
        import ast
        import glob

        problematic_files = []

        # Find all Python files in chunkhound, excluding test files themselves
        for py_file in glob.glob("chunkhound/**/*.py", recursive=True):
            if "/tests/" in py_file.replace("\\", "/"):
                continue
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for the problematic pattern: "ClassName" | None
            # This is a simple regex check, not a full AST analysis
            import re

            pattern = r':\s*"[^"]+"\s*\|\s*None'

            if re.search(pattern, content):
                # Found potential issue, let's verify it's not in a string
                try:
                    tree = ast.parse(content)
                    # This is where we'd do more sophisticated checking
                    # For now, just flag the file
                    problematic_files.append(py_file)
                except SyntaxError:
                    # If it's a syntax error, our other tests will catch it
                    pass

        if problematic_files:
            pytest.fail(
                f"Found problematic forward reference union patterns in:\n"
                + "\n".join(f"  - {f}" for f in problematic_files)
            )


class TestConfigurationSmoke:
    """Test that new configuration parameters don't break imports or config."""

    def test_rerank_format_configuration(self):
        """Verify rerank_format parameter doesn't break imports or config.

        This test ensures the new TEI reranking format configuration can be
        instantiated without errors, catching import-time or validation issues.
        """
        from chunkhound.core.config.embedding_config import EmbeddingConfig

        # Should not raise during import or instantiation with TEI format
        config_tei = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            base_url="http://localhost:8001",
            rerank_format="tei",
        )
        assert config_tei.rerank_format == "tei"

        # Should not raise with Cohere format
        config_cohere = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            base_url="http://localhost:8001",
            rerank_model="rerank-model",
            rerank_format="cohere",
        )
        assert config_cohere.rerank_format == "cohere"

        # Should not raise with auto format (default)
        config_auto = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            base_url="http://localhost:8001",
        )
        assert config_auto.rerank_format == "auto"


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v"])
