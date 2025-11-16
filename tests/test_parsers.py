"""
Parser validation tests.

Tests that all parsers can parse minimal valid code samples.
"""

import pytest
from chunkhound.core.types.common import FileId, Language
from chunkhound.parsers.parser_factory import get_parser_factory
from chunkhound.parsers.universal_engine import SetupError

# Minimal valid code snippets for each language
LANGUAGE_SAMPLES = {
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
    Language.HASKELL: "add x y = x + y",
    Language.RUST: "fn main() { }",
    Language.BASH: "echo hello",
    Language.MAKEFILE: "all:\n\techo hello",
    Language.C: "int main() { return 0; }",
    Language.CPP: "int main() { return 0; }",
    Language.OBJC: "@interface MyClass : NSObject\n@end",
    Language.MATLAB: "function result = hello()\nresult = 1;\nend",
    Language.MARKDOWN: "# Hello\nWorld",
    Language.JSON: '{"hello": "world"}',
    Language.YAML: "hello: world",
    Language.TOML: "hello = 'world'",
    Language.HCL: "resource \"aws_s3_bucket\" \"b\" {\n  bucket = \"my-bucket\"\n}\n",
    Language.TEXT: "hello world",
    Language.PDF: "hello world",  # PDF parser handles text content
    Language.ZIG: "fn main() void { }",
    Language.VUE: '<template><div>{{ message }}</div></template>\n<script setup lang="ts">\nconst message = "hello"\n</script>',
    Language.PHP: '<?php\nfunction hello() {\n  return "world";\n}\n?>',
    Language.SWIFT: "class MyClass {\n    func hello() -> String {\n        return \"world\"\n    }\n}",
    Language.DART: "void main() { }",
}


def create_large_array_content(language: Language, item_count: int = 150) -> str:
    """Create realistic large array content for testing line calculations.

    This function generates content with large arrays/lists to test that
    parsers correctly handle line number calculations when splitting chunks.
    """
    if language == Language.TOML:
        # Create a large dependencies array similar to the actual pyproject.toml that broke
        # Include realistic package names, version constraints, and comments
        dependencies = []
        for i in range(item_count):
            pkg_type = i % 6
            if pkg_type == 0:
                dependencies.append(f'"duckdb>={i % 3}.{i % 10}.0"')
            elif pkg_type == 1:
                dependencies.append(f'"tree-sitter-{["python", "rust", "go", "java"][i % 4]}>={i % 2}.{i % 20}.0,<{(i % 2)+1}.{(i % 20)+5}.0"')
            elif pkg_type == 2:
                dependencies.append(f'"fastapi>={i % 3}.{i % 10}.0"')
            elif pkg_type == 3:
                dependencies.append(f'"pytest>={i % 2}.{i % 15}.0"')
            elif pkg_type == 4:
                # Add a comment line occasionally to match real structure
                if i % 10 == 0:
                    dependencies.append(f'# "commented-package>=1.0.0",  # Using alternative instead')
                dependencies.append(f'"package-{i}>=1.{i % 10}.0"')
            else:
                dependencies.append(f'"pydantic>={i % 3}.{i % 8}.0"')

        deps_text = ',\n    '.join(dependencies)
        return f'''[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "test-project"
version = "1.0.0"
description = "Test project with large dependency array"
dependencies = [
    {deps_text}
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "black>=23.0.0",
]
'''

    elif language == Language.JSON:
        # Create a package.json-style file with large dependencies
        items = [f'"package-{i}": ">=1.{i % 10}.0"' for i in range(item_count)]
        items_text = ',\n        '.join(items)
        return f'''{{
    "name": "test-project",
    "dependencies": {{
        {items_text}
    }},
    "devDependencies": {{}}
}}'''

    elif language == Language.YAML:
        # Create a YAML file with large list
        items = [f'  - package-{i}>=1.{i % 10}.0' for i in range(item_count)]
        items_text = '\n'.join(items)
        return f'''name: test-project
dependencies:
{items_text}

build:
  requires: ["setuptools"]
'''

    elif language == Language.PYTHON:
        # Create a Python file with large list literal
        items = [f'"package-{i}"' for i in range(item_count)]
        items_text = ',\n    '.join(items)
        return f'''# Test module with large list
dependencies = [
    {items_text}
]

def setup():
    return dependencies
'''

    elif language == Language.JAVASCRIPT:
        # Create a JavaScript file with large array
        items = [f'"package-{i}"' for i in range(item_count)]
        items_text = ',\n    '.join(items)
        return f'''// Test module with large array
const dependencies = [
    {items_text}
];

module.exports = dependencies;
'''

    elif language == Language.TYPESCRIPT:
        # Create a TypeScript file with large array
        items = [f'"package-{i}"' for i in range(item_count)]
        items_text = ',\n    '.join(items)
        return f'''// Test module with large array
const dependencies: string[] = [
    {items_text}
];

export default dependencies;
'''

    elif language == Language.ZIG:
        # Create a Zig file with large array
        items = [f'"{i}"' for i in range(item_count)]
        items_text = ',\n    '.join(items)
        return f'''// Test module with large array
const dependencies = [_][]const u8{{
    {items_text}
}};

pub fn main() void {{
    const count = dependencies.len;
    _ = count;
}}
'''

    else:
        # Fallback - use the minimal sample for unsupported languages
        return LANGUAGE_SAMPLES.get(language, "")


class TestParserValidation:
    """Test that all parsers can parse minimal valid code."""

    @pytest.mark.parametrize("language", [lang for lang in Language if lang != Language.UNKNOWN])
    def test_parser_can_parse_minimal_code(self, language):
        """Test that each parser can parse a minimal valid code sample."""
        factory = get_parser_factory()
        
        # Create parser
        parser = factory.create_parser(language)
        assert parser is not None, f"Failed to create parser for {language.value}"
        
        # Get sample code
        sample_code = LANGUAGE_SAMPLES.get(language)
        assert sample_code is not None, f"No sample code defined for {language.value}"
        
        # Parse the sample
        try:
            chunks = parser.parse_content(sample_code, "test_file", FileId(1))
            assert isinstance(chunks, list), f"Parser for {language.value} didn't return a list"
            # Don't require chunks - some parsers might return empty for minimal code
        except SetupError as e:
            # SetupError indicates critical parser initialization failure (e.g., version incompatibility)
            # This should cause immediate test failure
            pytest.fail(f"CRITICAL: Parser setup failed for {language.value}: {e}")
        except Exception as e:
            pytest.fail(f"Parser for {language.value} failed to parse minimal code: {e}")

    @pytest.mark.parametrize("language", [lang for lang in Language if lang != Language.UNKNOWN])
    def test_parser_initializes_tree_sitter_language(self, language):
        """Test that parsers can initialize tree-sitter Language objects without version conflicts.

        This test specifically targets the tree-sitter language initialization where version
        compatibility is checked. This was the missing piece that allowed incompatible
        versions to pass CI tests.
        """
        factory = get_parser_factory()

        # Create parser - this should work even with version issues
        try:
            parser = factory.create_parser(language)
            assert parser is not None, f"Failed to create parser for {language.value}"
        except SetupError as e:
            # SetupError during parser creation indicates missing or incompatible dependencies
            pytest.fail(f"CRITICAL: Cannot create parser for {language.value}: {e}")

        # For text and PDF parsers, skip tree-sitter language initialization
        if language in (Language.TEXT, Language.PDF):
            return

        # Force tree-sitter Language object creation by accessing the engine's language
        # This is where version compatibility errors actually occur
        try:
            if hasattr(parser, 'engine') and parser.engine is not None:
                # Access the _language property which contains the Language object
                ts_language = parser.engine._language
                assert ts_language is not None, f"Tree-sitter language is None for {language.value}"
        except SetupError as e:
            # This is the critical error we want to catch - version incompatibility
            pytest.fail(f"CRITICAL: Tree-sitter version incompatibility for {language.value}: {e}")
        except Exception as e:
            # Check if this is a version incompatibility error
            if "Incompatible Language version" in str(e):
                pytest.fail(f"CRITICAL: Tree-sitter version incompatibility for {language.value}: {e}")
            else:
                pytest.fail(f"Unexpected error initializing tree-sitter language for {language.value}: {e}")

    @pytest.mark.parametrize("language,item_count", [
        (Language.TOML, 150),       # Original bug: large dependency arrays
        (Language.JSON, 150),       # Similar structure in package.json
        (Language.YAML, 150),       # Similar structure in YAML configs
        (Language.PYTHON, 100),     # Large list literals
        (Language.JAVASCRIPT, 100), # Large array literals
        (Language.TYPESCRIPT, 100), # Large array literals with types
        (Language.ZIG, 100),        # Large array literals in Zig
    ])
    def test_parser_handles_long_arrays(self, language, item_count):
        """Test that parsers correctly handle files with long arrays/lists.

        This test specifically validates the fix for the line calculation bug
        that occurred when parsing TOML files with large dependency arrays.
        The bug would cause invalid line ranges (start_line > end_line) when
        the universal parser split chunks containing large arrays.

        The test ensures:
        1. No chunks are skipped due to invalid line ranges
        2. All chunks have valid line numbers (start_line <= end_line)
        3. Line numbers stay within content bounds
        4. The fix works universally across all parser types
        """
        factory = get_parser_factory()

        # Create parser
        parser = factory.create_parser(language)
        assert parser is not None, f"Failed to create parser for {language.value}"

        # Generate large array content
        large_content = create_large_array_content(language, item_count)
        total_lines = large_content.count('\n') + 1

        print(f"\nTesting {language.value} with {item_count} array items ({total_lines} lines)")

        # Parse the content
        try:
            chunks = parser.parse_content(large_content, f"test_large_array.{language.value}", FileId(1))
            assert isinstance(chunks, list), f"Parser for {language.value} didn't return a list"

            # Critical validation: ensure no chunks have invalid line ranges
            invalid_chunks = []
            for chunk in chunks:
                if chunk.start_line > chunk.end_line:
                    invalid_chunks.append(f"{chunk.symbol}: {chunk.start_line} > {chunk.end_line}")
                elif chunk.start_line < 1:
                    invalid_chunks.append(f"{chunk.symbol}: start_line < 1 ({chunk.start_line})")
                elif chunk.end_line > total_lines:
                    invalid_chunks.append(f"{chunk.symbol}: end_line > total_lines ({chunk.end_line} > {total_lines})")

            # Fail the test if any chunks have invalid line ranges
            if invalid_chunks:
                error_msg = f"Found {len(invalid_chunks)} chunks with invalid line ranges in {language.value}:\n"
                error_msg += "\n".join(f"  - {error}" for error in invalid_chunks)
                pytest.fail(error_msg)

            print(f"✓ Successfully parsed {len(chunks)} chunks with valid line ranges")

        except Exception as e:
            pytest.fail(f"Parser for {language.value} failed to parse large array content: {e}")
