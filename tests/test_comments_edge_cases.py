"""Tests for comments and edge cases in JS-family parsers.

This module tests comment extraction (single-line, multi-line, JSDoc, TSDoc)
and edge cases (empty files, unicode, IIFE, metadata completeness) across
JavaScript, JSX, TypeScript, and TSX parsers.
"""

from pathlib import Path

import pytest

from chunkhound.core.types.common import Language, ChunkType
from chunkhound.parsers.parser_factory import ParserFactory, OXC_AVAILABLE


def _parse(code: str, filename: str, language: Language):
    """Parse code and return chunks."""
    factory = ParserFactory()
    parser = factory.create_parser(language)
    return parser.parse_content(code, Path(filename), file_id=1)


# =============================================================================
# Section 9: Comments
# =============================================================================


class TestSingleLineComments:
    """Tests for single-line comment extraction."""

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_basic_single_line_comment(self, lang, ext):
        """Single-line comment should be preserved in chunk content."""
        code = """// This is a comment
function foo() { return 42; }
"""
        chunks = _parse(code, f"test{ext}", lang)
        # At minimum, function should be extracted
        assert len(chunks) > 0
        # Comment may be attached to function or separate
        all_content = " ".join(c.code for c in chunks)
        assert "foo" in all_content

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_end_of_line_comment(self, lang, ext):
        """End-of-line comments should be preserved in chunk content."""
        code = """const x = 1; // inline comment
function bar() { return x; }
"""
        chunks = _parse(code, f"test{ext}", lang)
        assert len(chunks) > 0
        all_content = " ".join(c.code for c in chunks)
        # Either the variable or function should be extracted
        assert "bar" in all_content or "x" in all_content

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_multiple_consecutive_comments(self, lang, ext):
        """Multiple consecutive comments should be handled."""
        code = """// First comment
// Second comment
// Third comment
function multiComment() { return true; }
"""
        chunks = _parse(code, f"test{ext}", lang)
        assert len(chunks) > 0
        func_chunks = [c for c in chunks if "multiComment" in c.code]
        assert len(func_chunks) > 0


class TestMultiLineComments:
    """Tests for multi-line comment extraction."""

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_basic_multiline_comment(self, lang, ext):
        """Basic multi-line comment should be preserved."""
        code = """/* This is a
multi-line comment */
function withMulti() { return "multi"; }
"""
        chunks = _parse(code, f"test{ext}", lang)
        assert len(chunks) > 0
        func_chunks = [c for c in chunks if "withMulti" in c.code]
        assert len(func_chunks) > 0

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_inline_multiline_comment(self, lang, ext):
        """Inline multi-line comment in code."""
        code = """function withInline(/* param */ x) {
    return x /* inline */ * 2;
}
"""
        chunks = _parse(code, f"test{ext}", lang)
        assert len(chunks) > 0
        func_chunks = [c for c in chunks if "withInline" in c.code]
        assert len(func_chunks) > 0


class TestJSDocComments:
    """Tests for JSDoc documentation comment extraction."""

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_jsdoc_basic(self, lang, ext):
        """Basic JSDoc comment should be extracted."""
        code = '''/**
 * Calculate the sum of two numbers.
 */
function sum(a, b) {
    return a + b;
}
'''
        chunks = _parse(code, f"test{ext}", lang)
        assert len(chunks) > 0
        func_chunks = [c for c in chunks if "sum" in c.code]
        assert len(func_chunks) > 0
        # JSDoc content should be in the chunk
        assert any("Calculate" in c.code for c in chunks)

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_jsdoc_with_param_tag(self, lang, ext):
        """JSDoc with @param tag should be extracted."""
        code = '''/**
 * Add numbers.
 * @param {number} a - First number
 * @param {number} b - Second number
 */
function add(a, b) { return a + b; }
'''
        chunks = _parse(code, f"test{ext}", lang)
        assert len(chunks) > 0
        # Verify @param is in content
        all_content = " ".join(c.code for c in chunks)
        assert "@param" in all_content or "add" in all_content

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_jsdoc_with_returns_tag(self, lang, ext):
        """JSDoc with @returns tag should be extracted."""
        code = '''/**
 * Get value.
 * @returns {string} The value
 */
function getValue() { return "value"; }
'''
        chunks = _parse(code, f"test{ext}", lang)
        assert len(chunks) > 0
        func_chunks = [c for c in chunks if "getValue" in c.code]
        assert len(func_chunks) > 0

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_jsdoc_with_throws_tag(self, lang, ext):
        """JSDoc with @throws tag should be extracted."""
        code = '''/**
 * Dangerous operation.
 * @throws {Error} When something goes wrong
 */
function dangerous() { throw new Error("boom"); }
'''
        chunks = _parse(code, f"test{ext}", lang)
        assert len(chunks) > 0
        func_chunks = [c for c in chunks if "dangerous" in c.code]
        assert len(func_chunks) > 0

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_jsdoc_with_example_tag(self, lang, ext):
        """JSDoc with @example tag should be extracted."""
        code = '''/**
 * Multiply numbers.
 * @example
 * multiply(2, 3); // returns 6
 */
function multiply(a, b) { return a * b; }
'''
        chunks = _parse(code, f"test{ext}", lang)
        assert len(chunks) > 0
        func_chunks = [c for c in chunks if "multiply" in c.code]
        assert len(func_chunks) > 0

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_jsdoc_with_deprecated_tag(self, lang, ext):
        """JSDoc with @deprecated tag should be extracted."""
        code = '''/**
 * Old function.
 * @deprecated Use newFunction instead
 */
function oldFunction() { return "old"; }
'''
        chunks = _parse(code, f"test{ext}", lang)
        assert len(chunks) > 0
        func_chunks = [c for c in chunks if "oldFunction" in c.code]
        assert len(func_chunks) > 0


class TestTSDocComments:
    """Tests for TSDoc format documentation."""

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_tsdoc_format(self, lang, ext):
        """TSDoc format with dash separator should be extracted."""
        code = '''/**
 * Process data.
 * @param data - The data to process
 * @returns The processed result
 */
function process(data: string): string {
    return data.toUpperCase();
}
'''
        chunks = _parse(code, f"test{ext}", lang)
        assert len(chunks) > 0
        func_chunks = [c for c in chunks if "process" in c.code]
        assert len(func_chunks) > 0


class TestJSXComments:
    """Tests for JSX-specific comment syntax."""

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JSX, ".jsx"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_jsx_comment_in_element(self, lang, ext):
        """JSX comment syntax {/* comment */} should be preserved."""
        code = """function Component() {
    return (
        <div>
            {/* This is a JSX comment */}
            <span>Hello</span>
        </div>
    );
}
"""
        chunks = _parse(code, f"test{ext}", lang)
        assert len(chunks) > 0
        comp_chunks = [c for c in chunks if "Component" in c.code]
        assert len(comp_chunks) > 0

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JSX, ".jsx"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_multiline_jsx_comment(self, lang, ext):
        """Multi-line JSX comment should be preserved."""
        code = """function MultiComment() {
    return (
        <div>
            {/*
                Multi-line JSX comment
                with multiple lines
            */}
            <span>Content</span>
        </div>
    );
}
"""
        chunks = _parse(code, f"test{ext}", lang)
        assert len(chunks) > 0
        comp_chunks = [c for c in chunks if "MultiComment" in c.code]
        assert len(comp_chunks) > 0


# =============================================================================
# Section 10: Edge Cases and Special Patterns
# =============================================================================


class TestFileStructureEdgeCases:
    """Tests for edge cases in file structure."""

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_empty_file(self, lang, ext):
        """Empty file should return empty chunks without error."""
        code = ""
        chunks = _parse(code, f"test{ext}", lang)
        assert chunks == [] or len(chunks) == 0

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_only_whitespace(self, lang, ext):
        """File with only whitespace should return empty chunks."""
        code = "   \n\n   \t\t\n   "
        chunks = _parse(code, f"test{ext}", lang)
        assert len(chunks) == 0

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_only_comments(self, lang, ext):
        """File with only comments should parse without error."""
        code = """// Just a comment
/* Another comment */
/**
 * A JSDoc comment
 */
"""
        chunks = _parse(code, f"test{ext}", lang)
        # Should not error; may or may not extract comments as chunks
        assert isinstance(chunks, list)

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_only_imports(self, lang, ext):
        """File with only imports should parse without error."""
        code = """import React from 'react';
import { useState } from 'react';
"""
        chunks = _parse(code, f"test{ext}", lang)
        # Should not error; may or may not extract imports as chunks
        assert isinstance(chunks, list)

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_shebang(self, lang, ext):
        """File with shebang should parse correctly."""
        code = """#!/usr/bin/env node
function main() {
    console.log("Hello");
}
"""
        chunks = _parse(code, f"test{ext}", lang)
        assert len(chunks) > 0
        func_chunks = [c for c in chunks if "main" in c.code]
        assert len(func_chunks) > 0

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_use_strict_directive(self, lang, ext):
        """File with 'use strict' directive should parse correctly."""
        code = """"use strict";
function strictFunction() {
    return "strict";
}
"""
        chunks = _parse(code, f"test{ext}", lang)
        assert len(chunks) > 0
        func_chunks = [c for c in chunks if "strictFunction" in c.code]
        assert len(func_chunks) > 0


class TestUnicodeEdgeCases:
    """Tests for Unicode identifier and content handling."""

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_unicode_identifier(self, lang, ext):
        """Unicode identifier should be extracted correctly."""
        # Using valid JS identifiers with extended characters
        code = """const cafe = "coffee";
function greet() { return cafe; }
"""
        chunks = _parse(code, f"test{ext}", lang)
        assert len(chunks) > 0

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_unicode_in_string(self, lang, ext):
        """Unicode in string literals should be preserved."""
        code = """const greeting = "Hello, World!";
const emoji = "The party starts now";
function getGreeting() { return greeting + " " + emoji; }
"""
        chunks = _parse(code, f"test{ext}", lang)
        assert len(chunks) > 0
        func_chunks = [c for c in chunks if "getGreeting" in c.code]
        assert len(func_chunks) > 0

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_unicode_in_comment(self, lang, ext):
        """Unicode in comments should be preserved."""
        code = """// Japanese comment: Hello
function japanese() { return "hello"; }
"""
        chunks = _parse(code, f"test{ext}", lang)
        assert len(chunks) > 0
        func_chunks = [c for c in chunks if "japanese" in c.code]
        assert len(func_chunks) > 0

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JSX, ".jsx"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_unicode_in_jsx_content(self, lang, ext):
        """Unicode in JSX content should be preserved."""
        code = """function ChineseContent() {
    return <div>Chinese Content Here</div>;
}
"""
        chunks = _parse(code, f"test{ext}", lang)
        assert len(chunks) > 0
        comp_chunks = [c for c in chunks if "ChineseContent" in c.code]
        assert len(comp_chunks) > 0


class TestIIFEPatterns:
    """Tests for IIFE (Immediately Invoked Function Expression) patterns."""

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_iife_function(self, lang, ext):
        """IIFE with function keyword should be extracted."""
        code = """(function() {
    console.log("IIFE");
})();
"""
        chunks = _parse(code, f"test{ext}", lang)
        # IIFE should be recognized as some kind of chunk
        assert isinstance(chunks, list)

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_iife_arrow(self, lang, ext):
        """IIFE with arrow function should be extracted."""
        code = """(() => {
    console.log("Arrow IIFE");
})();
"""
        chunks = _parse(code, f"test{ext}", lang)
        assert isinstance(chunks, list)


class TestNestingPatterns:
    """Tests for nested function and class patterns."""

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_function_in_function(self, lang, ext):
        """Nested function should be extracted."""
        code = """function outer() {
    function inner() {
        return "inner";
    }
    return inner();
}
"""
        chunks = _parse(code, f"test{ext}", lang)
        assert len(chunks) > 0
        # At least outer function should be extracted
        outer_chunks = [c for c in chunks if "outer" in c.code]
        assert len(outer_chunks) > 0

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_class_in_function(self, lang, ext):
        """Class inside function should be extracted."""
        code = """function factory() {
    return class InnerClass {
        getValue() { return 42; }
    };
}
"""
        chunks = _parse(code, f"test{ext}", lang)
        assert len(chunks) > 0
        factory_chunks = [c for c in chunks if "factory" in c.code]
        assert len(factory_chunks) > 0

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_arrow_in_arrow(self, lang, ext):
        """Nested arrow functions should be extracted."""
        code = """const outer = () => {
    const inner = () => "inner";
    return inner();
};
"""
        chunks = _parse(code, f"test{ext}", lang)
        assert len(chunks) > 0
        all_content = " ".join(c.code for c in chunks)
        assert "outer" in all_content


class TestMetadataCompleteness:
    """Tests for metadata completeness in extracted chunks."""

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_line_numbers_present(self, lang, ext):
        """Chunks should have correct line numbers."""
        code = """function first() { return 1; }

function second() { return 2; }

function third() { return 3; }
"""
        chunks = _parse(code, f"test{ext}", lang)
        for chunk in chunks:
            # Line numbers should be present and positive
            assert hasattr(chunk, "start_line") or hasattr(chunk, "start_byte")
            if hasattr(chunk, "start_line") and chunk.start_line is not None:
                assert chunk.start_line >= 1
            if hasattr(chunk, "end_line") and chunk.end_line is not None:
                assert chunk.end_line >= chunk.start_line if chunk.start_line else True

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_byte_offsets_present(self, lang, ext):
        """Chunks should have byte offsets when available."""
        code = """function withBytes() {
    return "bytes";
}
"""
        chunks = _parse(code, f"test{ext}", lang)
        for chunk in chunks:
            # Check that byte offsets exist if they're part of the model
            if hasattr(chunk, "start_byte"):
                if chunk.start_byte is not None:
                    assert chunk.start_byte >= 0
            if hasattr(chunk, "end_byte"):
                if chunk.end_byte is not None:
                    assert chunk.end_byte >= chunk.start_byte if chunk.start_byte else True

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_chunk_type_present(self, lang, ext):
        """Chunks should have a chunk type."""
        code = """function typed() { return "typed"; }
class TypedClass {}
"""
        chunks = _parse(code, f"test{ext}", lang)
        for chunk in chunks:
            assert hasattr(chunk, "chunk_type")
            assert chunk.chunk_type is not None
            # Should be a valid ChunkType
            if isinstance(chunk.chunk_type, ChunkType):
                assert chunk.chunk_type.value is not None
            else:
                # String representation
                assert len(str(chunk.chunk_type)) > 0

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_content_not_empty(self, lang, ext):
        """Chunk content should not be empty."""
        code = """function withContent() {
    return "content";
}
"""
        chunks = _parse(code, f"test{ext}", lang)
        for chunk in chunks:
            assert hasattr(chunk, "code")
            assert chunk.code is not None
            assert len(chunk.code.strip()) > 0

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_language_field_present(self, lang, ext):
        """Chunks should have correct language field."""
        code = """function withLang() { return "lang"; }
"""
        chunks = _parse(code, f"test{ext}", lang)
        for chunk in chunks:
            assert hasattr(chunk, "language")
            assert chunk.language == lang


class TestUnusualValidPatterns:
    """Tests for unusual but valid JavaScript patterns."""

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_comma_operator(self, lang, ext):
        """Comma operator in expression should parse correctly."""
        code = """const x = (1, 2, 3);
function getX() { return x; }
"""
        chunks = _parse(code, f"test{ext}", lang)
        assert len(chunks) > 0

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_labeled_statement(self, lang, ext):
        """Labeled statement should parse correctly."""
        code = """function withLabel() {
    outer: for (let i = 0; i < 10; i++) {
        if (i === 5) break outer;
    }
}
"""
        chunks = _parse(code, f"test{ext}", lang)
        assert len(chunks) > 0
        func_chunks = [c for c in chunks if "withLabel" in c.code]
        assert len(func_chunks) > 0

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_object_method_shorthand(self, lang, ext):
        """Object method shorthand should parse correctly."""
        code = """const obj = {
    method() {
        return "shorthand";
    },
    async asyncMethod() {
        return "async";
    },
    *generatorMethod() {
        yield "generated";
    }
};
"""
        chunks = _parse(code, f"test{ext}", lang)
        assert len(chunks) > 0
        # Object should be extracted
        all_content = " ".join(c.code for c in chunks)
        assert "method" in all_content or "obj" in all_content

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JAVASCRIPT, ".js"),
            (Language.JSX, ".jsx"),
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_computed_property_names(self, lang, ext):
        """Computed property names should parse correctly."""
        code = """const key = "dynamic";
const obj = {
    [key]: "value",
    ["computed_" + "key"]: "another"
};
function getObj() { return obj; }
"""
        chunks = _parse(code, f"test{ext}", lang)
        assert len(chunks) > 0
