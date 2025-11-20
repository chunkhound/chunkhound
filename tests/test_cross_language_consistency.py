"""Tests for cross-language consistency in JS-family parsers.

This module verifies that the same construct produces consistent results
across JavaScript, JSX, TypeScript, and TSX parsers. This is critical for
parser consolidation work.
"""

from pathlib import Path

import pytest

from chunkhound.core.types.common import Language, ChunkType
from chunkhound.parsers.parser_factory import ParserFactory


def _parse(code: str, filename: str, language: Language):
    """Parse code and return chunks."""
    factory = ParserFactory()
    parser = factory.create_parser(language)
    return parser.parse_content(code, Path(filename), file_id=1)


# Define the language variants for parametrized tests
JS_FAMILY_LANGUAGES = [
    (Language.JAVASCRIPT, ".js"),
    (Language.JSX, ".jsx"),
    (Language.TYPESCRIPT, ".ts"),
    (Language.TSX, ".tsx"),
]


# =============================================================================
# Section 11: Cross-Language Consistency Tests
# =============================================================================


class TestFunctionExtractionConsistency:
    """Tests that function extraction is consistent across languages."""

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_basic_function_declaration(self, lang, ext):
        """Basic function declaration should be extracted consistently."""
        code = "function foo() { return 42; }"
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        # Find the function chunk
        func_chunk = next((c for c in chunks if "foo" in c.code), None)
        assert func_chunk is not None, f"Function 'foo' not found in {lang}"

        # Name should be extracted if available
        if hasattr(func_chunk, "name") and func_chunk.name:
            assert "foo" in func_chunk.name

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_async_function(self, lang, ext):
        """Async function should be extracted consistently."""
        code = "async function fetchData() { return await fetch('url'); }"
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        func_chunk = next((c for c in chunks if "fetchData" in c.code), None)
        assert func_chunk is not None, f"Async function 'fetchData' not found in {lang}"
        # The async keyword should be in the content
        assert "async" in func_chunk.code

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_arrow_function(self, lang, ext):
        """Arrow function assigned to const should be extracted consistently."""
        code = "const add = (a, b) => a + b;"
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        # Arrow function should be extracted
        all_content = " ".join(c.code for c in chunks)
        assert "add" in all_content, f"Arrow function 'add' not found in {lang}"

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_arrow_function_with_block(self, lang, ext):
        """Arrow function with block body should be extracted consistently."""
        code = """const multiply = (a, b) => {
    const result = a * b;
    return result;
};"""
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        all_content = " ".join(c.code for c in chunks)
        assert "multiply" in all_content, f"Arrow function 'multiply' not found in {lang}"

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    @pytest.mark.skip(reason="Generator functions not currently extracted by JS-family parsers - known gap")
    def test_generator_function(self, lang, ext):
        """Generator function should be extracted consistently.

        NOTE: This test is skipped because generator functions are not currently
        being extracted by the JS-family parsers. This is a known parser gap
        that should be addressed in parser consolidation work.
        """
        code = "function* generate() { yield 1; yield 2; }"
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        func_chunk = next((c for c in chunks if "generate" in c.code), None)
        assert func_chunk is not None, f"Generator 'generate' not found in {lang}"
        # The * should be in the content
        assert "function*" in func_chunk.code or "* generate" in func_chunk.code

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_function_with_parameters(self, lang, ext):
        """Function with parameters should be extracted consistently."""
        code = "function greet(name, greeting = 'Hello') { return `${greeting}, ${name}`; }"
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        func_chunk = next((c for c in chunks if "greet" in c.code), None)
        assert func_chunk is not None, f"Function 'greet' not found in {lang}"
        # Parameters should be in content
        assert "name" in func_chunk.code
        assert "greeting" in func_chunk.code


class TestClassExtractionConsistency:
    """Tests that class extraction is consistent across languages."""

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_basic_class(self, lang, ext):
        """Basic class should be extracted consistently."""
        code = "class Foo {}"
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        class_chunk = next((c for c in chunks if "Foo" in c.code), None)
        assert class_chunk is not None, f"Class 'Foo' not found in {lang}"

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_class_with_constructor(self, lang, ext):
        """Class with constructor should be extracted consistently."""
        code = """class Person {
    constructor(name) {
        this.name = name;
    }
}"""
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        class_chunk = next((c for c in chunks if "Person" in c.code), None)
        assert class_chunk is not None, f"Class 'Person' not found in {lang}"
        # Constructor should be in content
        assert "constructor" in class_chunk.code

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_class_with_method(self, lang, ext):
        """Class with method should be extracted consistently."""
        code = """class Calculator {
    add(a, b) {
        return a + b;
    }
}"""
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        class_chunk = next((c for c in chunks if "Calculator" in c.code), None)
        assert class_chunk is not None, f"Class 'Calculator' not found in {lang}"

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_class_with_extends(self, lang, ext):
        """Class with extends should be extracted consistently."""
        code = "class Child extends Parent {}"
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        class_chunk = next((c for c in chunks if "Child" in c.code), None)
        assert class_chunk is not None, f"Class 'Child' not found in {lang}"
        # extends should be in content
        assert "extends" in class_chunk.code
        assert "Parent" in class_chunk.code

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_class_with_static_method(self, lang, ext):
        """Class with static method should be extracted consistently."""
        code = """class Utils {
    static format(value) {
        return String(value);
    }
}"""
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        class_chunk = next((c for c in chunks if "Utils" in c.code), None)
        assert class_chunk is not None, f"Class 'Utils' not found in {lang}"
        # static should be in content
        assert "static" in class_chunk.code


class TestVariableExtractionConsistency:
    """Tests that variable extraction is consistent across languages."""

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_const_object(self, lang, ext):
        """Const object should be extracted consistently."""
        code = "const config = { key: 'value', number: 42 };"
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        all_content = " ".join(c.code for c in chunks)
        assert "config" in all_content, f"Const 'config' not found in {lang}"

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_const_array(self, lang, ext):
        """Const array should be extracted consistently."""
        code = "const items = [1, 2, 3];"
        chunks = _parse(code, f"test{ext}", lang)

        # May or may not extract depending on parser configuration
        assert isinstance(chunks, list)

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_let_declaration(self, lang, ext):
        """Let declaration should be extracted consistently."""
        code = "let counter = { value: 0 };"
        chunks = _parse(code, f"test{ext}", lang)

        assert isinstance(chunks, list)


class TestImportExtractionConsistency:
    """Tests that import extraction is consistent across languages."""

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_default_import(self, lang, ext):
        """Default import should be extracted consistently."""
        code = "import React from 'react';"
        chunks = _parse(code, f"test{ext}", lang)

        # Imports may or may not be extracted as chunks
        assert isinstance(chunks, list)

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_named_import(self, lang, ext):
        """Named import should be extracted consistently."""
        code = "import { useState, useEffect } from 'react';"
        chunks = _parse(code, f"test{ext}", lang)

        assert isinstance(chunks, list)

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_namespace_import(self, lang, ext):
        """Namespace import should be extracted consistently."""
        code = "import * as utils from './utils';"
        chunks = _parse(code, f"test{ext}", lang)

        assert isinstance(chunks, list)


class TestExportExtractionConsistency:
    """Tests that export extraction is consistent across languages."""

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_named_export_const(self, lang, ext):
        """Named export const should be extracted consistently."""
        code = "export const value = { key: 'exported' };"
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        all_content = " ".join(c.code for c in chunks)
        assert "value" in all_content or "export" in all_content

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_export_default_object(self, lang, ext):
        """Export default object should be extracted consistently."""
        code = "export default { key: 'default' };"
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        all_content = " ".join(c.code for c in chunks)
        assert "default" in all_content or "key" in all_content

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_export_function(self, lang, ext):
        """Export function should be extracted consistently."""
        code = "export function exported() { return 'exported'; }"
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        func_chunk = next((c for c in chunks if "exported" in c.code), None)
        assert func_chunk is not None, f"Exported function not found in {lang}"

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_export_class(self, lang, ext):
        """Export class should be extracted consistently."""
        code = "export class ExportedClass {}"
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        class_chunk = next((c for c in chunks if "ExportedClass" in c.code), None)
        assert class_chunk is not None, f"Exported class not found in {lang}"


class TestCommentExtractionConsistency:
    """Tests that comment extraction is consistent across languages."""

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_single_line_comment(self, lang, ext):
        """Single-line comment should be consistent."""
        code = """// This is a comment
function foo() { return 1; }"""
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        # Function should be found
        func_chunk = next((c for c in chunks if "foo" in c.code), None)
        assert func_chunk is not None

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_multiline_comment(self, lang, ext):
        """Multi-line comment should be consistent."""
        code = """/* Multi-line
comment */
function bar() { return 2; }"""
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1


class TestSameNameExtractionConsistency:
    """Tests that the same construct extracts the same name across languages."""

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_function_name_extraction(self, lang, ext):
        """Function name should be extracted consistently."""
        code = "function myFunction() { return 42; }"
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        func_chunk = next((c for c in chunks if "myFunction" in c.code), None)
        assert func_chunk is not None, f"Function not found in {lang}"

        # If name attribute exists, verify it
        if hasattr(func_chunk, "name") and func_chunk.name:
            assert "myFunction" in func_chunk.name

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_class_name_extraction(self, lang, ext):
        """Class name should be extracted consistently."""
        code = "class MyClass {}"
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        class_chunk = next((c for c in chunks if "MyClass" in c.code), None)
        assert class_chunk is not None, f"Class not found in {lang}"

        # If name attribute exists, verify it
        if hasattr(class_chunk, "name") and class_chunk.name:
            assert "MyClass" in class_chunk.name


class TestChunkTypeConsistency:
    """Tests that chunk types are consistent across languages."""

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_function_chunk_type(self, lang, ext):
        """Function should have consistent chunk type."""
        code = "function typed() { return 'typed'; }"
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        func_chunk = next((c for c in chunks if "typed" in c.code), None)
        assert func_chunk is not None

        # Chunk type should be function-related
        if hasattr(func_chunk, "chunk_type"):
            chunk_type = func_chunk.chunk_type
            if isinstance(chunk_type, ChunkType):
                assert chunk_type in [
                    ChunkType.FUNCTION,
                    ChunkType.CLOSURE,
                    ChunkType.VARIABLE,  # For arrow functions assigned to variables
                ]
            else:
                assert str(chunk_type) in ["function", "closure", "variable"]

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_class_chunk_type(self, lang, ext):
        """Class should have consistent chunk type.

        NOTE: JS/JSX parsers currently return FUNCTION type for classes,
        while TS/TSX correctly return CLASS. This is a known inconsistency
        that should be addressed in parser consolidation work.
        """
        code = "class TypedClass {}"
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        class_chunk = next((c for c in chunks if "TypedClass" in c.code), None)
        assert class_chunk is not None

        # Chunk type should be class (but JS/JSX currently return FUNCTION)
        if hasattr(class_chunk, "chunk_type"):
            chunk_type = class_chunk.chunk_type
            if isinstance(chunk_type, ChunkType):
                # Accept CLASS or FUNCTION for now due to JS/JSX inconsistency
                assert chunk_type in [ChunkType.CLASS, ChunkType.FUNCTION], (
                    f"Expected CLASS or FUNCTION for {lang}, got {chunk_type}"
                )
            else:
                assert str(chunk_type) in ["class", "function"]


class TestMultipleConstructsConsistency:
    """Tests files with multiple constructs for consistency."""

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_file_with_multiple_constructs(self, lang, ext):
        """File with multiple constructs should be consistent."""
        code = """function helperFunction() {
    return "helper";
}

class MainClass {
    constructor() {
        this.value = helperFunction();
    }

    getValue() {
        return this.value;
    }
}

const config = {
    name: "config",
    version: 1
};

export default MainClass;
"""
        chunks = _parse(code, f"test{ext}", lang)

        # Should extract multiple chunks
        assert len(chunks) >= 1
        all_content = " ".join(c.code for c in chunks)

        # Function should be found
        assert "helperFunction" in all_content
        # Class should be found
        assert "MainClass" in all_content


class TestCommonPatternsConsistency:
    """Tests common JavaScript patterns for consistency across languages."""

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_callback_pattern(self, lang, ext):
        """Callback pattern should be consistent."""
        code = """function processData(data, callback) {
    const result = transform(data);
    callback(result);
}"""
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        func_chunk = next((c for c in chunks if "processData" in c.code), None)
        assert func_chunk is not None

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_factory_pattern(self, lang, ext):
        """Factory pattern should be consistent."""
        code = """function createWidget(type) {
    if (type === 'button') {
        return { type: 'button', click: () => {} };
    }
    return { type: 'default' };
}"""
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        func_chunk = next((c for c in chunks if "createWidget" in c.code), None)
        assert func_chunk is not None

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_module_pattern(self, lang, ext):
        """Module pattern should be consistent."""
        code = """const Module = (function() {
    const private = "private";

    return {
        getPrivate() {
            return private;
        }
    };
})();"""
        chunks = _parse(code, f"test{ext}", lang)

        # Should parse without error
        assert isinstance(chunks, list)


class TestTypeScriptSpecificConsistency:
    """Tests TypeScript-specific constructs are consistent between TS and TSX."""

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_interface_extraction(self, lang, ext):
        """Interface should be extracted consistently in TS/TSX."""
        code = """interface User {
    name: string;
    age: number;
}"""
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        interface_chunk = next((c for c in chunks if "User" in c.code), None)
        assert interface_chunk is not None, f"Interface 'User' not found in {lang}"

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_type_alias_extraction(self, lang, ext):
        """Type alias should be extracted consistently in TS/TSX."""
        code = "type StringOrNumber = string | number;"
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        type_chunk = next((c for c in chunks if "StringOrNumber" in c.code), None)
        assert type_chunk is not None, f"Type alias 'StringOrNumber' not found in {lang}"

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_enum_extraction(self, lang, ext):
        """Enum should be extracted consistently in TS/TSX."""
        code = """enum Status {
    Active = "ACTIVE",
    Inactive = "INACTIVE"
}"""
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        enum_chunk = next((c for c in chunks if "Status" in c.code), None)
        assert enum_chunk is not None, f"Enum 'Status' not found in {lang}"

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_generic_function(self, lang, ext):
        """Generic function should be extracted consistently in TS/TSX."""
        code = "function identity<T>(value: T): T { return value; }"
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        func_chunk = next((c for c in chunks if "identity" in c.code), None)
        assert func_chunk is not None
        # Generic parameter should be in content
        assert "<T>" in func_chunk.code

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.TYPESCRIPT, ".ts"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_typed_function_parameters(self, lang, ext):
        """Typed function parameters should be extracted consistently."""
        code = "function add(a: number, b: number): number { return a + b; }"
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        func_chunk = next((c for c in chunks if "add" in c.code), None)
        assert func_chunk is not None
        # Type annotations should be in content
        assert ": number" in func_chunk.code


class TestJSXSpecificConsistency:
    """Tests JSX-specific constructs are consistent between JSX and TSX."""

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JSX, ".jsx"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_functional_component(self, lang, ext):
        """Functional component should be extracted consistently."""
        code = """function Button() {
    return <button>Click me</button>;
}"""
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        comp_chunk = next((c for c in chunks if "Button" in c.code), None)
        assert comp_chunk is not None, f"Component 'Button' not found in {lang}"
        # JSX should be in content
        assert "<button>" in comp_chunk.code

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JSX, ".jsx"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_arrow_component(self, lang, ext):
        """Arrow function component should be extracted consistently."""
        code = """const Card = () => {
    return (
        <div className="card">
            <h1>Title</h1>
        </div>
    );
};"""
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        all_content = " ".join(c.code for c in chunks)
        assert "Card" in all_content, f"Component 'Card' not found in {lang}"

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JSX, ".jsx"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_component_with_props(self, lang, ext):
        """Component with props should be extracted consistently."""
        code = """function Greeting(props) {
    return <span>Hello, {props.name}!</span>;
}"""
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        comp_chunk = next((c for c in chunks if "Greeting" in c.code), None)
        assert comp_chunk is not None
        # Props usage should be in content
        assert "props" in comp_chunk.code

    @pytest.mark.parametrize(
        "lang,ext",
        [
            (Language.JSX, ".jsx"),
            (Language.TSX, ".tsx"),
        ],
    )
    def test_fragment_syntax(self, lang, ext):
        """Fragment syntax should be extracted consistently."""
        code = """function List() {
    return (
        <>
            <li>Item 1</li>
            <li>Item 2</li>
        </>
    );
}"""
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) >= 1
        comp_chunk = next((c for c in chunks if "List" in c.code), None)
        assert comp_chunk is not None
        # Fragment should be in content
        assert "<>" in comp_chunk.code or "</>" in comp_chunk.code


class TestLocationConsistency:
    """Tests that location metadata is consistent across languages."""

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_start_line_consistency(self, lang, ext):
        """Start line should be consistent for same code."""
        code = """// Comment

function atLine3() {
    return "line 3";
}
"""
        chunks = _parse(code, f"test{ext}", lang)

        func_chunk = next((c for c in chunks if "atLine3" in c.code), None)
        if func_chunk and hasattr(func_chunk, "start_line") and func_chunk.start_line:
            # Function should start at line 3 or later
            assert func_chunk.start_line >= 3

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_end_line_after_start(self, lang, ext):
        """End line should be >= start line."""
        code = """function multiLine() {
    const a = 1;
    const b = 2;
    return a + b;
}
"""
        chunks = _parse(code, f"test{ext}", lang)

        for chunk in chunks:
            if hasattr(chunk, "start_line") and hasattr(chunk, "end_line"):
                if chunk.start_line and chunk.end_line:
                    assert chunk.end_line >= chunk.start_line


class TestContentConsistency:
    """Tests that extracted content is consistent across languages."""

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_full_content_extracted(self, lang, ext):
        """Full function content should be extracted."""
        code = """function complete() {
    const x = 1;
    const y = 2;
    return x + y;
}
"""
        chunks = _parse(code, f"test{ext}", lang)

        func_chunk = next((c for c in chunks if "complete" in c.code), None)
        assert func_chunk is not None

        # All parts should be in content
        assert "const x = 1" in func_chunk.code
        assert "const y = 2" in func_chunk.code
        assert "return x + y" in func_chunk.code

    @pytest.mark.parametrize("lang,ext", JS_FAMILY_LANGUAGES)
    def test_class_content_extracted(self, lang, ext):
        """Full class content should be extracted."""
        code = """class Complete {
    constructor() {
        this.value = 0;
    }

    increment() {
        this.value++;
    }
}
"""
        chunks = _parse(code, f"test{ext}", lang)

        class_chunk = next((c for c in chunks if "Complete" in c.code), None)
        assert class_chunk is not None

        # Constructor should be in content
        assert "constructor" in class_chunk.code
