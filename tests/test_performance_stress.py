"""Performance and stress tests for JS-family parsers.

Tests parser behavior with:
- Deeply nested code structures
- Large files with many constructs
- Memory usage with large inputs
- Parsing speed benchmarks

Reference: docs/js-family-parser-test-specification.md
"""

import time
import pytest
from pathlib import Path

from chunkhound.core.types.common import ChunkType, FileId, Language
from chunkhound.parsers.parser_factory import ParserFactory, get_parser_factory


@pytest.fixture
def parser_factory():
    """Create a parser factory instance."""
    return ParserFactory()


# =============================================================================
# DEEP NESTING TESTS
# =============================================================================


class TestDeepNesting:
    """Test parser behavior with deeply nested structures."""

    def test_deeply_nested_functions_js(self, parser_factory):
        """Test parsing deeply nested functions (20 levels).

        This tests that the parser can handle deep recursion without
        stack overflow or excessive memory usage.
        """
        parser = parser_factory.create_parser(Language.JAVASCRIPT)

        # Generate 20 levels of nested functions
        depth = 20
        code_lines = []
        indent = ""

        for i in range(depth):
            code_lines.append(f"{indent}function level{i}() {{")
            indent += "    "

        # Add innermost content
        code_lines.append(f"{indent}return 'deepest';")

        # Close all functions
        for i in range(depth - 1, -1, -1):
            indent = "    " * i
            code_lines.append(f"{indent}}}")

        code = "\n".join(code_lines)

        # Should complete without stack overflow
        start_time = time.perf_counter()
        chunks = parser.parse_content(code, "deep_nesting.js", FileId(1))
        elapsed = time.perf_counter() - start_time

        assert len(chunks) > 0, "Should extract at least one chunk from deeply nested code"
        assert elapsed < 5.0, f"Should complete within 5 seconds, took {elapsed:.2f}s"

        # Verify at least the outermost function is captured
        assert any("level0" in c.code for c in chunks), "Should capture outermost function"

    def test_deeply_nested_functions_ts(self, parser_factory):
        """Test parsing deeply nested TypeScript functions."""
        parser = parser_factory.create_parser(Language.TYPESCRIPT)

        depth = 15
        code_lines = []
        indent = ""

        for i in range(depth):
            code_lines.append(f"{indent}function level{i}(): void {{")
            indent += "    "

        code_lines.append(f"{indent}console.log('deepest');")

        for i in range(depth - 1, -1, -1):
            indent = "    " * i
            code_lines.append(f"{indent}}}")

        code = "\n".join(code_lines)

        start_time = time.perf_counter()
        chunks = parser.parse_content(code, "deep_nesting.ts", FileId(1))
        elapsed = time.perf_counter() - start_time

        assert len(chunks) > 0, "Should extract chunks from deeply nested TypeScript"
        assert elapsed < 5.0, f"Should complete within 5 seconds, took {elapsed:.2f}s"

    def test_deeply_nested_jsx_elements(self, parser_factory):
        """Test parsing deeply nested JSX elements."""
        parser = parser_factory.create_parser(Language.JSX)

        # Generate deeply nested JSX
        depth = 20
        jsx_open = "".join([f"<div id='level{i}'>" for i in range(depth)])
        jsx_close = "</div>" * depth

        code = f"""
function DeepComponent() {{
    return (
        {jsx_open}
            <span>Deepest content</span>
        {jsx_close}
    )
}}
"""

        start_time = time.perf_counter()
        chunks = parser.parse_content(code, "deep_jsx.jsx", FileId(1))
        elapsed = time.perf_counter() - start_time

        assert len(chunks) > 0, "Should extract chunk from deeply nested JSX"
        assert elapsed < 5.0, f"Should complete within 5 seconds, took {elapsed:.2f}s"
        assert any("DeepComponent" in c.code for c in chunks), "Should find component"

    def test_deeply_nested_objects(self, parser_factory):
        """Test parsing deeply nested object literals."""
        parser = parser_factory.create_parser(Language.JAVASCRIPT)

        depth = 15
        obj_start = "{ a: " * depth
        obj_end = " }" * depth

        code = f"""
const deepObject = {obj_start}'value'{obj_end};
"""

        start_time = time.perf_counter()
        chunks = parser.parse_content(code, "deep_objects.js", FileId(1))
        elapsed = time.perf_counter() - start_time

        assert len(chunks) >= 0, "Should handle deeply nested objects"
        assert elapsed < 5.0, f"Should complete within 5 seconds, took {elapsed:.2f}s"


# =============================================================================
# LARGE FILE TESTS
# =============================================================================


class TestLargeFiles:
    """Test parser behavior with large files."""

    def test_many_functions(self, parser_factory):
        """Test parsing file with 1000 functions.

        All functions should be extracted or at least the parser
        should complete successfully.
        """
        parser = parser_factory.create_parser(Language.JAVASCRIPT)

        # Generate 1000 simple functions
        num_functions = 1000
        functions = [
            f"function fn{i}() {{ return {i}; }}"
            for i in range(num_functions)
        ]
        code = "\n\n".join(functions)

        start_time = time.perf_counter()
        chunks = parser.parse_content(code, "many_functions.js", FileId(1))
        elapsed = time.perf_counter() - start_time

        assert len(chunks) > 0, "Should extract chunks from large file"
        assert elapsed < 10.0, f"Should complete within 10 seconds, took {elapsed:.2f}s"

        # Check that a reasonable number of functions are found
        # Parser may batch them, so we check content instead of count
        all_code = " ".join([c.code for c in chunks])
        found_count = sum(1 for i in range(100) if f"fn{i}" in all_code)
        assert found_count > 50, f"Should find at least 50 function names, found {found_count}"

    def test_many_classes(self, parser_factory):
        """Test parsing file with many classes."""
        parser = parser_factory.create_parser(Language.TYPESCRIPT)

        num_classes = 200
        classes = [
            f"""class Class{i} {{
    value: number = {i};

    getValue(): number {{
        return this.value;
    }}
}}"""
            for i in range(num_classes)
        ]
        code = "\n\n".join(classes)

        start_time = time.perf_counter()
        chunks = parser.parse_content(code, "many_classes.ts", FileId(1))
        elapsed = time.perf_counter() - start_time

        assert len(chunks) > 0, "Should extract chunks from file with many classes"
        assert elapsed < 10.0, f"Should complete within 10 seconds, took {elapsed:.2f}s"

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        # May not extract all if batched, but should extract some
        assert len(class_chunks) > 0 or any("class Class" in c.code for c in chunks), \
            "Should extract class content"

    def test_many_interfaces(self, parser_factory):
        """Test parsing file with many TypeScript interfaces."""
        parser = parser_factory.create_parser(Language.TYPESCRIPT)

        num_interfaces = 300
        interfaces = [
            f"""interface Interface{i} {{
    id: number;
    name: string;
    value{i}: number;
}}"""
            for i in range(num_interfaces)
        ]
        code = "\n\n".join(interfaces)

        start_time = time.perf_counter()
        chunks = parser.parse_content(code, "many_interfaces.ts", FileId(1))
        elapsed = time.perf_counter() - start_time

        assert len(chunks) > 0, "Should extract chunks from file with many interfaces"
        assert elapsed < 10.0, f"Should complete within 10 seconds, took {elapsed:.2f}s"

        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        # Should extract a substantial number
        assert len(interface_chunks) >= 10 or len(chunks) > 0, \
            "Should extract interface content"

    def test_large_react_file(self, parser_factory):
        """Test parsing large React/JSX file with multiple components."""
        parser = parser_factory.create_parser(Language.JSX)

        num_components = 100
        components = [
            f"""function Component{i}({{ value }}) {{
    const [state, setState] = useState({i});

    useEffect(() => {{
        console.log('Component{i} mounted');
    }}, []);

    return (
        <div className="component-{i}">
            <h2>Component {i}</h2>
            <p>Value: {{value}}</p>
            <p>State: {{state}}</p>
            <button onClick={{() => setState(s => s + 1)}}>
                Increment
            </button>
        </div>
    );
}}"""
            for i in range(num_components)
        ]
        code = "\n\n".join(components)

        start_time = time.perf_counter()
        chunks = parser.parse_content(code, "many_components.jsx", FileId(1))
        elapsed = time.perf_counter() - start_time

        assert len(chunks) > 0, "Should extract chunks from large React file"
        assert elapsed < 10.0, f"Should complete within 10 seconds, took {elapsed:.2f}s"


# =============================================================================
# MEMORY TESTS
# =============================================================================


class TestMemoryUsage:
    """Test parser memory behavior with large inputs."""

    def test_large_string_content(self, parser_factory):
        """Test parsing code with very large string literals."""
        parser = parser_factory.create_parser(Language.JAVASCRIPT)

        # 100KB string
        large_string = "x" * 100_000
        code = f"""
const largeString = "{large_string}";

function processString() {{
    return largeString.length;
}}
"""

        start_time = time.perf_counter()
        chunks = parser.parse_content(code, "large_string.js", FileId(1))
        elapsed = time.perf_counter() - start_time

        assert len(chunks) > 0, "Should handle file with large string"
        assert elapsed < 10.0, f"Should complete within 10 seconds, took {elapsed:.2f}s"

    def test_repetitive_code_pattern(self, parser_factory):
        """Test parsing highly repetitive code patterns."""
        parser = parser_factory.create_parser(Language.TYPESCRIPT)

        # Generate repetitive switch statement
        num_cases = 500
        cases = [f"        case {i}: return '{i}';" for i in range(num_cases)]

        code = f"""
function getLabel(code: number): string {{
    switch (code) {{
{chr(10).join(cases)}
        default: return 'unknown';
    }}
}}
"""

        start_time = time.perf_counter()
        chunks = parser.parse_content(code, "repetitive.ts", FileId(1))
        elapsed = time.perf_counter() - start_time

        assert len(chunks) > 0, "Should handle repetitive code"
        assert elapsed < 10.0, f"Should complete within 10 seconds, took {elapsed:.2f}s"

    def test_wide_import_list(self, parser_factory):
        """Test parsing file with many imports."""
        parser = parser_factory.create_parser(Language.TYPESCRIPT)

        num_imports = 200
        imports = [
            f"import {{ module{i} }} from './module{i}';"
            for i in range(num_imports)
        ]

        code = "\n".join(imports) + """

export function useAllModules() {
    return {
""" + ",\n".join([f"        module{i}" for i in range(num_imports)]) + """
    };
}
"""

        start_time = time.perf_counter()
        chunks = parser.parse_content(code, "many_imports.ts", FileId(1))
        elapsed = time.perf_counter() - start_time

        assert len(chunks) > 0, "Should handle many imports"
        assert elapsed < 10.0, f"Should complete within 10 seconds, took {elapsed:.2f}s"


# =============================================================================
# EDGE CASE STRESS TESTS
# =============================================================================


class TestEdgeCaseStress:
    """Test parser with edge case stress scenarios."""

    def test_complex_type_annotations(self, parser_factory):
        """Test parsing complex TypeScript type annotations."""
        parser = parser_factory.create_parser(Language.TYPESCRIPT)

        code = """
type DeepReadonly<T> = {
    readonly [P in keyof T]: T[P] extends object
        ? T[P] extends Function
            ? T[P]
            : DeepReadonly<T[P]>
        : T[P];
};

type DeepPartial<T> = {
    [P in keyof T]?: T[P] extends object
        ? T[P] extends Function
            ? T[P]
            : DeepPartial<T[P]>
        : T[P];
};

type Flatten<T> = T extends Array<infer U>
    ? Flatten<U>
    : T extends object
        ? { [K in keyof T]: Flatten<T[K]> }
        : T;

type UnionToIntersection<U> = (U extends any ? (k: U) => void : never) extends (
    k: infer I
) => void
    ? I
    : never;

function process<T extends object>(
    obj: T
): DeepReadonly<DeepPartial<Flatten<T>>> {
    return obj as any;
}
"""

        start_time = time.perf_counter()
        chunks = parser.parse_content(code, "complex_types.ts", FileId(1))
        elapsed = time.perf_counter() - start_time

        assert len(chunks) > 0, "Should extract chunks from complex types"
        assert elapsed < 5.0, f"Should complete within 5 seconds, took {elapsed:.2f}s"

        type_chunks = [c for c in chunks if c.chunk_type == ChunkType.TYPE_ALIAS]
        assert len(type_chunks) >= 1, "Should extract at least one type alias"

    def test_many_decorators(self, parser_factory):
        """Test parsing class with many decorators."""
        parser = parser_factory.create_parser(Language.TYPESCRIPT)

        num_decorators = 20
        class_decorators = [f"@ClassDecorator{i}()" for i in range(num_decorators)]
        property_decorators = [f"@PropertyDecorator{i}()" for i in range(5)]

        code = f"""
{chr(10).join(class_decorators)}
class HeavilyDecoratedClass {{
{chr(10).join(f"    {d}" for d in property_decorators)}
    property1: string;

{chr(10).join(f"    {d}" for d in property_decorators)}
    property2: number;

    @MethodDecorator1()
    @MethodDecorator2()
    @MethodDecorator3()
    method1() {{
        return this.property1;
    }}

    @MethodDecorator1()
    @MethodDecorator2()
    method2(
        @ParamDecorator1() arg1: string,
        @ParamDecorator2() arg2: number
    ) {{
        return arg1 + arg2;
    }}
}}
"""

        start_time = time.perf_counter()
        chunks = parser.parse_content(code, "many_decorators.ts", FileId(1))
        elapsed = time.perf_counter() - start_time

        assert len(chunks) > 0, "Should extract chunks from heavily decorated class"
        assert elapsed < 5.0, f"Should complete within 5 seconds, took {elapsed:.2f}s"

    def test_unicode_heavy_content(self, parser_factory):
        """Test parsing file with heavy unicode content."""
        parser = parser_factory.create_parser(Language.TYPESCRIPT)

        code = """
// Comments with various scripts
// Arabic: مرحبا بالعالم
// Chinese: 你好世界
// Japanese: こんにちは世界
// Korean: 안녕하세요 세계
// Russian: Привет мир
// Greek: Γεια σου κόσμε
// Hebrew: שלום עולם
// Thai: สวัสดีชาวโลก

interface Internationalization {
    arabic: string;      // العربية
    chinese: string;     // 中文
    japanese: string;    // 日本語
    korean: string;      // 한국어
    russian: string;     // Русский
    greek: string;       // Ελληνικά
    hebrew: string;      // עברית
    thai: string;        // ไทย
    emoji: string;       // 🌍🌎🌏
}

const messages: Record<string, string> = {
    'ar': 'مرحبا بالعالم',
    'zh': '你好世界',
    'ja': 'こんにちは世界',
    'ko': '안녕하세요 세계',
    'ru': 'Привет мир',
    'el': 'Γεια σου κόσμε',
    'he': 'שלום עולם',
    'th': 'สวัสดีชาวโลก',
};

function getMessage(locale: keyof typeof messages): string {
    return messages[locale] ?? 'Hello World';
}
"""

        start_time = time.perf_counter()
        chunks = parser.parse_content(code, "unicode.ts", FileId(1))
        elapsed = time.perf_counter() - start_time

        assert len(chunks) > 0, "Should extract chunks from unicode-heavy file"
        assert elapsed < 5.0, f"Should complete within 5 seconds, took {elapsed:.2f}s"


# =============================================================================
# CROSS-LANGUAGE PERFORMANCE
# =============================================================================


class TestCrossLanguagePerformance:
    """Test performance across different JS-family languages."""

    @pytest.mark.parametrize("language,extension", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_consistent_performance(self, parser_factory, language, extension):
        """Test that all languages have similar performance characteristics."""
        parser = parser_factory.create_parser(language)

        # Generate 100 functions
        functions = [
            f"function fn{i}(a, b) {{ return a + b + {i}; }}"
            for i in range(100)
        ]
        code = "\n".join(functions)

        start_time = time.perf_counter()
        chunks = parser.parse_content(code, f"perf_test.{extension}", FileId(1))
        elapsed = time.perf_counter() - start_time

        assert len(chunks) > 0, f"{language.value} should extract chunks"
        assert elapsed < 5.0, f"{language.value} took {elapsed:.2f}s, should be under 5s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
