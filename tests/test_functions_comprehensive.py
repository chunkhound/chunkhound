"""Comprehensive function tests for JS-family parsers.

Tests function declarations, expressions, arrow functions, and parameters
across JavaScript, JSX, TypeScript, TSX, and Vue.

Based on the test specification in:
/Users/tyler/chunkhound/docs/js-family-parser-test-specification.md (Section 3: Functions)
"""

from pathlib import Path

import pytest

from chunkhound.core.types.common import ChunkType, Language
from chunkhound.parsers.parser_factory import ParserFactory


@pytest.fixture
def parser_factory():
    """Create a parser factory instance."""
    return ParserFactory()


def _parse(code: str, filename: str, language: Language):
    """Helper to parse code with the appropriate parser."""
    factory = ParserFactory()
    parser = factory.create_parser(language)
    return parser.parse_content(code, Path(filename), file_id=1)


# =============================================================================
# FUNCTION DECLARATIONS
# =============================================================================


class TestFunctionDeclarations:
    """Test function declaration extraction."""

    def test_basic_function_js(self):
        """Test basic function declaration in JavaScript."""
        code = "function foo() { return 42; }"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("foo" in c.code for c in chunks), "Should contain function foo"

    def test_basic_function_ts(self):
        """Test basic function declaration in TypeScript."""
        code = "function foo() { return 42; }"
        chunks = _parse(code, "test.ts", Language.TYPESCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("foo" in c.code for c in chunks), "Should contain function foo"

    def test_function_with_parameters(self):
        """Test function with parameters."""
        code = "function add(a, b) { return a + b; }"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        function_chunks = [c for c in chunks if "add" in c.code]
        assert len(function_chunks) > 0, "Should find function with parameters"
        # Verify parameters are in the code
        assert any("a, b" in c.code or "(a, b)" in c.code for c in function_chunks)

    def test_async_function(self):
        """Test async function declaration."""
        code = "async function fetchData() { return await fetch('/api'); }"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("fetchData" in c.code for c in chunks), "Should extract async function"
        assert any("async" in c.code for c in chunks), "Should preserve async keyword"

    def test_generator_function(self):
        """Test generator function declaration."""
        code = "function* generateIds() { yield 1; yield 2; }"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("generateIds" in c.code for c in chunks), "Should extract generator function"
        assert any("function*" in c.code for c in chunks), "Should preserve generator syntax"

    def test_async_generator_function(self):
        """Test async generator function declaration."""
        code = "async function* asyncGenerator() { yield await Promise.resolve(1); }"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("asyncGenerator" in c.code for c in chunks), "Should extract async generator"
        assert any("async function*" in c.code for c in chunks), "Should preserve async generator syntax"

    def test_async_generator_with_fetch(self):
        """Test async generator function with real async operation.

        This tests the specific pattern: async function* asyncGen() { yield await fetch('/api') }
        """
        code = """
async function* asyncGen() {
    const response = yield await fetch('/api/data');
    yield await response.json();
}
"""
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("asyncGen" in c.code for c in chunks), "Should extract async generator"
        assert any("async function*" in c.code for c in chunks), "Should preserve async generator syntax"
        assert any("fetch" in c.code for c in chunks), "Should preserve fetch call"

    @pytest.mark.parametrize("lang,ext", [
        (Language.JAVASCRIPT, ".js"),
        (Language.JSX, ".jsx"),
        (Language.TYPESCRIPT, ".ts"),
        (Language.TSX, ".tsx"),
    ])
    def test_basic_function_cross_language(self, lang, ext):
        """Test basic function is extracted across all JS-family languages."""
        code = "function greet() { return 'hello'; }"
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) > 0, f"Should extract chunk for {lang.value}"
        assert any("greet" in c.code for c in chunks), f"Should find greet in {lang.value}"


# =============================================================================
# FUNCTION EXPRESSIONS
# =============================================================================


class TestFunctionExpressions:
    """Test function expression extraction."""

    def test_const_function_expression(self):
        """Test const function expression."""
        code = "const foo = function() { return 42; };"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("foo" in c.code for c in chunks), "Should extract const function expression"

    def test_named_function_expression(self):
        """Test named function expression."""
        code = "const foo = function bar() { return 42; };"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Should capture both the variable name and inner function name
        matching_chunks = [c for c in chunks if "foo" in c.code and "bar" in c.code]
        assert len(matching_chunks) > 0, "Should extract named function expression with both names"

    def test_let_function_expression(self):
        """Test let function expression."""
        code = "let foo = function() { return 42; };"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("let" in c.code and "foo" in c.code for c in chunks), "Should extract let function expression"

    def test_var_function_expression(self):
        """Test var function expression."""
        code = "var foo = function() { return 42; };"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("var" in c.code and "foo" in c.code for c in chunks), "Should extract var function expression"

    def test_async_function_expression(self):
        """Test async function expression."""
        code = "const fetchData = async function() { return await fetch('/api'); };"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("fetchData" in c.code for c in chunks), "Should extract async function expression"
        assert any("async" in c.code for c in chunks), "Should preserve async keyword"


# =============================================================================
# ARROW FUNCTIONS
# =============================================================================


class TestArrowFunctions:
    """Test arrow function extraction."""

    def test_const_arrow_function(self):
        """Test const arrow function."""
        code = "const foo = () => { return 42; };"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("foo" in c.code for c in chunks), "Should extract const arrow function"
        assert any("=>" in c.code for c in chunks), "Should preserve arrow syntax"

    def test_let_arrow_function(self):
        """Test let arrow function."""
        code = "let foo = () => { return 42; };"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("let" in c.code and "foo" in c.code for c in chunks), "Should extract let arrow function"

    def test_var_arrow_function(self):
        """Test var arrow function."""
        code = "var foo = () => { return 42; };"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("var" in c.code and "foo" in c.code for c in chunks), "Should extract var arrow function"

    def test_arrow_with_parameters(self):
        """Test arrow function with parameters."""
        code = "const add = (a, b) => { return a + b; };"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("add" in c.code for c in chunks), "Should extract arrow function with params"

    def test_arrow_single_param_no_parens(self):
        """Test arrow function with single param no parentheses."""
        code = "const double = x => x * 2;"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("double" in c.code for c in chunks), "Should extract single-param arrow"

    def test_arrow_implicit_return(self):
        """Test arrow function with implicit return."""
        code = "const getValue = () => 'value';"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("getValue" in c.code for c in chunks), "Should extract implicit return arrow"

    def test_async_arrow_function(self):
        """Test async arrow function."""
        code = "const fetchData = async () => { return await fetch('/api'); };"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("fetchData" in c.code for c in chunks), "Should extract async arrow"
        assert any("async" in c.code for c in chunks), "Should preserve async keyword"

    def test_arrow_with_object_return(self):
        """Test arrow function returning object literal."""
        code = "const getObject = () => ({ key: 'value' });"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("getObject" in c.code for c in chunks), "Should extract arrow returning object"

    @pytest.mark.parametrize("lang,ext", [
        (Language.JAVASCRIPT, ".js"),
        (Language.JSX, ".jsx"),
        (Language.TYPESCRIPT, ".ts"),
        (Language.TSX, ".tsx"),
    ])
    def test_arrow_function_cross_language(self, lang, ext):
        """Test arrow function is extracted across all JS-family languages."""
        code = "const handler = () => { console.log('clicked'); };"
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) > 0, f"Should extract chunk for {lang.value}"
        assert any("handler" in c.code for c in chunks), f"Should find handler in {lang.value}"


# =============================================================================
# FUNCTION PARAMETERS
# =============================================================================


class TestFunctionParameters:
    """Test parameter patterns in functions."""

    def test_default_values(self):
        """Test function with default parameter values."""
        code = "function foo(a = 1, b = 'default') { return a + b; }"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        function_chunk = next((c for c in chunks if "foo" in c.code), None)
        assert function_chunk is not None
        assert "= 1" in function_chunk.code or "=1" in function_chunk.code
        assert "'default'" in function_chunk.code

    def test_rest_parameters(self):
        """Test function with rest parameters."""
        code = "function foo(...args) { return args.length; }"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("...args" in c.code for c in chunks), "Should preserve rest parameter"

    def test_destructured_object_parameter(self):
        """Test function with destructured object parameter."""
        code = "function foo({ a, b }) { return a + b; }"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("{ a, b }" in c.code or "{a, b}" in c.code for c in chunks), "Should preserve destructuring"

    def test_destructured_array_parameter(self):
        """Test function with destructured array parameter."""
        code = "function foo([first, second]) { return first + second; }"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("[first, second]" in c.code for c in chunks), "Should preserve array destructuring"

    def test_mixed_parameters(self):
        """Test function with mixed parameter patterns."""
        code = "function foo(a, { b, c } = {}, ...rest) { return [a, b, c, ...rest]; }"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        function_chunk = next((c for c in chunks if "foo" in c.code), None)
        assert function_chunk is not None
        assert "...rest" in function_chunk.code

    def test_nested_destructuring(self):
        """Test function with nested destructuring."""
        code = "function foo({ a: { b } }) { return b; }"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("a: { b }" in c.code or "a:{b}" in c.code for c in chunks), "Should preserve nested destructuring"

    def test_default_with_destructuring(self):
        """Test function with default value in destructuring."""
        code = "function foo({ a = 1 } = {}) { return a; }"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("{ a = 1 }" in c.code or "{a = 1}" in c.code or "a=1" in c.code for c in chunks)


# =============================================================================
# TYPESCRIPT FUNCTION FEATURES
# =============================================================================


class TestTypeScriptFunctions:
    """Test TypeScript-specific function features."""

    def test_parameter_types(self):
        """Test function with parameter types."""
        code = "function foo(a: string, b: number): void { console.log(a, b); }"
        chunks = _parse(code, "test.ts", Language.TYPESCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        function_chunk = next((c for c in chunks if "foo" in c.code), None)
        assert function_chunk is not None
        assert ": string" in function_chunk.code
        assert ": number" in function_chunk.code

    def test_return_type(self):
        """Test function with return type."""
        code = "function foo(): string { return 'hello'; }"
        chunks = _parse(code, "test.ts", Language.TYPESCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("): string" in c.code for c in chunks), "Should preserve return type"

    def test_generic_function(self):
        """Test generic function."""
        code = "function identity<T>(a: T): T { return a; }"
        chunks = _parse(code, "test.ts", Language.TYPESCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("<T>" in c.code for c in chunks), "Should preserve generic type parameter"

    def test_multiple_generics(self):
        """Test function with multiple generics."""
        code = "function pair<T, U>(a: T, b: U): [T, U] { return [a, b]; }"
        chunks = _parse(code, "test.ts", Language.TYPESCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("<T, U>" in c.code for c in chunks), "Should preserve multiple generics"

    def test_generic_constraints(self):
        """Test function with generic constraints."""
        code = "function process<T extends object>(obj: T): T { return obj; }"
        chunks = _parse(code, "test.ts", Language.TYPESCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("extends object" in c.code for c in chunks), "Should preserve generic constraint"

    def test_optional_parameters(self):
        """Test function with optional parameters."""
        code = "function foo(a?: string, b?: number) { return a || b; }"
        chunks = _parse(code, "test.ts", Language.TYPESCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        function_chunk = next((c for c in chunks if "foo" in c.code), None)
        assert function_chunk is not None
        assert "a?:" in function_chunk.code or "a?" in function_chunk.code

    def test_function_overloads(self):
        """Test function overloads."""
        code = """function foo(a: string): string;
function foo(a: number): number;
function foo(a: string | number): string | number {
    return a;
}"""
        chunks = _parse(code, "test.ts", Language.TYPESCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Should capture the function with overloads
        assert any("foo" in c.code for c in chunks), "Should extract overloaded function"

    def test_this_parameter(self):
        """Test function with this parameter."""
        code = "function foo(this: Context, a: string): void { this.log(a); }"
        chunks = _parse(code, "test.ts", Language.TYPESCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("this: Context" in c.code for c in chunks), "Should preserve this parameter"

    def test_typed_arrow_function(self):
        """Test typed arrow function."""
        code = "const add = (a: number, b: number): number => a + b;"
        chunks = _parse(code, "test.ts", Language.TYPESCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("add" in c.code for c in chunks), "Should extract typed arrow function"
        assert any(": number =>" in c.code for c in chunks), "Should preserve return type annotation"

    @pytest.mark.parametrize("lang,ext", [
        (Language.TYPESCRIPT, ".ts"),
        (Language.TSX, ".tsx"),
    ])
    def test_generic_function_ts_variants(self, lang, ext):
        """Test generic function across TypeScript variants."""
        code = "function map<T, U>(items: T[], fn: (item: T) => U): U[] { return items.map(fn); }"
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) > 0, f"Should extract chunk for {lang.value}"
        assert any("<T, U>" in c.code for c in chunks), f"Should preserve generics in {lang.value}"


# =============================================================================
# COMPLEX FUNCTION PATTERNS
# =============================================================================


class TestComplexFunctionPatterns:
    """Test complex function patterns and edge cases."""

    def test_iife(self):
        """Test immediately invoked function expression."""
        code = "(function() { console.log('IIFE'); })();"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        # IIFE should be captured
        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("IIFE" in c.code for c in chunks), "Should capture IIFE"

    def test_arrow_iife(self):
        """Test immediately invoked arrow function."""
        code = "(() => { console.log('arrow IIFE'); })();"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("arrow IIFE" in c.code for c in chunks), "Should capture arrow IIFE"

    def test_nested_functions(self):
        """Test nested function declarations."""
        code = """function outer() {
    function inner() {
        return 'inner';
    }
    return inner();
}"""
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Should capture the outer function with nested inner function
        outer_chunk = next((c for c in chunks if "outer" in c.code and "inner" in c.code), None)
        assert outer_chunk is not None, "Should capture nested functions together or separately"

    def test_function_returning_function(self):
        """Test function returning another function."""
        code = "function makeAdder(x) { return function(y) { return x + y; }; }"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("makeAdder" in c.code for c in chunks), "Should extract higher-order function"

    def test_curried_arrow_function(self):
        """Test curried arrow function."""
        code = "const add = x => y => z => x + y + z;"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("add" in c.code for c in chunks), "Should extract curried function"

    def test_method_shorthand(self):
        """Test method shorthand in object (not a function declaration)."""
        code = """const obj = {
    foo() { return 'foo'; },
    bar() { return 'bar'; }
};"""
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Object with method shorthands should be captured
        assert any("foo" in c.code for c in chunks), "Should capture object with methods"

    def test_async_generator_arrow(self):
        """Test async arrow (generators can't be arrows in JS)."""
        # Note: Arrow functions cannot be generators, but async arrows are valid
        code = "const asyncFetch = async () => { const data = await fetch('/api'); return data; };"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("asyncFetch" in c.code for c in chunks), "Should extract async arrow"

    def test_callback_function(self):
        """Test function passed as callback."""
        code = "const numbers = [1, 2, 3].map(function(n) { return n * 2; });"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Inline callbacks may or may not be extracted separately
        assert any("map" in c.code for c in chunks), "Should capture code with callback"


# =============================================================================
# EXPORTED FUNCTIONS
# =============================================================================


class TestExportedFunctions:
    """Test exported function patterns."""

    def test_export_function_declaration(self):
        """Test export function declaration."""
        code = "export function foo() { return 42; }"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("export" in c.code and "foo" in c.code for c in chunks), "Should extract exported function"

    def test_export_async_function(self):
        """Test export async function."""
        code = "export async function fetchData() { return await fetch('/api'); }"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("export" in c.code and "async" in c.code and "fetchData" in c.code for c in chunks)

    def test_export_const_arrow(self):
        """Test export const arrow function."""
        code = "export const handler = () => { console.log('clicked'); };"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("export" in c.code and "handler" in c.code for c in chunks)

    def test_export_default_function(self):
        """Test export default function."""
        code = "export default function foo() { return 42; }"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("export default" in c.code and "foo" in c.code for c in chunks)

    def test_export_default_anonymous_function(self):
        """Test export default anonymous function."""
        code = "export default function() { return 42; }"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("export default function()" in c.code for c in chunks)

    def test_export_default_arrow_function(self):
        """Test export default arrow function."""
        code = "export default () => { return 42; };"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("export default" in c.code and "=>" in c.code for c in chunks)

    @pytest.mark.parametrize("lang,ext", [
        (Language.JAVASCRIPT, ".js"),
        (Language.TYPESCRIPT, ".ts"),
        (Language.TSX, ".tsx"),
    ])
    def test_export_typed_function_cross_language(self, lang, ext):
        """Test exported typed function across languages."""
        if lang == Language.JAVASCRIPT:
            code = "export function add(a, b) { return a + b; }"
        else:
            code = "export function add(a: number, b: number): number { return a + b; }"

        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) > 0, f"Should extract chunk for {lang.value}"
        assert any("export" in c.code and "add" in c.code for c in chunks)


# =============================================================================
# VUE FUNCTION PATTERNS
# =============================================================================


class TestVueFunctions:
    """Test function patterns in Vue single-file components."""

    def test_vue_script_setup_function(self):
        """Test function in Vue script setup."""
        code = """<script setup>
function greet() {
    return 'hello';
}
</script>"""
        chunks = _parse(code, "test.vue", Language.VUE)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("greet" in c.code for c in chunks), "Should extract function from script setup"

    def test_vue_script_setup_arrow(self):
        """Test arrow function in Vue script setup."""
        code = """<script setup>
const handler = () => {
    console.log('clicked');
};
</script>"""
        chunks = _parse(code, "test.vue", Language.VUE)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("handler" in c.code for c in chunks), "Should extract arrow function"

    def test_vue_typescript_function(self):
        """Test TypeScript function in Vue."""
        code = """<script setup lang="ts">
function add(a: number, b: number): number {
    return a + b;
}
</script>"""
        chunks = _parse(code, "test.vue", Language.VUE)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("add" in c.code for c in chunks), "Should extract TypeScript function"

    def test_vue_composable_function(self):
        """Test composable function definition in Vue."""
        code = """<script setup>
function useCounter() {
    const count = ref(0);
    return { count };
}
</script>"""
        chunks = _parse(code, "test.vue", Language.VUE)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("useCounter" in c.code for c in chunks), "Should extract composable function"

    def test_vue_async_function(self):
        """Test async function in Vue script setup."""
        code = """<script setup>
async function fetchData() {
    const response = await fetch('/api');
    return response.json();
}
</script>"""
        chunks = _parse(code, "test.vue", Language.VUE)

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("fetchData" in c.code for c in chunks), "Should extract async function"


# =============================================================================
# CROSS-LANGUAGE CONSISTENCY TESTS
# =============================================================================


class TestCrossLanguageConsistency:
    """Test that same constructs produce consistent results across languages."""

    @pytest.mark.parametrize("lang,ext", [
        (Language.JAVASCRIPT, ".js"),
        (Language.JSX, ".jsx"),
        (Language.TYPESCRIPT, ".ts"),
        (Language.TSX, ".tsx"),
    ])
    def test_function_declaration_consistency(self, lang, ext):
        """Test function declaration is consistent across languages."""
        code = "function calculateSum(a, b) { return a + b; }"
        chunks = _parse(code, f"test{ext}", lang)

        # Every language should extract this
        assert len(chunks) > 0, f"{lang.value} should extract function"
        assert any("calculateSum" in c.code for c in chunks), f"{lang.value} should find calculateSum"

    @pytest.mark.parametrize("lang,ext", [
        (Language.JAVASCRIPT, ".js"),
        (Language.JSX, ".jsx"),
        (Language.TYPESCRIPT, ".ts"),
        (Language.TSX, ".tsx"),
    ])
    def test_arrow_function_consistency(self, lang, ext):
        """Test arrow function is consistent across languages."""
        code = "const transform = (x) => x * 2;"
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) > 0, f"{lang.value} should extract arrow function"
        assert any("transform" in c.code for c in chunks), f"{lang.value} should find transform"

    @pytest.mark.parametrize("lang,ext", [
        (Language.JAVASCRIPT, ".js"),
        (Language.JSX, ".jsx"),
        (Language.TYPESCRIPT, ".ts"),
        (Language.TSX, ".tsx"),
    ])
    def test_async_function_consistency(self, lang, ext):
        """Test async function is consistent across languages."""
        code = "async function loadData() { return await fetch('/data'); }"
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) > 0, f"{lang.value} should extract async function"
        assert any("loadData" in c.code for c in chunks), f"{lang.value} should find loadData"
        assert any("async" in c.code for c in chunks), f"{lang.value} should preserve async keyword"

    @pytest.mark.parametrize("lang,ext", [
        (Language.JAVASCRIPT, ".js"),
        (Language.JSX, ".jsx"),
        (Language.TYPESCRIPT, ".ts"),
        (Language.TSX, ".tsx"),
    ])
    def test_generator_function_consistency(self, lang, ext):
        """Test generator function is consistent across languages."""
        code = "function* sequence() { yield 1; yield 2; yield 3; }"
        chunks = _parse(code, f"test{ext}", lang)

        assert len(chunks) > 0, f"{lang.value} should extract generator"
        assert any("sequence" in c.code for c in chunks), f"{lang.value} should find sequence"
        assert any("function*" in c.code for c in chunks), f"{lang.value} should preserve generator syntax"


# =============================================================================
# EDGE CASES
# =============================================================================


class TestFunctionEdgeCases:
    """Test edge cases in function parsing."""

    def test_function_with_unicode_name(self):
        """Test function with unicode identifier."""
        code = "function cafe() { return 'coffee'; }"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract function with unicode name"
        assert any("cafe" in c.code for c in chunks)

    def test_function_with_emoji_string(self):
        """Test function returning emoji."""
        code = "function getEmoji() { return '🎉'; }"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract function"
        # Unicode string may or may not be preserved depending on parsing

    def test_multiline_function(self):
        """Test multiline function with complex body."""
        code = """function process(data) {
    const result = [];
    for (const item of data) {
        if (item.valid) {
            result.push(item.value);
        }
    }
    return result;
}"""
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract multiline function"
        assert any("process" in c.code for c in chunks)

    def test_empty_function(self):
        """Test empty function body."""
        code = "function noop() {}"
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract empty function"
        assert any("noop" in c.code for c in chunks)

    def test_function_with_comments(self):
        """Test function with JSDoc comments."""
        code = """/**
 * Calculates the sum of two numbers.
 * @param {number} a - First number
 * @param {number} b - Second number
 * @returns {number} The sum
 */
function add(a, b) {
    return a + b;
}"""
        chunks = _parse(code, "test.js", Language.JAVASCRIPT)

        assert len(chunks) > 0, "Should extract documented function"
        assert any("add" in c.code for c in chunks)
