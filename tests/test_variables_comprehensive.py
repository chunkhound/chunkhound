"""Comprehensive variable parsing tests for JS-family parsers.

This module tests variable extraction across JavaScript, JSX, TypeScript, TSX, and Vue.
Tests cover declaration types, initializer types, multiple declarations, destructuring,
and TypeScript-specific variable features.

Based on: docs/js-family-parser-test-specification.md (Section 5: Variables)
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


def _get_symbols(chunks):
    """Extract symbol names from chunks, handling None values."""
    return {c.symbol for c in chunks if c.symbol}


# =============================================================================
# DECLARATION TYPES
# =============================================================================

class TestDeclarationTypes:
    """Tests for const/let/var declarations across JS-family languages."""

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_const_object_literal_extracted(self, language, ext):
        """Const with object literal should be extracted."""
        code = "const config = { api: 'http://localhost', timeout: 5000 }"
        chunks = _parse(code, f"test.{ext}", language)
        symbols = _get_symbols(chunks)
        assert "config" in symbols, \
            f"Const object literal 'config' should be extracted for {language}"
        assert any("api" in c.code and "timeout" in c.code for c in chunks), \
            "Object content should be in chunk"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_let_object_literal_extracted(self, language, ext):
        """Let with object literal should be extracted."""
        code = "let settings = { debug: true, verbose: false }"
        chunks = _parse(code, f"test.{ext}", language)
        assert "settings" in _get_symbols(chunks), \
            f"Let object literal 'settings' should be extracted for {language}"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_var_object_literal_extracted(self, language, ext):
        """Var with object literal should be extracted (known gap to verify fix)."""
        code = "var legacyConfig = { mode: 'compat', version: 1 }"
        chunks = _parse(code, f"test.{ext}", language)
        assert "legacyConfig" in _get_symbols(chunks), \
            f"Var object literal 'legacyConfig' should be extracted for {language}"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_uninitialized_let_may_not_extract(self, language, ext):
        """Uninitialized let declarations may or may not be extracted."""
        code = "let uninitializedVar"
        chunks = _parse(code, f"test.{ext}", language)
        # Document behavior - uninitialized vars may not be extracted
        # This is acceptable behavior

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_uninitialized_var_may_not_extract(self, language, ext):
        """Uninitialized var declarations may or may not be extracted."""
        code = "var uninitializedVar"
        chunks = _parse(code, f"test.{ext}", language)
        # Document behavior - uninitialized vars may not be extracted


# =============================================================================
# INITIALIZER TYPES
# =============================================================================

class TestInitializerTypes:
    """Tests for different initializer types in variable declarations."""

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_object_literal_extracted(self, language, ext):
        """Object literals should be extracted."""
        code = "const obj = { key: 'value', nested: { inner: true } }"
        chunks = _parse(code, f"test.{ext}", language)
        assert "obj" in _get_symbols(chunks), \
            f"Object literal 'obj' should be extracted for {language}"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_array_literal_extracted(self, language, ext):
        """Array literals should be extracted."""
        code = "const arr = [1, 2, 3, { id: 1 }, ['nested']]"
        chunks = _parse(code, f"test.{ext}", language)
        assert "arr" in _get_symbols(chunks), \
            f"Array literal 'arr' should be extracted for {language}"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_function_expression_extracted(self, language, ext):
        """Function expressions should be extracted."""
        code = "const fn = function() { return 42 }"
        chunks = _parse(code, f"test.{ext}", language)
        assert "fn" in _get_symbols(chunks), \
            f"Function expression 'fn' should be extracted for {language}"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_arrow_function_extracted(self, language, ext):
        """Arrow functions should be extracted."""
        code = "const arrowFn = () => { return 'arrow' }"
        chunks = _parse(code, f"test.{ext}", language)
        assert "arrowFn" in _get_symbols(chunks), \
            f"Arrow function 'arrowFn' should be extracted for {language}"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_class_expression_extracted(self, language, ext):
        """Class expressions should be extracted."""
        code = "const MyClass = class { constructor() { this.value = 1 } }"
        chunks = _parse(code, f"test.{ext}", language)
        assert "MyClass" in _get_symbols(chunks), \
            f"Class expression 'MyClass' should be extracted for {language}"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_simple_string_may_not_extract(self, language, ext):
        """Simple string literals may not be extracted (filtering behavior)."""
        code = "const str = 'hello'"
        chunks = _parse(code, f"test.{ext}", language)
        # Simple primitives may be filtered out - document behavior

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_simple_number_may_not_extract(self, language, ext):
        """Simple number literals may not be extracted (filtering behavior)."""
        code = "const num = 42"
        chunks = _parse(code, f"test.{ext}", language)
        # Simple primitives may be filtered out - document behavior

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_template_literal_may_extract(self, language, ext):
        """Template literals may be extracted."""
        code = "const tpl = `Hello ${name}, welcome!`"
        chunks = _parse(code, f"test.{ext}", language)
        # Document behavior for template literals

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_regexp_literal_may_extract(self, language, ext):
        """RegExp literals may be extracted."""
        code = "const pattern = /[a-z]+/gi"
        chunks = _parse(code, f"test.{ext}", language)
        # Document behavior for regexp literals

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_bigint_literal_may_extract(self, language, ext):
        """BigInt literals may be extracted."""
        code = "const big = 9007199254740991n"
        chunks = _parse(code, f"test.{ext}", language)
        # Document behavior for BigInt literals


# =============================================================================
# MULTIPLE DECLARATIONS
# =============================================================================

class TestMultipleDeclarations:
    """Tests for multiple variable declarations in a single statement."""

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_multiple_const_objects_extracted(self, language, ext):
        """Multiple const declarations with objects should extract at least one."""
        code = "const a = { x: 1 }, b = { y: 2 }, c = { z: 3 }"
        chunks = _parse(code, f"test.{ext}", language)
        # At least the first or all should be extracted
        symbols = _get_symbols(chunks)
        assert "a" in symbols or "b" in symbols or "c" in symbols, \
            f"At least one of multiple const declarations should be extracted for {language}"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_multiple_let_uninitialized(self, language, ext):
        """Multiple uninitialized let declarations."""
        code = "let x, y, z"
        chunks = _parse(code, f"test.{ext}", language)
        # Uninitialized declarations may not be extracted

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_mixed_initialization(self, language, ext):
        """Mixed initialized and uninitialized in same statement."""
        code = "var i = 0, len"
        chunks = _parse(code, f"test.{ext}", language)
        # The initialized one may be extracted


# =============================================================================
# DESTRUCTURING - OBJECT
# =============================================================================

class TestObjectDestructuring:
    """Tests for object destructuring patterns."""

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_object_destructuring_basic(self, language, ext):
        """Basic object destructuring from object literal."""
        code = """
const source = { a: 1, b: 2, c: 3 }
const { a, b } = source
"""
        chunks = _parse(code, f"test.{ext}", language)
        # The source object should be extracted
        assert "source" in _get_symbols(chunks), \
            f"Source object should be extracted for {language}"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_object_destructuring_with_defaults(self, language, ext):
        """Object destructuring with default values."""
        code = "const { a = 1, b = 'default' } = obj"
        chunks = _parse(code, f"test.{ext}", language)
        # Document behavior for destructuring with defaults

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_object_destructuring_with_rename(self, language, ext):
        """Object destructuring with property renaming."""
        code = "const { originalName: renamedVar } = obj"
        chunks = _parse(code, f"test.{ext}", language)
        # Document behavior for renaming destructuring

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_object_destructuring_with_rest(self, language, ext):
        """Object destructuring with rest element."""
        code = "const { a, ...rest } = obj"
        chunks = _parse(code, f"test.{ext}", language)
        # Document behavior for rest destructuring

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_nested_object_destructuring(self, language, ext):
        """Nested object destructuring."""
        code = "const { outer: { inner } } = obj"
        chunks = _parse(code, f"test.{ext}", language)
        # Document behavior for nested destructuring


# =============================================================================
# DESTRUCTURING - ARRAY
# =============================================================================

class TestArrayDestructuring:
    """Tests for array destructuring patterns."""

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_array_destructuring_basic(self, language, ext):
        """Basic array destructuring from array literal."""
        code = """
const source = [1, 2, 3]
const [first, second] = source
"""
        chunks = _parse(code, f"test.{ext}", language)
        # The source array should be extracted
        assert "source" in _get_symbols(chunks), \
            f"Source array should be extracted for {language}"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_array_destructuring_with_skip(self, language, ext):
        """Array destructuring with skipped elements."""
        code = "const [first, , third] = arr"
        chunks = _parse(code, f"test.{ext}", language)
        # Document behavior for skipping elements

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_array_destructuring_with_rest(self, language, ext):
        """Array destructuring with rest element."""
        code = "const [first, ...rest] = arr"
        chunks = _parse(code, f"test.{ext}", language)
        # Document behavior for array rest

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_array_destructuring_with_defaults(self, language, ext):
        """Array destructuring with default values."""
        code = "const [first = 'default', second = 0] = arr"
        chunks = _parse(code, f"test.{ext}", language)
        # Document behavior for array defaults

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_nested_array_destructuring(self, language, ext):
        """Nested array destructuring."""
        code = "const [[a, b], [c, d]] = arr"
        chunks = _parse(code, f"test.{ext}", language)
        # Document behavior for nested array destructuring


# =============================================================================
# MIXED DESTRUCTURING
# =============================================================================

class TestMixedDestructuring:
    """Tests for complex mixed destructuring patterns."""

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_object_with_array_property(self, language, ext):
        """Object destructuring with array inside."""
        code = "const { items: [first, second] } = obj"
        chunks = _parse(code, f"test.{ext}", language)
        # Document behavior for mixed destructuring

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_array_with_object_elements(self, language, ext):
        """Array destructuring with object inside."""
        code = "const [{ name }, { value }] = arr"
        chunks = _parse(code, f"test.{ext}", language)
        # Document behavior for mixed destructuring


# =============================================================================
# TYPESCRIPT-SPECIFIC VARIABLES
# =============================================================================

class TestTypeScriptVariables:
    """Tests for TypeScript-specific variable features."""

    @pytest.mark.parametrize("language,ext", [
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_type_annotation_object(self, language, ext):
        """Variable with type annotation for object."""
        code = "const user: { name: string; age: number } = { name: 'John', age: 30 }"
        chunks = _parse(code, f"test.{ext}", language)
        assert "user" in _get_symbols(chunks), \
            f"Typed object variable 'user' should be extracted for {language}"

    @pytest.mark.parametrize("language,ext", [
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_type_annotation_array(self, language, ext):
        """Variable with type annotation for array."""
        code = "const items: string[] = ['a', 'b', 'c']"
        chunks = _parse(code, f"test.{ext}", language)
        assert "items" in _get_symbols(chunks), \
            f"Typed array variable 'items' should be extracted for {language}"

    @pytest.mark.parametrize("language,ext", [
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_complex_type_generic(self, language, ext):
        """Variable with complex generic type."""
        code = "const map: Map<string, number> = new Map()"
        chunks = _parse(code, f"test.{ext}", language)
        # Document behavior for complex generic types

    @pytest.mark.parametrize("language,ext", [
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_as_const_assertion(self, language, ext):
        """Variable with 'as const' assertion."""
        code = "const config = { api: 'http://localhost', timeout: 5000 } as const"
        chunks = _parse(code, f"test.{ext}", language)
        assert "config" in _get_symbols(chunks), \
            f"'as const' variable 'config' should be extracted for {language}"

    @pytest.mark.parametrize("language,ext", [
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_satisfies_operator(self, language, ext):
        """Variable with 'satisfies' operator."""
        code = """
type Config = { api: string; timeout: number }
const config = { api: 'http://localhost', timeout: 5000 } satisfies Config
"""
        chunks = _parse(code, f"test.{ext}", language)
        assert "config" in _get_symbols(chunks), \
            f"'satisfies' variable 'config' should be extracted for {language}"

    @pytest.mark.parametrize("language,ext", [
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_union_type_annotation(self, language, ext):
        """Variable with union type annotation."""
        code = "const value: string | number = getValue()"
        chunks = _parse(code, f"test.{ext}", language)
        # Document behavior for union types

    @pytest.mark.parametrize("language,ext", [
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_intersection_type_annotation(self, language, ext):
        """Variable with intersection type annotation."""
        code = "const combo: TypeA & TypeB = getCombined()"
        chunks = _parse(code, f"test.{ext}", language)
        # Document behavior for intersection types

    @pytest.mark.parametrize("language,ext", [
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_readonly_tuple_type(self, language, ext):
        """Variable with readonly tuple type."""
        code = "const tuple: readonly [string, number] = ['hello', 42]"
        chunks = _parse(code, f"test.{ext}", language)
        # Document behavior for readonly tuple types


# =============================================================================
# VUE VARIABLES
# =============================================================================

class TestVueVariables:
    """Tests for variable extraction in Vue SFC files."""

    def test_vue_script_setup_const_object(self):
        """Const object in Vue script setup should be extracted."""
        code = """
<script setup>
const config = { api: 'http://localhost' }
</script>

<template>
  <div>{{ config.api }}</div>
</template>
"""
        chunks = _parse(code, "test.vue", Language.VUE)
        assert "config" in _get_symbols(chunks), \
            "Vue script setup const object 'config' should be extracted"

    def test_vue_script_setup_array(self):
        """Array in Vue script setup should be extracted."""
        code = """
<script setup>
const items = ['item1', 'item2', 'item3']
</script>

<template>
  <ul>
    <li v-for="item in items" :key="item">{{ item }}</li>
  </ul>
</template>
"""
        chunks = _parse(code, "test.vue", Language.VUE)
        assert "items" in _get_symbols(chunks), \
            "Vue script setup array 'items' should be extracted"

    def test_vue_typescript_typed_variable(self):
        """TypeScript typed variable in Vue script setup."""
        code = """
<script setup lang="ts">
interface User {
  name: string
  age: number
}
const user: User = { name: 'John', age: 30 }
</script>

<template>
  <div>{{ user.name }}</div>
</template>
"""
        chunks = _parse(code, "test.vue", Language.VUE)
        assert "user" in _get_symbols(chunks), \
            "Vue TypeScript typed variable 'user' should be extracted"

    def test_vue_reactive_ref(self):
        """Reactive ref variable in Vue script setup."""
        code = """
<script setup>
import { ref } from 'vue'
const count = ref(0)
const config = { initial: 0 }
</script>

<template>
  <button @click="count++">{{ count }}</button>
</template>
"""
        chunks = _parse(code, "test.vue", Language.VUE)
        # Ref declarations and config should be extracted
        assert "config" in _get_symbols(chunks), \
            "Vue config object should be extracted"

    def test_vue_computed_variable(self):
        """Computed variable in Vue script setup."""
        code = """
<script setup>
import { ref, computed } from 'vue'
const count = ref(0)
const doubled = computed(() => count.value * 2)
const config = { multiplier: 2 }
</script>

<template>
  <div>{{ doubled }}</div>
</template>
"""
        chunks = _parse(code, "test.vue", Language.VUE)
        assert "config" in _get_symbols(chunks), \
            "Vue config object should be extracted"


# =============================================================================
# EXPORTED VARIABLES
# =============================================================================

class TestExportedVariables:
    """Tests for exported variable declarations."""

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_export_const_object(self, language, ext):
        """Export const with object literal."""
        code = "export const config = { api: 'http://localhost' }"
        chunks = _parse(code, f"test.{ext}", language)
        assert "config" in _get_symbols(chunks), \
            f"Exported const object 'config' should be extracted for {language}"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_export_const_array(self, language, ext):
        """Export const with array literal."""
        code = "export const items = ['a', 'b', 'c']"
        chunks = _parse(code, f"test.{ext}", language)
        assert "items" in _get_symbols(chunks), \
            f"Exported const array 'items' should be extracted for {language}"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_export_let_object(self, language, ext):
        """Export let with object literal."""
        code = "export let settings = { debug: true }"
        chunks = _parse(code, f"test.{ext}", language)
        assert "settings" in _get_symbols(chunks), \
            f"Exported let object 'settings' should be extracted for {language}"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_export_var_object(self, language, ext):
        """Export var with object literal."""
        code = "export var legacy = { compat: true }"
        chunks = _parse(code, f"test.{ext}", language)
        assert "legacy" in _get_symbols(chunks), \
            f"Exported var object 'legacy' should be extracted for {language}"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_export_default_object(self, language, ext):
        """Export default object literal."""
        code = "export default { api: 'http://localhost', timeout: 5000 }"
        chunks = _parse(code, f"test.{ext}", language)
        assert len(chunks) > 0, \
            f"Export default object should be extracted for {language}"
        assert any("api" in c.code for c in chunks), \
            "Object content should be in chunk"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.JSX, "jsx"),
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_export_default_array(self, language, ext):
        """Export default array literal."""
        code = "export default [1, 2, 3, { id: 1 }]"
        chunks = _parse(code, f"test.{ext}", language)
        assert len(chunks) > 0, \
            f"Export default array should be extracted for {language}"


# =============================================================================
# COMMONJS VARIABLES
# =============================================================================

class TestCommonJSVariables:
    """Tests for CommonJS-style variable exports."""

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.TYPESCRIPT, "ts"),
    ])
    def test_module_exports_object(self, language, ext):
        """module.exports object literal."""
        code = "module.exports = { api: 'http://localhost', timeout: 5000 }"
        chunks = _parse(code, f"test.{ext}", language)
        assert len(chunks) > 0, \
            f"module.exports object should be extracted for {language}"
        assert any("api" in c.code for c in chunks), \
            "Object content should be in chunk"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.TYPESCRIPT, "ts"),
    ])
    def test_module_exports_array(self, language, ext):
        """module.exports array literal."""
        code = "module.exports = [1, 2, 3]"
        chunks = _parse(code, f"test.{ext}", language)
        assert len(chunks) > 0, \
            f"module.exports array should be extracted for {language}"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.TYPESCRIPT, "ts"),
    ])
    def test_exports_property_assignment(self, language, ext):
        """exports.property = value pattern."""
        code = """
exports.config = { api: 'http://localhost' }
exports.items = ['a', 'b', 'c']
"""
        chunks = _parse(code, f"test.{ext}", language)
        # Document behavior for exports.property pattern


# =============================================================================
# CROSS-LANGUAGE CONSISTENCY
# =============================================================================

class TestCrossLanguageConsistency:
    """Tests to verify consistent behavior across JS-family languages."""

    def test_object_literal_consistency(self):
        """Object literal extraction should be consistent across languages."""
        code = "const config = { api: 'http://localhost', timeout: 5000 }"

        js_chunks = _parse(code, "test.js", Language.JAVASCRIPT)
        jsx_chunks = _parse(code, "test.jsx", Language.JSX)
        ts_chunks = _parse(code, "test.ts", Language.TYPESCRIPT)
        tsx_chunks = _parse(code, "test.tsx", Language.TSX)

        # All should extract the config variable
        js_names = _get_symbols(js_chunks)
        jsx_names = _get_symbols(jsx_chunks)
        ts_names = _get_symbols(ts_chunks)
        tsx_names = _get_symbols(tsx_chunks)

        assert "config" in js_names, "JS should extract 'config'"
        assert "config" in jsx_names, "JSX should extract 'config'"
        assert "config" in ts_names, "TS should extract 'config'"
        assert "config" in tsx_names, "TSX should extract 'config'"

    def test_array_literal_consistency(self):
        """Array literal extraction should be consistent across languages."""
        code = "const items = ['a', 'b', 'c']"

        js_chunks = _parse(code, "test.js", Language.JAVASCRIPT)
        jsx_chunks = _parse(code, "test.jsx", Language.JSX)
        ts_chunks = _parse(code, "test.ts", Language.TYPESCRIPT)
        tsx_chunks = _parse(code, "test.tsx", Language.TSX)

        js_names = _get_symbols(js_chunks)
        jsx_names = _get_symbols(jsx_chunks)
        ts_names = _get_symbols(ts_chunks)
        tsx_names = _get_symbols(tsx_chunks)

        assert "items" in js_names, "JS should extract 'items'"
        assert "items" in jsx_names, "JSX should extract 'items'"
        assert "items" in ts_names, "TS should extract 'items'"
        assert "items" in tsx_names, "TSX should extract 'items'"

    def test_arrow_function_consistency(self):
        """Arrow function extraction should be consistent across languages."""
        code = "const handler = () => { return 'result' }"

        js_chunks = _parse(code, "test.js", Language.JAVASCRIPT)
        jsx_chunks = _parse(code, "test.jsx", Language.JSX)
        ts_chunks = _parse(code, "test.ts", Language.TYPESCRIPT)
        tsx_chunks = _parse(code, "test.tsx", Language.TSX)

        js_names = _get_symbols(js_chunks)
        jsx_names = _get_symbols(jsx_chunks)
        ts_names = _get_symbols(ts_chunks)
        tsx_names = _get_symbols(tsx_chunks)

        assert "handler" in js_names, "JS should extract 'handler'"
        assert "handler" in jsx_names, "JSX should extract 'handler'"
        assert "handler" in ts_names, "TS should extract 'handler'"
        assert "handler" in tsx_names, "TSX should extract 'handler'"

    def test_export_const_consistency(self):
        """Exported const extraction should be consistent across languages."""
        code = "export const settings = { debug: true, verbose: false }"

        js_chunks = _parse(code, "test.js", Language.JAVASCRIPT)
        jsx_chunks = _parse(code, "test.jsx", Language.JSX)
        ts_chunks = _parse(code, "test.ts", Language.TYPESCRIPT)
        tsx_chunks = _parse(code, "test.tsx", Language.TSX)

        js_names = _get_symbols(js_chunks)
        jsx_names = _get_symbols(jsx_chunks)
        ts_names = _get_symbols(ts_chunks)
        tsx_names = _get_symbols(tsx_chunks)

        assert "settings" in js_names, "JS should extract 'settings'"
        assert "settings" in jsx_names, "JSX should extract 'settings'"
        assert "settings" in ts_names, "TS should extract 'settings'"
        assert "settings" in tsx_names, "TSX should extract 'settings'"


# =============================================================================
# COMPLEX REAL-WORLD PATTERNS
# =============================================================================

class TestRealWorldPatterns:
    """Tests for real-world variable patterns commonly found in codebases."""

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.TYPESCRIPT, "ts"),
    ])
    def test_config_file_pattern(self, language, ext):
        """Typical config file pattern."""
        code = """
const config = {
  database: {
    host: process.env.DB_HOST || 'localhost',
    port: parseInt(process.env.DB_PORT || '5432'),
    name: process.env.DB_NAME || 'mydb'
  },
  server: {
    port: parseInt(process.env.PORT || '3000'),
    host: '0.0.0.0'
  },
  features: {
    enableAuth: true,
    enableCache: false
  }
}

module.exports = config
"""
        chunks = _parse(code, f"config.{ext}", language)
        assert "config" in _get_symbols(chunks), \
            f"Config object should be extracted for {language}"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.TYPESCRIPT, "ts"),
    ])
    def test_constants_file_pattern(self, language, ext):
        """Typical constants file pattern."""
        code = """
export const API_ENDPOINTS = {
  users: '/api/users',
  products: '/api/products',
  orders: '/api/orders'
}

export const HTTP_STATUS = {
  OK: 200,
  CREATED: 201,
  BAD_REQUEST: 400,
  UNAUTHORIZED: 401,
  NOT_FOUND: 404,
  SERVER_ERROR: 500
}

export const ROUTES = [
  { path: '/', component: 'Home' },
  { path: '/about', component: 'About' },
  { path: '/contact', component: 'Contact' }
]
"""
        chunks = _parse(code, f"constants.{ext}", language)
        names = _get_symbols(chunks)
        assert "API_ENDPOINTS" in names, "API_ENDPOINTS should be extracted"
        assert "HTTP_STATUS" in names, "HTTP_STATUS should be extracted"
        assert "ROUTES" in names, "ROUTES should be extracted"

    @pytest.mark.parametrize("language,ext", [
        (Language.TYPESCRIPT, "ts"),
        (Language.TSX, "tsx"),
    ])
    def test_typed_constants_pattern(self, language, ext):
        """TypeScript typed constants pattern."""
        code = """
interface ApiConfig {
  baseUrl: string
  timeout: number
  retries: number
}

const API_CONFIG: ApiConfig = {
  baseUrl: 'https://api.example.com',
  timeout: 5000,
  retries: 3
}

type Theme = 'light' | 'dark' | 'system'

const DEFAULT_THEME: Theme = 'system'

const THEME_CONFIG = {
  light: { bg: '#fff', text: '#000' },
  dark: { bg: '#000', text: '#fff' }
} as const
"""
        chunks = _parse(code, f"config.{ext}", language)
        names = _get_symbols(chunks)
        assert "API_CONFIG" in names, "API_CONFIG should be extracted"
        assert "THEME_CONFIG" in names, "THEME_CONFIG should be extracted"

    def test_vue_composable_pattern(self):
        """Vue composable with multiple variables pattern."""
        code = """
<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'

interface User {
  id: number
  name: string
  email: string
}

const users = ref<User[]>([])
const isLoading = ref(false)
const error = ref<string | null>(null)

const config = {
  apiUrl: 'https://api.example.com',
  pageSize: 10
}

const userCount = computed(() => users.value.length)
const hasUsers = computed(() => users.value.length > 0)

async function fetchUsers() {
  isLoading.value = true
  try {
    const response = await fetch(`${config.apiUrl}/users`)
    users.value = await response.json()
  } catch (e) {
    error.value = e.message
  } finally {
    isLoading.value = false
  }
}

onMounted(fetchUsers)
</script>

<template>
  <div>
    <div v-if="isLoading">Loading...</div>
    <div v-else-if="error">{{ error }}</div>
    <ul v-else>
      <li v-for="user in users" :key="user.id">{{ user.name }}</li>
    </ul>
    <p>Total users: {{ userCount }}</p>
  </div>
</template>
"""
        chunks = _parse(code, "UserList.vue", Language.VUE)
        names = _get_symbols(chunks)
        assert "config" in names, "Config object should be extracted"
        assert "fetchUsers" in names, "Function should be extracted"


# =============================================================================
# EDGE CASES
# =============================================================================

class TestVariableEdgeCases:
    """Tests for edge cases in variable extraction."""

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.TYPESCRIPT, "ts"),
    ])
    def test_empty_object(self, language, ext):
        """Empty object literal."""
        code = "const empty = {}"
        chunks = _parse(code, f"test.{ext}", language)
        # Document behavior for empty objects

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.TYPESCRIPT, "ts"),
    ])
    def test_empty_array(self, language, ext):
        """Empty array literal."""
        code = "const empty = []"
        chunks = _parse(code, f"test.{ext}", language)
        # Document behavior for empty arrays

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.TYPESCRIPT, "ts"),
    ])
    def test_deeply_nested_object(self, language, ext):
        """Deeply nested object literal."""
        code = """
const nested = {
  level1: {
    level2: {
      level3: {
        level4: {
          value: 'deep'
        }
      }
    }
  }
}
"""
        chunks = _parse(code, f"test.{ext}", language)
        assert "nested" in _get_symbols(chunks), \
            f"Deeply nested object should be extracted for {language}"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.TYPESCRIPT, "ts"),
    ])
    def test_computed_property_names(self, language, ext):
        """Object with computed property names."""
        code = """
const key = 'dynamicKey'
const obj = {
  [key]: 'value',
  [`prefix_${key}`]: 'another'
}
"""
        chunks = _parse(code, f"test.{ext}", language)
        assert "obj" in _get_symbols(chunks), \
            f"Object with computed properties should be extracted for {language}"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.TYPESCRIPT, "ts"),
    ])
    def test_shorthand_property_names(self, language, ext):
        """Object with shorthand property names."""
        code = """
const name = 'John'
const age = 30
const user = { name, age }
"""
        chunks = _parse(code, f"test.{ext}", language)
        assert "user" in _get_symbols(chunks), \
            f"Object with shorthand properties should be extracted for {language}"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.TYPESCRIPT, "ts"),
    ])
    def test_spread_in_object(self, language, ext):
        """Object with spread operator."""
        code = """
const defaults = { a: 1, b: 2 }
const extended = { ...defaults, c: 3 }
"""
        chunks = _parse(code, f"test.{ext}", language)
        names = _get_symbols(chunks)
        assert "defaults" in names or "extended" in names, \
            f"At least one object should be extracted for {language}"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.TYPESCRIPT, "ts"),
    ])
    def test_spread_in_array(self, language, ext):
        """Array with spread operator."""
        code = """
const first = [1, 2, 3]
const combined = [...first, 4, 5, 6]
"""
        chunks = _parse(code, f"test.{ext}", language)
        names = _get_symbols(chunks)
        assert "first" in names or "combined" in names, \
            f"At least one array should be extracted for {language}"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.TYPESCRIPT, "ts"),
    ])
    def test_method_shorthand_in_object(self, language, ext):
        """Object with method shorthand."""
        code = """
const obj = {
  name: 'test',
  getValue() {
    return this.name
  },
  async fetchData() {
    return await fetch('/api')
  }
}
"""
        chunks = _parse(code, f"test.{ext}", language)
        assert "obj" in _get_symbols(chunks), \
            f"Object with methods should be extracted for {language}"

    @pytest.mark.parametrize("language,ext", [
        (Language.JAVASCRIPT, "js"),
        (Language.TYPESCRIPT, "ts"),
    ])
    def test_getter_setter_in_object(self, language, ext):
        """Object with getter and setter."""
        code = """
const obj = {
  _value: 0,
  get value() {
    return this._value
  },
  set value(v) {
    this._value = v
  }
}
"""
        chunks = _parse(code, f"test.{ext}", language)
        assert "obj" in _get_symbols(chunks), \
            f"Object with getter/setter should be extracted for {language}"
