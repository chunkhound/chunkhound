"""Comprehensive import pattern tests for JS-family parsers.

This module tests all import patterns across JavaScript, JSX, TypeScript, TSX, and Vue
to ensure consistent and complete extraction of import statements.

Categories covered:
1. ES6 Imports (default, named, multiple, namespace, combined, side-effect, aliased, multi-line)
2. TypeScript-Specific Imports (type import, inline type, type namespace)
3. Dynamic Imports (await import, promise import)
4. CommonJS Imports (require, destructured require, aliased require)
"""

import pytest
from pathlib import Path

from chunkhound.core.types.common import Language, ChunkType
from chunkhound.parsers.parser_factory import ParserFactory


def create_parser(language: Language):
    """Create a parser for the given language."""
    factory = ParserFactory()
    return factory.create_parser(language)


def parse_code(code: str, language: Language, filename: str = None):
    """Parse code and return chunks."""
    if filename is None:
        ext_map = {
            Language.JAVASCRIPT: "test.js",
            Language.JSX: "test.jsx",
            Language.TYPESCRIPT: "test.ts",
            Language.TSX: "test.tsx",
            Language.VUE: "test.vue",
        }
        filename = ext_map.get(language, "test.js")

    parser = create_parser(language)
    return parser.parse_content(code, Path(filename), file_id=1)


# Languages that support all ES6 import patterns
ES6_LANGUAGES = [
    (Language.JAVASCRIPT, ".js"),
    (Language.JSX, ".jsx"),
    (Language.TYPESCRIPT, ".ts"),
    (Language.TSX, ".tsx"),
]

# Languages that support TypeScript-specific imports
TS_LANGUAGES = [
    (Language.TYPESCRIPT, ".ts"),
    (Language.TSX, ".tsx"),
]

# Languages that currently extract ES6 imports (TS/TSX do, JS/JSX don't yet)
# JS and JSX parsers use JavaScript mapping which doesn't have import queries yet
ES6_IMPORT_SUPPORTED = [
    pytest.param(Language.JAVASCRIPT, ".js", marks=pytest.mark.xfail(reason="JS parser doesn't extract ES6 imports yet")),
    pytest.param(Language.JSX, ".jsx", marks=pytest.mark.xfail(reason="JSX parser doesn't extract ES6 imports yet")),
    pytest.param(Language.TYPESCRIPT, ".ts"),
    pytest.param(Language.TSX, ".tsx"),
]


class TestES6DefaultImport:
    """Test default import extraction across all JS-family languages."""

    @pytest.mark.parametrize("lang,ext", ES6_IMPORT_SUPPORTED)
    def test_default_import_extracted(self, lang, ext):
        """Test that default imports are extracted as chunks."""
        code = "import React from 'react';"
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Default import should be extracted for {lang.value}"
        import_chunks = [c for c in chunks if "import" in c.code.lower()]
        assert len(import_chunks) > 0, f"Should find import content for {lang.value}"
        assert any("React" in c.code for c in import_chunks), \
            f"Import should contain 'React' for {lang.value}"

    @pytest.mark.parametrize("lang,ext", ES6_IMPORT_SUPPORTED)
    def test_default_import_with_relative_path(self, lang, ext):
        """Test default import with relative path."""
        code = "import MyComponent from './components/MyComponent';"
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Relative import should be extracted for {lang.value}"
        assert any("MyComponent" in c.code for c in chunks), \
            f"Import should contain 'MyComponent' for {lang.value}"

    @pytest.mark.xfail(reason="JS parser doesn't extract ES6 imports yet")
    def test_default_import_javascript_direct(self):
        """Direct unit test for JavaScript default import."""
        code = "import React from 'react';"
        parser = create_parser(Language.JAVASCRIPT)
        chunks = parser.parse_content(code, Path("test.js"), file_id=1)

        assert len(chunks) > 0, "Should extract default import"
        assert any("React" in c.code for c in chunks)


class TestES6NamedImport:
    """Test named import extraction across all JS-family languages."""

    @pytest.mark.parametrize("lang,ext", ES6_IMPORT_SUPPORTED)
    def test_single_named_import(self, lang, ext):
        """Test single named import extraction."""
        code = "import { useState } from 'react';"
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Named import should be extracted for {lang.value}"
        assert any("useState" in c.code for c in chunks), \
            f"Import should contain 'useState' for {lang.value}"

    @pytest.mark.parametrize("lang,ext", ES6_IMPORT_SUPPORTED)
    def test_multiple_named_imports(self, lang, ext):
        """Test multiple named imports in single statement."""
        code = "import { useState, useEffect, useCallback } from 'react';"
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Multiple named imports should be extracted for {lang.value}"
        import_content = " ".join(c.code for c in chunks)
        assert "useState" in import_content, f"Should contain useState for {lang.value}"
        assert "useEffect" in import_content, f"Should contain useEffect for {lang.value}"
        assert "useCallback" in import_content, f"Should contain useCallback for {lang.value}"

    @pytest.mark.parametrize("lang,ext", ES6_IMPORT_SUPPORTED)
    def test_aliased_named_import(self, lang, ext):
        """Test aliased named import (import { foo as bar })."""
        code = "import { readFile as read } from 'fs';"
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Aliased import should be extracted for {lang.value}"
        import_content = " ".join(c.code for c in chunks)
        assert "readFile" in import_content or "read" in import_content, \
            f"Should contain alias info for {lang.value}"

    @pytest.mark.parametrize("lang,ext", ES6_IMPORT_SUPPORTED)
    def test_multiple_aliased_imports(self, lang, ext):
        """Test multiple aliased imports."""
        code = "import { foo as bar, baz as qux } from 'module';"
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Multiple aliased imports should be extracted for {lang.value}"


class TestES6NamespaceImport:
    """Test namespace import extraction (import * as)."""

    @pytest.mark.parametrize("lang,ext", ES6_IMPORT_SUPPORTED)
    def test_namespace_import(self, lang, ext):
        """Test namespace import extraction."""
        code = "import * as utils from './utils';"
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Namespace import should be extracted for {lang.value}"
        assert any("utils" in c.code for c in chunks), \
            f"Import should contain namespace name 'utils' for {lang.value}"

    @pytest.mark.parametrize("lang,ext", ES6_IMPORT_SUPPORTED)
    def test_namespace_import_npm_package(self, lang, ext):
        """Test namespace import from npm package."""
        code = "import * as lodash from 'lodash';"
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Namespace import from npm should be extracted for {lang.value}"
        assert any("lodash" in c.code for c in chunks)


class TestES6CombinedImport:
    """Test combined default + named imports."""

    @pytest.mark.parametrize("lang,ext", ES6_IMPORT_SUPPORTED)
    def test_combined_default_and_named(self, lang, ext):
        """Test combined default + named import."""
        code = "import React, { useState, useEffect } from 'react';"
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Combined import should be extracted for {lang.value}"
        import_content = " ".join(c.code for c in chunks)
        assert "React" in import_content, f"Should contain default 'React' for {lang.value}"
        assert "useState" in import_content, f"Should contain named 'useState' for {lang.value}"

    @pytest.mark.parametrize("lang,ext", ES6_IMPORT_SUPPORTED)
    def test_combined_default_and_namespace(self, lang, ext):
        """Test combined default + namespace import."""
        code = "import React, * as ReactAll from 'react';"
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Combined default+namespace should be extracted for {lang.value}"


class TestES6SideEffectImport:
    """Test side-effect imports (import 'module')."""

    @pytest.mark.parametrize("lang,ext", ES6_IMPORT_SUPPORTED)
    def test_side_effect_import_css(self, lang, ext):
        """Test side-effect CSS import."""
        code = "import './styles.css';"
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Side-effect import should be extracted for {lang.value}"
        assert any("styles.css" in c.code for c in chunks), \
            f"Import should contain module path for {lang.value}"

    @pytest.mark.parametrize("lang,ext", ES6_IMPORT_SUPPORTED)
    def test_side_effect_import_polyfill(self, lang, ext):
        """Test side-effect polyfill import."""
        code = "import 'core-js/stable';"
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Polyfill import should be extracted for {lang.value}"

    @pytest.mark.parametrize("lang,ext", ES6_IMPORT_SUPPORTED)
    def test_multiple_side_effect_imports(self, lang, ext):
        """Test multiple side-effect imports."""
        code = """import './styles.css';
import './global.css';
import 'normalize.css';"""
        chunks = parse_code(code, lang, f"test{ext}")

        # Should extract all three imports
        assert len(chunks) > 0, f"Multiple side-effect imports should be extracted for {lang.value}"


class TestES6MultiLineImport:
    """Test multi-line import extraction."""

    @pytest.mark.parametrize("lang,ext", ES6_IMPORT_SUPPORTED)
    def test_multi_line_named_import(self, lang, ext):
        """Test multi-line named import."""
        code = """import {
    useState,
    useEffect,
    useCallback,
    useMemo
} from 'react';"""
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Multi-line import should be extracted for {lang.value}"
        import_content = " ".join(c.code for c in chunks)
        assert "useState" in import_content, f"Should contain useState for {lang.value}"
        assert "useMemo" in import_content, f"Should contain useMemo for {lang.value}"

    @pytest.mark.parametrize("lang,ext", ES6_IMPORT_SUPPORTED)
    def test_multi_line_with_aliases(self, lang, ext):
        """Test multi-line import with aliases."""
        code = """import {
    readFile as read,
    writeFile as write,
    stat as getStat
} from 'fs/promises';"""
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Multi-line aliased import should be extracted for {lang.value}"

    @pytest.mark.parametrize("lang,ext", ES6_IMPORT_SUPPORTED)
    def test_multi_line_with_trailing_comma(self, lang, ext):
        """Test multi-line import with trailing comma."""
        code = """import {
    a,
    b,
    c,
} from 'module';"""
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Import with trailing comma should be extracted for {lang.value}"


class TestTypeScriptTypeImport:
    """Test TypeScript type-only imports."""

    @pytest.mark.parametrize("lang,ext", TS_LANGUAGES)
    def test_type_import(self, lang, ext):
        """Test type-only import."""
        code = "import type { Props } from './types';"
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Type import should be extracted for {lang.value}"
        assert any("type" in c.code and "Props" in c.code for c in chunks), \
            f"Import should contain 'type' and 'Props' for {lang.value}"

    @pytest.mark.parametrize("lang,ext", TS_LANGUAGES)
    def test_multiple_type_imports(self, lang, ext):
        """Test multiple type-only imports."""
        code = "import type { User, Product, Order } from './models';"
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Multiple type imports should be extracted for {lang.value}"
        import_content = " ".join(c.code for c in chunks)
        assert "User" in import_content
        assert "Product" in import_content

    @pytest.mark.parametrize("lang,ext", TS_LANGUAGES)
    def test_inline_type_import(self, lang, ext):
        """Test inline type import mixed with value imports."""
        code = "import { type Props, useState, type State } from 'react';"
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Inline type import should be extracted for {lang.value}"
        import_content = " ".join(c.code for c in chunks)
        assert "Props" in import_content
        assert "useState" in import_content

    @pytest.mark.parametrize("lang,ext", TS_LANGUAGES)
    def test_type_namespace_import(self, lang, ext):
        """Test type-only namespace import."""
        code = "import type * as Types from './types';"
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Type namespace import should be extracted for {lang.value}"
        assert any("Types" in c.code for c in chunks)

    @pytest.mark.parametrize("lang,ext", TS_LANGUAGES)
    def test_type_default_import(self, lang, ext):
        """Test type-only default import."""
        code = "import type Config from './config';"
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Type default import should be extracted for {lang.value}"


class TestDynamicImport:
    """Test dynamic import extraction."""

    @pytest.mark.parametrize("lang,ext", ES6_LANGUAGES)
    def test_await_import(self, lang, ext):
        """Test await import expression."""
        code = """async function loadModule() {
    const module = await import('./module');
    return module.default;
}"""
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Dynamic import should be extracted for {lang.value}"
        # The function containing the dynamic import should be extracted
        content = " ".join(c.code for c in chunks)
        assert "import" in content and "module" in content, \
            f"Should capture dynamic import for {lang.value}"

    @pytest.mark.parametrize("lang,ext", ES6_LANGUAGES)
    def test_promise_import(self, lang, ext):
        """Test promise-style dynamic import."""
        code = """import('./module').then(m => {
    console.log(m.default);
});"""
        chunks = parse_code(code, lang, f"test{ext}")

        # Dynamic imports may be captured as expressions or within other constructs
        # At minimum the file should parse without error
        assert isinstance(chunks, list), f"Should return list for {lang.value}"

    @pytest.mark.parametrize("lang,ext", ES6_LANGUAGES)
    def test_conditional_import(self, lang, ext):
        """Test conditional dynamic import."""
        code = """async function conditionalLoad(condition) {
    if (condition) {
        const { default: module } = await import('./heavy-module');
        return module;
    }
    return null;
}"""
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Conditional import should be extracted for {lang.value}"


class TestCommonJSImport:
    """Test CommonJS require() imports."""

    @pytest.mark.parametrize("lang,ext", ES6_LANGUAGES)
    def test_basic_require(self, lang, ext):
        """Test basic require statement."""
        code = "const fs = require('fs');"
        chunks = parse_code(code, lang, f"test{ext}")

        # CommonJS requires may or may not be extracted depending on parser
        # but the file should parse without error
        assert isinstance(chunks, list), f"Should return list for {lang.value}"

    @pytest.mark.parametrize("lang,ext", ES6_LANGUAGES)
    def test_destructured_require(self, lang, ext):
        """Test destructured require statement."""
        code = "const { readFile, writeFile } = require('fs');"
        chunks = parse_code(code, lang, f"test{ext}")

        assert isinstance(chunks, list), f"Should return list for {lang.value}"

    @pytest.mark.parametrize("lang,ext", ES6_LANGUAGES)
    def test_aliased_destructured_require(self, lang, ext):
        """Test aliased destructured require."""
        code = "const { readFile: read } = require('fs');"
        chunks = parse_code(code, lang, f"test{ext}")

        assert isinstance(chunks, list), f"Should return list for {lang.value}"

    @pytest.mark.parametrize("lang,ext", ES6_LANGUAGES)
    def test_multiple_requires(self, lang, ext):
        """Test multiple require statements."""
        code = """const fs = require('fs');
const path = require('path');
const os = require('os');"""
        chunks = parse_code(code, lang, f"test{ext}")

        assert isinstance(chunks, list), f"Should return list for {lang.value}"


class TestVueImports:
    """Test Vue Single File Component import extraction."""

    def test_vue_script_setup_imports(self):
        """Test imports in Vue script setup."""
        code = """<template>
  <div>{{ message }}</div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import type { User } from './types';

const message = ref('Hello');
</script>"""
        chunks = parse_code(code, Language.VUE, "test.vue")

        assert len(chunks) > 0, "Vue imports should be extracted"
        content = " ".join(c.code for c in chunks)
        assert "ref" in content or "import" in content, "Should capture Vue imports"

    def test_vue_standard_script_imports(self):
        """Test imports in Vue standard script."""
        code = """<template>
  <div>Hello</div>
</template>

<script>
import MyComponent from './MyComponent.vue';
import { helper } from './utils';

export default {
  components: { MyComponent }
}
</script>"""
        chunks = parse_code(code, Language.VUE, "test.vue")

        assert len(chunks) > 0, "Vue standard script imports should be extracted"


class TestCrossLanguageConsistency:
    """Test that imports produce consistent results across languages.

    Note: These tests currently fail for JS/JSX because those parsers don't
    extract static ES6 imports yet. They pass for TypeScript/TSX.
    """

    @pytest.mark.xfail(reason="JS/JSX parsers don't extract ES6 imports yet - only TS/TSX pass")
    def test_default_import_consistency(self):
        """Test that default import extraction is consistent across languages."""
        code = "import React from 'react';"

        results = {}
        for lang, ext in ES6_LANGUAGES:
            chunks = parse_code(code, lang, f"test{ext}")
            results[lang.value] = {
                "count": len(chunks),
                "has_react": any("React" in c.code for c in chunks)
            }

        # All languages should extract the import
        for lang_name, result in results.items():
            assert result["count"] > 0, f"{lang_name} should extract import"
            assert result["has_react"], f"{lang_name} should contain 'React'"

    @pytest.mark.xfail(reason="JS/JSX parsers don't extract ES6 imports yet - only TS/TSX pass")
    def test_named_import_consistency(self):
        """Test that named import extraction is consistent across languages."""
        code = "import { useState, useEffect } from 'react';"

        results = {}
        for lang, ext in ES6_LANGUAGES:
            chunks = parse_code(code, lang, f"test{ext}")
            content = " ".join(c.code for c in chunks)
            results[lang.value] = {
                "count": len(chunks),
                "has_useState": "useState" in content,
                "has_useEffect": "useEffect" in content
            }

        # All languages should extract with same names
        for lang_name, result in results.items():
            assert result["count"] > 0, f"{lang_name} should extract import"
            assert result["has_useState"], f"{lang_name} should contain 'useState'"
            assert result["has_useEffect"], f"{lang_name} should contain 'useEffect'"

    @pytest.mark.xfail(reason="JS/JSX parsers don't extract ES6 imports yet - only TS/TSX pass")
    def test_namespace_import_consistency(self):
        """Test that namespace import extraction is consistent."""
        code = "import * as utils from './utils';"

        for lang, ext in ES6_LANGUAGES:
            chunks = parse_code(code, lang, f"test{ext}")
            assert len(chunks) > 0, f"{lang.value} should extract namespace import"
            assert any("utils" in c.code for c in chunks), \
                f"{lang.value} should contain 'utils'"


class TestImportMetadata:
    """Test that import chunks have correct metadata."""

    def test_import_line_numbers(self):
        """Test that imports have correct line numbers."""
        code = """// Header comment
import React from 'react';
import { useState } from 'react';

function Component() {}"""

        for lang, ext in ES6_LANGUAGES:
            chunks = parse_code(code, lang, f"test{ext}")

            for chunk in chunks:
                # All chunks should have valid line numbers
                assert chunk.start_line >= 1, \
                    f"start_line should be >= 1 for {lang.value}"
                assert chunk.end_line >= chunk.start_line, \
                    f"end_line should be >= start_line for {lang.value}"

    def test_import_content_completeness(self):
        """Test that import content is complete."""
        code = "import { useState, useEffect, useCallback } from 'react';"

        for lang, ext in ES6_LANGUAGES:
            chunks = parse_code(code, lang, f"test{ext}")

            if len(chunks) > 0:
                import_chunk = chunks[0]
                # Content should be the full import statement
                assert "import" in import_chunk.code.lower()
                assert "from" in import_chunk.code.lower()


class TestComplexImportPatterns:
    """Test complex real-world import patterns."""

    @pytest.mark.parametrize("lang,ext", ES6_IMPORT_SUPPORTED)
    def test_mixed_import_file(self, lang, ext):
        """Test file with multiple import types."""
        code = """// Multiple import types
import React, { useState, useEffect } from 'react';
import * as utils from './utils';
import './styles.css';

export function Component() {
    return null;
}"""
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Mixed imports should be extracted for {lang.value}"
        content = " ".join(c.code for c in chunks)
        assert "React" in content or "utils" in content or "styles" in content

    @pytest.mark.parametrize("lang,ext", TS_LANGUAGES)
    def test_typescript_mixed_value_type_imports(self, lang, ext):
        """Test TypeScript file with both value and type imports."""
        code = """import React from 'react';
import type { FC, ReactNode } from 'react';
import { useState } from 'react';
import type * as Types from './types';

export const App: FC = () => null;"""
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Mixed value/type imports should be extracted for {lang.value}"

    @pytest.mark.parametrize("lang,ext", ES6_IMPORT_SUPPORTED)
    def test_deeply_nested_import_path(self, lang, ext):
        """Test import with deeply nested path."""
        code = "import { helper } from '../../../../../../shared/utils/helpers';"
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Deeply nested path import should be extracted for {lang.value}"

    @pytest.mark.parametrize("lang,ext", ES6_IMPORT_SUPPORTED)
    def test_import_with_special_characters(self, lang, ext):
        """Test import with special characters in path."""
        code = "import data from './data-2023_01.json';"
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Special char path import should be extracted for {lang.value}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.parametrize("lang,ext", ES6_LANGUAGES)
    def test_empty_named_import(self, lang, ext):
        """Test empty named import (unusual but valid)."""
        code = "import {} from 'module';"
        chunks = parse_code(code, lang, f"test{ext}")

        # Should parse without error
        assert isinstance(chunks, list), f"Empty import should not crash for {lang.value}"

    @pytest.mark.parametrize("lang,ext", ES6_IMPORT_SUPPORTED)
    def test_import_at_end_of_file(self, lang, ext):
        """Test import at end of file without newline."""
        code = "import React from 'react'"  # No semicolon or newline
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Import at EOF should be extracted for {lang.value}"

    @pytest.mark.parametrize("lang,ext", ES6_LANGUAGES)
    def test_import_with_comments(self, lang, ext):
        """Test import with inline comments."""
        code = """import {
    useState,  // for state management
    useEffect, // for side effects
    useRef     // for refs
} from 'react';"""
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Import with comments should be extracted for {lang.value}"

    @pytest.mark.parametrize("lang,ext", ES6_IMPORT_SUPPORTED)
    def test_file_only_imports(self, lang, ext):
        """Test file containing only imports."""
        code = """import React from 'react';
import { useState } from 'react';
import './styles.css';"""
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Import-only file should be parsed for {lang.value}"

    @pytest.mark.parametrize("lang,ext", ES6_IMPORT_SUPPORTED)
    def test_many_imports(self, lang, ext):
        """Test file with many imports."""
        imports = [f"import mod{i} from 'module{i}';" for i in range(20)]
        code = "\n".join(imports)
        chunks = parse_code(code, lang, f"test{ext}")

        assert len(chunks) > 0, f"Many imports should be extracted for {lang.value}"


class TestImportChunkType:
    """Test that imports are correctly classified."""

    @pytest.mark.parametrize("lang,ext", ES6_LANGUAGES)
    def test_import_chunk_type_classification(self, lang, ext):
        """Test that imports get appropriate chunk type."""
        code = "import React from 'react';"
        chunks = parse_code(code, lang, f"test{ext}")

        if len(chunks) > 0:
            # Imports should typically be classified as IMPORT or similar
            import_chunks = [c for c in chunks if "import" in c.code.lower()]
            for chunk in import_chunks:
                # The chunk type depends on implementation
                # but should be a valid ChunkType
                assert chunk.chunk_type is not None, \
                    f"Import should have chunk_type for {lang.value}"


# Direct unit tests for specific parsers
class TestJavaScriptParserImports:
    """Direct unit tests for JavaScript parser import extraction."""

    @pytest.mark.xfail(reason="JS parser doesn't extract ES6 imports yet")
    def test_js_default_import(self):
        """Test JavaScript default import extraction."""
        parser = create_parser(Language.JAVASCRIPT)
        chunks = parser.parse_content(
            "import React from 'react';",
            Path("test.js"),
            file_id=1
        )
        assert len(chunks) > 0
        assert any("React" in c.code for c in chunks)

    @pytest.mark.xfail(reason="JS parser doesn't extract ES6 imports yet")
    def test_js_named_import(self):
        """Test JavaScript named import extraction."""
        parser = create_parser(Language.JAVASCRIPT)
        chunks = parser.parse_content(
            "import { useState, useEffect } from 'react';",
            Path("test.js"),
            file_id=1
        )
        assert len(chunks) > 0


class TestTypeScriptParserImports:
    """Direct unit tests for TypeScript parser import extraction."""

    def test_ts_type_import(self):
        """Test TypeScript type import extraction."""
        parser = create_parser(Language.TYPESCRIPT)
        chunks = parser.parse_content(
            "import type { Props } from './types';",
            Path("test.ts"),
            file_id=1
        )
        assert len(chunks) > 0
        assert any("type" in c.code and "Props" in c.code for c in chunks)

    def test_ts_inline_type_import(self):
        """Test TypeScript inline type import extraction."""
        parser = create_parser(Language.TYPESCRIPT)
        chunks = parser.parse_content(
            "import { type Props, useState } from 'react';",
            Path("test.ts"),
            file_id=1
        )
        assert len(chunks) > 0


class TestJSXParserImports:
    """Direct unit tests for JSX parser import extraction."""

    @pytest.mark.xfail(reason="JSX parser doesn't extract ES6 imports yet")
    def test_jsx_default_import(self):
        """Test JSX default import extraction."""
        parser = create_parser(Language.JSX)
        chunks = parser.parse_content(
            "import React from 'react';",
            Path("test.jsx"),
            file_id=1
        )
        assert len(chunks) > 0

    @pytest.mark.xfail(reason="JSX parser doesn't extract ES6 imports yet")
    def test_jsx_combined_import(self):
        """Test JSX combined import extraction."""
        parser = create_parser(Language.JSX)
        chunks = parser.parse_content(
            "import React, { useState } from 'react';",
            Path("test.jsx"),
            file_id=1
        )
        assert len(chunks) > 0


class TestTSXParserImports:
    """Direct unit tests for TSX parser import extraction."""

    def test_tsx_type_import(self):
        """Test TSX type import extraction."""
        parser = create_parser(Language.TSX)
        chunks = parser.parse_content(
            "import type { FC } from 'react';",
            Path("test.tsx"),
            file_id=1
        )
        assert len(chunks) > 0

    def test_tsx_mixed_imports(self):
        """Test TSX mixed value and type imports."""
        parser = create_parser(Language.TSX)
        chunks = parser.parse_content(
            """import React, { useState } from 'react';
import type { FC, ReactNode } from 'react';""",
            Path("test.tsx"),
            file_id=1
        )
        assert len(chunks) > 0
