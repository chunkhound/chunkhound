"""JS/TS/JSX/TSX top-level import and call statements are searchable (#449).

A spec file is almost entirely top-level `describe(...)`/`it(...)` calls plus
imports. Those must produce chunks, without duplicating what the CommonJS
export patterns already capture.
"""

from pathlib import Path

import pytest

from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import ParserFactory


def _parse(code: str, filename: str, language: Language):
    parser = ParserFactory().create_parser(language)
    return parser.parse_content(code, Path(filename), file_id=1)


SPEC = (
    'import { add } from "./math";\n'
    "\n"
    "const fixture = { a: 1, b: 2 };\n"
    "\n"
    'describe("add", () => {\n'
    '  it("sums two numbers", () => {\n'
    "    expect(add(fixture.a, fixture.b)).toBe(3);\n"
    "  });\n"
    "});\n"
)


@pytest.mark.parametrize(
    ("filename", "language"),
    [
        ("math.spec.ts", Language.TYPESCRIPT),
        ("math.spec.tsx", Language.TSX),
        ("math.spec.js", Language.JAVASCRIPT),
        ("math.spec.jsx", Language.JSX),
    ],
)
def test_spec_file_imports_and_describe_blocks_are_chunked(filename, language):
    chunks = _parse(SPEC, filename, language)
    code = "\n".join(c.code for c in chunks)

    assert "import { add }" in code
    assert 'describe("add"' in code
    assert "sums two numbers" in code
    # What was already chunked is still there, exactly once.
    assert sum("const fixture" in c.code for c in chunks) == 1
    assert sum("sums two numbers" in c.code for c in chunks) == 1


def test_commonjs_exports_are_not_duplicated_by_call_statement_chunks():
    code = (
        "const fs = require('fs');\n"
        "\n"
        "module.exports = {\n"
        "  serviceUrl: process.env.SERVICE_URL,\n"
        "};\n"
        "\n"
        "exports.other = { key: process.env.OTHER_KEY };\n"
        "\n"
        "app.listen(PORT_VALUE);\n"
        "await main();\n"
    )
    chunks = _parse(code, "server.mjs", Language.JAVASCRIPT)

    assert sum("SERVICE_URL" in c.code for c in chunks) == 1
    assert sum("OTHER_KEY" in c.code for c in chunks) == 1
    assert any("app.listen(PORT_VALUE)" in c.code for c in chunks)
    assert any("await main()" in c.code for c in chunks)


LANGUAGES = [
    ("x.spec.ts", Language.TYPESCRIPT),
    ("x.spec.tsx", Language.TSX),
    ("x.spec.js", Language.JAVASCRIPT),
    ("x.spec.jsx", Language.JSX),
]

NESTED = (
    "(function () {\n"
    "  setup();\n"
    "})();\n"
    "function helper() { return 1; }\n"
    'describe("math", () => {\n'
    "  function inner() {\n"
    "    return 2;\n"
    "  }\n"
    "  class Fixture {\n"
    "    value() {\n"
    "      return inner();\n"
    "    }\n"
    "  }\n"
    '  it("adds", () => {\n'
    "    expect(new Fixture().value()).toBe(2);\n"
    "  });\n"
    "});\n"
)


@pytest.mark.parametrize(("filename", "language"), LANGUAGES)
def test_definitions_nested_in_a_call_are_chunked_once(filename, language):
    chunks = _parse(NESTED, filename, language)

    # Named definitions inside describe(...) keep their own chunk, and their
    # code is not repeated in the describe block.
    assert sum("function inner()" in c.code for c in chunks) == 1
    assert sum("class Fixture" in c.code for c in chunks) == 1
    assert any(c.symbol == "inner" for c in chunks)
    # The statement's own code around them is still searchable, once.
    assert sum('describe("math"' in c.code for c in chunks) == 1
    assert sum("expect(new Fixture().value())" in c.code for c in chunks) == 1


@pytest.mark.parametrize(("filename", "language"), LANGUAGES)
def test_call_statements_do_not_merge_across_definitions(filename, language):
    chunks = _parse(NESTED, filename, language)

    # The IIFE and describe(...) are close enough for the block merge, but
    # helper() sits between them, so no chunk may hold both.
    assert not any("setup()" in c.code and 'describe("math"' in c.code for c in chunks)
    names = {c.symbol for c in chunks if "setup()" in c.code}
    assert names == {"iife_line_1"}


@pytest.mark.parametrize(("filename", "language"), LANGUAGES)
@pytest.mark.parametrize(
    ("code", "name"),
    [
        (
            "it('says \"hi\"', () => {\n  expect(greet()).toBe('hi');\n});\n",
            'it("says "hi"")',
        ),
        (
            "test(`adds ${n}`, () => {\n  expect(add(n)).toBe(n);\n});\n",
            'test("adds ${n}")',
        ),
        ("(async () => {\n  await main();\n})();\n", "iife_line_1"),
    ],
)
def test_call_statement_names(filename, language, code, name):
    assert [c.symbol for c in _parse(code, filename, language)] == [name]
