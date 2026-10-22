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
