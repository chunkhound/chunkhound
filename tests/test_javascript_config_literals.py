"""JavaScript/TypeScript config-literal parsing tests.

Ensures top-level object/array exports and assignments are chunked so
string/env-like tokens are embedded and discoverable.
"""

from pathlib import Path

from chunkhound.core.types.common import Language, ChunkType
from chunkhound.parsers.parser_factory import ParserFactory


def _parse(code: str, filename: str, language: Language):
    factory = ParserFactory()
    parser = factory.create_parser(language)
    return parser.parse_content(code, Path(filename), file_id=1)


def test_js_export_default_object_literal_chunked():
    code = (
        "export default {\n"
        "  serviceUrl: process.env.SERVICE_URL,\n"
        "  apiKey: process.env.API_KEY\n"
        "};\n"
    )
    chunks = _parse(code, "config.js", Language.JAVASCRIPT)
    assert len(chunks) > 0
    assert any("SERVICE_URL" in c.code for c in chunks)
    # Type may vary by grammar/provider; presence is the key signal


def test_ts_export_default_object_literal_chunked():
    code = (
        "const cfg: { serviceUrl: string } = {\n"
        "  serviceUrl: process.env.SERVICE_URL as string\n"
        "};\n"
        "export default cfg;\n"
    )
    chunks = _parse(code, "config.ts", Language.TYPESCRIPT)
    assert len(chunks) > 0
    assert any("SERVICE_URL" in c.code for c in chunks)
    # Type may vary by grammar/provider; presence is the key signal


def test_js_module_exports_object_literal_chunked():
    code = (
        "module.exports = {\n"
        "  serviceUrl: process.env.SERVICE_URL,\n"
        "  apiKey: process.env.API_KEY\n"
        "};\n"
    )
    chunks = _parse(code, "config.js", Language.JAVASCRIPT)
    assert len(chunks) > 0
    assert any("SERVICE_URL" in c.code for c in chunks)


def test_js_named_export_const_object_literal_chunked():
    code = (
        "export const config = {\n"
        "  serviceUrl: process.env.SERVICE_URL\n"
        "};\n"
    )
    chunks = _parse(code, "config.js", Language.JAVASCRIPT)
    assert len(chunks) > 0
    assert any("SERVICE_URL" in c.code for c in chunks)


def test_js_const_then_export_named_binding_chunked():
    code = (
        "const config = { serviceUrl: process.env.SERVICE_URL };\n"
        "export { config };\n"
    )
    chunks = _parse(code, "config.js", Language.JAVASCRIPT)
    assert len(chunks) > 0
    assert any("SERVICE_URL" in c.code for c in chunks)


def test_js_export_default_array_literal_chunked():
    code = (
        "export default [\n"
        "  process.env.SERVICE_URL,\n"
        "  'other'\n"
        "];\n"
    )
    chunks = _parse(code, "config.js", Language.JAVASCRIPT)
    assert len(chunks) > 0
    assert any("SERVICE_URL" in c.code for c in chunks)


def test_ts_named_export_const_object_literal_chunked():
    code = (
        "export const config: { serviceUrl: string } = {\n"
        "  serviceUrl: process.env.SERVICE_URL as string\n"
        "};\n"
    )
    chunks = _parse(code, "config.ts", Language.TYPESCRIPT)
    assert len(chunks) > 0
    assert any("SERVICE_URL" in c.code for c in chunks)
