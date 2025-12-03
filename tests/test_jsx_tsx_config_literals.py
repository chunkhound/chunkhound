"""JSX/TSX config-literal parsing tests.

Ensures top-level object/array exports in JSX/TSX are chunked so
string/env-like tokens are embedded and discoverable.
"""

from pathlib import Path

from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import ParserFactory


def _parse(code: str, filename: str, language: Language):
    factory = ParserFactory()
    parser = factory.create_parser(language)
    return parser.parse_content(code, Path(filename), file_id=1)


def test_tsx_named_export_const_object_literal_chunked():
    code = (
        "export const config: { serviceUrl: string } = {\n"
        "  serviceUrl: process.env.SERVICE_URL as string\n"
        "};\n"
    )
    chunks = _parse(code, "config.tsx", Language.TSX)
    assert len(chunks) > 0
    assert any("SERVICE_URL" in c.code for c in chunks)


def test_jsx_export_default_array_literal_chunked():
    code = (
        "export default [\n"
        "  process.env.SERVICE_URL,\n"
        "  'a'\n"
        "];\n"
    )
    chunks = _parse(code, "config.jsx", Language.JSX)
    assert len(chunks) > 0
    assert any("SERVICE_URL" in c.code for c in chunks)

