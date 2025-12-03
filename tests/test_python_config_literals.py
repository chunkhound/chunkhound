"""Python config-literal parsing tests.

Ensures top-level dict/list assignments are chunked so string/env-like tokens
are embedded and discoverable.
"""

from pathlib import Path

from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import ParserFactory


def _parse(code: str, filename: str):
    factory = ParserFactory()
    parser = factory.create_parser(Language.PYTHON)
    return parser.parse_content(code, Path(filename), file_id=1)


def test_python_top_level_assignment_dict_chunked():
    code = (
        "import os\n"
        "CONFIG = {\n"
        "  'service_url': os.getenv('SERVICE_URL'),\n"
        "  'api_key': os.getenv('API_KEY')\n"
        "}\n"
    )
    chunks = _parse(code, "settings.py")
    assert len(chunks) > 0
    assert any("SERVICE_URL" in c.code for c in chunks)
    # Type may vary by grammar/provider; presence is the key signal


def test_python_top_level_assignment_list_chunked():
    code = (
        "import os\n"
        "SETTINGS = [\n"
        "  os.getenv('SERVICE_URL'),\n"
        "  'fallback'\n"
        "]\n"
    )
    chunks = _parse(code, "settings.py")
    assert len(chunks) > 0
    assert any("SERVICE_URL" in c.code for c in chunks)


def test_python_top_level_nested_dict_chunked():
    code = (
        "import os\n"
        "CONFIG = {\n"
        "  'nested': { 'service_url': os.getenv('SERVICE_URL') }\n"
        "}\n"
    )
    chunks = _parse(code, "settings.py")
    assert len(chunks) > 0
    assert any("SERVICE_URL" in c.code for c in chunks)
