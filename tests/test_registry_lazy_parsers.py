"""Tests for lazy language parser instantiation in the registry."""

from collections import defaultdict

from chunkhound.core.types.common import Language
from chunkhound.registry import ProviderRegistry


def test_registry_lazily_instantiates_language_parsers(monkeypatch):
    """Ensure parser factories are only invoked when a language is requested."""

    class FakeParserFactory:
        def __init__(self):
            self.calls = defaultdict(int)

        def get_available_languages(self):
            return {Language.PYTHON: True, Language.JAVA: True}

        def create_parser(self, language, cast_config=None):
            self.calls[language] += 1
            return f"parser-{language.value}"

    fake_factory = FakeParserFactory()
    monkeypatch.setattr("chunkhound.registry.get_parser_factory", lambda: fake_factory)

    registry = ProviderRegistry()
    registry._language_parsers.clear()

    # Register factories without instantiating parsers.
    registry._setup_language_parsers()
    assert fake_factory.calls == {}

    # First access materializes parser exactly once.
    parser_python = registry.get_language_parser(Language.PYTHON)
    assert parser_python == "parser-python"
    assert fake_factory.calls[Language.PYTHON] == 1

    # Second access reuses cached parser.
    parser_python_again = registry.get_language_parser(Language.PYTHON)
    assert parser_python_again == parser_python
    assert fake_factory.calls[Language.PYTHON] == 1

    # Another language materializes independently.
    parser_java = registry.get_language_parser(Language.JAVA)
    assert parser_java == "parser-java"
    assert fake_factory.calls[Language.JAVA] == 1
