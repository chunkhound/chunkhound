"""
Regression test for issue #267: C# parser error message uses wrong pip package name.

The correct PyPI package is `tree-sitter-c-sharp`, not `tree-sitter-csharp`.
"""

import pytest
from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import LANGUAGE_CONFIGS


def test_csharp_pip_package_name_is_correct():
    """LanguageConfig for C# must reference tree-sitter-c-sharp, not tree-sitter-csharp."""
    config = LANGUAGE_CONFIGS[Language.CSHARP]
    assert config.pip_package == "tree-sitter-c-sharp", (
        f"Wrong pip package name: got '{config.pip_package}'. "
        "Users following this name would try 'pip install tree-sitter-csharp' "
        "which does not exist on PyPI."
    )


def test_csharp_setup_error_references_correct_package():
    """SetupError for C# must suggest installing tree-sitter-c-sharp."""
    from chunkhound.parsers.universal_engine import SetupError

    config = LANGUAGE_CONFIGS[Language.CSHARP]
    err = SetupError(
        parser=config.language_name,
        missing_dependency=config.pip_package,
        install_command=f"pip install {config.pip_package}",
        original_error="simulated",
    )
    assert "tree-sitter-c-sharp" in str(err)
    assert "tree-sitter-csharp" not in str(err)


def test_other_languages_pip_package_defaults_correctly():
    """Languages without an explicit pip_package still derive the name correctly."""
    for lang, config in LANGUAGE_CONFIGS.items():
        if lang == Language.CSHARP:
            continue
        # All others should follow the auto-derived pattern
        expected = f"tree-sitter-{config.language_name}"
        assert config.pip_package == expected, (
            f"Language {lang}: expected pip_package='{expected}', got '{config.pip_package}'"
        )
