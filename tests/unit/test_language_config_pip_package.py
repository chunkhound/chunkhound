"""
Regression test for issue #267: C# parser error message uses wrong pip package name.

The correct PyPI package is `tree-sitter-c-sharp`, not `tree-sitter-csharp`.
"""

from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import LANGUAGE_CONFIGS


def test_csharp_pip_package_name_is_correct():
    """C# LanguageConfig must use tree-sitter-c-sharp, not tree-sitter-csharp."""
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


# Languages whose PyPI package name diverges from the auto-derived
# f"tree-sitter-{language_name}" pattern.
EXPLICIT_PIP_PACKAGES = {
    Language.CSHARP: "tree-sitter-c-sharp",   # PyPI: tree-sitter-c-sharp
    Language.MAKEFILE: "tree-sitter-make",     # PyPI: tree-sitter-make (not tree-sitter-makefile)
    Language.SCSS: "tree-sitter-language-pack",  # bundled in language-pack, no standalone
}


def test_explicit_pip_package_overrides_are_correct():
    """Every language with a non-default pip_package must match the canonical PyPI name."""
    for lang, expected in EXPLICIT_PIP_PACKAGES.items():
        config = LANGUAGE_CONFIGS[lang]
        assert config.pip_package == expected, (
            f"{lang}: expected pip_package='{expected}', got '{config.pip_package}'"
        )


def test_non_overridden_languages_use_default_formula():
    """Languages not in the explicit-override table use tree-sitter-{language_name}."""
    for lang, config in LANGUAGE_CONFIGS.items():
        if lang in EXPLICIT_PIP_PACKAGES:
            continue
        expected = f"tree-sitter-{config.language_name}"
        assert config.pip_package == expected, (
            f"{lang}: expected default pip_package='{expected}', got '{config.pip_package}'. "
            "If the PyPI name differs, add an entry to EXPLICIT_PIP_PACKAGES."
        )
