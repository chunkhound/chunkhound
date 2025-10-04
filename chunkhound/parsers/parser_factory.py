"""Parser factory for creating unified parsers with all language mappings.

This module provides the ParserFactory class that:
1. Imports all tree-sitter language modules explicitly at the top (no dynamic imports)
2. Maps languages to their appropriate tree-sitter modules and mappings
3. Creates UniversalParser instances with the correct language configuration
4. Provides a clean interface for the rest of the system
5. Handles cases where tree-sitter modules aren't available gracefully

All tree-sitter language modules are imported explicitly to avoid dynamic import
complexity and ensure better error handling during startup.
"""

import logging
from pathlib import Path
from typing import Any

from chunkhound.core.types.common import Language

# Import all language mappings
from chunkhound.parsers.mappings import (
    BashMapping,
    CMapping,
    CppMapping,
    CSharpMapping,
    GoMapping,
    GroovyMapping,
    HaskellMapping,
    JavaMapping,
    JavaScriptMapping,
    JsonMapping,
    JSXMapping,
    KotlinMapping,
    MakefileMapping,
    MarkdownMapping,
    MatlabMapping,
    PDFMapping,
    PythonMapping,
    RustMapping,
    TextMapping,
    TomlMapping,
    TSXMapping,
    TypeScriptMapping,
    YamlMapping,
    ZigMapping,
)
from chunkhound.parsers.mappings.base import BaseMapping
from chunkhound.parsers.universal_engine import SetupError, TreeSitterEngine
from chunkhound.parsers.universal_parser import CASTConfig, UniversalParser

logger = logging.getLogger(__name__)

# Explicit tree-sitter language imports
# Import all available tree-sitter languages explicitly to avoid dynamic import complexity

# Core language support
try:
    import tree_sitter_python as ts_python

    PYTHON_AVAILABLE = True
except ImportError:
    ts_python = None
    PYTHON_AVAILABLE = False

try:
    import tree_sitter_javascript as ts_javascript

    JAVASCRIPT_AVAILABLE = True
except ImportError:
    ts_javascript = None
    JAVASCRIPT_AVAILABLE = False

try:
    import tree_sitter_typescript as ts_typescript

    TYPESCRIPT_AVAILABLE = True
except ImportError:
    ts_typescript = None
    TYPESCRIPT_AVAILABLE = False

try:
    import tree_sitter_java as ts_java

    JAVA_AVAILABLE = True
except ImportError:
    ts_java = None
    JAVA_AVAILABLE = False

try:
    import tree_sitter_c as ts_c

    C_AVAILABLE = True
except ImportError:
    ts_c = None
    C_AVAILABLE = False

try:
    import tree_sitter_cpp as ts_cpp

    CPP_AVAILABLE = True
except ImportError:
    ts_cpp = None
    CPP_AVAILABLE = False

try:
    import tree_sitter_c_sharp as ts_csharp

    CSHARP_AVAILABLE = True
except ImportError:
    ts_csharp = None
    CSHARP_AVAILABLE = False

try:
    import tree_sitter_go as ts_go

    GO_AVAILABLE = True
except ImportError:
    ts_go = None
    GO_AVAILABLE = False

try:
    import tree_sitter_haskell as ts_haskell

    HASKELL_AVAILABLE = True
except ImportError:
    ts_haskell = None
    HASKELL_AVAILABLE = False

try:
    import tree_sitter_rust as ts_rust

    RUST_AVAILABLE = True
except ImportError:
    ts_rust = None
    RUST_AVAILABLE = False

try:
    import tree_sitter_bash as ts_bash

    BASH_AVAILABLE = True
except ImportError:
    ts_bash = None
    BASH_AVAILABLE = False

try:
    import tree_sitter_kotlin as ts_kotlin

    KOTLIN_AVAILABLE = True
except ImportError:
    ts_kotlin = None
    KOTLIN_AVAILABLE = False

try:
    import tree_sitter_groovy as ts_groovy

    GROOVY_AVAILABLE = True
except ImportError:
    ts_groovy = None
    GROOVY_AVAILABLE = False

try:
    from tree_sitter_language_pack import get_language

    _matlab_lang = get_language("matlab")
    if _matlab_lang:
        # Create a module-like wrapper for compatibility with LanguageConfig
        class _MatlabLanguageWrapper:
            def language(self):
                return _matlab_lang

        ts_matlab = _MatlabLanguageWrapper()
        MATLAB_AVAILABLE = True
    else:
        ts_matlab = None
        MATLAB_AVAILABLE = False
except ImportError:
    ts_matlab = None
    MATLAB_AVAILABLE = False

if not HASKELL_AVAILABLE:
    try:
        from tree_sitter_language_pack import get_language as _get_language_haskell

        _haskell_lang = _get_language_haskell("haskell")
        if _haskell_lang:

            class _HaskellLanguageWrapper:
                def language(self):
                    return _haskell_lang

            ts_haskell = _HaskellLanguageWrapper()
            HASKELL_AVAILABLE = True
    except ImportError:
        pass

# Markup and config languages
try:
    import tree_sitter_json as ts_json

    JSON_AVAILABLE = True
except ImportError:
    ts_json = None
    JSON_AVAILABLE = False

try:
    import tree_sitter_yaml as ts_yaml

    YAML_AVAILABLE = True
except ImportError:
    ts_yaml = None
    YAML_AVAILABLE = False

try:
    import tree_sitter_toml as ts_toml

    TOML_AVAILABLE = True
except ImportError:
    ts_toml = None
    TOML_AVAILABLE = False

try:
    import tree_sitter_markdown as ts_markdown

    MARKDOWN_AVAILABLE = True
except ImportError:
    ts_markdown = None
    MARKDOWN_AVAILABLE = False

# Build system languages
try:
    import tree_sitter_make as ts_make

    MAKEFILE_AVAILABLE = True
except ImportError:
    ts_make = None
    MAKEFILE_AVAILABLE = False

try:
    import tree_sitter_zig as ts_zig

    ZIG_AVAILABLE = True
except ImportError:
    ts_zig = None
    ZIG_AVAILABLE = False

# Additional language extensions (these might use the same parser as base language)
JSX_AVAILABLE = JAVASCRIPT_AVAILABLE  # JSX uses JavaScript parser
TSX_AVAILABLE = TYPESCRIPT_AVAILABLE  # TSX uses TypeScript parser


class LanguageConfig:
    """Configuration for a language including its tree-sitter module and mapping."""

    def __init__(
        self,
        tree_sitter_module: Any,
        mapping_class: type[BaseMapping],
        available: bool,
        language_name: str,
    ):
        self.tree_sitter_module = tree_sitter_module
        self.mapping_class = mapping_class
        self.available = available
        self.language_name = language_name

    def _handle_language_result(self, result):
        """Handle language module result, supporting both old and new APIs."""
        from tree_sitter import Language

        # If result is already a Language object, return as-is
        if isinstance(result, Language):
            return result
        else:
            # In tree-sitter 0.25.x, language modules should return Language objects directly
            # If we get an integer (old API), try to handle it but warn about compatibility
            if isinstance(result, int):
                import warnings

                warnings.warn(
                    f"tree-sitter-{self.language_name.lower()} is using deprecated API (returned integer {result}). "
                    f"Consider upgrading to a version compatible with tree-sitter 0.25+",
                    DeprecationWarning,
                    stacklevel=3,
                )
                # Try to create Language object from integer (deprecated but still supported in some versions)
                try:
                    return Language(result)
                except Exception as e:
                    raise SetupError(
                        parser=self.language_name,
                        missing_dependency=f"Compatible tree-sitter-{self.language_name.lower()} for tree-sitter 0.25.x",
                        install_command=f"pip install --upgrade tree-sitter-{self.language_name.lower()}",
                        original_error=f"Cannot create Language from integer {result}: {e}",
                    ) from e
            return Language(result)

    def get_tree_sitter_language(self):
        """Get the tree-sitter Language object from the module."""
        if not self.available or not self.tree_sitter_module:
            raise SetupError(
                parser=self.language_name,
                missing_dependency=f"tree-sitter-{self.language_name.lower()}",
                install_command=f"pip install tree-sitter-{self.language_name.lower()}",
                original_error="Tree-sitter module not available",
            )

        # Special handling for TypeScript/TSX which have different attribute names
        if self.language_name == "typescript":
            lang_func = self.tree_sitter_module.language_typescript
            result = lang_func() if callable(lang_func) else lang_func
            return self._handle_language_result(result)
        elif self.language_name == "tsx":
            lang_func = self.tree_sitter_module.language_tsx
            result = lang_func() if callable(lang_func) else lang_func
            return self._handle_language_result(result)
        elif self.language_name == "jsx":
            lang_func = self.tree_sitter_module.language_tsx
            result = lang_func() if callable(lang_func) else lang_func
            return self._handle_language_result(result)
        elif self.language_name == "javascript" and hasattr(
            self.tree_sitter_module, "language_javascript"
        ):
            # Some versions use language_javascript
            lang_func = self.tree_sitter_module.language_javascript
            result = lang_func() if callable(lang_func) else lang_func
            return self._handle_language_result(result)
        else:
            # Standard case - most tree-sitter modules use .language function
            lang_func = self.tree_sitter_module.language
            result = lang_func() if callable(lang_func) else lang_func
            return self._handle_language_result(result)


# Language configuration mapping
LANGUAGE_CONFIGS: dict[Language, LanguageConfig] = {
    Language.PYTHON: LanguageConfig(
        ts_python, PythonMapping, PYTHON_AVAILABLE, "python"
    ),
    Language.JAVASCRIPT: LanguageConfig(
        ts_javascript, JavaScriptMapping, JAVASCRIPT_AVAILABLE, "javascript"
    ),
    Language.TYPESCRIPT: LanguageConfig(
        ts_typescript, TypeScriptMapping, TYPESCRIPT_AVAILABLE, "typescript"
    ),
    Language.JAVA: LanguageConfig(ts_java, JavaMapping, JAVA_AVAILABLE, "java"),
    Language.C: LanguageConfig(ts_c, CMapping, C_AVAILABLE, "c"),
    Language.CPP: LanguageConfig(ts_cpp, CppMapping, CPP_AVAILABLE, "cpp"),
    Language.CSHARP: LanguageConfig(
        ts_csharp, CSharpMapping, CSHARP_AVAILABLE, "csharp"
    ),
    Language.GO: LanguageConfig(ts_go, GoMapping, GO_AVAILABLE, "go"),
    Language.HASKELL: LanguageConfig(
        ts_haskell, HaskellMapping, HASKELL_AVAILABLE, "haskell"
    ),
    Language.RUST: LanguageConfig(ts_rust, RustMapping, RUST_AVAILABLE, "rust"),
    Language.ZIG: LanguageConfig(ts_zig, ZigMapping, ZIG_AVAILABLE, "zig"),
    Language.BASH: LanguageConfig(ts_bash, BashMapping, BASH_AVAILABLE, "bash"),
    Language.KOTLIN: LanguageConfig(
        ts_kotlin, KotlinMapping, KOTLIN_AVAILABLE, "kotlin"
    ),
    Language.GROOVY: LanguageConfig(
        ts_groovy, GroovyMapping, GROOVY_AVAILABLE, "groovy"
    ),
    Language.MATLAB: LanguageConfig(
        ts_matlab, MatlabMapping, MATLAB_AVAILABLE, "matlab"
    ),
    Language.JSON: LanguageConfig(ts_json, JsonMapping, JSON_AVAILABLE, "json"),
    Language.YAML: LanguageConfig(ts_yaml, YamlMapping, YAML_AVAILABLE, "yaml"),
    Language.TOML: LanguageConfig(ts_toml, TomlMapping, TOML_AVAILABLE, "toml"),
    Language.MARKDOWN: LanguageConfig(
        ts_markdown, MarkdownMapping, MARKDOWN_AVAILABLE, "markdown"
    ),
    Language.MAKEFILE: LanguageConfig(
        ts_make, MakefileMapping, MAKEFILE_AVAILABLE, "makefile"
    ),
    Language.JSX: LanguageConfig(
        ts_typescript, JSXMapping, JSX_AVAILABLE, "jsx"
    ),  # JSX uses TSX grammar
    Language.TSX: LanguageConfig(
        ts_typescript, TSXMapping, TSX_AVAILABLE, "tsx"
    ),  # TSX uses TS parser with tsx language
    Language.TEXT: LanguageConfig(
        None, TextMapping, True, "text"
    ),  # Text doesn't need tree-sitter
    Language.PDF: LanguageConfig(
        None, PDFMapping, True, "pdf"
    ),  # PDF doesn't need tree-sitter
}

# File extension to language mapping
EXTENSION_TO_LANGUAGE: dict[str, Language] = {
    # Python
    ".py": Language.PYTHON,
    ".pyi": Language.PYTHON,
    ".pyw": Language.PYTHON,
    # JavaScript & TypeScript
    ".js": Language.JAVASCRIPT,
    ".mjs": Language.JAVASCRIPT,
    ".cjs": Language.JAVASCRIPT,
    ".jsx": Language.JSX,
    ".ts": Language.TYPESCRIPT,
    ".tsx": Language.TSX,
    ".mts": Language.TYPESCRIPT,
    ".cts": Language.TYPESCRIPT,
    # Java & JVM Languages
    ".java": Language.JAVA,
    ".kt": Language.KOTLIN,
    ".kts": Language.KOTLIN,
    ".groovy": Language.GROOVY,
    ".gvy": Language.GROOVY,
    ".gy": Language.GROOVY,
    ".gsh": Language.GROOVY,
    # C/C++
    ".c": Language.C,
    ".h": Language.C,
    ".cpp": Language.CPP,
    ".cxx": Language.CPP,
    ".cc": Language.CPP,
    ".c++": Language.CPP,
    ".hpp": Language.CPP,
    ".hxx": Language.CPP,
    ".hh": Language.CPP,
    ".h++": Language.CPP,
    # C#
    ".cs": Language.CSHARP,
    ".csx": Language.CSHARP,
    # Other languages
    ".go": Language.GO,
    ".hs": Language.HASKELL,
    ".lhs": Language.HASKELL,
    ".hs-boot": Language.HASKELL,
    ".hsig": Language.HASKELL,
    ".hsc": Language.HASKELL,
    ".rs": Language.RUST,
    ".zig": Language.ZIG,
    ".sh": Language.BASH,
    ".bash": Language.BASH,
    ".zsh": Language.BASH,
    ".fish": Language.BASH,
    ".m": Language.MATLAB,
    # Config & Data
    ".json": Language.JSON,
    ".yaml": Language.YAML,
    ".yml": Language.YAML,
    ".toml": Language.TOML,
    ".md": Language.MARKDOWN,
    ".markdown": Language.MARKDOWN,
    ".mdown": Language.MARKDOWN,
    ".mkd": Language.MARKDOWN,
    ".mdx": Language.MARKDOWN,
    # Build systems
    "makefile": Language.MAKEFILE,
    "Makefile": Language.MAKEFILE,
    "GNUmakefile": Language.MAKEFILE,
    ".mk": Language.MAKEFILE,
    ".mak": Language.MAKEFILE,
    # Text files (fallback)
    ".txt": Language.TEXT,
    ".text": Language.TEXT,
    ".log": Language.TEXT,
    ".cfg": Language.TEXT,
    ".conf": Language.TEXT,
    ".ini": Language.TEXT,
    # PDF files
    ".pdf": Language.PDF,
}


class ParserFactory:
    """Factory for creating unified parsers with all language mappings.

    This factory provides a clean interface for creating UniversalParser instances
    with the appropriate language configuration. It handles tree-sitter module
    availability and provides fallback options.
    """

    def __init__(self, default_cast_config: CASTConfig | None = None):
        """Initialize parser factory.

        Args:
            default_cast_config: Default cAST configuration for all parsers
        """
        self.default_cast_config = default_cast_config or CASTConfig()
        self._parser_cache: dict[Language, UniversalParser] = {}

    def create_parser(
        self,
        language: Language,
        cast_config: CASTConfig | None = None,
    ) -> UniversalParser:
        """Create a universal parser for the specified language.

        Args:
            language: Programming language to create parser for
            cast_config: Optional cAST configuration (uses default if not provided)

        Returns:
            UniversalParser instance configured for the language

        Raises:
            SetupError: If the required tree-sitter module is not available
            ValueError: If the language is not supported
        """
        # Use cache to avoid recreating parsers
        cache_key = language
        if cache_key in self._parser_cache:
            return self._parser_cache[cache_key]

        if language not in LANGUAGE_CONFIGS:
            raise ValueError(f"Unsupported language: {language}")

        config = LANGUAGE_CONFIGS[language]
        cast_config = cast_config or self.default_cast_config
        # Haskell-specific cAST tuning: avoid greedy merging of adjacent definitions
        if language == Language.HASKELL:
            from chunkhound.parsers.universal_parser import CASTConfig as _CAST

            cast_config = _CAST(
                max_chunk_size=cast_config.max_chunk_size,
                min_chunk_size=cast_config.min_chunk_size,
                merge_threshold=cast_config.merge_threshold,
                preserve_structure=cast_config.preserve_structure,
                greedy_merge=False,
                safe_token_limit=cast_config.safe_token_limit,
            )

        # Import TreeSitterEngine here to avoid circular imports

        # Special handling for text and PDF files (no tree-sitter required)
        if language in (Language.TEXT, Language.PDF):
            # Text and PDF mappings don't need tree-sitter engine
            mapping = config.mapping_class()
            # For text and PDF files, we'll handle this specially in the parser
            parser = UniversalParser(None, mapping, cast_config)  # type: ignore
            self._parser_cache[cache_key] = parser
            return parser

        if not config.available:
            raise SetupError(
                parser=config.language_name,
                missing_dependency=f"tree-sitter-{config.language_name.lower()}",
                install_command=f"pip install tree-sitter-{config.language_name.lower()}",
                original_error="Tree-sitter module not available",
            )

        try:
            # Get tree-sitter language object
            ts_language = config.get_tree_sitter_language()

            # Create engine and mapping
            engine = TreeSitterEngine(config.language_name, ts_language)
            mapping = config.mapping_class()

            # Create parser
            parser = UniversalParser(
                engine,
                mapping,
                cast_config,
            )

            # Cache for future use
            self._parser_cache[cache_key] = parser

            return parser

        except Exception as e:
            raise SetupError(
                parser=config.language_name,
                missing_dependency=f"tree-sitter-{config.language_name.lower()}",
                install_command=f"pip install tree-sitter-{config.language_name.lower()}",
                original_error=str(e),
            ) from e

    def create_parser_for_file(
        self, file_path: Path, cast_config: CASTConfig | None = None
    ) -> UniversalParser:
        """Create a parser appropriate for the given file.

        Args:
            file_path: Path to the file to parse
            cast_config: Optional cAST configuration

        Returns:
            UniversalParser instance appropriate for the file

        Raises:
            SetupError: If the required tree-sitter module is not available
            ValueError: If the file type is not supported
        """
        language = self.detect_language(file_path)
        return self.create_parser(language, cast_config)

    def detect_language(self, file_path: Path) -> Language:
        """Detect the programming language of a file based on its extension.

        Args:
            file_path: Path to the file to analyze

        Returns:
            Detected Language enum value
        """
        # Check by extension first
        extension = file_path.suffix.lower()
        if extension in EXTENSION_TO_LANGUAGE:
            return EXTENSION_TO_LANGUAGE[extension]

        # Check by filename (for files like Makefile)
        filename = file_path.name
        if filename in EXTENSION_TO_LANGUAGE:
            return EXTENSION_TO_LANGUAGE[filename]

        # Check by stem (for files like Makefile.debug)
        stem = file_path.stem.lower()
        if stem in ["makefile", "gnumakefile"]:
            return Language.MAKEFILE

        # Fallback to text for unknown files
        return Language.TEXT

    def get_available_languages(self) -> dict[Language, bool]:
        """Get a dictionary of all languages and their availability status.

        Returns:
            Dictionary mapping Language to availability boolean
        """
        return {
            language: config.available for language, config in LANGUAGE_CONFIGS.items()
        }

    def get_supported_extensions(self) -> dict[str, Language]:
        """Get all supported file extensions and their associated languages.

        Returns:
            Dictionary mapping file extensions to Language enum values
        """
        return EXTENSION_TO_LANGUAGE.copy()

    def is_language_available(self, language: Language) -> bool:
        """Check if a specific language is available (tree-sitter module installed).

        Args:
            language: Language to check

        Returns:
            True if the language is supported and available
        """
        return LANGUAGE_CONFIGS.get(
            language, LanguageConfig(None, TextMapping, False, "unknown")
        ).available

    def get_missing_dependencies(self) -> dict[Language, str]:
        """Get a list of missing dependencies for unavailable languages.

        Returns:
            Dictionary mapping Language to installation command for missing languages
        """
        missing = {}
        for language, config in LANGUAGE_CONFIGS.items():
            if not config.available and language not in (Language.TEXT, Language.PDF):
                missing[language] = (
                    f"pip install tree-sitter-{config.language_name.lower()}"
                )
        return missing

    def clear_cache(self) -> None:
        """Clear the parser cache to free memory."""
        self._parser_cache.clear()

    def get_statistics(self) -> dict[str, Any]:
        """Get factory statistics.

        Returns:
            Dictionary with factory statistics
        """
        available_count = sum(
            1 for config in LANGUAGE_CONFIGS.values() if config.available
        )
        total_count = len(LANGUAGE_CONFIGS)
        cached_count = len(self._parser_cache)

        return {
            "total_languages": total_count,
            "available_languages": available_count,
            "unavailable_languages": total_count - available_count,
            "cached_parsers": cached_count,
            "supported_extensions": len(EXTENSION_TO_LANGUAGE),
            "availability_ratio": available_count / total_count
            if total_count > 0
            else 0.0,
        }


# Global factory instance for convenience
_global_factory: ParserFactory | None = None


def get_parser_factory(cast_config: CASTConfig | None = None) -> ParserFactory:
    """Get the global parser factory instance.

    Args:
        cast_config: Optional cAST configuration for the factory

    Returns:
        Global ParserFactory instance
    """
    global _global_factory
    if _global_factory is None or cast_config is not None:
        _global_factory = ParserFactory(cast_config)
    return _global_factory


def create_parser_for_file(
    file_path: Path, cast_config: CASTConfig | None = None
) -> UniversalParser:
    """Convenience function to create a parser for a file.

    Args:
        file_path: Path to the file to parse
        cast_config: Optional cAST configuration

    Returns:
        UniversalParser instance appropriate for the file
    """
    factory = get_parser_factory(cast_config)
    return factory.create_parser_for_file(file_path, cast_config)


def create_parser_for_language(
    language: Language, cast_config: CASTConfig | None = None
) -> UniversalParser:
    """Convenience function to create a parser for a language.

    Args:
        language: Programming language to create parser for
        cast_config: Optional cAST configuration

    Returns:
        UniversalParser instance configured for the language
    """
    factory = get_parser_factory(cast_config)
    return factory.create_parser(language, cast_config)
