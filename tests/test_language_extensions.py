"""Tests for language extension consistency across modules.

This test suite ensures that extension lists remain synchronized across:
- Language.get_all_extensions() in core/types/common.py
- EXTENSION_TO_LANGUAGE in parsers/parser_factory.py
- Realtime service fallback in services/realtime_indexing_service.py

These tests prevent the extension list desynchronization bug that caused
newly added languages (Zig, PHP variants, etc.) to be filtered out during
real-time filesystem events.
"""

from pathlib import Path

import pytest

from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import EXTENSION_TO_LANGUAGE
from chunkhound.services.realtime_indexing_service import SimpleEventHandler


class TestExtensionConsistency:
    """Test suite for extension list synchronization."""

    def test_language_extensions_match_parser_factory(self):
        """Language.get_all_extensions() must match parser_factory.EXTENSION_TO_LANGUAGE.

        This is the primary consistency check - ensures the Language enum's
        get_all_extensions() method returns exactly the same extensions as
        the parser factory supports.
        """
        lang_exts = Language.get_all_extensions()
        factory_exts = {
            ext for ext in EXTENSION_TO_LANGUAGE.keys() if ext.startswith(".")
        }

        # Check for missing extensions
        missing = factory_exts - lang_exts
        assert not missing, (
            f"Language.get_all_extensions() missing {len(missing)} extensions "
            f"that are supported by parsers: {sorted(missing)}"
        )

        # Check for extra extensions (shouldn't happen with current implementation)
        extra = lang_exts - factory_exts
        assert not extra, (
            f"Language.get_all_extensions() has {len(extra)} extensions "
            f"not in parser factory: {sorted(extra)}"
        )

        # Verify exact match
        assert lang_exts == factory_exts, (
            "Extension lists must match exactly. "
            f"Missing: {sorted(factory_exts - lang_exts)}, "
            f"Extra: {sorted(lang_exts - factory_exts)}"
        )

    def test_realtime_service_supports_all_extensions(self):
        """Realtime service fallback must index all parser-supported extensions.

        This ensures that when no config is provided, the realtime service's
        fallback mechanism can index all file types that the parser system supports.
        """
        # Create handler without config (triggers fallback mode)
        handler = SimpleEventHandler(None, None, None)

        # Test each extension in EXTENSION_TO_LANGUAGE
        unsupported = []
        for ext in EXTENSION_TO_LANGUAGE.keys():
            if ext.startswith("."):
                test_file = Path(f"test{ext}")
                if not handler._should_index(test_file):
                    unsupported.append(ext)

        assert not unsupported, (
            f"Realtime service cannot index {len(unsupported)} supported extensions: "
            f"{sorted(unsupported)}"
        )

    def test_language_filename_patterns_match_parser_factory(self):
        """Language.get_all_filename_patterns() must match parser_factory.

        This ensures filename patterns (like Makefile, Dockerfile) are derived
        from parser_factory and stay in sync.
        """
        lang_patterns = Language.get_all_filename_patterns()
        factory_patterns = {
            key.lower()
            for key in EXTENSION_TO_LANGUAGE.keys()
            if not key.startswith(".")
        }

        # Check for missing patterns
        missing = factory_patterns - lang_patterns
        assert not missing, (
            f"Language.get_all_filename_patterns() missing {len(missing)} patterns "
            f"from parser_factory: {sorted(missing)}"
        )

        # Check for extra patterns
        extra = lang_patterns - factory_patterns
        assert not extra, (
            f"Language.get_all_filename_patterns() has {len(extra)} patterns "
            f"not in parser_factory: {sorted(extra)}"
        )

        # Verify exact match
        assert lang_patterns == factory_patterns, (
            "Filename pattern lists must match exactly. "
            f"Missing: {sorted(factory_patterns - lang_patterns)}, "
            f"Extra: {sorted(lang_patterns - factory_patterns)}"
        )

    def test_realtime_service_supports_filename_patterns(self):
        """Realtime service must index filename-based patterns like Makefile.

        Some languages are detected by filename rather than extension (e.g., Makefile).
        The realtime service must support these as well.
        """
        handler = SimpleEventHandler(None, None, None)

        # Test filename patterns from EXTENSION_TO_LANGUAGE
        filename_patterns = [
            key for key in EXTENSION_TO_LANGUAGE.keys() if not key.startswith(".")
        ]

        unsupported = []
        for filename in filename_patterns:
            test_file = Path(filename)
            if not handler._should_index(test_file):
                unsupported.append(filename)

        assert not unsupported, (
            f"Realtime service cannot index {len(unsupported)} filename patterns: "
            f"{sorted(unsupported)}"
        )

    def test_realtime_service_filename_case_insensitive(self):
        """Realtime service must handle filename patterns case-insensitively.

        Tests all case variants of filename patterns to ensure cross-platform
        compatibility (macOS/Windows are case-insensitive).
        """
        handler = SimpleEventHandler(None, None, None)

        # Get all filename patterns from parser_factory
        filename_patterns = [
            key for key in EXTENSION_TO_LANGUAGE.keys() if not key.startswith(".")
        ]

        failures = []
        for pattern in filename_patterns:
            # Test exact case
            if not handler._should_index(Path(pattern)):
                failures.append(f"{pattern} (exact case)")

            # Test lowercase
            if not handler._should_index(Path(pattern.lower())):
                failures.append(f"{pattern.lower()} (lowercase)")

            # Test uppercase
            if not handler._should_index(Path(pattern.upper())):
                failures.append(f"{pattern.upper()} (uppercase)")

        assert not failures, (
            f"Realtime service failed to index {len(failures)} filename variants: "
            f"{failures}"
        )

    def test_makefile_variants_regression(self):
        """Regression test: Ensure all Makefile variants are detected.

        This is a critical bug fix - the original hardcoded check missed
        'Makefile' (capital M) which is the MOST COMMON form in real projects.
        """
        handler = SimpleEventHandler(None, None, None)

        # These are the most common Makefile variants in real projects
        makefile_variants = [
            "Makefile",  # Most common (capital M) - CRITICAL
            "makefile",  # Lowercase variant
            "GNUmakefile",  # GNU Make specific
            "gnumakefile",  # Lowercase GNU variant
        ]

        failures = []
        for variant in makefile_variants:
            test_file = Path(f"some/path/{variant}")
            if not handler._should_index(test_file):
                failures.append(variant)

        assert not failures, (
            f"CRITICAL BUG: Failed to detect Makefile variants {failures}. "
            f"These are commonly used in real projects and MUST be supported."
        )

    def test_specific_new_languages_supported(self):
        """Verify that specific newly added languages work in real-time indexing.

        This is a regression test for the original bug report - Zig, PHP variants,
        and other recently added languages must work in real-time mode.
        """
        handler = SimpleEventHandler(None, None, None)

        # Languages that were reported as broken
        test_cases = [
            (".zig", "Zig language"),
            (".php", "PHP base"),
            (".phtml", "PHP HTML template"),
            (".php3", "PHP3 legacy"),
            (".php4", "PHP4 legacy"),
            (".php5", "PHP5 legacy"),
            (".phps", "PHP source"),
            (".pdf", "PDF documents"),
            (".vue", "Vue.js components"),
        ]

        failures = []
        for ext, description in test_cases:
            test_file = Path(f"test{ext}")
            if not handler._should_index(test_file):
                failures.append(f"{ext} ({description})")

        assert not failures, (
            f"Realtime service cannot index newly added languages: "
            f"{', '.join(failures)}"
        )

    def test_file_patterns_include_all_extensions(self):
        """Language.get_file_patterns() must include patterns for all extensions.

        This ensures that the glob patterns used for file discovery include
        all parser-supported extensions.
        """
        patterns = Language.get_file_patterns()
        extensions = Language.get_all_extensions()

        # Extract extensions from patterns (e.g., "**/*.py" -> ".py")
        pattern_extensions = set()
        for pattern in patterns:
            if pattern.startswith("**/") and pattern[3] == "*":
                # Extension pattern like "**/*.py"
                ext = pattern[4:]  # Skip "***/" to get extension
                if ext.startswith("."):
                    pattern_extensions.add(ext)

        # Check that all extensions have patterns
        missing = extensions - pattern_extensions
        assert not missing, (
            f"Language.get_file_patterns() missing {len(missing)} extension patterns: "
            f"{sorted(missing)}"
        )

    def test_no_unsupported_extensions_in_realtime_fallback(self):
        """Realtime service fallback should not accept extensions not in parser system.

        This prevents the service from accepting files that can't actually be parsed.
        """
        handler = SimpleEventHandler(None, None, None)

        # Test extensions that are NOT in the parser system
        unsupported_extensions = [
            ".rb",  # Ruby (no parser)
            ".scala",  # Scala (no parser)
            ".r",  # R language (no parser)
            ".xyz",  # Completely invalid
        ]

        false_positives = []
        for ext in unsupported_extensions:
            test_file = Path(f"test{ext}")
            if handler._should_index(test_file):
                false_positives.append(ext)

        assert not false_positives, (
            f"Realtime service incorrectly accepts {len(false_positives)} "
            f"unsupported extensions: {sorted(false_positives)}"
        )


class TestExtensionCoverage:
    """Test coverage of specific language families."""

    @pytest.mark.parametrize(
        "ext",
        [
            ".py",
            ".pyi",
            ".pyw",  # Python variants
            ".js",
            ".mjs",
            ".cjs",  # JavaScript variants
            ".ts",
            ".mts",
            ".cts",  # TypeScript variants
            ".tsx",
            ".jsx",  # React variants
            ".php",
            ".phtml",
            ".php3",
            ".php4",
            ".php5",
            ".phps",  # PHP variants
            ".zig",  # Zig
            ".vue",  # Vue
        ],
    )
    def test_language_variant_extensions_supported(self, ext):
        """Test that all language variants are in get_all_extensions().

        Note: We only test that the extension is in get_all_extensions(),
        not that Language.from_file_extension() maps it correctly.
        That's because from_file_extension() has a simplified mapping
        (only common variants), while the parser factory supports ALL variants.
        """
        assert ext in Language.get_all_extensions(), (
            f"{ext} not in Language.get_all_extensions() "
            f"- this means it won't be indexed in real-time mode"
        )

    def test_all_parser_factory_extensions_have_language(self):
        """Every extension in EXTENSION_TO_LANGUAGE must map to a valid Language enum."""
        for ext, language in EXTENSION_TO_LANGUAGE.items():
            if ext.startswith("."):
                assert isinstance(language, Language), (
                    f"Extension {ext} maps to invalid language: {language}"
                )
                assert language != Language.UNKNOWN, (
                    f"Extension {ext} maps to UNKNOWN language"
                )
