"""Grammar availability flags for optional tree-sitter grammars.

This module is intentionally lightweight — it imports nothing from ChunkHound
so that both ``common.py`` and ``parser_factory.py`` can import from it
without creating a circular dependency.
"""

from tree_sitter_language_pack import get_language as _get_lang

# SCSS grammar is bundled in tree-sitter-language-pack but may not be present
# in all builds.  ``get_language`` returns None when the grammar is missing.
SCSS_AVAILABLE: bool = _get_lang("scss") is not None
