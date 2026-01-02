"""Haskell language mapping for the unified parser architecture.

This mapping extracts semantic concepts from Haskell source using the
Tree-sitter grammar. It leverages the base mapping adapter so function
definitions, data/newtype declarations, type synonyms, and type classes can be
fed into the universal ConceptExtractor.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.base import MAX_CONSTANT_VALUE_LENGTH, BaseMapping
from chunkhound.parsers.universal_engine import UniversalConcept

if TYPE_CHECKING:
    from tree_sitter import Node as TSNode

try:
    from tree_sitter import Node as TSNode

    TREE_SITTER_AVAILABLE = True
except ImportError:  # pragma: no cover - handled in runtime environments
    TREE_SITTER_AVAILABLE = False
    TSNode = Any  # type: ignore


class HaskellMapping(BaseMapping):
    """Haskell-specific mapping implementation."""

    def __init__(self) -> None:
        super().__init__(Language.HASKELL)

    # BaseMapping abstract methods -------------------------------------------------
    def get_function_query(self) -> str:
        """Capture top-level function bindings."""
        return """
            (function
                name: (_) @function_name
            ) @function_def

            (bind
                name: (_) @function_name
            ) @function_def

            (pattern_synonym
                (signature
                    synonym: (_) @function_name
                )
            ) @function_def

            (pattern_synonym
                (equation
                    synonym: (_) @function_name
                )
            ) @function_def
        """

    def get_class_query(self) -> str:
        """Capture algebraic data types, newtypes, type synonyms, and type classes."""
        return """
            (data_type
                name: (_) @class_name
            ) @class_def

            (newtype
                name: (_) @class_name
            ) @class_def

            (type_synonym
                name: (_) @class_name
            ) @class_def

            (type_family
                name: (_) @class_name
            ) @class_def

            (data_family
                name: (_) @class_name
            ) @class_def

            (instance
                name: (_) @class_name
            ) @class_def

            (class
                name: (_) @class_name
            ) @class_def
        """

    def get_method_query(self) -> str:
        """Capture methods defined inside type classes."""
        return """
            (class
                declarations: (class_declarations
                    (function
                        name: (_) @method_name
                    ) @method_def
                )
            )

            (class
                declarations: (_
                    (function
                        name: (_) @method_name
                    ) @method_def
                )
            )

            (class
                declarations: (_
                    (bind
                        name: (_) @method_name
                    ) @method_def
                )
            )

            (instance
                declarations: (_
                    (function
                        name: (_) @method_name
                    ) @method_def
                )
            )

            (instance
                declarations: (_
                    (bind
                        name: (_) @method_name
                    ) @method_def
                )
            )
        """

    def get_comment_query(self) -> str:
        """Capture line, block, and Haddock comments."""
        return """
            (comment) @comment
            (haddock) @comment
        """

    def extract_function_name(self, node: TSNode | None, source: str) -> str:
        """Extract the bound function name, falling back when necessary."""
        if not TREE_SITTER_AVAILABLE or node is None:
            return self.get_fallback_name(node, "function")

        # Functions and binds expose 'name'; pattern synonyms expose 'synonym'
        name_node = node.child_by_field_name("name")
        if name_node is None and node.type == "pattern_synonym":
            name_node = node.child_by_field_name("synonym")
        if name_node is None and node.child_count > 0:
            name_node = node.child(0)

        if name_node is not None:
            text = self.get_node_text(name_node, source).strip()
            if text:
                return text

        return self.get_fallback_name(node, "function")

    def extract_class_name(self, node: TSNode | None, source: str) -> str:
        """Extract the declared type name for data/newtype/class/type synonym."""
        if not TREE_SITTER_AVAILABLE or node is None:
            return self.get_fallback_name(node, "type")

        name_node = node.child_by_field_name("name")
        if name_node is None and node.child_count > 0:
            name_node = node.child(0)

        text = ""
        if name_node is not None:
            text = self.get_node_text(name_node, source).strip()

        if node.type == "instance":
            type_patterns = node.child_by_field_name("type_patterns")
            if type_patterns is not None:
                patterns_text = self.get_node_text(type_patterns, source).strip()
                if patterns_text:
                    text = f"{text} {patterns_text}".strip()
        else:
            param_field = node.child_by_field_name(
                "type_params"
            ) or node.child_by_field_name("patterns")
            if param_field is not None:
                params_text = self.get_node_text(param_field, source).strip()
                if params_text:
                    text = f"{text} {params_text}".strip()

        if text:
            return text

        return self.get_fallback_name(node, "type")

    # Optional overrides -----------------------------------------------------------
    # Uses BaseMapping default filtering behaviour.

    # LanguageMapping protocol methods --------------------------------------------
    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:  # type: ignore[override]
        """Provide universal concept queries for Haskell.

        Returns:
            Tree-sitter query string for the requested universal concept, or None.
        """
        if concept == UniversalConcept.DEFINITION:
            # Unify core Haskell definitions under a single @definition with @name
            # Keep this conservative to ensure compatibility across grammar versions.
            return """
            (function) @definition

            (bind) @definition

            (pattern_synonym) @definition

            ; Capture let bindings (inside expressions)
            (local_binds
                (bind) @definition
            )
            """

        elif concept == UniversalConcept.BLOCK:
            # Methods (functions/binds) defined within class or instance declarations
            return """
            (class
                declarations: (_
                    (function
                        name: (_) @method_name
                    ) @definition
                )
            )

            (class
                declarations: (_
                    (bind
                        name: (_) @method_name
                    ) @definition
                )
            )

            (instance
                declarations: (_
                    (function
                        name: (_) @method_name
                    ) @definition
                )
            )

            (instance
                declarations: (_
                    (bind
                        name: (_) @method_name
                    ) @definition
                )
            )
            """

        elif concept == UniversalConcept.COMMENT:
            return """
            (comment) @definition
            (haddock) @definition
            """

        elif concept == UniversalConcept.IMPORT:
            # Capture import declarations (module name will be parsed from text)
            return """
            (import) @definition
            """

        elif concept == UniversalConcept.STRUCTURE:
            # Capture module header (module X where ...) if present
            return """
            (module) @definition
            """

        return None

    def extract_name(
        self, concept: UniversalConcept, captures: dict[str, TSNode], content: bytes
    ) -> str:  # type: ignore[override]
        """Extract name for a universal concept using Haskell semantics."""
        # Decode once
        source = content.decode("utf-8", errors="replace")

        # Prefer unified name capture when present
        if concept == UniversalConcept.DEFINITION:
            if "name" in captures:
                text = self.get_node_text(captures["name"], source).strip()
                if text:
                    return text
            # Fallback to function/class extractors
            def_node = captures.get("definition") or next(iter(captures.values()), None)
            # Heuristic: use function extractor when node appears to be function-like
            if def_node is not None:
                if def_node.type in {"function", "bind", "pattern_synonym"}:
                    return self.extract_function_name(def_node, source)
                else:
                    return self.extract_class_name(def_node, source)
            return "unnamed_definition"

        elif concept == UniversalConcept.BLOCK:
            if "method_name" in captures:
                text = self.get_node_text(captures["method_name"], source).strip()
                if text:
                    return text
            # Fallback to function name from definition node
            def_node = captures.get("definition") or next(iter(captures.values()), None)
            return (
                self.extract_method_name(def_node, source)
                if def_node
                else "unnamed_block"
            )

        elif concept == UniversalConcept.COMMENT:
            # Location-based comment name for consistency
            node = captures.get("definition") or next(iter(captures.values()), None)
            if node is not None:
                line = node.start_point[0] + 1
                return f"comment_line_{line}"
            return "unnamed_comment"

        elif concept == UniversalConcept.IMPORT:
            # Parse module name from the import line text
            node = captures.get("definition") or next(iter(captures.values()), None)
            if node is not None:
                text = self.get_node_text(node, source)
                return self._extract_import_module_name(text)
            return "import_unknown"

        elif concept == UniversalConcept.STRUCTURE:
            node = captures.get("definition") or next(iter(captures.values()), None)
            if node is not None:
                text = self.get_node_text(node, source)
                return self._extract_module_name(text)
            return "module_unknown"

        return f"unnamed_{concept.value}"

    def extract_content(
        self, concept: UniversalConcept, captures: dict[str, TSNode], content: bytes
    ) -> str:  # type: ignore[override]
        """Extract raw content for the captured node."""
        source = content.decode("utf-8", errors="replace")
        node = captures.get("definition") or next(iter(captures.values()), None)
        return self.get_node_text(node, source) if node is not None else ""

    def extract_metadata(
        self, concept: UniversalConcept, captures: dict[str, TSNode], content: bytes
    ) -> dict[str, Any]:  # type: ignore[override]
        """Provide light metadata for concepts."""
        meta: dict[str, Any] = {
            "concept": concept.value,
            "language": self.language.value,
        }
        node = captures.get("definition") or next(iter(captures.values()), None)
        if node is not None:
            meta["node_type"] = getattr(node, "type", "")
        if concept == UniversalConcept.IMPORT:
            # Also include full import text
            src = content.decode("utf-8", errors="replace")
            if node is not None:
                meta["import_text"] = self.get_node_text(node, src).strip()
        return meta

    # Helpers ----------------------------------------------------------------------
    def _extract_import_module_name(self, text: str) -> str:
        """Heuristic extraction of module name from an import line."""
        # Remove leading/trailing spaces and normalize whitespace
        stripped = " ".join((text or "").strip().split())
        if not stripped.lower().startswith("import "):
            return "import_unknown"
        # Remove leading 'import'
        rest = stripped[len("import ") :]
        # Drop optional qualifiers and take the first token as module name
        tokens = rest.replace("(", " ").replace(")", " ").split()
        filtered: list[str] = []
        skip_next = False
        keywords = {"qualified", "safe", "{-#", "#-"}
        for i, tok in enumerate(tokens):
            if skip_next:
                skip_next = False
                continue
            low = tok.lower()
            if low in keywords:
                continue
            if low == "as" or low == "hiding":
                # Stop before alias/hiding details
                break
            filtered.append(tok)
            break
        return filtered[0] if filtered else "import_unknown"

    def _extract_module_name(self, text: str) -> str:
        """Extract module name from a module header line."""
        stripped = " ".join((text or "").strip().split())
        if not stripped.lower().startswith("module "):
            return "module_unknown"
        rest = stripped[len("module ") :]
        # Up to the first 'where'
        if " where" in rest:
            rest = rest.split(" where", 1)[0]
        return rest.strip() or "module_unknown"

    def extract_constants(
        self, concept: UniversalConcept, captures: dict[str, TSNode], content: bytes
    ) -> list[dict[str, str]] | None:
        """Extract constant definitions from Haskell code.

        Identifies bindings with simple constant patterns (literals, lists, etc.)
        from multiple contexts:
        - Top-level bindings: `maxValue = 100`
        - Let bindings: `let localConst = 42 in ...`
        - Where clauses: `... where x = 10`

        Rejects function definitions (bindings with parameters) and complex
        expressions (lambdas, case, let, do, etc.).

        Args:
            concept: The universal concept being extracted
            captures: Dictionary of capture names to tree-sitter nodes
            content: Source code as bytes

        Returns:
            List of constant dictionaries with 'name' and 'value' keys, or None
        """
        if concept != UniversalConcept.DEFINITION:
            return None

        # Get the definition node
        def_node = captures.get("definition")
        if not def_node or def_node.type not in ["function", "bind"]:
            return None

        # Skip function definitions (they have patterns)
        if def_node.type == "function":
            return None

        source = content.decode("utf-8")

        # Extract name from variable node
        name_node = def_node.child_by_field_name("name")
        if not name_node:
            return None

        name = self.get_node_text(name_node, source).strip()

        # Must start with lowercase (Haskell naming convention)
        if not name or not name[0].islower():
            return None

        # Get the match node (contains = and RHS)
        match_node = def_node.child_by_field_name("match")
        if not match_node:
            return None

        # Find the RHS value node (skip the '=' token)
        value_node = None
        for child in match_node.children:
            if child.type != "=":
                value_node = child
                break

        if not value_node:
            return None

        # Check if the value is a simple constant
        # Accept: literal, list, variable, record, tuple, string
        # Reject: lambda, case, let, where, do, function application with complex args
        simple_value_types = {
            "literal", "list", "variable", "record", "tuple", "string",
            "quoted_name", "con_unit", "integer", "float", "char",
        }

        # Reject complex expressions
        if value_node.type in {"lambda", "case", "let_in", "do"}:
            return None

        # For other types, check if children are simple (heuristic)
        # This handles constructors like "Just 42" or "Config {...}"
        if value_node.type not in simple_value_types:
            # Check if it contains lambda, case, let, etc. in subtree
            if self._contains_complex_expression(value_node):
                return None

        # Extract value text
        value = self.get_node_text(value_node, source).strip()

        # Truncate long values
        if len(value) > MAX_CONSTANT_VALUE_LENGTH:
            value = value[:MAX_CONSTANT_VALUE_LENGTH] + "..."

        return [{"name": name, "value": value}]

    def _contains_complex_expression(self, node: TSNode) -> bool:
        """Check if node contains complex expressions (lambda, case, let, etc.)."""
        complex_types = {"lambda", "case", "let_in", "do", "where"}

        if node.type in complex_types:
            return True

        # Recursively check children (limit depth to avoid performance issues)
        for child in node.children:
            if self._contains_complex_expression(child):
                return True

        return False

    def resolve_import_path(
        self, import_text: str, base_dir: Path, source_file: Path
    ) -> Path | None:
        """Resolve import path for Haskell.

        Haskell module resolution is complex and typically handled by the build system.

        Args:
            import_text: The import statement text
            base_dir: Base directory of the project
            source_file: Path to the file containing the import

        Returns:
            None (Haskell module resolution is complex, not file-based)
        """
        # Haskell module resolution is complex, return None
        return None
