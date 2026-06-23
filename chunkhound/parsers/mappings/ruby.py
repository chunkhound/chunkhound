"""Ruby language mapping for unified parser architecture.

This module provides Ruby-specific tree-sitter queries and extraction logic
for mapping Ruby AST nodes to universal semantic concepts used by the unified
parser.

Ruby's tree-sitter grammar exposes real named nodes (``class``, ``module``,
``method``, ``singleton_method``), so this mapping follows the named-node
pattern used by ``python.py`` / ``java.py`` rather than the call-node pattern
used by ``elixir.py``.
"""

import re
from pathlib import Path
from typing import Any

from tree_sitter import Node as TSNode

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.base import MAX_CONSTANT_VALUE_LENGTH, BaseMapping
from chunkhound.parsers.universal_engine import UniversalConcept


class RubyMapping(BaseMapping):
    """Ruby-specific tree-sitter mapping implementation.

    Handles Ruby's core definition constructs:
    - Modules and classes (with inheritance via ``superclass``)
    - Instance methods (``method``) and class methods (``singleton_method``)
    - Constant assignments (UPPER_SNAKE_CASE / CamelCase constants)
    - Comments
    - ``require`` / ``require_relative`` imports
    """

    def __init__(self) -> None:
        """Initialize Ruby mapping."""
        super().__init__(Language.RUBY)

    # BaseMapping required (legacy) methods -------------------------------

    def get_function_query(self) -> str:
        """Get tree-sitter query pattern for Ruby method definitions."""
        # name: (_) matches every method-name node shape - identifier
        # (plain_method), setter (name=), and operator ([], <=>, +, ...) -
        # mirroring upstream tree-sitter-ruby tags.scm. (identifier) alone
        # silently drops setter and operator methods.
        return """
            (method
                name: (_) @function_name
            ) @function_def

            (singleton_method
                name: (_) @function_name
            ) @function_def
        """

    def get_class_query(self) -> str:
        """Get tree-sitter query pattern for Ruby class/module definitions."""
        return """
            (class
                name: (_) @class_name
            ) @class_def

            (module
                name: (_) @class_name
            ) @class_def
        """

    def get_comment_query(self) -> str:
        """Get tree-sitter query pattern for Ruby comments."""
        return """
            (comment) @comment
        """

    def extract_function_name(self, node: TSNode | None, source: str) -> str:
        """Extract method name from a method definition node."""
        if node is None:
            return self.get_fallback_name(node, "method")

        name_node = node.child_by_field_name("name")
        if name_node:
            return self.get_node_text(name_node, source).strip()

        return self.get_fallback_name(node, "method")

    def extract_class_name(self, node: TSNode | None, source: str) -> str:
        """Extract class/module name from a class or module definition node."""
        if node is None:
            return self.get_fallback_name(node, "class")

        name_node = node.child_by_field_name("name")
        if name_node:
            return self.get_node_text(name_node, source).strip()

        return self.get_fallback_name(node, "class")

    # LanguageMapping protocol methods -----------------------------------

    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        """Get tree-sitter query for a universal concept in Ruby."""

        if concept == UniversalConcept.DEFINITION:
            # name: (_) covers both plain constants (Foo) and scoped names
            # (Foo::Bar) for class/module definitions.
            return """
            (class
                name: (_) @name
            ) @definition

            (module
                name: (_) @name
            ) @definition

            ; name: (_) matches identifier (plain_method), setter (name=),
            ; and operator ([], <=>, +, ...) method names, matching upstream
            ; tags.scm. (identifier) alone drops setter/operator methods.
            (method
                name: (_) @name
            ) @definition

            (singleton_method
                name: (_) @name
            ) @definition

            ; Constant assignment (left side is a constant node)
            (assignment
                left: (constant) @lhs
                right: (_) @rhs
            ) @definition
            """

        elif concept == UniversalConcept.BLOCK:
            return """
            (do_block) @block
            (block) @block
            (if) @block
            (unless) @block
            (while) @block
            (until) @block
            (for) @block
            (case) @block
            (begin) @block
            """

        elif concept == UniversalConcept.COMMENT:
            return """
            (comment) @definition
            """

        elif concept == UniversalConcept.IMPORT:
            return """
            (call
                method: (identifier) @_m
                (#match? @_m "^(require|require_relative|load|autoload)$")
            ) @definition
            """

        elif concept == UniversalConcept.STRUCTURE:
            return """
            (program) @definition
            """

        return None

    def extract_name(
        self, concept: UniversalConcept, captures: dict[str, TSNode], content: bytes
    ) -> str:
        """Extract a name from captures for this concept."""

        source = content.decode("utf-8")

        if concept == UniversalConcept.DEFINITION:
            # Prefer explicit name capture (class/module/method/singleton_method)
            if "name" in captures:
                name = self.get_node_text(captures["name"], source).strip()
                if name:
                    return name

            # Constant assignment: use the left-hand constant
            if "lhs" in captures:
                name = self.get_node_text(captures["lhs"], source).strip()
                if name:
                    return name

            if "definition" in captures:
                node = captures["definition"]
                line = node.start_point[0] + 1
                return f"definition_line_{line}"

            return "unnamed_definition"

        elif concept == UniversalConcept.BLOCK:
            if "block" in captures:
                node = captures["block"]
                line = node.start_point[0] + 1
                return f"{node.type}_line_{line}"

            return "unnamed_block"

        elif concept == UniversalConcept.COMMENT:
            if "definition" in captures:
                node = captures["definition"]
                line = node.start_point[0] + 1
                return f"comment_line_{line}"

            return "unnamed_comment"

        elif concept == UniversalConcept.IMPORT:
            if "definition" in captures:
                def_text = self.get_node_text(captures["definition"], source).strip()
                match = re.search(
                    r"(?:require_relative|require|load|autoload)\s*[\(\s]*"
                    r'["\']([^"\']+)["\']',
                    def_text,
                )
                if match:
                    target = match.group(1)
                    leaf = target.split("/")[-1]
                    return f"require_{leaf}"

            return "unnamed_import"

        elif concept == UniversalConcept.STRUCTURE:
            return "file_structure"

        return "unnamed"

    def extract_content(
        self, concept: UniversalConcept, captures: dict[str, TSNode], content: bytes
    ) -> str:
        """Extract the source content for this concept."""

        source = content.decode("utf-8")

        if concept == UniversalConcept.BLOCK and "block" in captures:
            return self.get_node_text(captures["block"], source)
        elif "definition" in captures:
            return self.get_node_text(captures["definition"], source)
        elif captures:
            node = list(captures.values())[0]
            return self.get_node_text(node, source)

        return ""

    def extract_metadata(
        self, concept: UniversalConcept, captures: dict[str, TSNode], content: bytes
    ) -> dict[str, Any]:
        """Extract Ruby-specific metadata from captures."""

        source = content.decode("utf-8")
        metadata: dict[str, Any] = {}

        if concept == UniversalConcept.DEFINITION:
            def_node = captures.get("definition")
            if def_node:
                metadata["node_type"] = def_node.type

                if def_node.type == "class":
                    metadata["kind"] = "class"
                    superclass = def_node.child_by_field_name("superclass")
                    if superclass:
                        # superclass node text includes the leading "< "
                        text = self.get_node_text(superclass, source).strip()
                        metadata["superclass"] = text.lstrip("<").strip()
                elif def_node.type == "module":
                    metadata["kind"] = "module"
                    # Ruby modules are namespaces (and mixin containers); the
                    # engine has no "module" kind, so steer it explicitly rather
                    # than falling through to the FUNCTION default.
                    metadata["chunk_type_hint"] = "namespace"
                elif def_node.type == "method":
                    metadata["kind"] = "method"
                elif def_node.type == "singleton_method":
                    metadata["kind"] = "singleton_method"
                    metadata["is_class_method"] = True
                elif def_node.type == "assignment":
                    metadata["kind"] = "constant"

        elif concept == UniversalConcept.IMPORT:
            if "definition" in captures:
                import_text = self.get_node_text(captures["definition"], source).strip()
                match = re.search(
                    r"(?:require_relative|require|load|autoload)\s*[\(\s]*"
                    r'["\']([^"\']+)["\']',
                    import_text,
                )
                if match:
                    metadata["module"] = match.group(1)
                if import_text.startswith("require_relative"):
                    metadata["import_type"] = "require_relative"
                elif import_text.startswith("require"):
                    metadata["import_type"] = "require"

        elif concept == UniversalConcept.COMMENT:
            if "definition" in captures:
                comment_text = self.get_node_text(captures["definition"], source)
                clean_text = self.clean_comment_text(comment_text)
                comment_type = "regular"
                is_doc = False

                if clean_text:
                    upper_text = clean_text.upper()
                    if any(
                        prefix in upper_text
                        for prefix in ["TODO:", "FIXME:", "HACK:", "NOTE:", "WARNING:"]
                    ):
                        comment_type = "annotation"
                        is_doc = True
                    elif clean_text.startswith("#!/"):
                        comment_type = "shebang"
                        is_doc = True

                metadata["comment_type"] = comment_type
                if is_doc:
                    metadata["is_doc_comment"] = True

        return metadata

    def clean_comment_text(self, text: str) -> str:
        """Strip Ruby comment markers from comment text."""
        cleaned = text.strip()
        if cleaned.startswith("#"):
            cleaned = cleaned[1:]
        return cleaned.strip()

    def extract_constants(
        self, concept: UniversalConcept, captures: dict[str, TSNode], content: bytes
    ) -> list[dict[str, str]] | None:
        """Extract constant definitions from Ruby code.

        Ruby constants begin with an uppercase letter; the grammar exposes the
        left-hand side as a ``constant`` node, so any captured constant
        assignment qualifies.
        """
        if concept != UniversalConcept.DEFINITION:
            return None

        source = content.decode("utf-8")

        lhs = captures.get("lhs")
        if lhs is None or lhs.type != "constant":
            return None

        name = self.get_node_text(lhs, source).strip()
        if not name:
            return None

        value = ""
        rhs = captures.get("rhs")
        if rhs is not None:
            value = self.get_node_text(rhs, source).strip()
            if len(value) > MAX_CONSTANT_VALUE_LENGTH:
                value = value[:MAX_CONSTANT_VALUE_LENGTH]

        return [{"name": name, "value": value}]

    def resolve_import_paths(
        self, import_text: str, base_dir: Path, source_file: Path
    ) -> list[Path]:
        """Resolve a Ruby require/require_relative target to a file path."""
        # require_relative is resolved relative to the requiring file.
        match = re.search(r'require_relative\s*[\(\s]*["\']([^"\']+)["\']', import_text)
        if match:
            target = match.group(1)
            if not target.endswith(".rb"):
                target += ".rb"
            resolved = (source_file.parent / target).resolve()
            if resolved.exists():
                return [resolved]
            return []

        # require is resolved against load paths; approximate with base_dir and
        # common Rails source roots.
        match = re.search(r'require\s*[\(\s]*["\']([^"\']+)["\']', import_text)
        if match:
            target = match.group(1)
            if not target.endswith(".rb"):
                target += ".rb"
            for root in (base_dir, base_dir / "lib", base_dir / "app"):
                candidate = root / target
                if candidate.exists():
                    return [candidate]

        return []
