"""Elixir language mapping for unified parser architecture.

This module provides Elixir-specific tree-sitter queries and extraction logic
for the universal concept system. Elixir's Tree-sitter grammar represents all
constructs as `call` nodes — pattern matching on identifier text (defmodule,
def, etc.) is required to classify chunks.
"""

import re
from pathlib import Path
from typing import Any

from tree_sitter import Node

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.base import BaseMapping
from chunkhound.parsers.universal_engine import UniversalConcept

# Elixir keywords that define modules/types
_MODULE_KEYWORDS = r"^(defmodule|defprotocol|defimpl)$"

# Elixir keywords that define functions/macros
_FUNCTION_KEYWORDS = (
    r"^(def|defp|defmacro|defmacrop|defguard|defguardp|defdelegate|defstruct)$"
)

# Elixir keywords for imports/dependencies
_IMPORT_KEYWORDS = r"^(use|import|alias|require)$"

# Module attribute names that define types/specs
_TYPE_ATTR_KEYWORDS = r"^(spec|type|typep|opaque|callback)$"

# Module attribute names for documentation
_DOC_ATTR_KEYWORDS = r"^(doc|moduledoc)$"

# Keyword -> metadata kind mapping
_KIND_MAP: dict[str, str] = {
    "defmodule": "class",
    "defprotocol": "interface",
    "defimpl": "class",
    "def": "function",
    "defp": "function",
    "defmacro": "macro",
    "defmacrop": "macro",
    "defguard": "function",
    "defguardp": "function",
    "defdelegate": "function",
    "defstruct": "struct",
    "spec": "type",
    "type": "type",
    "typep": "type",
    "opaque": "type",
    "callback": "type",
}


class ElixirMapping(BaseMapping):
    """Elixir-specific tree-sitter mapping for universal concepts."""

    def __init__(self) -> None:
        super().__init__(Language.ELIXIR)

    # --- BaseMapping required methods ---

    def get_function_query(self) -> str:
        return f"""
        (call
            target: (identifier) @keyword
            (#match? @keyword "{_FUNCTION_KEYWORDS}")
        ) @func_def
        """

    def get_class_query(self) -> str:
        return f"""
        (call
            target: (identifier) @keyword
            (#match? @keyword "{_MODULE_KEYWORDS}")
        ) @class_def
        """

    def get_comment_query(self) -> str:
        return "(comment) @comment"

    def extract_function_name(self, node: Node | None, source: str) -> str:
        if node is None:
            return self.get_fallback_name(node, "function")
        keyword, name = self._extract_keyword_and_name(node, source)
        if name:
            return name
        return self.get_fallback_name(node, "function")

    def extract_class_name(self, node: Node | None, source: str) -> str:
        if node is None:
            return self.get_fallback_name(node, "module")
        keyword, name = self._extract_keyword_and_name(node, source)
        if name:
            return name
        return self.get_fallback_name(node, "module")

    # --- LanguageMapping protocol methods ---

    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        if concept == UniversalConcept.DEFINITION:
            return f"""
            (call
                target: (identifier) @keyword
                (#match? @keyword "{_MODULE_KEYWORDS}")
            ) @definition

            (call
                target: (identifier) @keyword
                (#match? @keyword "{_FUNCTION_KEYWORDS}")
            ) @definition

            (unary_operator
                operator: "@"
                operand: (call
                    target: (identifier) @attr_name
                    (#match? @attr_name "{_TYPE_ATTR_KEYWORDS}")
                )
            ) @definition
            """

        elif concept == UniversalConcept.COMMENT:
            return f"""
            (comment) @definition

            (unary_operator
                operator: "@"
                operand: (call
                    target: (identifier) @attr_name
                    (#match? @attr_name "{_DOC_ATTR_KEYWORDS}")
                )
            ) @definition
            """

        elif concept == UniversalConcept.IMPORT:
            return f"""
            (call
                target: (identifier) @keyword
                (#match? @keyword "{_IMPORT_KEYWORDS}")
            ) @definition
            """

        elif concept == UniversalConcept.BLOCK:
            return """
            (do_block) @block
            """

        elif concept == UniversalConcept.STRUCTURE:
            return """
            (source) @definition
            """

        return None

    def extract_name(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        source = content.decode("utf-8")

        if concept == UniversalConcept.DEFINITION:
            def_node = captures.get("definition")
            if not def_node:
                return "unnamed_definition"

            # Module attribute (@spec, @type, etc.)
            if def_node.type == "unary_operator":
                return self._extract_attr_name(def_node, source)

            # Call node (defmodule, def, defp, etc.)
            keyword, name = self._extract_keyword_and_name(def_node, source)
            if name:
                return name
            return "unnamed_definition"

        elif concept == UniversalConcept.COMMENT:
            def_node = captures.get("definition")
            if not def_node:
                return "unnamed_comment"
            if def_node.type == "comment":
                line = def_node.start_point[0] + 1
                return f"comment_line_{line}"
            if def_node.type == "unary_operator":
                return self._extract_attr_name(def_node, source)
            return "unnamed_comment"

        elif concept == UniversalConcept.IMPORT:
            def_node = captures.get("definition")
            if not def_node:
                return "unnamed_import"
            keyword, name = self._extract_keyword_and_name(def_node, source)
            if keyword and name:
                return f"{keyword}_{name}"
            if keyword:
                return keyword
            return "unnamed_import"

        elif concept == UniversalConcept.BLOCK:
            block_node = captures.get("block")
            if block_node:
                line = block_node.start_point[0] + 1
                return f"do_block_line_{line}"
            return "unnamed_block"

        elif concept == UniversalConcept.STRUCTURE:
            return "elixir_source"

        return "unnamed"

    def extract_content(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
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
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> dict[str, Any]:
        source = content.decode("utf-8")
        metadata: dict[str, Any] = {}

        if concept == UniversalConcept.DEFINITION:
            def_node = captures.get("definition")
            if not def_node:
                return metadata

            # Module attribute (@spec, @type, etc.)
            if def_node.type == "unary_operator":
                attr_keyword = self._get_attr_keyword(def_node, source)
                if attr_keyword:
                    metadata["kind"] = _KIND_MAP.get(attr_keyword, "type")
                    metadata["node_type"] = "attribute"
                return metadata

            # Call node
            keyword = self._get_call_keyword(def_node, source)
            if keyword:
                metadata["kind"] = _KIND_MAP.get(keyword, "function")
                metadata["node_type"] = def_node.type

                # Count body lines for functions
                if keyword in ("def", "defp", "defmacro", "defmacrop",
                               "defguard", "defguardp"):
                    body = self.find_child_by_type(def_node, "do_block")
                    if body:
                        body_text = self.get_node_text(body, source)
                        metadata["body_lines"] = len(body_text.splitlines())

        elif concept == UniversalConcept.COMMENT:
            def_node = captures.get("definition")
            if def_node:
                if def_node.type == "comment":
                    text = self.get_node_text(def_node, source)
                    clean = self.clean_comment_text(text)
                    if clean:
                        upper = clean.upper()
                        if any(p in upper for p in
                               ["TODO:", "FIXME:", "HACK:", "NOTE:", "WARNING:"]):
                            metadata["comment_type"] = "annotation"
                        else:
                            metadata["comment_type"] = "regular"
                elif def_node.type == "unary_operator":
                    attr_keyword = self._get_attr_keyword(def_node, source)
                    if attr_keyword == "moduledoc":
                        metadata["comment_type"] = "moduledoc"
                        metadata["is_doc_comment"] = True
                    elif attr_keyword == "doc":
                        metadata["comment_type"] = "doc"
                        metadata["is_doc_comment"] = True

        elif concept == UniversalConcept.IMPORT:
            def_node = captures.get("definition")
            if def_node:
                keyword = self._get_call_keyword(def_node, source)
                if keyword:
                    metadata["import_type"] = keyword
                # Extract the target module alias
                args_node = self.find_child_by_type(def_node, "arguments")
                if args_node:
                    alias_node = self.find_child_by_type(args_node, "alias")
                    if alias_node:
                        metadata["module"] = self.get_node_text(
                            alias_node, source
                        ).strip()

        elif concept == UniversalConcept.BLOCK:
            if "block" in captures:
                metadata["block_type"] = "do_block"

        return metadata

    def clean_comment_text(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("#"):
            cleaned = cleaned[1:]
        return cleaned.strip()

    def resolve_import_paths(
        self, import_text: str, base_dir: Path, source_file: Path
    ) -> list[Path]:
        # Elixir imports are module aliases, not file paths.
        # We can try to resolve alias/import to lib/ paths.
        match = re.search(
            r"(?:alias|import|use|require)\s+([\w.]+)", import_text
        )
        if not match:
            return []

        module_name = match.group(1)
        # Convert MyApp.Accounts.User -> my_app/accounts/user.ex
        parts = module_name.split(".")
        snake_parts = [self._to_snake_case(p) for p in parts]
        rel_path = "/".join(snake_parts) + ".ex"

        search_dirs: list[Path] = []

        # Detect umbrella app structure from source_file (apps/*/lib/...)
        try:
            rel_source = source_file.relative_to(base_dir)
            if len(rel_source.parts) >= 2 and rel_source.parts[0] == "apps":
                apps_dir = base_dir / "apps"
                if apps_dir.is_dir():
                    for app_dir in sorted(apps_dir.iterdir()):
                        lib_dir = app_dir / "lib"
                        if lib_dir.is_dir():
                            search_dirs.append(lib_dir)
        except ValueError:
            pass

        # Standard Mix search paths (always included as fallback)
        search_dirs.append(base_dir / "lib")
        search_dirs.append(base_dir)

        for search_dir in search_dirs:
            candidate = search_dir / rel_path
            if candidate.exists():
                return [candidate]

        return []

    # --- Private helpers ---

    def _get_call_keyword(self, node: Node, source: str) -> str | None:
        """Get the keyword identifier from a call node (e.g., 'def', 'defmodule')."""
        if node.type != "call":
            return None
        target = self.find_child_by_type(node, "identifier")
        if target:
            return self.get_node_text(target, source).strip()
        return None

    def _get_attr_keyword(self, node: Node, source: str) -> str | None:
        """Get the attribute keyword from a unary_operator node."""
        if node.type != "unary_operator":
            return None
        call_node = self.find_child_by_type(node, "call")
        if call_node:
            target = self.find_child_by_type(call_node, "identifier")
            if target:
                return self.get_node_text(target, source).strip()
        return None

    def _extract_keyword_and_name(
        self, node: Node, source: str
    ) -> tuple[str | None, str | None]:
        """Extract the keyword and name from a call node.

        Returns (keyword, name) tuple. For defmodule, name is the alias.
        For def/defp, name is the function name from nested arguments.
        """
        keyword = self._get_call_keyword(node, source)
        if not keyword:
            return (None, None)

        args_node = self.find_child_by_type(node, "arguments")
        if not args_node:
            return (keyword, None)

        # Module-level: defmodule/defprotocol/defimpl -> alias is first arg
        if keyword in ("defmodule", "defprotocol", "defimpl"):
            alias_node = self.find_child_by_type(args_node, "alias")
            if alias_node:
                return (keyword, self.get_node_text(alias_node, source).strip())
            return (keyword, None)

        # defstruct has no name — use parent module context
        if keyword == "defstruct":
            return (keyword, "defstruct")

        # Import-type: use/import/alias/require -> alias is first arg
        if keyword in ("use", "import", "alias", "require"):
            alias_node = self.find_child_by_type(args_node, "alias")
            if alias_node:
                return (keyword, self.get_node_text(alias_node, source).strip())
            # Could be an atom or identifier (e.g., require :crypto)
            for child in args_node.children:
                if child.type in ("atom", "identifier"):
                    return (keyword, self.get_node_text(child, source).strip())
            return (keyword, None)

        # Function-level: def/defp/defmacro/defguard/defdelegate
        # Pattern: arguments -> call(func_name, args) for arity > 0
        #          arguments -> identifier for arity 0
        #          arguments -> binary_operator (when clause for guards)
        name = self._extract_func_name_from_args(args_node, source)
        return (keyword, name)

    def _extract_func_name_from_args(
        self, args_node: Node, source: str
    ) -> str | None:
        """Extract function name from a def/defp arguments node.

        Handles: def foo(a, b), def foo, defguard is_pos(x) when x > 0
        """
        for child in args_node.children:
            if child.type == "call":
                # def foo(a, b) -> call target is "foo"
                target = self.find_child_by_type(child, "identifier")
                if target:
                    return self.get_node_text(target, source).strip()

            elif child.type == "identifier":
                # def foo (zero arity, no parens)
                return self.get_node_text(child, source).strip()

            elif child.type == "binary_operator":
                # defguard is_positive(x) when x > 0
                # The left side of `when` is the call
                inner_call = self.find_child_by_type(child, "call")
                if inner_call:
                    target = self.find_child_by_type(inner_call, "identifier")
                    if target:
                        return self.get_node_text(target, source).strip()
                # Or it could be an identifier (zero-arity guard)
                inner_id = self.find_child_by_type(child, "identifier")
                if inner_id:
                    return self.get_node_text(inner_id, source).strip()

        return None

    def _extract_attr_name(self, node: Node, source: str) -> str:
        """Extract a readable name from a module attribute node.

        For @spec foo(...), returns 'spec_foo'.
        For @type t, returns 'type_t'.
        For @doc, returns 'doc'.
        """
        attr_keyword = self._get_attr_keyword(node, source)
        if not attr_keyword:
            return "unnamed_attr"

        # For type attrs, try to extract the type/spec name
        if attr_keyword in ("spec", "type", "typep", "opaque", "callback"):
            call_node = self.find_child_by_type(node, "call")
            if call_node:
                inner_args = self.find_child_by_type(call_node, "arguments")
                if inner_args:
                    name = self._extract_type_name_from_args(
                        inner_args, source
                    )
                    if name:
                        return f"{attr_keyword}_{name}"

        # For @doc/@moduledoc, just return the keyword
        if attr_keyword in ("doc", "moduledoc"):
            return attr_keyword

        return attr_keyword

    def _extract_type_name_from_args(
        self, args_node: Node, source: str
    ) -> str | None:
        """Extract the type/spec name from attribute arguments.

        @spec foo(integer()) :: atom() -> 'foo'
        @type t :: %__MODULE__{} -> 't'
        """
        for child in args_node.children:
            if child.type == "binary_operator":
                # Left side of :: is the name/call
                left = self.find_child_by_type(child, "call")
                if left:
                    target = self.find_child_by_type(left, "identifier")
                    if target:
                        return self.get_node_text(target, source).strip()
                left_id = self.find_child_by_type(child, "identifier")
                if left_id:
                    return self.get_node_text(left_id, source).strip()
            elif child.type == "call":
                target = self.find_child_by_type(child, "identifier")
                if target:
                    return self.get_node_text(target, source).strip()
            elif child.type == "identifier":
                return self.get_node_text(child, source).strip()
        return None

    @staticmethod
    def _to_snake_case(name: str) -> str:
        """Convert CamelCase to snake_case."""
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
