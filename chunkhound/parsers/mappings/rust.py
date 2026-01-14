"""Rust language mapping for unified parser architecture.

This module provides Rust-specific tree-sitter queries and extraction logic
for the universal concept system. It maps Rust's AST nodes to universal
semantic concepts used by the unified parser.
"""

import re
from pathlib import Path
from typing import Any

from tree_sitter import Node

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.base import MAX_CONSTANT_VALUE_LENGTH, BaseMapping
from chunkhound.parsers.universal_engine import UniversalConcept


class RustMapping(BaseMapping):
    """Rust-specific tree-sitter mapping for universal concepts."""

    def __init__(self) -> None:
        """Initialize Rust mapping."""
        super().__init__(Language.RUST)

    # BaseMapping required methods
    def get_function_query(self) -> str:
        """Get tree-sitter query pattern for function definitions."""
        return """
        (function_item
            name: (identifier) @func_name
        ) @func_def
        """

    def get_class_query(self) -> str:
        """Get tree-sitter query pattern for struct definitions.

        Returns patterns for Rust's equivalent of classes (structs and enums).
        """
        return """
        (struct_item
            name: (type_identifier) @struct_name
        ) @struct_def

        (enum_item
            name: (type_identifier) @enum_name
        ) @enum_def
        """

    def get_comment_query(self) -> str:
        """Get tree-sitter query pattern for comments."""
        return """
        (line_comment) @comment
        (block_comment) @comment
        """

    def extract_function_name(self, node: Node | None, source: str) -> str:
        """Extract function name from a function definition node."""
        if node is None:
            return self.get_fallback_name(node, "function")

        # Find the function name child
        name_node = self.find_child_by_type(node, "identifier")
        if name_node:
            return self.get_node_text(name_node, source).strip()

        return self.get_fallback_name(node, "function")

    def extract_class_name(self, node: Node | None, source: str) -> str:
        """Extract struct/enum name from a definition node."""
        if node is None:
            return self.get_fallback_name(node, "struct")

        # Find the type_identifier child
        name_node = self.find_child_by_type(node, "type_identifier")
        if name_node:
            return self.get_node_text(name_node, source).strip()

        return self.get_fallback_name(node, "struct")

    # LanguageMapping protocol methods
    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        """Get tree-sitter query for universal concept in Rust."""

        if concept == UniversalConcept.DEFINITION:
            return """
            (function_item
                name: (identifier) @name
            ) @definition

            (impl_item
                type: (_) @impl_type
                body: (declaration_list
                    (function_item
                        name: (identifier) @name
                    ) @method
                )*
            ) @definition

            (struct_item
                name: (type_identifier) @name
            ) @definition

            (enum_item
                name: (type_identifier) @name
            ) @definition

            (trait_item
                name: (type_identifier) @name
            ) @definition

            (type_item
                name: (type_identifier) @name
            ) @definition

            (const_item
                name: (identifier) @name
            ) @definition

            (static_item
                name: (identifier) @name
            ) @definition

            (macro_definition
                name: (identifier) @name
            ) @definition

            (let_declaration
                pattern: (identifier) @name
            ) @definition
            """

        elif concept == UniversalConcept.BLOCK:
            return """
            (block) @block

            (if_expression
                consequence: (block) @block
            )

            (while_expression
                body: (block) @block
            )

            (for_expression
                body: (block) @block
            )

            (loop_expression
                body: (block) @block
            )

            (match_expression) @block

            (closure_expression
                body: (block) @block
            )
            """

        elif concept == UniversalConcept.COMMENT:
            return """
            (line_comment) @definition
            (block_comment) @definition
            """

        elif concept == UniversalConcept.IMPORT:
            return """
            (use_declaration) @definition

            (mod_item
                name: (identifier) @mod_name
            ) @definition

            (extern_crate_declaration
                name: (identifier) @crate_name
            ) @definition
            """

        elif concept == UniversalConcept.STRUCTURE:
            return """
            (source_file
                (attribute_item)* @attributes
                (use_declaration)* @imports
            ) @definition
            """

        return None

    def extract_name(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract name from captures for this concept."""

        # Convert bytes to string for processing
        source = content.decode("utf-8")

        if concept == UniversalConcept.DEFINITION:
            # Try to get the name from various capture groups
            if "name" in captures:
                name_node = captures["name"]
                name = self.get_node_text(name_node, source).strip()

                # For impl methods, prepend the impl type
                if "impl_type" in captures:
                    impl_type_node = captures["impl_type"]
                    impl_type = self.get_node_text(impl_type_node, source).strip()
                    # Clean up generic parameters for display
                    if "<" in impl_type:
                        impl_type = impl_type.split("<")[0]
                    return f"{impl_type}::{name}"

                return name

            # For impl blocks without specific method
            if "impl_type" in captures:
                impl_type_node = captures["impl_type"]
                impl_type = self.get_node_text(impl_type_node, source).strip()
                return f"impl_{impl_type}"

            return "unnamed_definition"

        elif concept == UniversalConcept.BLOCK:
            # Use location-based naming for blocks
            if "definition" in captures:
                node = captures["definition"]
                line = node.start_point[0] + 1
                return f"block_line_{line}"
            elif "block" in captures:
                node = captures["block"]
                line = node.start_point[0] + 1
                return f"block_line_{line}"
            elif "match_arm" in captures:
                node = captures["match_arm"]
                line = node.start_point[0] + 1
                return f"match_arm_line_{line}"

            return "unnamed_block"

        elif concept == UniversalConcept.COMMENT:
            # Use location-based naming for comments
            if "definition" in captures:
                node = captures["definition"]
                line = node.start_point[0] + 1
                comment_text = self.get_node_text(node, source)

                # Check for doc comments
                if comment_text.startswith("///") or comment_text.startswith("//!"):
                    return f"doc_comment_line_{line}"
                elif comment_text.startswith("/**") or comment_text.startswith("/*!"):
                    return f"doc_comment_line_{line}"
                else:
                    return f"comment_line_{line}"

            return "unnamed_comment"

        elif concept == UniversalConcept.IMPORT:
            if "use_clause" in captures:
                use_node = captures["use_clause"]
                use_text = self.get_node_text(use_node, source).strip()
                # Extract the last part of the use path
                parts = use_text.replace("::", "/").split("/")
                if parts:
                    last_part = parts[-1].strip("{}")
                    # Handle glob imports
                    if last_part == "*":
                        if len(parts) > 1:
                            return f"use_{parts[-2]}_all"
                        return "use_glob"
                    return f"use_{last_part}"
                return "use_unknown"
            elif "mod_name" in captures:
                mod_node = captures["mod_name"]
                mod_name = self.get_node_text(mod_node, source).strip()
                return f"mod_{mod_name}"
            elif "crate_name" in captures:
                crate_node = captures["crate_name"]
                crate_name = self.get_node_text(crate_node, source).strip()
                return f"extern_crate_{crate_name}"

            return "unnamed_import"

        elif concept == UniversalConcept.STRUCTURE:
            return "file_structure"

        return "unnamed"

    def extract_content(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract content from captures for this concept."""

        # Convert bytes to string for processing
        source = content.decode("utf-8")

        if concept == UniversalConcept.BLOCK and "block" in captures:
            node = captures["block"]
            return self.get_node_text(node, source)
        elif "definition" in captures:
            node = captures["definition"]
            return self.get_node_text(node, source)
        elif captures:
            # Use the first available capture
            node = list(captures.values())[0]
            return self.get_node_text(node, source)

        return ""

    def extract_metadata(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> dict[str, Any]:
        """Extract Rust-specific metadata."""

        source = content.decode("utf-8")
        metadata = {}

        if concept == UniversalConcept.DEFINITION:
            # Extract definition-specific metadata
            def_node = captures.get("definition")
            if def_node:
                metadata["node_type"] = def_node.type

                # For functions, extract parameters, return type, and attributes
                if def_node.type == "function_item":
                    metadata["kind"] = "function"
                    params = self._extract_function_parameters(def_node, source)
                    metadata["parameters"] = params
                    return_type = self._extract_function_return_type(def_node, source)
                    if return_type:
                        metadata["return_type"] = return_type

                    # Check for async
                    if self._has_async_modifier(def_node, source):
                        metadata["is_async"] = True

                    # Check for unsafe
                    if self._has_unsafe_modifier(def_node, source):
                        metadata["is_unsafe"] = True

                    # Extract visibility
                    visibility = self._extract_visibility(def_node, source)
                    if visibility:
                        metadata["visibility"] = visibility

                # For impl blocks
                elif def_node.type == "impl_item":
                    metadata["kind"] = "impl"
                    if "impl_type" in captures:
                        impl_type_node = captures["impl_type"]
                        metadata["impl_type"] = self.get_node_text(
                            impl_type_node, source
                        ).strip()

                    # Check if it's a trait impl
                    if self._is_trait_impl(def_node, source):
                        metadata["impl_kind"] = "trait"
                        trait_name = self._extract_trait_name(def_node, source)
                        if trait_name:
                            metadata["trait_name"] = trait_name
                    else:
                        metadata["impl_kind"] = "inherent"

                # For structs and enums
                elif def_node.type in ["struct_item", "enum_item"]:
                    metadata["kind"] = def_node.type.replace("_item", "")

                    # Extract generics
                    generics = self._extract_generics(def_node, source)
                    if generics:
                        metadata["generics"] = generics

                    # Extract visibility
                    visibility = self._extract_visibility(def_node, source)
                    if visibility:
                        metadata["visibility"] = visibility

                # For traits
                elif def_node.type == "trait_item":
                    metadata["kind"] = "trait"

                    # Extract generics
                    generics = self._extract_generics(def_node, source)
                    if generics:
                        metadata["generics"] = generics

                    # Check if unsafe
                    if self._has_unsafe_modifier(def_node, source):
                        metadata["is_unsafe"] = True

                # For type aliases
                elif def_node.type == "type_item":
                    metadata["kind"] = "type_alias"

                    # Extract the aliased type
                    aliased_type = self._extract_aliased_type(def_node, source)
                    if aliased_type:
                        metadata["aliased_type"] = aliased_type

                # For constants and statics
                elif def_node.type in ["const_item", "static_item"]:
                    metadata["kind"] = def_node.type.replace("_item", "")

                    # Extract type
                    type_annotation = self._extract_type_annotation(def_node, source)
                    if type_annotation:
                        metadata["type"] = type_annotation

                # For macros
                elif def_node.type == "macro_definition":
                    metadata["kind"] = "macro"

                # For let bindings
                elif def_node.type == "let_declaration":
                    metadata["kind"] = "variable"

                    # Extract type if present
                    type_annotation = self._extract_type_annotation(def_node, source)
                    if type_annotation:
                        metadata["type"] = type_annotation

                    # Check if mutable
                    if self._is_mutable_binding(def_node, source):
                        metadata["is_mutable"] = True

        elif concept == UniversalConcept.IMPORT:
            if "use_clause" in captures:
                use_node = captures["use_clause"]
                use_text = self.get_node_text(use_node, source).strip()
                metadata["use_path"] = use_text

                # Detect glob imports
                if use_text.endswith("*"):
                    metadata["is_glob"] = True

                # Detect aliased imports
                if " as " in use_text:
                    metadata["is_aliased"] = True
                    parts = use_text.split(" as ")
                    if len(parts) == 2:
                        metadata["original_name"] = parts[0].strip()
                        metadata["alias"] = parts[1].strip()

            elif "mod_name" in captures:
                mod_node = captures["mod_name"]
                metadata["module_name"] = self.get_node_text(mod_node, source).strip()

            elif "crate_name" in captures:
                crate_node = captures["crate_name"]
                metadata["crate_name"] = self.get_node_text(crate_node, source).strip()

        elif concept == UniversalConcept.COMMENT:
            if "definition" in captures:
                comment_node = captures["definition"]
                comment_text = self.get_node_text(comment_node, source)

                # Determine comment type and if it's documentation
                if comment_text.startswith("///"):
                    metadata["comment_type"] = "line_doc"
                    metadata["is_doc_comment"] = True
                elif comment_text.startswith("//!"):
                    metadata["comment_type"] = "line_module_doc"
                    metadata["is_doc_comment"] = True
                elif comment_text.startswith("/**"):
                    metadata["comment_type"] = "block_doc"
                    metadata["is_doc_comment"] = True
                elif comment_text.startswith("/*!"):
                    metadata["comment_type"] = "block_module_doc"
                    metadata["is_doc_comment"] = True
                elif comment_text.startswith("//"):
                    metadata["comment_type"] = "line"
                elif comment_text.startswith("/*"):
                    metadata["comment_type"] = "block"

        return metadata

    def _extract_function_parameters(self, func_node: Node, source: str) -> list[str]:
        """Extract parameter types from a Rust function node."""
        parameters = []

        # Find parameters node
        params_node = None
        for child in self.walk_tree(func_node):
            if child and child.type == "parameters":
                params_node = child
                break

        if params_node:
            # Find parameter nodes
            param_nodes = self.find_children_by_type(params_node, "parameter")
            for param_node in param_nodes:
                # Extract parameter pattern and type
                pattern_text = ""
                type_text = ""

                for i in range(param_node.child_count):
                    child = param_node.child(i)
                    if child is None:
                        continue

                    if child.type in ["identifier", "self"]:
                        pattern_text = self.get_node_text(child, source).strip()
                    elif child.type == ":" and i + 1 < param_node.child_count:
                        # Next node should be the type
                        type_child = param_node.child(i + 1)
                        if type_child:
                            type_text = self.get_node_text(type_child, source).strip()

                if pattern_text and type_text:
                    parameters.append(f"{pattern_text}: {type_text}")
                elif pattern_text:
                    parameters.append(pattern_text)

        return parameters

    def _extract_function_return_type(self, func_node: Node, source: str) -> str | None:
        """Extract return type from a Rust function node."""

        # Look for -> after parameters
        for i in range(func_node.child_count):
            child = func_node.child(i)
            if child and self.get_node_text(child, source).strip() == "->":
                # Next non-whitespace node should be the return type
                if i + 1 < func_node.child_count:
                    type_child = func_node.child(i + 1)
                    if type_child and type_child.type != "block":
                        return self.get_node_text(type_child, source).strip()

        return None

    def _has_async_modifier(self, node: Node, source: str) -> bool:
        """Check if a function has async modifier."""
        # Look for async keyword before the function
        for child in self.walk_tree(node):
            if child and child.type == "async":
                return True
        return False

    def _has_unsafe_modifier(self, node: Node, source: str) -> bool:
        """Check if a function/trait has unsafe modifier."""
        # Look for unsafe keyword
        for child in self.walk_tree(node):
            if child and child.type == "unsafe":
                return True
        return False

    def _extract_visibility(self, node: Node, source: str) -> str | None:
        """Extract visibility modifier from a node."""
        # Look for visibility_modifier
        vis_node = self.find_child_by_type(node, "visibility_modifier")
        if vis_node:
            return self.get_node_text(vis_node, source).strip()
        return None

    def _is_trait_impl(self, impl_node: Node, source: str) -> bool:
        """Check if an impl block implements a trait."""
        # Look for 'for' keyword in impl
        for child in self.walk_tree(impl_node):
            if child and self.get_node_text(child, source).strip() == "for":
                return True
        return False

    def _extract_trait_name(self, impl_node: Node, source: str) -> str | None:
        """Extract trait name from trait impl block."""
        # Find pattern: impl TraitName for Type
        found_impl = False
        for i in range(impl_node.child_count):
            child = impl_node.child(i)
            if child is None:
                continue

            text = self.get_node_text(child, source).strip()
            if text == "impl":
                found_impl = True
            elif found_impl and text != "for" and child.type == "type_identifier":
                # This might be the trait name
                next_child = (
                    impl_node.child(i + 1) if i + 1 < impl_node.child_count else None
                )
                if (
                    next_child
                    and self.get_node_text(next_child, source).strip() == "for"
                ):
                    return text

        return None

    def _extract_generics(self, node: Node, source: str) -> list[str]:
        """Extract generic parameters from a node."""
        generics = []

        # Find type_parameters node
        type_params_node = self.find_child_by_type(node, "type_parameters")
        if type_params_node:
            # Find type_parameter nodes
            param_nodes = self.find_children_by_type(type_params_node, "type_parameter")
            for param_node in param_nodes:
                param_name = self.get_node_text(param_node, source).strip()
                generics.append(param_name)

        return generics

    def _extract_aliased_type(self, type_node: Node, source: str) -> str | None:
        """Extract the aliased type from a type alias."""
        # Look for = followed by the type
        found_equals = False
        for i in range(type_node.child_count):
            child = type_node.child(i)
            if child is None:
                continue

            if self.get_node_text(child, source).strip() == "=":
                found_equals = True
            elif found_equals and child.type != ";":
                # This should be the aliased type
                return self.get_node_text(child, source).strip()

        return None

    def _extract_type_annotation(self, node: Node, source: str) -> str | None:
        """Extract type annotation from a node."""
        # Look for : followed by type
        found_colon = False
        for i in range(node.child_count):
            child = node.child(i)
            if child is None:
                continue

            if self.get_node_text(child, source).strip() == ":":
                found_colon = True
            elif found_colon and child.type not in ["=", ";"]:
                # This should be the type
                return self.get_node_text(child, source).strip()

        return None

    def _is_mutable_binding(self, let_node: Node, source: str) -> bool:
        """Check if a let binding is mutable."""
        # Look for 'mut' keyword
        for child in self.walk_tree(let_node):
            if child and self.get_node_text(child, source).strip() == "mut":
                return True
        return False

    def extract_constants(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> list[dict[str, str]] | None:
        """Extract constant definitions with their values.

        Returns list of dicts with 'name', 'value', and optionally 'type' keys,
        or None if this is not a constant definition.

        In Rust, this includes:
        - const items (const MAX: i32 = 100)
        - static items (static COUNTER: i32 = 0)
        - immutable let bindings (let MAX = 100) - mutable let bindings are excluded
        """
        if concept != UniversalConcept.DEFINITION:
            return None

        # Get the definition node
        def_node = captures.get("definition")
        if not def_node or def_node.type not in ["const_item", "static_item", "let_declaration"]:
            return None

        source = content.decode("utf-8")

        # For let_declaration, skip mutable bindings
        if def_node.type == "let_declaration":
            if self._is_mutable_binding(def_node, source):
                return None

        # Extract metadata to get kind and type
        metadata = self.extract_metadata(concept, captures, content)
        if metadata.get("kind") not in ["const", "static", "variable"]:
            return None

        # Extract constant name
        name_node = captures.get("name")
        if not name_node:
            return None

        const_name = self.get_node_text(name_node, source).strip()

        # Extract value by finding the assignment after "="
        const_value = None
        found_equals = False
        for i in range(def_node.child_count):
            child = def_node.child(i)
            if child is None:
                continue

            if self.get_node_text(child, source).strip() == "=":
                found_equals = True
            elif found_equals and child.type != ";":
                # Extract the value, truncating to 50 chars if longer
                value_text = self.get_node_text(child, source).strip()
                if len(value_text) > MAX_CONSTANT_VALUE_LENGTH:
                    const_value = value_text[:MAX_CONSTANT_VALUE_LENGTH]
                else:
                    const_value = value_text
                break

        # Build result dict
        result: dict[str, str] = {"name": const_name}

        if const_value is not None:
            result["value"] = const_value

        # Include type if available
        if "type" in metadata:
            result["type"] = metadata["type"]

        # For static items, check if mutable
        if def_node.type == "static_item":
            # Look for 'mut' keyword
            for child in self.walk_tree(def_node):
                if child and self.get_node_text(child, source).strip() == "mut":
                    result["mutable"] = "true"
                    break

        return [result]

    def resolve_import_path(
        self, import_text: str, base_dir: Path, source_file: Path
    ) -> Path | None:
        """Resolve Rust use/mod statement to file path."""

        # Handle mod declarations: mod foo;
        mod_match = re.search(r"mod\s+(\w+)\s*;", import_text)
        if mod_match:
            mod_name = mod_match.group(1)
            source_dir = source_file.parent

            # Try mod_name.rs
            mod_file = source_dir / f"{mod_name}.rs"
            if mod_file.exists():
                return mod_file

            # Try mod_name/mod.rs
            mod_file = source_dir / mod_name / "mod.rs"
            if mod_file.exists():
                return mod_file

            return None

        # Handle use statements: use crate::foo::bar;
        use_match = re.search(
            r"use\s+(?:crate|self|super)::(\w+(?:::\w+)*)", import_text
        )
        if use_match:
            path_parts = use_match.group(1).split("::")

            # Start from src/ for crate:: imports
            search_dir = base_dir / "src"
            if not search_dir.exists():
                search_dir = base_dir

            # Walk the module path
            # All but last (which might be item, not module)
            for part in path_parts[:-1]:
                mod_file = search_dir / f"{part}.rs"
                if mod_file.exists():
                    return mod_file
                mod_dir = search_dir / part
                if mod_dir.is_dir():
                    search_dir = mod_dir

            # Try the last part
            last = path_parts[-1]
            mod_file = search_dir / f"{last}.rs"
            if mod_file.exists():
                return mod_file
            mod_file = search_dir / last / "mod.rs"
            if mod_file.exists():
                return mod_file

        return None
