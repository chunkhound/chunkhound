"""Ruby language mapping for unified parser architecture.

This module provides Ruby-specific tree-sitter queries and extraction logic
for the universal concept system. It maps Ruby's AST nodes to universal
semantic concepts used by the unified parser.

Supports:
- Basic Ruby syntax (classes, modules, methods, constants)
- Rails DSL patterns (associations, validations, callbacks, scopes)
- JBuilder templates
- All Ruby file extensions (.rb, .rake, .gemspec, .jbuilder, Gemfile, etc.)
"""

import re
from pathlib import Path
from typing import Any

from tree_sitter import Node

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.base import MAX_CONSTANT_VALUE_LENGTH, BaseMapping
from chunkhound.parsers.universal_engine import UniversalConcept


class RubyMapping(BaseMapping):
    """Ruby-specific tree-sitter mapping with Rails DSL support."""

    # Rails DSL detection constants (Phase 2)
    ASSOCIATIONS = {
        "belongs_to",
        "has_one",
        "has_many",
        "has_and_belongs_to_many",
    }
    VALIDATIONS = {
        "validates",
        "validates_presence_of",
        "validates_uniqueness_of",
        "validates_length_of",
        "validates_numericality_of",
        "validates_format_of",
        "validates_inclusion_of",
        "validates_exclusion_of",
        "validates_acceptance_of",
        "validates_confirmation_of",
    }
    CALLBACKS = {
        "before_validation",
        "after_validation",
        "before_save",
        "after_save",
        "before_create",
        "after_create",
        "before_update",
        "after_update",
        "before_destroy",
        "after_destroy",
        "around_save",
        "around_create",
        "around_update",
        "around_destroy",
    }
    SCOPES = {"scope", "default_scope"}

    def __init__(self) -> None:
        """Initialize Ruby mapping."""
        super().__init__(Language.RUBY)

    # BaseMapping required methods
    def get_function_query(self) -> str:
        """Get tree-sitter query pattern for function definitions."""
        return """
        (method
            name: (identifier) @func_name
        ) @func_def

        (singleton_method
            object: (self)
            name: (identifier) @func_name
        ) @func_def
        """

    def get_class_query(self) -> str:
        """Get tree-sitter query pattern for class definitions."""
        return """
        (class
            name: (constant) @class_name
        ) @class_def

        (module
            name: (constant) @class_name
        ) @class_def
        """

    def get_comment_query(self) -> str:
        """Get tree-sitter query pattern for comments."""
        return """
        (comment) @comment
        """

    def extract_function_name(self, node: Node | None, source: str) -> str:
        """Extract function name from a method definition node."""
        if node is None:
            return self.get_fallback_name(node, "method")

        # Try to find the name node
        name_node = self.find_child_by_type(node, "identifier")
        if name_node:
            return self.get_node_text(name_node, source).strip()

        return self.get_fallback_name(node, "method")

    def extract_class_name(self, node: Node | None, source: str) -> str:
        """Extract class name from a class/module definition node."""
        if node is None:
            return self.get_fallback_name(node, "class")

        # Try to find the constant node (class/module name)
        name_node = self.find_child_by_type(node, "constant")
        if name_node:
            name = self.get_node_text(name_node, source).strip()

            # Handle namespaced classes (e.g., Foo::Bar)
            # Return the last part for simple names
            if "::" in name:
                return name.split("::")[-1]
            return name

        return self.get_fallback_name(node, "class")

    # LanguageMapping protocol methods
    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        """Get tree-sitter query for universal concept in Ruby."""

        if concept == UniversalConcept.DEFINITION:
            return """
            (class
                name: (constant) @name
            ) @definition

            (module
                name: (constant) @name
            ) @definition

            (method
                name: (identifier) @name
            ) @definition

            (singleton_method
                object: (self)
                name: (identifier) @name
            ) @definition

            (assignment
                left: (constant) @name
            ) @definition
            """

        elif concept == UniversalConcept.BLOCK:
            return """
            (do_block) @block

            (block) @block
            """

        elif concept == UniversalConcept.COMMENT:
            return """
            (comment) @definition
            """

        elif concept == UniversalConcept.IMPORT:
            return """
            (call
                method: (identifier) @method
                (#match? @method "^(require|require_relative|load)$")
            ) @definition
            """

        elif concept == UniversalConcept.STRUCTURE:
            return """
            (program) @definition
            """

        return None  # type: ignore[unreachable]

    def extract_name(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract name from captures for this concept."""

        source = content.decode("utf-8")

        if concept == UniversalConcept.DEFINITION:
            # Try to get the name from capture groups
            if "name" in captures:
                name_node = captures["name"]
                name = self.get_node_text(name_node, source).strip()

                # Remove leading : from symbols
                if name.startswith(":"):
                    name = name[1:]

                # For namespaced names, return full name
                return name

            return "unnamed_definition"

        elif concept == UniversalConcept.BLOCK:
            # Use location-based naming for blocks
            if "block" in captures:
                node = captures["block"]
                line = node.start_point[0] + 1
                block_type = node.type
                return f"{block_type}_line_{line}"

            return "unnamed_block"

        elif concept == UniversalConcept.COMMENT:
            # Use location-based naming for comments
            if "definition" in captures:
                node = captures["definition"]
                line = node.start_point[0] + 1
                return f"comment_line_{line}"

            return "unnamed_comment"

        elif concept == UniversalConcept.IMPORT:
            if "definition" in captures:
                def_node = captures["definition"]
                def_text = self.get_node_text(def_node, source).strip()

                # Extract the module name from require("module") or require 'module'
                match = re.search(
                    r'require(?:_relative)?\s*[\(\s]*["\']([^"\']+)["\']', def_text
                )
                if match:
                    module_name = match.group(1)
                    # Get just the module name for cleaner names
                    if "/" in module_name:
                        module_name = module_name.split("/")[-1]
                    return f"require_{module_name}"

            return "unnamed_require"

        elif concept == UniversalConcept.STRUCTURE:
            return "ruby_program"

        return "unnamed"  # type: ignore[unreachable]

    def extract_content(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract content from captures for this concept."""

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
        """Extract Ruby-specific metadata including Rails DSL patterns."""

        source = content.decode("utf-8")
        metadata: dict[str, Any] = {}

        if concept == UniversalConcept.DEFINITION:
            def_node = captures.get("definition")
            if def_node:
                metadata["node_type"] = def_node.type

                # Classes and modules
                if def_node.type in ("class", "module"):
                    metadata["kind"] = def_node.type

                    # Extract superclass for classes
                    if def_node.type == "class":
                        superclass_node = self.find_child_by_type(
                            def_node, "superclass"
                        )
                        if superclass_node:
                            superclass = self.get_node_text(
                                superclass_node, source
                            ).strip()
                            # Remove leading <
                            if superclass.startswith("<"):
                                superclass = superclass[1:].strip()
                            metadata["superclass"] = superclass

                    # Phase 2: Extract Rails DSL patterns
                    rails_patterns = self._extract_rails_patterns(def_node, source)
                    if rails_patterns:
                        metadata.update(rails_patterns)

                # Methods
                elif def_node.type in ("method", "singleton_method"):
                    metadata["kind"] = "method"

                    if def_node.type == "singleton_method":
                        metadata["is_class_method"] = True

                    # Extract method body size as complexity metric
                    body_text = self.get_node_text(def_node, source)
                    metadata["body_lines"] = len(body_text.splitlines())

                # Constants
                elif def_node.type == "assignment":
                    metadata["kind"] = "constant"

        elif concept == UniversalConcept.BLOCK:
            if "block" in captures:
                block_node = captures["block"]
                metadata["block_type"] = block_node.type

        elif concept == UniversalConcept.IMPORT:
            if "definition" in captures:
                import_node = captures["definition"]
                import_text = self.get_node_text(import_node, source).strip()

                # Detect import type
                if "require_relative" in import_text:
                    metadata["import_type"] = "require_relative"
                elif "require" in import_text:
                    metadata["import_type"] = "require"
                elif "load" in import_text:
                    metadata["import_type"] = "load"

                # Extract the module being required
                match = re.search(
                    r'(?:require|require_relative|load)\s*[\(\s]*["\']([^"\']+)["\']',
                    import_text,
                )
                if match:
                    metadata["module"] = match.group(1)

        elif concept == UniversalConcept.COMMENT:
            if "definition" in captures:
                comment_node = captures["definition"]
                comment_text = self.get_node_text(comment_node, source)

                # Clean and analyze comment
                clean_text = self.clean_comment_text(comment_text)

                # Detect special comment types
                is_doc = False
                comment_type = "regular"

                if clean_text:
                    upper_text = clean_text.upper()
                    if any(
                        prefix in upper_text
                        for prefix in ["TODO:", "FIXME:", "HACK:", "NOTE:", "WARNING:"]
                    ):
                        comment_type = "annotation"
                        is_doc = True
                    elif clean_text.startswith("#!"):
                        comment_type = "shebang"
                        is_doc = True
                    elif len(clean_text) > 50 and any(
                        word in clean_text.lower()
                        for word in [
                            "param",
                            "return",
                            "example",
                            "usage",
                            "@param",
                            "@return",
                        ]
                    ):
                        comment_type = "documentation"
                        is_doc = True

                metadata["comment_type"] = comment_type
                if is_doc:
                    metadata["is_doc_comment"] = True

        return metadata

    def clean_comment_text(self, text: str) -> str:
        """Clean Ruby comment text by removing comment markers.

        Args:
            text: Raw comment text

        Returns:
            Cleaned comment text
        """
        cleaned = text.strip()

        # Remove Ruby single-line comment marker
        if cleaned.startswith("#"):
            cleaned = cleaned[1:]

        # Multi-line comments =begin...=end are typically handled as separate nodes
        if cleaned.startswith("=begin"):
            cleaned = cleaned[6:]
        if cleaned.endswith("=end"):
            cleaned = cleaned[:-4]

        return cleaned.strip()

    def resolve_import_path(
        self, import_text: str, base_dir: Path, source_file: Path
    ) -> Path | None:
        """Resolve import path from Ruby require/require_relative/load.

        Args:
            import_text: The text of the import statement
            base_dir: Base directory of the indexed codebase
            source_file: Path to the file containing the import

        Returns:
            Resolved absolute path if found, None otherwise
        """
        # Extract the path from require/require_relative/load
        match = re.search(
            r'(?:require|require_relative|load)\s*[\(\s]*["\']([^"\']+)["\']',
            import_text,
        )
        if not match:
            return None

        module_path = match.group(1)

        # require_relative - relative to source file
        if "require_relative" in import_text:
            # Try with .rb extension
            for ext in [".rb", ""]:
                resolved = (source_file.parent / (module_path + ext)).resolve()
                if resolved.exists():
                    return resolved

        # require/load - search in base_dir and lib/
        else:
            for ext in [".rb", ""]:
                # Try relative to base directory
                full_path = base_dir / (module_path + ext)
                if full_path.exists():
                    return full_path

                # Try in lib/ directory (common Ruby pattern)
                lib_path = base_dir / "lib" / (module_path + ext)
                if lib_path.exists():
                    return lib_path

        return None

    def extract_constants(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> list[dict[str, str]] | None:
        """Extract constant definitions from Ruby code.

        Ruby constants start with uppercase letters (CONSTANT, ClassName, etc).

        Args:
            concept: The universal concept being extracted
            captures: Dictionary of capture names to tree-sitter nodes
            content: Source code as bytes

        Returns:
            List of constant dictionaries with 'name' and 'value' keys, or None
        """
        if concept != UniversalConcept.DEFINITION:
            return None

        source = content.decode("utf-8")

        # Get the definition node to extract constant name and value
        def_node = captures.get("definition")
        if not def_node or def_node.type != "assignment":
            return None

        # Extract constant name
        name_node = captures.get("name")
        if name_node:
            name = self.get_node_text(name_node, source).strip()

            # Match UPPER_SNAKE_CASE pattern for true constants
            # (not classes which also use constants)
            if name and re.match(r"^_?[A-Z][A-Z0-9_]*$", name):
                # Extract value from right side of assignment
                value = ""
                for child in def_node.children:
                    if child.type not in ("constant", "="):
                        value = self.get_node_text(child, source).strip()
                        if len(value) > MAX_CONSTANT_VALUE_LENGTH:
                            value = value[:MAX_CONSTANT_VALUE_LENGTH]
                        break

                return [{"name": name, "value": value}]

        return None

    # Rails DSL extraction methods (Phase 2)
    def _extract_rails_patterns(self, class_node: Node, source: str) -> dict[str, Any]:
        """Extract Rails DSL patterns from class body.

        Detects associations, validations, callbacks, and scopes.

        Args:
            class_node: The class definition node
            source: Source code as string

        Returns:
            Dictionary with detected Rails patterns
        """
        patterns: dict[str, Any] = {}

        # Lists to collect patterns
        associations: list[dict[str, Any]] = []
        validations: list[dict[str, Any]] = []
        callbacks: list[dict[str, Any]] = []
        scopes: list[dict[str, Any]] = []

        # Walk through class body looking for method calls
        for child in self.walk_tree(class_node):
            if child and child.type == "call":
                # Get method name
                method_node = self.find_child_by_type(child, "identifier")
                if not method_node:
                    continue

                method_name = self.get_node_text(method_node, source).strip()

                # Check if this is a Rails DSL method
                if method_name in self.ASSOCIATIONS:
                    assoc = self._parse_association(child, method_name, source)
                    if assoc:
                        associations.append(assoc)

                elif method_name in self.VALIDATIONS:
                    validation = self._parse_validation(child, method_name, source)
                    if validation:
                        validations.append(validation)

                elif method_name in self.CALLBACKS:
                    callback = self._parse_callback(child, method_name, source)
                    if callback:
                        callbacks.append(callback)

                elif method_name in self.SCOPES:
                    scope = self._parse_scope(child, source)
                    if scope:
                        scopes.append(scope)

        # Add to patterns if found
        if associations:
            patterns["associations"] = associations
        if validations:
            patterns["validations"] = validations
        if callbacks:
            patterns["callbacks"] = callbacks
        if scopes:
            patterns["scopes"] = scopes

        # Mark as Rails model if any patterns found
        if patterns:
            patterns["rails_model"] = True

        return patterns

    def _parse_association(
        self, call_node: Node, assoc_type: str, source: str
    ) -> dict[str, Any] | None:
        """Parse Rails association (belongs_to, has_many, etc).

        Args:
            call_node: The method call node
            assoc_type: Type of association (belongs_to, has_many, etc)
            source: Source code as string

        Returns:
            Dictionary with association info or None
        """
        # Find argument_list
        args_node = self.find_child_by_type(call_node, "argument_list")
        if not args_node:
            return None

        assoc_info: dict[str, Any] = {"type": assoc_type}

        # Extract first symbol (association name)
        for child in args_node.children:
            if child.type in ("simple_symbol", "symbol"):
                name_text = self.get_node_text(child, source).strip()
                # Remove leading :
                if name_text.startswith(":"):
                    name_text = name_text[1:]
                assoc_info["name"] = name_text
                break

        # Extract hash options (class_name, foreign_key, dependent, etc)
        for child in args_node.children:
            if child.type == "hash":
                options = self._parse_hash_options(child, source)
                if options:
                    assoc_info.update(options)
                break

        # Only return if we found a name
        if "name" in assoc_info:
            return assoc_info

        return None

    def _parse_validation(
        self, call_node: Node, val_type: str, source: str
    ) -> dict[str, Any] | None:
        """Parse Rails validation.

        Args:
            call_node: The method call node
            val_type: Type of validation (validates, validates_presence_of, etc)
            source: Source code as string

        Returns:
            Dictionary with validation info or None
        """
        # Find argument_list
        args_node = self.find_child_by_type(call_node, "argument_list")
        if not args_node:
            return None

        val_info: dict[str, Any] = {}

        # Extract field name(s) - can be multiple symbols
        fields: list[str] = []
        for child in args_node.children:
            if child.type in ("simple_symbol", "symbol"):
                field_text = self.get_node_text(child, source).strip()
                # Remove leading :
                if field_text.startswith(":"):
                    field_text = field_text[1:]
                fields.append(field_text)

        if fields:
            # For single field, store as string
            if len(fields) == 1:
                val_info["field"] = fields[0]
            else:
                val_info["fields"] = fields

        # Extract validation rules from hash
        rules: list[str] = []
        for child in args_node.children:
            if child.type == "hash":
                hash_text = self.get_node_text(child, source).strip()
                # Extract rule names from keys
                # Simple extraction - look for common patterns
                if "presence" in hash_text:
                    rules.append("presence")
                if "uniqueness" in hash_text:
                    rules.append("uniqueness")
                if "length" in hash_text:
                    rules.append("length")
                if "format" in hash_text:
                    rules.append("format")
                if "numericality" in hash_text:
                    rules.append("numericality")
                break

        # For specific validation methods like validates_presence_of
        if "_" in val_type:
            # Extract rule from method name (validates_presence_of -> presence)
            rule = val_type.replace("validates_", "").replace("_of", "")
            rules.append(rule)

        if rules:
            val_info["rules"] = rules

        # Only return if we found field(s)
        if "field" in val_info or "fields" in val_info:
            return val_info

        return None

    def _parse_callback(
        self, call_node: Node, callback_type: str, source: str
    ) -> dict[str, Any] | None:
        """Parse Rails callback (before_save, after_create, etc).

        Args:
            call_node: The method call node
            callback_type: Type of callback
            source: Source code as string

        Returns:
            Dictionary with callback info or None
        """
        # Find argument_list
        args_node = self.find_child_by_type(call_node, "argument_list")
        if not args_node:
            return None

        callback_info: dict[str, Any] = {"type": callback_type}

        # Extract method name - usually a symbol
        for child in args_node.children:
            if child.type in ("simple_symbol", "symbol"):
                method_text = self.get_node_text(child, source).strip()
                # Remove leading :
                if method_text.startswith(":"):
                    method_text = method_text[1:]
                callback_info["method"] = method_text
                break

        # Only return if we found a method
        if "method" in callback_info:
            return callback_info

        return None

    def _parse_scope(self, call_node: Node, source: str) -> dict[str, Any] | None:
        """Parse Rails scope definition.

        Args:
            call_node: The method call node
            source: Source code as string

        Returns:
            Dictionary with scope info or None
        """
        # Find argument_list
        args_node = self.find_child_by_type(call_node, "argument_list")
        if not args_node:
            return None

        scope_info: dict[str, Any] = {}

        # Extract scope name - usually first symbol or identifier
        for child in args_node.children:
            if child.type in ("simple_symbol", "symbol"):
                name_text = self.get_node_text(child, source).strip()
                # Remove leading :
                if name_text.startswith(":"):
                    name_text = name_text[1:]
                scope_info["name"] = name_text
                break
            elif child.type == "identifier":
                scope_info["name"] = self.get_node_text(child, source).strip()
                break

        # Only return if we found a name
        if "name" in scope_info:
            return scope_info

        return None

    def _parse_hash_options(self, hash_node: Node, source: str) -> dict[str, str]:
        """Parse Ruby hash options from association/validation args.

        Args:
            hash_node: The hash node
            source: Source code as string

        Returns:
            Dictionary of option key-value pairs
        """
        options: dict[str, str] = {}

        # Walk through pairs in hash
        for child in hash_node.children:
            if child.type == "pair":
                # Get key and value
                key_node = None
                value_node = None

                for pair_child in child.children:
                    if pair_child.type in ("simple_symbol", "symbol", "identifier"):
                        if key_node is None:
                            key_node = pair_child
                        else:
                            value_node = pair_child
                    elif pair_child.type == "string":
                        value_node = pair_child

                if key_node:
                    key = self.get_node_text(key_node, source).strip()
                    # Remove leading : from symbol keys
                    if key.startswith(":"):
                        key = key[1:]

                    if value_node:
                        value = self.get_node_text(value_node, source).strip()
                        # Remove quotes from strings
                        if value.startswith(("'", '"')):
                            value = value[1:-1]
                        # Remove leading : from symbol values
                        if value.startswith(":"):
                            value = value[1:]
                        options[key] = value

        return options
