"""Makefile language mapping for unified parser architecture.

This module provides Makefile-specific tree-sitter queries and extraction logic
for the universal concept system. It maps Makefile's AST nodes to universal
semantic concepts used by the unified parser.
"""

import re
from pathlib import Path
from typing import Any

from tree_sitter import Node

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.base import MAX_CONSTANT_VALUE_LENGTH, BaseMapping
from chunkhound.parsers.universal_engine import UniversalConcept


class MakefileMapping(BaseMapping):
    """Makefile-specific tree-sitter mapping for universal concepts."""

    def __init__(self) -> None:
        """Initialize Makefile mapping."""
        super().__init__(Language.MAKEFILE)

    # BaseMapping required methods
    def get_function_query(self) -> str:
        """Get tree-sitter query pattern for function definitions (Make macros)."""
        return """
        (variable_assignment
            name: (_) @func_name
        ) @func_def
        """

    def get_class_query(self) -> str:
        """Get tree-sitter query pattern for class definitions (not applicable to Makefiles)."""
        return ""

    def get_comment_query(self) -> str:
        """Get tree-sitter query pattern for comments."""
        return """
        (comment) @comment
        """

    def extract_function_name(self, node: Node | None, source: str) -> str:
        """Extract function/macro name from a variable assignment node."""
        if node is None:
            return self.get_fallback_name(node, "variable")

        # Find the variable name child
        name_node = self.find_child_by_type(node, "_")
        if name_node:
            return self.get_node_text(name_node, source).strip()

        return self.get_fallback_name(node, "variable")

    def extract_class_name(self, node: Node | None, source: str) -> str:
        """Extract class name from a class definition node (not applicable to Makefiles)."""
        return ""

    # LanguageMapping protocol methods
    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        """Get tree-sitter query for universal concept in Makefile."""

        if concept == UniversalConcept.DEFINITION:
            return """
            (rule
                (targets) @targets
            ) @definition
            
            (variable_assignment
                name: (_) @name
            ) @definition
            
            (define_directive
                name: (_) @name
            ) @definition
            """

        elif concept == UniversalConcept.BLOCK:
            return """
            (rule
                (recipe) @recipe
            ) @definition
            
            (conditional
                condition: (_) @condition
                (_) @block
            ) @definition
            
            (define_directive
                name: (_) @name
                (_) @block
            ) @definition
            """

        elif concept == UniversalConcept.COMMENT:
            return """
            (comment) @definition
            """

        elif concept == UniversalConcept.IMPORT:
            return """
            (include_directive
                filenames: (_) @include_path
            ) @definition
            
            (include_directive
                filenames: (_) @include_path
            ) @definition
            
            (include_directive
                filenames: (_) @include_path
            ) @definition
            """

        elif concept == UniversalConcept.STRUCTURE:
            return """
            (makefile) @definition
            """

        # All cases handled above
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
                return name
            elif "targets" in captures:
                targets_node = captures["targets"]
                targets_text = self.get_node_text(targets_node, source).strip()
                # Get the first target as the primary name
                first_target = (
                    targets_text.split()[0] if targets_text else "unnamed_target"
                )
                return first_target

            return "unnamed_definition"

        elif concept == UniversalConcept.BLOCK:
            # Use location-based naming for blocks
            if "definition" in captures:
                node = captures["definition"]
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
            if "include_path" in captures:
                path_node = captures["include_path"]
                path = self.get_node_text(path_node, source).strip()
                # Remove quotes if present
                path = path.strip("\"'")
                # Get just the filename for cleaner names
                if "/" in path:
                    path = path.split("/")[-1]
                return f"include_{path}"

            return "unnamed_include"

        elif concept == UniversalConcept.STRUCTURE:
            return "makefile"

        # All cases handled above
        return "unnamed"

    def extract_content(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract content from captures for this concept."""

        # Convert bytes to string for processing
        source = content.decode("utf-8")

        if "definition" in captures:
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
        """Extract Makefile-specific metadata."""

        source = content.decode("utf-8")
        metadata = {}

        if concept == UniversalConcept.DEFINITION:
            # Extract definition specific metadata
            def_node = captures.get("definition")
            if def_node:
                metadata["node_type"] = def_node.type

                # For rules, extract targets and prerequisites
                if def_node.type == "rule":
                    metadata["kind"] = "rule"

                    # Extract all targets
                    if "targets" in captures:
                        targets_node = captures["targets"]
                        targets_text = self.get_node_text(targets_node, source).strip()
                        targets_list = [
                            t.strip() for t in targets_text.split() if t.strip()
                        ]
                        metadata["targets"] = targets_list

                        # Detect special target types
                        special_targets = {
                            ".PHONY",
                            ".SILENT",
                            ".SUFFIXES",
                            ".PRECIOUS",
                            ".INTERMEDIATE",
                            ".DELETE_ON_ERROR",
                            ".IGNORE",
                            ".LOW_RESOLUTION_TIME",
                            ".SECONDARY",
                            ".EXPORT_ALL_VARIABLES",
                        }

                        has_special = any(
                            target in special_targets for target in targets_list
                        )
                        if has_special:
                            metadata["has_special_targets"] = True

                    # Extract prerequisites/dependencies
                    prerequisites = self._extract_prerequisites(def_node, source)
                    if prerequisites:
                        metadata["prerequisites"] = prerequisites

                        # Enhanced dependency analysis
                        if "targets" in captures and prerequisites:
                            metadata["dependencies"] = {
                                "provides": targets_list,
                                "requires": prerequisites,
                            }

                            # Pattern rule detection
                            if self._is_pattern_rule(targets_list):
                                metadata["is_pattern_rule"] = True
                                metadata["pattern_stem"] = self._extract_pattern_stem(
                                    targets_list[0]
                                )

                            # Enhanced phony target detection
                            if ".PHONY" in prerequisites or self._is_phony_target(
                                targets_list, prerequisites
                            ):
                                metadata["is_phony"] = True

                    # Extract recipe/commands
                    recipe_node = self.find_child_by_type(def_node, "recipe")
                    if recipe_node:
                        recipe_text = self.get_node_text(recipe_node, source)
                        commands = self._extract_recipe_commands(recipe_text)
                        metadata["commands"] = commands
                        metadata["command_count"] = len(commands)

                        # Recipe complexity analysis
                        recipe_analysis = self._analyze_recipe_complexity(recipe_text)
                        metadata.update(recipe_analysis)

                # For variable assignments
                elif def_node.type == "variable_assignment":
                    metadata["kind"] = "variable"

                    # Extract assignment operator
                    assignment_op = self._extract_assignment_operator(def_node, source)
                    if assignment_op:
                        metadata["assignment_type"] = assignment_op

                    # Extract variable value (for simple cases)
                    value_node = self._find_value_node(def_node)
                    if value_node:
                        value = self.get_node_text(value_node, source).strip()
                        # Only store short values to avoid clutter
                        if len(value) < 200:
                            metadata["value"] = value

                # For define directives (multi-line variables)
                elif def_node.type == "define_directive":
                    metadata["kind"] = "define"

                    # Extract the body content
                    body_node = self.find_child_by_type(def_node, "body")
                    if body_node:
                        body_text = self.get_node_text(body_node, source)
                        metadata["body_lines"] = len(body_text.splitlines())

        elif concept == UniversalConcept.BLOCK:
            if "definition" in captures:
                block_node = captures["definition"]
                metadata["block_type"] = block_node.type

                # For recipe blocks, count commands
                if block_node.type == "rule":
                    recipe_node = self.find_child_by_type(block_node, "recipe")
                    if recipe_node:
                        recipe_text = self.get_node_text(recipe_node, source)
                        commands = self._extract_recipe_commands(recipe_text)
                        metadata["command_count"] = len(commands)

                        # Detect command types
                        command_types = set()
                        for cmd in commands:
                            if cmd.startswith("@"):
                                command_types.add("silent")
                            if cmd.startswith("-"):
                                command_types.add("ignore_errors")
                            if cmd.startswith("+"):
                                command_types.add("always_exec")

                        if command_types:
                            metadata["command_modifiers"] = list(command_types)

        elif concept == UniversalConcept.IMPORT:
            if "definition" in captures:
                import_node = captures["definition"]
                import_type = import_node.type.replace("_directive", "")
                metadata["include_type"] = import_type

                if "include_path" in captures:
                    path_node = captures["include_path"]
                    path = self.get_node_text(path_node, source).strip().strip("\"'")
                    metadata["include_path"] = path

                    # Determine if it's a relative or absolute path
                    if path.startswith("/"):
                        metadata["path_type"] = "absolute"
                    elif path.startswith("./") or path.startswith("../"):
                        metadata["path_type"] = "relative"
                    else:
                        metadata["path_type"] = "relative_simple"

        elif concept == UniversalConcept.COMMENT:
            if "definition" in captures:
                comment_node = captures["definition"]
                comment_text = self.get_node_text(comment_node, source)

                # Clean and analyze comment
                clean_text = self.clean_comment_text(comment_text)

                # Detect special comment types
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
                    elif any(
                        word in clean_text.lower()
                        for word in ["target", "rule", "variable", "usage", "example"]
                    ):
                        comment_type = "documentation"
                        is_doc = True
                    elif clean_text.startswith("===") or clean_text.startswith("---"):
                        comment_type = "section_header"
                        is_doc = True

                metadata["comment_type"] = comment_type
                if is_doc:
                    metadata["is_doc_comment"] = True

        return metadata

    def _extract_prerequisites(self, rule_node: Node, source: str) -> list[str]:
        """Extract prerequisites/dependencies from a rule."""
        prerequisites = []

        # Look for prerequisites in the rule
        for child in self.walk_tree(rule_node):
            if child and child.type == "prerequisites":
                prereq_text = self.get_node_text(child, source).strip()
                # Split on whitespace and filter out empty strings
                prereq_list = [p.strip() for p in prereq_text.split() if p.strip()]
                prerequisites.extend(prereq_list)
                break

        return prerequisites

    def _extract_assignment_operator(
        self, assignment_node: Node, source: str
    ) -> str | None:
        """Extract the assignment operator from a variable assignment."""
        assignment_text = self.get_node_text(assignment_node, source)

        # Look for common Make assignment operators
        operators = ["::=", ":=", "+=", "?=", "!=", "="]
        for op in operators:
            if op in assignment_text:
                return op

        return None

    def _find_value_node(self, assignment_node: Node) -> Node | None:
        """Find the value node in a variable assignment."""
        # Look for common value node types
        value_types = ["text", "variable_reference", "function_call", "string"]

        for child in self.walk_tree(assignment_node):
            if child and child.type in value_types:
                return child

        return None

    def _extract_recipe_commands(self, recipe_text: str) -> list[str]:
        """Extract individual commands from a recipe block."""
        commands = []

        # Split by lines and process each command
        lines = recipe_text.splitlines()
        for line in lines:
            # Skip empty lines and lines that are just whitespace
            stripped = line.strip()
            if stripped:
                # Remove the leading tab that Make requires
                if line.startswith("\t"):
                    command = line[1:]
                else:
                    command = line

                # Handle line continuations
                if command.strip().endswith("\\"):
                    command = command.strip()[:-1].strip()

                if command.strip():
                    commands.append(command.strip())

        return commands

    def _is_pattern_rule(self, targets: list[str]) -> bool:
        """Check if this is a pattern rule (contains % wildcards)."""
        return any("%" in target for target in targets)

    def _is_phony_target(self, targets: list[str], prerequisites: list[str]) -> bool:
        """Heuristic to detect if this might be a phony target."""
        phony_indicators = {
            "all",
            "clean",
            "install",
            "test",
            "check",
            "dist",
            "distclean",
            "mostlyclean",
            "maintainer-clean",
        }

        return any(target.lower() in phony_indicators for target in targets)

    def _extract_pattern_stem(self, pattern_target: str) -> str:
        """Extract the stem part of a pattern rule target."""
        if "%" in pattern_target:
            return pattern_target.split("%")[0]
        return pattern_target

    def _analyze_recipe_complexity(self, recipe_text: str) -> dict[str, Any]:
        """Analyze recipe complexity for better chunking hints."""
        commands = self._extract_recipe_commands(recipe_text)

        analysis = {
            "has_shell_constructs": any(
                any(
                    construct in cmd
                    for construct in ["if", "for", "while", "&&", "||", "|"]
                )
                for cmd in commands
            ),
            "has_multiline_commands": any("\\" in cmd for cmd in commands),
            "estimated_complexity": "high" if len(commands) > 5 else "low",
        }

        return analysis

    def _detect_function_type(self, function_name: str) -> str:
        """Detect built-in vs user-defined function types."""
        builtin_functions = {
            "call",
            "eval",
            "foreach",
            "if",
            "or",
            "and",
            "strip",
            "filter",
            "filter-out",
            "sort",
            "word",
            "words",
            "wordlist",
            "firstword",
            "lastword",
            "dir",
            "notdir",
            "suffix",
            "basename",
            "addsuffix",
            "addprefix",
            "join",
            "wildcard",
            "realpath",
            "abspath",
            "shell",
        }

        return "builtin" if function_name in builtin_functions else "user_defined"

    def extract_constants(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> list[dict[str, str]] | None:
        """Extract constant definitions from Makefile.

        Identifies variable assignments (VAR = value, VAR := value) as constants.

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
        if not def_node or def_node.type != "variable_assignment":
            return None

        source = content.decode("utf-8")

        # Extract variable name
        name_node = captures.get("name")
        if not name_node:
            return None

        name = self.get_node_text(name_node, source).strip()

        # Extract value
        value_node = self._find_value_node(def_node)
        value = ""
        if value_node:
            value = self.get_node_text(value_node, source).strip()

        # Truncate long values
        if len(value) > MAX_CONSTANT_VALUE_LENGTH:
            value = value[:MAX_CONSTANT_VALUE_LENGTH] + "..."

        return [{"name": name, "value": value}]

    def resolve_import_path(
        self, import_text: str, base_dir: Path, source_file: Path
    ) -> Path | None:
        """Resolve import path from Makefile include directive.

        Args:
            import_text: The text of the import statement
            base_dir: Base directory of the indexed codebase
            source_file: Path to the file containing the import

        Returns:
            Resolved absolute path if found, None otherwise
        """
        # include file.mk or -include file.mk
        match = re.search(r"-?include\s+(\S+)", import_text)
        if not match:
            return None

        path = match.group(1)

        # Try relative to source file first
        resolved = (source_file.parent / path).resolve()
        if resolved.exists():
            return resolved

        # Try relative to base directory
        full_path = base_dir / path
        if full_path.exists():
            return full_path

        return None
