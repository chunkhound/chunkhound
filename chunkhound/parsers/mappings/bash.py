"""Bash language mapping for unified parser architecture.

This module provides Bash-specific tree-sitter queries and extraction logic
for the universal concept system. It maps Bash's AST nodes to universal
semantic concepts used by the unified parser.
"""

import re
from pathlib import Path
from typing import Any

from tree_sitter import Node

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.base import MAX_CONSTANT_VALUE_LENGTH, BaseMapping
from chunkhound.parsers.universal_engine import UniversalConcept


class BashMapping(BaseMapping):
    """Bash-specific tree-sitter mapping for universal concepts."""

    def __init__(self) -> None:
        """Initialize Bash mapping."""
        super().__init__(Language.BASH)

    # BaseMapping required methods
    def get_function_query(self) -> str:
        """Get tree-sitter query pattern for function definitions."""
        return """
        (function_definition
            name: (word) @func_name
        ) @func_def
        """

    def get_class_query(self) -> str:
        """Get tree-sitter query pattern for class definitions (not applicable to Bash)."""
        return ""

    def get_comment_query(self) -> str:
        """Get tree-sitter query pattern for comments."""
        return """
        (comment) @comment
        """

    def extract_function_name(self, node: Node | None, source: str) -> str:
        """Extract function name from a function definition node."""
        if node is None:
            return self.get_fallback_name(node, "function")

        # Find the function name child
        name_node = self.find_child_by_type(node, "word")
        if name_node:
            return self.get_node_text(name_node, source).strip()

        return self.get_fallback_name(node, "function")

    def extract_class_name(self, node: Node | None, source: str) -> str:
        """Extract class name from a class definition node (not applicable to Bash)."""
        return ""

    # LanguageMapping protocol methods
    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        """Get tree-sitter query for universal concept in Bash."""

        if concept == UniversalConcept.DEFINITION:
            return """
            (function_definition
                name: (word) @name
            ) @definition
            
            (variable_assignment
                name: (variable_name) @name
            ) @definition
            
            (for_statement
                variable: (variable_name) @name
            ) @definition

            (declaration_command) @definition
            """

        elif concept == UniversalConcept.BLOCK:
            return """
            (compound_statement) @block
            
            (if_statement
                (compound_statement) @block
            )
            
            (for_statement
                (do_group
                    (compound_statement) @block
                )
            )
            
            (while_statement
                (do_group
                    (compound_statement) @block
                )
            )
            
            (case_statement) @block
            
            (subshell) @block
            """

        elif concept == UniversalConcept.COMMENT:
            return """
            (comment) @definition
            """

        elif concept == UniversalConcept.IMPORT:
            return """
            (command
                name: (command_name) @cmd_name
                (#match? @cmd_name "^(source|\\.)$")
            ) @definition
            """

        elif concept == UniversalConcept.STRUCTURE:
            return """
            (program) @definition
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
            if "cmd_name" in captures:
                cmd_node = captures["cmd_name"]
                cmd = self.get_node_text(cmd_node, source).strip()
                # Extract the sourced file from the command arguments
                if "definition" in captures:
                    def_node = captures["definition"]
                    def_text = self.get_node_text(def_node, source).strip()
                    # Simple extraction of the first argument after source/. command
                    parts = def_text.split()
                    if len(parts) > 1:
                        source_file = parts[1].strip("\"'")
                        # Get just the filename for cleaner names
                        if "/" in source_file:
                            source_file = source_file.split("/")[-1]
                        return f"source_{source_file}"
                return f"source_{cmd}"

            return "unnamed_source"

        elif concept == UniversalConcept.STRUCTURE:
            return "bash_script"

        # All cases handled above
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
        """Extract Bash-specific metadata."""

        source = content.decode("utf-8")
        metadata = {}

        if concept == UniversalConcept.DEFINITION:
            # Extract definition specific metadata
            def_node = captures.get("definition")
            if def_node:
                metadata["node_type"] = def_node.type

                # For functions, extract basic info
                if def_node.type == "function_definition":
                    metadata["kind"] = "function"
                    # Extract function body size as a complexity metric
                    body_node = self.find_child_by_type(def_node, "compound_statement")
                    if body_node:
                        body_text = self.get_node_text(body_node, source)
                        metadata["body_lines"] = len(body_text.splitlines())

                # For variable assignments
                elif def_node.type == "variable_assignment":
                    metadata["kind"] = "variable"
                    # Extract variable value if it's a simple assignment
                    for child in self.walk_tree(def_node):
                        if child and child.type in ["string", "word", "concatenation"]:
                            value = self.get_node_text(child, source).strip()
                            # Only store short values to avoid clutter
                            if len(value) < 100:
                                metadata["value"] = value
                            break

                # For loop variables
                elif def_node.type == "for_statement":
                    metadata["kind"] = "loop_variable"
                    # Extract the list being iterated over
                    for child in self.walk_tree(def_node):
                        if (
                            child
                            and child.type == "word"
                            and child != captures.get("name")
                        ):
                            # This might be part of the iteration list
                            list_text = self.get_node_text(child, source).strip()
                            if list_text and not list_text.startswith("$"):
                                metadata["iteration_list"] = list_text
                            break

        elif concept == UniversalConcept.BLOCK:
            if "block" in captures:
                block_node = captures["block"]
                metadata["block_type"] = block_node.type

                # Count statements in the block for complexity
                if block_node.type == "compound_statement":
                    statements = 0
                    for child in self.walk_tree(block_node):
                        if child and child.type in [
                            "command",
                            "pipeline",
                            "variable_assignment",
                            "if_statement",
                            "for_statement",
                            "while_statement",
                        ]:
                            statements += 1
                    metadata["statement_count"] = statements

        elif concept == UniversalConcept.IMPORT:
            if "definition" in captures:
                import_node = captures["definition"]
                import_text = self.get_node_text(import_node, source).strip()

                # Extract the file being sourced
                parts = import_text.split()
                if len(parts) > 1:
                    source_file = parts[1].strip("\"'")
                    metadata["source_file"] = source_file

                    # Determine if it's a relative or absolute path
                    if source_file.startswith("/"):
                        metadata["path_type"] = "absolute"
                    elif source_file.startswith("./") or source_file.startswith("../"):
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
                    elif clean_text.startswith("#!/"):
                        comment_type = "shebang"
                        is_doc = True
                    elif len(clean_text) > 50 and any(
                        word in clean_text.lower()
                        for word in ["function", "parameter", "return", "usage"]
                    ):
                        comment_type = "documentation"
                        is_doc = True

                metadata["comment_type"] = comment_type
                if is_doc:
                    metadata["is_doc_comment"] = True

        return metadata

    def _extract_pipeline_commands(self, pipeline_node: Node, source: str) -> list[str]:
        """Extract command names from a pipeline."""
        commands = []

        for child in self.walk_tree(pipeline_node):
            if child and child.type == "command":
                cmd_name_node = self.find_child_by_type(child, "command_name")
                if cmd_name_node:
                    cmd_name = self.get_node_text(cmd_name_node, source).strip()
                    commands.append(cmd_name)

        return commands

    def _is_builtin_command(self, command: str) -> bool:
        """Check if a command is a Bash builtin."""
        builtins = {
            "alias",
            "bind",
            "builtin",
            "caller",
            "command",
            "declare",
            "echo",
            "enable",
            "help",
            "let",
            "local",
            "logout",
            "mapfile",
            "printf",
            "read",
            "readarray",
            "source",
            "type",
            "typeset",
            "ulimit",
            "unalias",
            "set",
            "unset",
            "export",
            "cd",
            "pwd",
            "pushd",
            "popd",
            "dirs",
            "jobs",
            "bg",
            "fg",
            "kill",
            "wait",
            "trap",
            "exit",
            "return",
            "break",
            "continue",
            "test",
            "[",
            "eval",
            "exec",
            "shift",
            "getopts",
            "hash",
            "history",
            "fc",
            "compgen",
            "complete",
            "shopt",
        }
        return command in builtins

    def extract_constants(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> list[dict[str, str]] | None:
        """Extract constant definitions from Bash code.

        Identifies readonly variables and declare -r variables as constants.

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
        if not def_node or def_node.type != "declaration_command":
            return None

        source = content.decode("utf-8")
        node_text = self.get_node_text(def_node, source)

        # Only extract readonly or declare -r declarations (not declare without -r)
        if not (node_text.startswith("readonly ") or "declare -r" in node_text):
            return None

        # Extract variable name and value
        # Pattern: readonly VAR=value or declare -r VAR=value
        match = re.match(
            r"(?:readonly|declare\s+-r)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)",
            node_text.strip(),
        )
        if not match:
            return None

        name = match.group(1)
        value = match.group(2).strip()

        # Truncate long values
        if len(value) > MAX_CONSTANT_VALUE_LENGTH:
            value = value[:MAX_CONSTANT_VALUE_LENGTH] + "..."

        return [{"name": name, "value": value}]

    def resolve_import_path(
        self, import_text: str, base_dir: Path, source_file: Path
    ) -> Path | None:
        """Resolve import path from Bash source/. command.

        Args:
            import_text: The text of the import statement
            base_dir: Base directory of the indexed codebase
            source_file: Path to the file containing the import

        Returns:
            Resolved absolute path if found, None otherwise
        """
        # source ./file.sh or . ./file.sh
        match = re.search(r'(?:source|\.)\s+["\']?([^\s"\']+)', import_text)
        if not match:
            return None

        path = match.group(1)

        # Try relative to source file first
        if path.startswith("./") or path.startswith("../"):
            resolved = (source_file.parent / path).resolve()
            if resolved.exists():
                return resolved

        # Try relative to base directory
        full_path = base_dir / path
        if full_path.exists():
            return full_path

        return None
