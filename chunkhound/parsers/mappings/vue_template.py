"""Vue template language mapping for directive parsing.

This module provides Vue template-specific tree-sitter queries and extraction
logic for mapping Vue template AST nodes to semantic chunks.

## Supported Features
- Conditional rendering (v-if, v-else-if, v-else)
- List rendering (v-for)
- Event handlers (@click, @submit, etc.)
- Property bindings (:prop, v-bind)
- Two-way binding (v-model)
- Component usage (PascalCase tags)
- Interpolations ({{ variable }})
- Slot usage

## Limitations
- Does not parse nested JavaScript expressions within directives
- Component props are extracted as strings, not parsed as JS
- Event handler expressions are captured but not analyzed
"""

from pathlib import Path
from typing import Any

from tree_sitter import Node as TSNode

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.base import BaseMapping
from chunkhound.parsers.universal_engine import UniversalConcept


class VueTemplateMapping(BaseMapping):
    """Vue template language mapping for directive parsing.

    This mapping handles Vue template syntax including directives, components,
    and interpolations. It extends BaseMapping to provide Vue-specific queries
    and extraction logic.
    """

    def __init__(self) -> None:
        """Initialize Vue template mapping."""
        super().__init__(Language.VUE)

    def get_function_query(self) -> str:
        """Vue templates don't have functions.

        Returns:
            Empty string (no functions in templates)
        """
        return ""

    def get_class_query(self) -> str:
        """Vue templates don't have classes.

        Returns:
            Empty string (no classes in templates)
        """
        return ""

    def get_comment_query(self) -> str:
        """Get tree-sitter query pattern for Vue template comments.

        Returns:
            Tree-sitter query string for finding template comments
        """
        return """
            (comment) @definition
        """

    def extract_function_name(self, node: TSNode | None, source: str) -> str:
        """Not applicable for Vue templates.

        Args:
            node: Tree-sitter node
            source: Source code string

        Returns:
            Empty string
        """
        return ""

    def extract_class_name(self, node: TSNode | None, source: str) -> str:
        """Not applicable for Vue templates.

        Args:
            node: Tree-sitter node
            source: Source code string

        Returns:
            Empty string
        """
        return ""

    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        """Get tree-sitter query for universal concept in Vue templates.

        Args:
            concept: The universal concept to query for

        Returns:
            Tree-sitter query string or None if concept not supported
        """
        if concept == UniversalConcept.DEFINITION:
            # Directives, components, and interpolations are "definitions"
            return self._get_directive_query()

        elif concept == UniversalConcept.BLOCK:
            # Conditional and loop blocks
            return self._get_block_query()

        elif concept == UniversalConcept.COMMENT:
            # Template comments
            return self.get_comment_query()

        elif concept == UniversalConcept.STRUCTURE:
            # Component structure
            return self._get_component_query()

        return None

    def _get_directive_query(self) -> str:
        """Get query for all Vue directives.

        Returns:
            Tree-sitter query for directives
        """
        return """
            ; All directive attributes
            (directive_attribute) @definition

            ; Interpolations {{ variable }}
            (interpolation
              (raw_text)? @interpolation_expr
            ) @definition

            ; Component usage (PascalCase tags)
            (element
              (start_tag
                (tag_name) @component_name
                (#match? @component_name "^[A-Z]")
              )
            ) @definition

            ; Self-closing components
            (self_closing_tag
              (tag_name) @component_name
              (#match? @component_name "^[A-Z]")
            ) @definition
        """

    def _get_block_query(self) -> str:
        """Get query for template blocks (conditional, loops).

        Returns:
            Tree-sitter query for blocks
        """
        return """
            ; Elements with v-if create conditional blocks
            (element
              (start_tag
                (directive_attribute
                  (directive_name) @directive_name
                  (#eq? @directive_name "v-if")
                )
              )
            ) @block

            ; Elements with v-for create loop blocks
            (element
              (start_tag
                (directive_attribute
                  (directive_name) @directive_name
                  (#eq? @directive_name "v-for")
                )
              )
            ) @block
        """

    def _get_component_query(self) -> str:
        """Component queries are now part of the DEFINITION query.

        Returns:
            Empty string - components handled in _get_directive_query
        """
        return ""

    def _parse_directive_attribute(self, directive_attr_text: str) -> dict[str, str] | None:
        """Parse a Vue directive attribute into its components.

        Args:
            directive_attr_text: The raw directive attribute text (e.g., 'v-if="condition"')

        Returns:
            Dict with keys: 'directive', 'argument', 'value' or None if not a valid directive
        """
        directive_attr_text = directive_attr_text.strip()

        # Handle bare directives (no = sign)
        if "=" not in directive_attr_text:
            return {
                "directive": directive_attr_text,
                "argument": "",
                "value": "",
            }

        # Parse directives with = sign
        parts = directive_attr_text.split("=", 1)
        if len(parts) != 2:
            return None

        directive_part = parts[0].strip()
        value_part = parts[1].strip().strip('"').strip("'")

        # Parse directive and argument
        # Handle v-on: and @ syntax
        if directive_part.startswith("@") or directive_part.startswith("v-on:"):
            if directive_part.startswith("@"):
                argument = directive_part[1:]  # Remove @
                directive = "@"
            else:
                argument = directive_part[5:]  # Remove v-on:
                directive = "v-on"
        # Handle v-bind: and : syntax
        elif directive_part.startswith(":") or directive_part.startswith("v-bind:"):
            if directive_part.startswith(":"):
                argument = directive_part[1:]  # Remove :
                directive = ":"
            else:
                argument = directive_part[7:]  # Remove v-bind:
                directive = "v-bind"
        # Handle v-slot: and # syntax
        elif directive_part.startswith("v-slot:") or directive_part.startswith("#"):
            if directive_part.startswith("v-slot:"):
                argument = directive_part[7:]  # Remove v-slot:
                directive = "v-slot"
            else:
                argument = directive_part[1:]  # Remove #
                directive = "v-slot"
        # Handle other directives (v-if, v-for, v-model, etc.)
        else:
            directive = directive_part
            argument = ""

        return {
            "directive": directive,
            "argument": argument,
            "value": value_part,
        }

    def extract_name(
        self, concept: UniversalConcept, captures: dict[str, TSNode], content: bytes
    ) -> str:
        """Extract name from captures for this concept.

        Args:
            concept: The universal concept being extracted
            captures: Dictionary of capture names to tree-sitter nodes
            content: Source code as bytes

        Returns:
            Extracted name string
        """
        source = content.decode("utf-8")

        if concept == UniversalConcept.DEFINITION:
            # Handle directive_attribute nodes
            if "definition" in captures:
                directive_attr_node = captures["definition"]
                directive_attr_text = self.get_node_text(directive_attr_node, source).strip()

                parsed = self._parse_directive_attribute(directive_attr_text)
                if parsed:
                    directive = parsed["directive"]
                    argument = parsed["argument"]
                    value = parsed["value"]

                    # Generate names based on directive type
                    if directive in ["v-if", "v-else-if"]:
                        expr = self.get_expression_preview(value, max_length=20)
                        return f"v-if_{expr}"
                    elif directive == "v-for":
                        expr = self.get_expression_preview(value, max_length=20)
                        return f"v-for_{expr}"
                    elif directive == "v-model":
                        expr = self.get_expression_preview(value, max_length=20)
                        return f"v-model_{expr}"
                    elif directive == "@":
                        return f"@{argument}"
                    elif directive == "v-on":
                        return f"@{argument}"
                    elif directive == ":":
                        return f":{argument}"
                    elif directive == "v-bind":
                        return f":{argument}"
                    elif directive == "v-slot":
                        return f"v-slot:{argument}"
                    elif directive == "v-else":
                        return "v-else"

            # Handle interpolations
            if "interpolation_expr" in captures:
                expr_node = captures["interpolation_expr"]
                expr = self.get_node_text(expr_node, source).strip()
                # Truncate long expressions
                if len(expr) > 30:
                    expr = expr[:27] + "..."
                return f"{{{{ {expr} }}}}"

            # Handle components
            if "component_name" in captures:
                component_node = captures["component_name"]
                component = self.get_node_text(component_node, source).strip()
                return f"Component_{component}"

        elif concept == UniversalConcept.BLOCK:
            # Use location-based naming for blocks
            if "block" in captures:
                node = captures["block"]
                line = node.start_point[0] + 1
                return f"block_line_{line}"

        elif concept == UniversalConcept.COMMENT:
            if "definition" in captures:
                node = captures["definition"]
                line = node.start_point[0] + 1
                return f"template_comment_line_{line}"

        elif concept == UniversalConcept.STRUCTURE:
            if "component_name" in captures:
                component_node = captures["component_name"]
                component = self.get_node_text(component_node, source).strip()
                return f"Component_{component}"

        return "unnamed"

    def extract_content(
        self, concept: UniversalConcept, captures: dict[str, TSNode], content: bytes
    ) -> str:
        """Extract content from captures for this concept.

        Args:
            concept: The universal concept being extracted
            captures: Dictionary of capture names to tree-sitter nodes
            content: Source code as bytes

        Returns:
            Extracted content as string
        """
        source = content.decode("utf-8")

        if "definition" in captures:
            node = captures["definition"]
            return self.get_node_text(node, source)
        elif "block" in captures:
            node = captures["block"]
            return self.get_node_text(node, source)
        elif captures:
            # Use the first available capture
            node = list(captures.values())[0]
            return self.get_node_text(node, source)

        return ""

    def extract_metadata(
        self, concept: UniversalConcept, captures: dict[str, TSNode], content: bytes
    ) -> dict[str, Any]:
        """Extract Vue template-specific metadata from captures.

        Args:
            concept: The universal concept being extracted
            captures: Dictionary of capture names to tree-sitter nodes
            content: Source code as bytes

        Returns:
            Dictionary of metadata
        """
        source = content.decode("utf-8")
        metadata: dict[str, Any] = {
            "vue_section": "template",
            "is_vue_sfc": True,
        }

        if concept == UniversalConcept.DEFINITION:
            # Handle directive_attribute nodes
            if "definition" in captures:
                directive_attr_node = captures["definition"]
                directive_attr_text = self.get_node_text(directive_attr_node, source).strip()

                parsed = self._parse_directive_attribute(directive_attr_text)
                if parsed:
                    directive = parsed["directive"]
                    argument = parsed["argument"]
                    value = parsed["value"]

                    # Set metadata based on directive type
                    if directive in ["v-if", "v-else-if"]:
                        metadata["directive_type"] = directive
                        metadata["condition"] = value
                    elif directive == "v-for":
                        metadata["directive_type"] = directive
                        metadata["loop_expression"] = value
                        # Try to parse "item in items" pattern
                        if " in " in value:
                            loop_parts = value.split(" in ", 1)
                            if len(loop_parts) == 2:
                                metadata["loop_variable"] = loop_parts[0].strip()
                                metadata["loop_iterable"] = loop_parts[1].strip()
                    elif directive == "v-model":
                        metadata["directive_type"] = directive
                        metadata["model_binding"] = value
                    elif directive == "@":
                        metadata["directive_type"] = "event_handler"
                        metadata["event_name"] = argument
                        metadata["handler_expression"] = value
                    elif directive == "v-on":
                        metadata["directive_type"] = "event_handler"
                        metadata["event_name"] = argument
                        metadata["handler_expression"] = value
                    elif directive == ":":
                        metadata["directive_type"] = "property_binding"
                        metadata["property_name"] = argument
                        metadata["binding_expression"] = value
                    elif directive == "v-bind":
                        metadata["directive_type"] = "property_binding"
                        metadata["property_name"] = argument
                        metadata["binding_expression"] = value
                    elif directive == "v-slot":
                        metadata["directive_type"] = "slot"
                        metadata["slot_name"] = argument
                    elif directive == "v-else":
                        metadata["directive_type"] = "v-else"

            # Handle interpolations
            if "interpolation_expr" in captures:
                metadata["directive_type"] = "interpolation"
                expr_node = captures["interpolation_expr"]
                metadata["interpolation_expression"] = self.get_node_text(
                    expr_node, source
                ).strip()

            # Handle components
            if "component_name" in captures:
                metadata["directive_type"] = "component_usage"
                component_node = captures["component_name"]
                metadata["component_name"] = self.get_node_text(
                    component_node, source
                ).strip()

        elif concept == UniversalConcept.BLOCK:
            if "directive_name" in captures:
                directive_node = captures["directive_name"]
                directive = self.get_node_text(directive_node, source).strip()
                metadata["block_type"] = directive

        elif concept == UniversalConcept.COMMENT:
            metadata["comment_type"] = "template_comment"

        return metadata

    def resolve_import_paths(
        self,
        import_text: str,
        base_dir: Path,
        source_file: Path,
    ) -> list[Path]:
        """Vue templates don't have imports.

        Imports in Vue SFCs are in the <script> section,
        handled by VueMapping (vue.py), not VueTemplateMapping.

        Args:
            import_text: The raw import statement text
            base_dir: Project root directory
            source_file: File containing the import

        Returns:
            Empty list - templates don't have imports
        """
        return []
