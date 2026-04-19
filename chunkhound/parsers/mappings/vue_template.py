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
        """Get query for component usage.

        Returns:
            Tree-sitter query for component usage
        """
        return """
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

                # Check if it's a directive attribute (contains = sign)
                if "=" in directive_attr_text:
                    # Parse the directive attribute
                    parts = directive_attr_text.split("=", 1)
                    if len(parts) == 2:
                        directive_part = parts[0].strip()
                        value_part = parts[1].strip().strip('"').strip("'")

                        # Check directive type based on prefix
                        if directive_part.startswith("v-if") or directive_part.startswith("v-else-if"):
                            expr = self.get_expression_preview(value_part, max_length=20)
                            return f"v-if_{expr}"
                        elif directive_part.startswith("v-for"):
                            expr = self.get_expression_preview(value_part, max_length=20)
                            return f"v-for_{expr}"
                        elif directive_part.startswith("v-model"):
                            expr = self.get_expression_preview(value_part, max_length=20)
                            return f"v-model_{expr}"
                        elif directive_part.startswith("@"):
                            # Event handler like @click="handler"
                            event_name = directive_part[1:]  # Remove @
                            return f"@{event_name}"
                        elif directive_part.startswith(":"):
                            # Property binding like :prop="value"
                            prop_name = directive_part[1:]  # Remove :
                            return f":{prop_name}"
                        elif directive_part.startswith("v-slot:") or directive_part == "#":
                            # Slot usage
                            if directive_part.startswith("v-slot:"):
                                slot_name = directive_part[7:]  # Remove v-slot:
                            else:
                                slot_name = "default"
                            return f"v-slot:{slot_name}"

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

            # Handle slots
            if "slot_name" in captures:
                slot_node = captures["slot_name"]
                slot = self.get_node_text(slot_node, source).strip()
                return f"v-slot:{slot}"

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

                # Check if it's a directive attribute (contains = sign)
                if "=" in directive_attr_text:
                    # Parse the directive attribute
                    parts = directive_attr_text.split("=", 1)
                    if len(parts) == 2:
                        directive_part = parts[0].strip()
                        value_part = parts[1].strip().strip('"').strip("'")

                        # Check directive type based on prefix
                        if directive_part.startswith("v-if") or directive_part.startswith("v-else-if"):
                            metadata["directive_type"] = directive_part
                            metadata["condition"] = value_part
                        elif directive_part.startswith("v-for"):
                            metadata["directive_type"] = directive_part
                            metadata["loop_expression"] = value_part
                            # Try to parse "item in items" pattern
                            if " in " in value_part:
                                loop_parts = value_part.split(" in ", 1)
                                if len(loop_parts) == 2:
                                    metadata["loop_variable"] = loop_parts[0].strip()
                                    metadata["loop_iterable"] = loop_parts[1].strip()
                        elif directive_part.startswith("v-model"):
                            metadata["directive_type"] = directive_part
                            metadata["model_binding"] = value_part
                        elif directive_part.startswith("@"):
                            # Event handler like @click="handler"
                            metadata["directive_type"] = "event_handler"
                            metadata["event_name"] = directive_part[1:]  # Remove @
                            metadata["handler_expression"] = value_part
                        elif directive_part.startswith(":"):
                            # Property binding like :prop="value"
                            metadata["directive_type"] = "property_binding"
                            metadata["property_name"] = directive_part[1:]  # Remove :
                            metadata["binding_expression"] = value_part
                        elif directive_part.startswith("v-slot:") or directive_part == "#":
                            # Slot usage
                            metadata["directive_type"] = "slot"
                            if directive_part.startswith("v-slot:"):
                                metadata["slot_name"] = directive_part[7:]  # Remove v-slot:
                            else:
                                metadata["slot_name"] = "default"

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
