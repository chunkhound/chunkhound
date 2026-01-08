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
from typing import TYPE_CHECKING, Any

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.base import BaseMapping
from chunkhound.parsers.universal_engine import UniversalConcept

if TYPE_CHECKING:
    from tree_sitter import Node as TSNode

try:
    from tree_sitter import Node as TSNode

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    TSNode = Any  # type: ignore


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

    def extract_function_name(self, node: "TSNode | None", source: str) -> str:
        """Not applicable for Vue templates.

        Args:
            node: Tree-sitter node
            source: Source code string

        Returns:
            Empty string
        """
        return ""

    def extract_class_name(self, node: "TSNode | None", source: str) -> str:
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
            ; Conditional rendering (v-if, v-else-if)
            (directive_attribute
              (directive_name) @directive_name
              (#match? @directive_name "^v-if$|^v-else-if$")
              (quoted_attribute_value
                (attribute_value) @condition_expr
              )?
            ) @definition

            ; List rendering (v-for)
            (directive_attribute
              (directive_name) @directive_name
              (#eq? @directive_name "v-for")
              (quoted_attribute_value
                (attribute_value) @loop_expr
              )?
            ) @definition

            ; Event handlers (@click, @submit, v-on:click, etc.)
            (directive_attribute
              (directive_name) @event_prefix
              (#match? @event_prefix "^@|^v-on$")
              (directive_argument)? @event_name
              (quoted_attribute_value
                (attribute_value) @handler_expr
              )?
            ) @definition

            ; Property bindings (:prop, v-bind:prop)
            (directive_attribute
              (directive_name) @bind_prefix
              (#match? @bind_prefix "^:|^v-bind$")
              (directive_argument)? @prop_name
              (quoted_attribute_value
                (attribute_value) @bind_expr
              )?
            ) @definition

            ; Two-way binding (v-model)
            (directive_attribute
              (directive_name) @directive_name
              (#eq? @directive_name "v-model")
              (directive_argument)? @model_arg
              (quoted_attribute_value
                (attribute_value) @model_expr
              )?
            ) @definition

            ; Interpolations {{ variable }}
            (interpolation
              (raw_text)? @interpolation_expr
            ) @definition

            ; Slot usage
            (directive_attribute
              (directive_name) @directive_name
              (#match? @directive_name "^v-slot$|^#")
              (directive_argument)? @slot_name
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
        self, concept: UniversalConcept, captures: dict[str, "TSNode"], content: bytes
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
            # Check for directive types
            if "directive_name" in captures:
                directive_node = captures["directive_name"]
                directive = self.get_node_text(directive_node, source).strip()

                # Handle different directive types
                if directive in ("v-if", "v-else-if"):
                    if "condition_expr" in captures:
                        expr_text = self.get_node_text(captures["condition_expr"], source).strip()
                        expr = self.get_expression_preview(expr_text, max_length=20)
                    else:
                        expr = "expr"
                    return f"v-if_{expr}"
                elif directive == "v-for":
                    if "loop_expr" in captures:
                        expr_text = self.get_node_text(captures["loop_expr"], source).strip()
                        expr = self.get_expression_preview(expr_text, max_length=20)
                    else:
                        expr = "expr"
                    return f"v-for_{expr}"
                elif directive == "v-model":
                    if "model_expr" in captures:
                        expr_text = self.get_node_text(captures["model_expr"], source).strip()
                        expr = self.get_expression_preview(expr_text, max_length=20)
                    else:
                        expr = "expr"
                    return f"v-model_{expr}"

            # Handle event handlers
            if "event_prefix" in captures:
                if "event_name" in captures:
                    event_node = captures["event_name"]
                    event = self.get_node_text(event_node, source).strip()
                    return f"@{event}"
                return "@event"

            # Handle property bindings
            if "bind_prefix" in captures:
                if "prop_name" in captures:
                    prop_node = captures["prop_name"]
                    prop = self.get_node_text(prop_node, source).strip()
                    return f":{prop}"
                return ":prop"

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
        self, concept: UniversalConcept, captures: dict[str, "TSNode"], content: bytes
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
        self, concept: UniversalConcept, captures: dict[str, "TSNode"], content: bytes
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
            # Extract directive-specific metadata
            if "directive_name" in captures:
                directive_node = captures["directive_name"]
                directive = self.get_node_text(directive_node, source).strip()
                metadata["directive_type"] = directive

                # Extract directive arguments and values
                if "condition_expr" in captures:
                    expr_node = captures["condition_expr"]
                    metadata["condition"] = self.get_node_text(
                        expr_node, source
                    ).strip()

                elif "loop_expr" in captures:
                    expr_node = captures["loop_expr"]
                    loop_expr = self.get_node_text(expr_node, source).strip()
                    metadata["loop_expression"] = loop_expr
                    # Try to parse "item in items" pattern
                    if " in " in loop_expr:
                        parts = loop_expr.split(" in ", 1)
                        if len(parts) == 2:
                            metadata["loop_variable"] = parts[0].strip()
                            metadata["loop_iterable"] = parts[1].strip()

                elif "model_expr" in captures:
                    expr_node = captures["model_expr"]
                    metadata["model_binding"] = self.get_node_text(
                        expr_node, source
                    ).strip()

                if "model_arg" in captures:
                    arg_node = captures["model_arg"]
                    metadata["model_modifier"] = self.get_node_text(
                        arg_node, source
                    ).strip()

            # Handle event handlers
            if "event_prefix" in captures:
                metadata["directive_type"] = "event_handler"
                if "event_name" in captures:
                    event_node = captures["event_name"]
                    metadata["event_name"] = self.get_node_text(
                        event_node, source
                    ).strip()
                if "handler_expr" in captures:
                    handler_node = captures["handler_expr"]
                    metadata["handler_expression"] = self.get_node_text(
                        handler_node, source
                    ).strip()

            # Handle property bindings
            if "bind_prefix" in captures:
                metadata["directive_type"] = "property_binding"
                if "prop_name" in captures:
                    prop_node = captures["prop_name"]
                    metadata["property_name"] = self.get_node_text(
                        prop_node, source
                    ).strip()
                if "bind_expr" in captures:
                    expr_node = captures["bind_expr"]
                    metadata["binding_expression"] = self.get_node_text(
                        expr_node, source
                    ).strip()

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

            # Handle slots
            if "slot_name" in captures:
                metadata["directive_type"] = "slot"
                slot_node = captures["slot_name"]
                metadata["slot_name"] = self.get_node_text(slot_node, source).strip()

        elif concept == UniversalConcept.BLOCK:
            if "directive_name" in captures:
                directive_node = captures["directive_name"]
                directive = self.get_node_text(directive_node, source).strip()
                metadata["block_type"] = directive

        elif concept == UniversalConcept.COMMENT:
            metadata["comment_type"] = "template_comment"

        return metadata

    def resolve_import_path(
        self,
        import_text: str,
        base_dir: Path,
        source_file: Path,
    ) -> Path | None:
        """Vue templates don't have imports.

        Imports in Vue SFCs are in the <script> section,
        handled by VueMapping (vue.py), not VueTemplateMapping.

        Args:
            import_text: The raw import statement text
            base_dir: Project root directory
            source_file: File containing the import

        Returns:
            None - templates don't have imports
        """
        return None
