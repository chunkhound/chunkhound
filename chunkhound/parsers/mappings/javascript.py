"""JavaScript language mapping for unified parser architecture.

This module provides JavaScript-specific tree-sitter queries and extraction logic
for semantic code analysis. It handles JavaScript-specific language features like
arrow functions, ES6 classes, JSDoc comments, and modern module syntax.
"""

from typing import TYPE_CHECKING, Any

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.base import BaseMapping
from chunkhound.parsers.universal_engine import UniversalConcept
from chunkhound.parsers.mappings._shared.js_family_extraction import (
    JSFamilyExtraction,
)
from chunkhound.parsers.mappings._shared.js_query_patterns import (
    TOP_LEVEL_LEXICAL_CONFIG,
    TOP_LEVEL_VAR_CONFIG,
    COMMONJS_MODULE_EXPORTS,
    COMMONJS_NESTED_EXPORTS,
    COMMONJS_EXPORTS_SHORTHAND,
)
from chunkhound.parsers.mappings._shared.js_family_extraction import (
    JSFamilyExtraction,
)
from chunkhound.parsers.mappings._shared.js_query_patterns import (
    TOP_LEVEL_LEXICAL_CONFIG,
    TOP_LEVEL_VAR_CONFIG,
    COMMONJS_MODULE_EXPORTS,
    COMMONJS_NESTED_EXPORTS,
    COMMONJS_EXPORTS_SHORTHAND,
)

if TYPE_CHECKING:
    from tree_sitter import Node as TSNode

try:
    from tree_sitter import Node as TSNode

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    TSNode = None


class JavaScriptMapping(BaseMapping, JSFamilyExtraction):
    """JavaScript language mapping for tree-sitter parsing.

    Provides JavaScript-specific queries and extraction methods for:
    - Functions (regular, arrow, async, generator)
    - ES6 Classes with methods and constructors
    - Module imports/exports
    - JSDoc comments
    - Object literals and methods
    """

    def __init__(self):
        """Initialize JavaScript mapping."""
        super().__init__(Language.JAVASCRIPT)

    def get_function_query(self) -> str:
        """Get tree-sitter query for JavaScript function definitions.

        Handles multiple function declaration styles:
        - function declarations: function foo() {}
        - function expressions: const foo = function() {}
        - arrow functions: const foo = () => {}
        - async functions: async function foo() {}
        - generator functions: function* foo() {}
        - methods in objects: { foo() {} }

        Returns:
            Tree-sitter query string for finding function definitions
        """
        return """
        (function_declaration
            name: (identifier) @function_name
        ) @function_def

        (function_expression
            name: (identifier) @func_expr_name
        ) @func_expr_def

        (variable_declarator
            name: (identifier) @arrow_func_name
            value: (arrow_function) @arrow_function
        ) @arrow_func_def

        (variable_declarator
            name: (identifier) @var_func_name
            value: (function_expression) @function_expression
        ) @var_func_def

        """

    def get_class_query(self) -> str:
        """Get tree-sitter query for JavaScript class definitions.

        Handles:
        - ES6 class declarations: class Foo extends Bar {}
        - Class expressions: const Foo = class {}

        Returns:
            Tree-sitter query string for finding class definitions
        """
        return """
        (class_declaration
            name: (identifier) @class_name
        ) @class_def

        (variable_declarator
            name: (identifier) @var_class_name
            value: (class) @class_expr
        ) @var_class_def
        """

    def get_method_query(self) -> str:
        """Get tree-sitter query for JavaScript method definitions.

        Handles:
        - Class methods: methodName() {}
        - Constructors: constructor() {}
        - Getters/setters: get/set prop() {}
        - Async methods: async methodName() {}
        - Generator methods: *methodName() {}

        Returns:
            Tree-sitter query string for finding method definitions
        """
        return """
        (method_definition
            (property_identifier) @method_name
        ) @method_def
        """

    def get_comment_query(self) -> str:
        """Get tree-sitter query for JavaScript comments.

        Handles:
        - Single-line comments: // comment
        - Multi-line comments: /* comment */
        - JSDoc comments: /** @param ... */

        Returns:
            Tree-sitter query string for finding comments
        """
        return """
        (comment) @comment
        """

    # Universal Concept integration -------------------------------------------------
    def get_query_for_concept(self, concept: "UniversalConcept") -> str | None:  # type: ignore[override]
        """Provide a richer DEFINITION query including top-level config patterns.

        - Keep standard function/class patterns
        - Add export statements and top-level declarations/assignments so
          object/array configs (e.g., `export default { ... }`, `module.exports = {}`)
          become chunks.
        """
        if concept == UniversalConcept.DEFINITION:
            return (
                "\n".join(
                    [
                        """
                        ; Standard definitions
                        (function_declaration
                            name: (identifier) @name
                        ) @definition

                        (class_declaration
                            name: (identifier) @name
                        ) @definition

                        ; Top-level export (default or named)
                        (export_statement) @definition
                        """,
                        TOP_LEVEL_LEXICAL_CONFIG,
                        TOP_LEVEL_VAR_CONFIG,
                        # Function/arrow declarators at top level
                        """
                        (program
                            (lexical_declaration
                                (variable_declarator
                                    name: (identifier) @name
                                    value: (function_expression)
                                ) @definition
                            )
                        )
                        (program
                            (lexical_declaration
                                (variable_declarator
                                    name: (identifier) @name
                                    value: (arrow_function)
                                ) @definition
                            )
                        )
                        (program
                            (variable_declaration
                                (variable_declarator
                                    name: (identifier) @name
                                    value: (function_expression)
                                ) @definition
                            )
                        )
                        (program
                            (variable_declaration
                                (variable_declarator
                                    name: (identifier) @name
                                    value: (arrow_function)
                                ) @definition
                            )
                        )
                        """,
                        COMMONJS_MODULE_EXPORTS,
                        COMMONJS_NESTED_EXPORTS,
                        COMMONJS_EXPORTS_SHORTHAND,
                    ]
                )
            )

        elif concept == UniversalConcept.COMMENT:
            return """
            (comment) @definition
            """

        # Use default handling for other concepts
        return None

    # Extraction trio inherited from JSFamilyExtraction to avoid duplication

    def get_docstring_query(self) -> str:
        """Get tree-sitter query for JSDoc comments.

        JSDoc comments are multi-line comments that start with /**
        and contain structured documentation tags.

        Returns:
            Tree-sitter query string for finding JSDoc comments
        """
        return """
        (comment) @jsdoc
        (#match? @jsdoc "^/\\*\\*")
        """

    def get_import_query(self) -> str:
        """Get tree-sitter query for JavaScript import/export statements.

        Handles:
        - ES6 imports: import { foo } from 'module'
        - Default imports: import foo from 'module'
        - Namespace imports: import * as foo from 'module'
        - Dynamic imports: import('module')
        - CommonJS requires: const foo = require('module')
        - Exports: export { foo }, export default foo

        Returns:
            Tree-sitter query string for finding import/export statements
        """
        return """
        (import_statement) @import

        (export_statement) @export

        (variable_declarator
            name: [(identifier) (destructuring_pattern)] @require.name
            value: (call_expression
                function: (identifier) @require_fn
                (#eq? @require_fn "require")
                arguments: (arguments (string) @module)
            )
        ) @require
        """

    def extract_function_name(self, node: "TSNode | None", source: str) -> str:
        """Extract function name from a JavaScript function definition.

        Handles various function patterns and provides meaningful fallbacks.

        Args:
            node: Tree-sitter function definition node
            source: Source code string

        Returns:
            Function name or fallback name if extraction fails
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return "unknown_function"

        # Try to find identifier node for function name
        name_node = self.find_child_by_type(node, "identifier")
        if name_node:
            return self.get_node_text(name_node, source)

        # For arrow functions assigned to variables
        if node.type == "variable_declarator":
            name_node = node.child(0)
            if name_node and name_node.type == "identifier":
                return self.get_node_text(name_node, source)

        # For method definitions
        if node.type == "method_definition":
            for i in range(node.child_count):
                child = node.child(i)
                if child and child.type in ["identifier", "property_identifier"]:
                    return self.get_node_text(child, source)

        # For object method shorthand
        if node.type == "pair":
            key_node = node.child(0)
            if key_node:
                if key_node.type in ["identifier", "property_identifier"]:
                    return self.get_node_text(key_node, source)
                elif key_node.type == "string":
                    return self.clean_string_literal(
                        self.get_node_text(key_node, source)
                    )

        return self.get_fallback_name(node, "function")

    def extract_class_name(self, node: "TSNode | None", source: str) -> str:
        """Extract class name from a JavaScript class definition.

        Args:
            node: Tree-sitter class definition node
            source: Source code string

        Returns:
            Class name or fallback name if extraction fails
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return "unknown_class"

        # Direct class declaration
        if node.type == "class_declaration":
            name_node = self.find_child_by_type(node, "identifier")
            if name_node:
                return self.get_node_text(name_node, source)

        # Class expression assigned to variable
        if node.type == "variable_declarator":
            name_node = node.child(0)
            if name_node and name_node.type == "identifier":
                return self.get_node_text(name_node, source)

        return self.get_fallback_name(node, "class")

    def extract_method_name(self, node: "TSNode | None", source: str) -> str:
        """Extract method name from a JavaScript method definition.

        Args:
            node: Tree-sitter method definition node
            source: Source code string

        Returns:
            Method name or fallback name if extraction fails
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return "unknown_method"

        # Method definition in class or object
        if node.type == "method_definition":
            for i in range(node.child_count):
                child = node.child(i)
                if child and child.type in ["identifier", "property_identifier"]:
                    return self.get_node_text(child, source)

        return self.get_fallback_name(node, "method")

    def extract_parameters(self, node: "TSNode | None", source: str) -> list[str]:
        """Extract parameter names from a JavaScript function/method.

        Handles:
        - Simple parameters: function(a, b) {}
        - Default parameters: function(a = 1, b = 2) {}
        - Rest parameters: function(...args) {}
        - Destructured parameters: function({a, b}) {}

        Args:
            node: Tree-sitter function/method definition node
            source: Source code string

        Returns:
            List of parameter names
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return []

        parameters = []

        # Find the parameters node
        params_node = self.find_child_by_type(node, "formal_parameters")
        if not params_node:
            # Try alternative parameter node types for arrow functions
            params_node = self.find_child_by_type(node, "parameters")

        if not params_node:
            return parameters

        # Extract parameters
        for i in range(params_node.child_count):
            child = params_node.child(i)
            if not child:
                continue

            if child.type == "identifier":
                param_name = self.get_node_text(child, source).strip()
                if param_name and param_name not in [",", "(", ")"]:
                    parameters.append(param_name)

            elif child.type == "assignment_pattern":
                # Default parameter: param = value
                left_child = child.child(0)
                if left_child and left_child.type == "identifier":
                    param_name = self.get_node_text(left_child, source).strip()
                    if param_name:
                        parameters.append(f"{param_name} = ...")

            elif child.type == "rest_pattern":
                # Rest parameter: ...args
                rest_child = child.child(1)  # Skip the "..." operator
                if rest_child and rest_child.type == "identifier":
                    param_name = self.get_node_text(rest_child, source).strip()
                    if param_name:
                        parameters.append(f"...{param_name}")

            elif child.type in ["object_pattern", "array_pattern"]:
                # Destructured parameter: {a, b} or [a, b]
                pattern_text = self.get_node_text(child, source).strip()
                if pattern_text:
                    parameters.append(pattern_text)

        return parameters

    def should_include_node(self, node: "TSNode | None", source: str) -> bool:
        """Determine if a JavaScript node should be included as a chunk.

        Filters out:
        - Empty functions/methods
        - Test files (when appropriate)
        - Generated code patterns

        Args:
            node: Tree-sitter node
            source: Source code string

        Returns:
            True if node should be included, False otherwise
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return False

        # Filter out very small nodes (likely incomplete)
        node_text = self.get_node_text(node, source)
        if len(node_text.strip()) < 10:
            return False

        # Filter out single-line comments that are too short
        if node.type == "comment":
            cleaned_text = self.clean_comment_text(node_text)
            if len(cleaned_text.strip()) < 5:
                return False

        return True

    def is_constructor(self, node: "TSNode | None", source: str) -> bool:
        """Check if a method node is a constructor.

        Args:
            node: Tree-sitter method definition node
            source: Source code string

        Returns:
            True if the method is a constructor
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return False

        if node.type != "method_definition":
            return False

        # Check if method name is "constructor"
        for i in range(node.child_count):
            child = node.child(i)
            if child and child.type in ["identifier", "property_identifier"]:
                method_name = self.get_node_text(child, source)
                return method_name == "constructor"

        return False

    def is_async_function(self, node: "TSNode | None", source: str) -> bool:
        """Check if a function is async.

        Args:
            node: Tree-sitter function definition node
            source: Source code string

        Returns:
            True if the function is async
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return False

        # Check for async keyword in the node text
        node_text = self.get_node_text(node, source)
        return "async " in node_text

    def is_generator_function(self, node: "TSNode | None", source: str) -> bool:
        """Check if a function is a generator.

        Args:
            node: Tree-sitter function definition node
            source: Source code string

        Returns:
            True if the function is a generator (contains *)
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return False

        # Check for generator syntax in the node text
        node_text = self.get_node_text(node, source)
        return "function*" in node_text or "*" in node_text.split("(")[0]

    def extract_jsdoc_tags(
        self, node: "TSNode | None", source: str
    ) -> dict[str, list[str]]:
        """Extract JSDoc tags from a JSDoc comment.

        Args:
            node: Tree-sitter comment node containing JSDoc
            source: Source code string

        Returns:
            Dictionary mapping tag names to their values
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return {}

        comment_text = self.get_node_text(node, source)
        if not comment_text.strip().startswith("/**"):
            return {}

        tags = {}
        lines = comment_text.split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith("* @"):
                # Extract tag and value
                tag_part = line[2:].strip()  # Remove "* "
                if " " in tag_part:
                    tag_name, tag_value = tag_part.split(" ", 1)
                    tag_name = tag_name[1:]  # Remove "@"
                    if tag_name not in tags:
                        tags[tag_name] = []
                    tags[tag_name].append(tag_value.strip())
                else:
                    tag_name = tag_part[1:]  # Remove "@"
                    if tag_name not in tags:
                        tags[tag_name] = []
                    tags[tag_name].append("")

        return tags
