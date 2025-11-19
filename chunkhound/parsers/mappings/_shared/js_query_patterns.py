"""Shared tree-sitter query fragments for JS-family mappings (JS/TS/JSX)."""

# Import statement pattern
IMPORT_STATEMENT = """
(import_statement) @definition
"""

# Top-level const/let with object/array initializer
TOP_LEVEL_LEXICAL_CONFIG = """
; Top-level const/let with object/array initializer
(program
    (lexical_declaration
        (variable_declarator
            name: (identifier) @name
            value: [(object) (array)] @init
        )
    ) @definition
)
"""

# Top-level var with object/array initializer (JS/JSX only)
TOP_LEVEL_VAR_CONFIG = """
; Top-level var with object/array initializer
(program
    (variable_declaration
        (variable_declarator
            name: (identifier) @name
            value: [(object) (array)] @init
        )
    ) @definition
)
"""

# Top-level const/let with call_expression initializer
# Captures patterns like: const props = defineProps(), const logger = createLogger()
TOP_LEVEL_LEXICAL_CALL = """
; Top-level const/let with call_expression initializer
(program
    (lexical_declaration
        (variable_declarator
            name: (identifier) @name
            value: (call_expression)
        )
    ) @definition
)
"""

# Top-level var with call_expression initializer
TOP_LEVEL_VAR_CALL = """
; Top-level var with call_expression initializer
(program
    (variable_declaration
        (variable_declarator
            name: (identifier) @name
            value: (call_expression)
        )
    ) @definition
)
"""

# Top-level const/let with primitive initializer (string, number, boolean, etc.)
# Captures patterns like: const message = 'Hello', const count = 42
TOP_LEVEL_LEXICAL_PRIMITIVE = """
; Top-level const/let with string initializer
(program
    (lexical_declaration
        (variable_declarator
            name: (identifier) @name
            value: (string)
        )
    ) @definition
)

; Top-level const/let with number initializer
(program
    (lexical_declaration
        (variable_declarator
            name: (identifier) @name
            value: (number)
        )
    ) @definition
)

; Top-level const/let with template_string initializer
(program
    (lexical_declaration
        (variable_declarator
            name: (identifier) @name
            value: (template_string)
        )
    ) @definition
)

; Top-level const/let with boolean initializer
(program
    (lexical_declaration
        (variable_declarator
            name: (identifier) @name
            value: [(true) (false)]
        )
    ) @definition
)

; Top-level const/let with null/undefined
(program
    (lexical_declaration
        (variable_declarator
            name: (identifier) @name
            value: [(null) (undefined)]
        )
    ) @definition
)
"""

# CommonJS patterns
COMMONJS_MODULE_EXPORTS = """
; CommonJS assignment: module.exports = ...
(program
    (expression_statement
        (assignment_expression
            left: (member_expression
                object: (identifier) @lhs_module
                property: (property_identifier) @lhs_exports
            )
            right: [(object) (array)] @init
        ) @definition
        (#eq? @lhs_module "module")
        (#eq? @lhs_exports "exports")
    )
)
"""

COMMONJS_NESTED_EXPORTS = """
; CommonJS nested assignment: module.exports.something = ...
(program
    (expression_statement
        (assignment_expression
            left: (member_expression
                object: (member_expression
                    object: (identifier) @lhs_module_n
                    property: (property_identifier) @lhs_exports_n
                )
            )
            right: [(object) (array)] @init
        ) @definition
        (#eq? @lhs_module_n "module")
        (#eq? @lhs_exports_n "exports")
    )
)
"""

COMMONJS_EXPORTS_SHORTHAND = """
; CommonJS assignment: exports.something = ...
(program
    (expression_statement
        (assignment_expression
            left: (member_expression
                object: (identifier) @lhs_exports
            )
            right: [(object) (array)] @init
        ) @definition
        (#eq? @lhs_exports "exports")
    )
)
"""


def class_declaration_query(name_type: str = "identifier") -> str:
    """Generate class declaration query with appropriate name node type.

    Args:
        name_type: The node type for class names. Use "identifier" for JavaScript
                   grammar, "type_identifier" for TypeScript/TSX grammars.

    Returns:
        A tree-sitter query string for class declarations.
    """
    return f'''
        (class_declaration
            name: ({name_type}) @name
        ) @definition
    '''

