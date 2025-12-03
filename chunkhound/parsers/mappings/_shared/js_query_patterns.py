"""Shared tree-sitter query fragments for JS-family mappings (JS/TS/JSX)."""

# Top-level const/let with object/array initializer
TOP_LEVEL_LEXICAL_CONFIG = """
; Top-level const/let with object/array initializer
(program
    (lexical_declaration
        (variable_declarator
            name: (identifier) @name
            value: [(object) (array)] @init
        ) @definition
    )
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
        ) @definition
    )
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

