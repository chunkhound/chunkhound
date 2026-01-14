"""Shared tree-sitter query fragments for JS-family mappings (JS/TS/JSX)."""

# const/let with any value type (any scope)
LEXICAL_DECLARATION_CONFIG = """
; const/let with any value type (any scope)
(lexical_declaration
    (variable_declarator
        name: (identifier) @name
        value: [(object) (array) (number) (string) (true) (false) (null) (undefined)] @init
    )
) @definition
"""

# var with any value type (JS/JSX only, any scope)
VAR_DECLARATION_CONFIG = """
; var with any value type (any scope)
(variable_declaration
    (variable_declarator
        name: (identifier) @name
        value: [(object) (array) (number) (string) (true) (false) (null) (undefined)] @init
    )
) @definition
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

