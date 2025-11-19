# JS-Family Parsing Gap Analysis

This document analyzes the parsing feature parity across TypeScript, TSX, JavaScript, and JSX mappings to ensure all valid syntax is captured consistently.

## Executive Summary

**Current State:** The TypeScript parsing fix in commit `a7b1fe70` does not apply consistently across all JS-family languages. Several critical features are missing.

**Goal:** All JS-family languages should parse ALL valid syntax for their respective grammars. TypeScript/TSX should be supersets of JavaScript/JSX.

---

## Feature Matrix

### Universal Concept: IMPORT

| Language | Status | Notes |
|----------|--------|-------|
| TypeScript | :white_check_mark: | `(import_statement) @definition` |
| TSX | :white_check_mark: | Inherits from TypeScript |
| JavaScript | :x: **MISSING** | No IMPORT handler in `get_query_for_concept()` |
| JSX | :x: **MISSING** | No IMPORT handler in `get_query_for_concept()` |

**Impact:** Import statements in `.js` and `.jsx` files will NOT be extracted as chunks.

---

### Universal Concept: DEFINITION

#### Standard Definitions

| Pattern | TypeScript | TSX | JavaScript | JSX |
|---------|------------|-----|------------|-----|
| `function_declaration` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| `class_declaration` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| `export_statement` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

#### Top-Level Variable Configurations (object/array initializers)

| Pattern | TypeScript | TSX | JavaScript | JSX |
|---------|------------|-----|------------|-----|
| `TOP_LEVEL_LEXICAL_CONFIG` (const/let) | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| `TOP_LEVEL_VAR_CONFIG` (var) | :x: **MISSING** | :x: **MISSING** | :white_check_mark: | :white_check_mark: |

**Impact:** TypeScript/TSX won't capture top-level `var config = {...}` patterns.

**Example not captured in TS/TSX:**
```typescript
var config = {
  apiUrl: "https://api.example.com",
  timeout: 5000
};
```

#### Function/Arrow in Lexical Declarations (const/let)

| Pattern | TypeScript | TSX | JavaScript | JSX |
|---------|------------|-----|------------|-----|
| `const/let name = function() {}` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| `const/let name = () => {}` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

#### Function/Arrow in Variable Declarations (var)

| Pattern | TypeScript | TSX | JavaScript | JSX |
|---------|------------|-----|------------|-----|
| `var name = function() {}` | :x: **MISSING** | :x: **MISSING** | :white_check_mark: | :x: **MISSING** |
| `var name = () => {}` | :x: **MISSING** | :x: **MISSING** | :white_check_mark: | :x: **MISSING** |

**Impact:** TypeScript/TSX/JSX won't capture functions declared with `var`.

**Example not captured in TS/TSX/JSX:**
```typescript
var handleClick = function(event) {
  console.log(event);
};

var processData = (data) => {
  return data.map(transform);
};
```

#### CommonJS Patterns

| Pattern | TypeScript | TSX | JavaScript | JSX |
|---------|------------|-----|------------|-----|
| `module.exports = {...}` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| `module.exports.x = {...}` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| `exports.x = {...}` | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

#### TypeScript-Specific Constructs

| Pattern | TypeScript | TSX | JavaScript | JSX |
|---------|------------|-----|------------|-----|
| `interface_declaration` | :white_check_mark: | :white_check_mark: | N/A | N/A |
| `enum_declaration` | :white_check_mark: | :white_check_mark: | N/A | N/A |
| `type_alias_declaration` | :white_check_mark: | :white_check_mark: | N/A | N/A |
| `internal_module` (namespace) | :white_check_mark: | :white_check_mark: | N/A | N/A |
| `ambient_declaration` | :white_check_mark: | :white_check_mark: | N/A | N/A |

---

### Universal Concept: COMMENT

| Language | Status |
|----------|--------|
| TypeScript | :white_check_mark: |
| TSX | :white_check_mark: |
| JavaScript | :white_check_mark: |
| JSX | :white_check_mark: |

---

## Code Quality Issues

### Duplicate Imports

**JavaScript (`javascript.py` lines 13-32):**
```python
from chunkhound.parsers.mappings._shared.js_family_extraction import (
    JSFamilyExtraction,
)
from chunkhound.parsers.mappings._shared.js_query_patterns import (
    TOP_LEVEL_LEXICAL_CONFIG,
    TOP_LEVEL_VAR_CONFIG,
    ...
)
# DUPLICATED AGAIN:
from chunkhound.parsers.mappings._shared.js_family_extraction import (
    JSFamilyExtraction,
)
from chunkhound.parsers.mappings._shared.js_query_patterns import (
    ...
)
```

**JSX (`jsx.py` lines 13-26):**
Same issue - imports are duplicated.

---

## Required Fixes

### Critical (Must Fix)

1. **Add IMPORT concept to JavaScript**
   - File: `chunkhound/parsers/mappings/javascript.py`
   - Location: `get_query_for_concept()` method
   - Add handler for `UniversalConcept.IMPORT`

2. **Add IMPORT concept to JSX**
   - File: `chunkhound/parsers/mappings/jsx.py`
   - Location: `get_query_for_concept()` method
   - Add handler for `UniversalConcept.IMPORT`

3. **Add TOP_LEVEL_VAR_CONFIG to TypeScript**
   - File: `chunkhound/parsers/mappings/typescript.py`
   - Location: `get_query_for_concept()` DEFINITION handler
   - Add `TOP_LEVEL_VAR_CONFIG` pattern

4. **Add var function/arrow patterns to TypeScript**
   - File: `chunkhound/parsers/mappings/typescript.py`
   - Add patterns for `var name = function(){}` and `var name = () => {}`

5. **Add var function/arrow patterns to JSX**
   - File: `chunkhound/parsers/mappings/jsx.py`
   - Add patterns for `var name = function(){}` and `var name = () => {}`

### Important (Should Fix)

6. **Clean up duplicate imports in JavaScript**
   - File: `chunkhound/parsers/mappings/javascript.py`
   - Remove duplicate import blocks (lines 23-32)

7. **Clean up duplicate imports in JSX**
   - File: `chunkhound/parsers/mappings/jsx.py`
   - Remove duplicate import blocks (lines 20-26)

8. **Add tests for TSX import extraction**
   - Verify TSX inherits IMPORT handling correctly from TypeScript

9. **Add tests for Vue SFC import extraction**
   - Verify Vue script sections parse imports correctly

### Minor (Nice to Have)

10. **Remove test files from repository root**
    - `test_parsing_comparison.py`
    - `test_py_parsing.py`
    - `test_ts_parsing.ts`

---

## Implementation Details

### Fix 1 & 2: Add IMPORT to JavaScript and JSX

```python
def get_query_for_concept(self, concept: "UniversalConcept") -> str | None:
    if concept == UniversalConcept.IMPORT:
        return """
        (import_statement) @definition
        """
    elif concept == UniversalConcept.DEFINITION:
        # ... existing code
```

### Fix 3 & 4: Add var patterns to TypeScript

In the DEFINITION handler, add:
```python
TOP_LEVEL_VAR_CONFIG,
# var function/arrow declarators
"""
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
```

### Fix 5: Add var patterns to JSX

Same as TypeScript - add the var function/arrow patterns to the DEFINITION handler.

---

## Rationale

### Why TypeScript needs `var` patterns

1. **Valid TypeScript syntax** - `var` is valid in TypeScript for backward compatibility
2. **Legacy code** - Many projects have legacy TypeScript using `var`
3. **Migration support** - Projects migrating from JS to TS may still have `var`
4. **Consistency** - Parser should capture ALL valid syntax, not just "recommended" syntax

### Why all languages need IMPORT

1. **Core language feature** - Imports are fundamental to module systems
2. **Searchability** - Users need to find where modules are imported
3. **Dependency analysis** - Understanding code dependencies requires import information
4. **Consistency** - All JS-family languages support ES6 imports

---

## Testing Checklist

After implementing fixes, verify:

- [ ] TypeScript parses `var config = {...}`
- [ ] TypeScript parses `var fn = function() {}`
- [ ] TypeScript parses `var fn = () => {}`
- [ ] TypeScript parses `import ... from '...'`
- [ ] TSX inherits all TypeScript patterns
- [ ] JavaScript parses `import ... from '...'`
- [ ] JSX parses `import ... from '...'`
- [ ] JSX parses `var fn = function() {}`
- [ ] JSX parses `var fn = () => {}`
- [ ] Vue SFC script sections parse imports
- [ ] No duplicate chunks created
- [ ] Existing tests still pass

---

## Files to Modify

| File | Changes |
|------|---------|
| `chunkhound/parsers/mappings/typescript.py` | Add TOP_LEVEL_VAR_CONFIG, var function/arrow patterns |
| `chunkhound/parsers/mappings/javascript.py` | Add IMPORT concept, remove duplicate imports |
| `chunkhound/parsers/mappings/jsx.py` | Add IMPORT concept, var function/arrow patterns, remove duplicate imports |
| `tests/test_parsers.py` | Add tests for new patterns |

---

## Version

- **Created:** 2025-11-19
- **Commit analyzed:** `a7b1fe70c842ff3e0981b47d0748bd9263acc4fe`
- **Status:** Pending implementation
