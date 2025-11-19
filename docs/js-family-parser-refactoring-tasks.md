# JS-Family Parser Refactoring Tasks

This document defines the tasks for consolidating JS-family parsers with dependency information and parallelization strategy.

---

## Overview

**Goal:** Refactor so TypeScript extends JavaScript, with shared JSXExtras mixin for React patterns.

**Key Principle:** TypeScript is a superset of JavaScript, so TypeScriptMapping should extend JavaScriptMapping.

---

## Task Dependency Graph

```
Phase 1 (Parallel)
├── Task 1: Create JSXExtras mixin
└── Task 2: Update shared query patterns
         │
         ▼
Phase 2 (Sequential - Critical)
└── Task 3: Refactor TypeScriptMapping to extend JavaScriptMapping
         │
         ▼
Phase 3 (Parallel)
├── Task 4: Update JSXMapping with JSXExtras
├── Task 5: Update TSXMapping with JSXExtras
├── Task 6: Add IMPORT concept to JavaScriptMapping
└── Task 7: Clean up duplicate imports
         │
         ▼
Phase 4 (Sequential)
└── Task 8: Run tests and fix issues
```

---

## Phase 1: Foundation (PARALLEL)

### Task 1: Create JSXExtras Mixin

**File to create:** `chunkhound/parsers/mappings/_shared/jsx_extras.py`

**Extract from JSXMapping:**
- `get_jsx_element_query()`
- `get_jsx_expression_query()`
- `get_hook_query()`
- `extract_component_name()`
- `extract_jsx_element_name()`
- `extract_hook_name()`
- `is_react_component()`
- `extract_jsx_props()`
- `clean_jsx_text()`
- `should_include_node()` (JSX-specific parts)

**Extract from TSXMapping:**
- `extract_component_props_type()`
- `extract_hook_types()`
- `get_props_interface_query()`

**Complexity:** Medium
**Risk:** Low

---

### Task 2: Update Shared Query Patterns

**File to modify:** `chunkhound/parsers/mappings/_shared/js_query_patterns.py`

**Changes:**
1. Add IMPORT_STATEMENT pattern:
```python
IMPORT_STATEMENT = """
(import_statement) @definition
"""
```

2. Verify TOP_LEVEL_VAR_CONFIG exists

3. Add helper function:
```python
def class_declaration_query(name_type: str = "identifier") -> str:
    return f'''
        (class_declaration
            name: ({name_type}) @name
        ) @definition
    '''
```

**Complexity:** Low
**Risk:** Low

---

## Phase 2: Core Inheritance (SEQUENTIAL - CRITICAL)

### Task 3: Refactor TypeScriptMapping to Extend JavaScriptMapping

**File to modify:** `chunkhound/parsers/mappings/typescript.py`

**Changes:**

1. **Change inheritance:**
```python
# Before
class TypeScriptMapping(BaseMapping, JSFamilyExtraction):

# After
class TypeScriptMapping(JavaScriptMapping):
```

2. **Update imports:**
- Remove JSFamilyExtraction import
- Add JavaScriptMapping import

3. **Override `get_query_for_concept()`:**
```python
def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
    if concept == UniversalConcept.DEFINITION:
        # Get base JS patterns
        base_query = super().get_query_for_concept(concept)

        if base_query is None:
            return None

        # CRITICAL: Replace identifier with type_identifier for class names
        base_query = base_query.replace(
            "name: (identifier)",
            "name: (type_identifier)"
        )

        # Add TypeScript-specific patterns
        ts_specific = """
            (interface_declaration
                name: (type_identifier) @name
            ) @definition

            (enum_declaration
                name: (identifier) @name
            ) @definition

            (type_alias_declaration
                name: (type_identifier) @name
            ) @definition

            (internal_module
                name: (identifier) @name
            ) @definition

            (ambient_declaration
                (module name: (_) @name)
            ) @definition
        """

        return base_query + ts_specific

    # Inherit IMPORT and COMMENT from JavaScript
    return super().get_query_for_concept(concept)
```

4. **Override `extract_class_name()` for type_identifier:**
```python
def extract_class_name(self, node, source) -> str:
    if node is None:
        return self.get_fallback_name(node, "class")

    # TypeScript uses type_identifier for class names
    name_node = self.find_child_by_type(node, "type_identifier")
    if name_node:
        return self.get_node_text(name_node, source)

    return self.get_fallback_name(node, "class")
```

5. **Remove duplicated methods** that are now inherited:
- Basic extraction methods from JSFamilyExtraction
- Any methods identical to JavaScriptMapping

**Complexity:** High
**Risk:** Medium (core change affecting all TS/TSX parsing)

---

## Phase 3: Apply Changes (PARALLEL)

### Task 4: Update JSXMapping with JSXExtras

**File to modify:** `chunkhound/parsers/mappings/jsx.py`

**Changes:**

1. **Add mixin:**
```python
from chunkhound.parsers.mappings._shared.jsx_extras import JSXExtras

class JSXMapping(JavaScriptMapping, JSXExtras):
```

2. **Remove methods** now in JSXExtras

3. **Ensure class query uses type_identifier** (already done for tsx grammar)

**Complexity:** Low
**Risk:** Low

---

### Task 5: Update TSXMapping with JSXExtras

**File to modify:** `chunkhound/parsers/mappings/tsx.py`

**Changes:**

1. **Add mixin:**
```python
from chunkhound.parsers.mappings._shared.jsx_extras import JSXExtras

class TSXMapping(TypeScriptMapping, JSXExtras):
```

2. **Remove methods** now in JSXExtras

**Complexity:** Low
**Risk:** Low

---

### Task 6: Add IMPORT Concept to JavaScriptMapping

**File to modify:** `chunkhound/parsers/mappings/javascript.py`

**Changes:**

Add IMPORT handler to `get_query_for_concept()`:
```python
def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
    if concept == UniversalConcept.IMPORT:
        return """
        (import_statement) @definition
        """
    elif concept == UniversalConcept.DEFINITION:
        # ... existing code
```

**Complexity:** Low
**Risk:** Low

---

### Task 7: Clean Up Duplicate Imports

**Files to modify:**
- `chunkhound/parsers/mappings/javascript.py` (lines 13-32)
- `chunkhound/parsers/mappings/jsx.py` (lines 13-26)

**Changes:** Remove duplicate import blocks

**Complexity:** Low
**Risk:** Low

---

## Phase 4: Validation (SEQUENTIAL)

### Task 8: Run Tests and Fix Issues

**Commands:**
```bash
# Run smoke tests first
uv run pytest tests/test_smoke.py -v

# Run all comprehensive tests
uv run pytest tests/test_*_comprehensive.py -v

# Run cross-language consistency tests
uv run pytest tests/test_cross_language_consistency.py -v

# Run TypeScript tests
uv run pytest tests/test_typescript_*.py -v

# Run JSX/React tests
uv run pytest tests/test_jsx_react.py -v
```

**Expected outcomes:**
- All smoke tests pass
- Previously xfailed tests for JS/JSX IMPORT should now pass
- Previously xfailed tests for TS var patterns should now pass
- Cross-language consistency tests should pass
- No new failures

**Fix any issues** that arise from the refactoring.

**Complexity:** Medium-High
**Risk:** Medium

---

## Success Criteria

1. ✅ TypeScriptMapping extends JavaScriptMapping
2. ✅ JSXExtras mixin created and used by JSX and TSX
3. ✅ IMPORT concept works for JS, JSX, TS, TSX, Vue
4. ✅ var patterns work for TS, TSX
5. ✅ No duplicate imports in mapping files
6. ✅ All smoke tests pass
7. ✅ Cross-language consistency tests pass
8. ✅ Previously xfailed import tests now pass
9. ✅ No regressions in existing functionality

---

## Files Modified/Created

### Created
- `chunkhound/parsers/mappings/_shared/jsx_extras.py`

### Modified
- `chunkhound/parsers/mappings/_shared/js_query_patterns.py`
- `chunkhound/parsers/mappings/typescript.py` (major)
- `chunkhound/parsers/mappings/javascript.py`
- `chunkhound/parsers/mappings/jsx.py`
- `chunkhound/parsers/mappings/tsx.py`

---

## Reference Documents

- **Architecture:** `docs/js-family-parser-consolidation-proposal.md`
- **Gap Analysis:** `docs/js-family-parsing-gap-analysis.md`
- **Test Specification:** `docs/js-family-parser-test-specification.md`
- **Test Coverage:** `docs/js-family-test-verification-report.md`

---

## Version

- **Created:** 2025-11-19
- **Status:** Ready for Implementation
