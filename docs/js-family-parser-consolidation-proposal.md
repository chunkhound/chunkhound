# JS-Family Parser Consolidation Proposal

## Executive Summary

The current JS-family parser architecture has TypeScript and JavaScript as **sibling classes**, both extending BaseMapping independently. This causes code duplication and makes it harder to propagate fixes across all languages.

**Proposal:** Refactor so that **TypeScript extends JavaScript**, reflecting the actual language relationship where TypeScript is a superset of JavaScript.

---

## Current Architecture

```
BaseMapping (abstract)
├─ JavaScriptMapping + JSFamilyExtraction
│  └─ JSXMapping
├─ TypeScriptMapping + JSFamilyExtraction  ← Parallel to JavaScript!
│  ├─ TSXMapping
│  └─ VueMapping
```

### Problems with Current Architecture

1. **Code Duplication** - JS and TS both implement nearly identical patterns:
   - `function_declaration`
   - `class_declaration`
   - `export_statement`
   - `TOP_LEVEL_LEXICAL_CONFIG`
   - COMMONJS patterns
   - function/arrow declarators

2. **Fix Propagation Failure** - The commit `a7b1fe70` fixed TypeScript but left JavaScript/JSX broken because they're separate hierarchies.

3. **Inconsistent Features** - TypeScript is missing `var` patterns that JavaScript has, even though `var` is valid TypeScript.

4. **Semantic Mismatch** - TypeScript IS a superset of JavaScript linguistically, but the code doesn't reflect this relationship.

---

## Tree-Sitter Grammar Analysis

### Grammar Usage by Language

| Language | Tree-Sitter Grammar | Package | Module Function |
|----------|--------------------| --------|-----------------|
| JavaScript | `javascript` | tree-sitter-javascript | `language()` |
| TypeScript | `typescript` | tree-sitter-typescript | `language_typescript()` |
| JSX | `tsx` | tree-sitter-typescript | `language_tsx()` |
| TSX | `tsx` | tree-sitter-typescript | `language_tsx()` |
| Vue | `typescript` | tree-sitter-typescript | `language_typescript()` |

**Key Finding:** JSX already uses the TSX grammar (not JavaScript grammar)! This is defined in `parser_factory.py` lines 480-482.

### Critical: AST Node Type Differences

Different grammars produce **different AST node types** for the same constructs:

| Construct | JavaScript Grammar | TypeScript/TSX Grammar |
|-----------|-------------------|------------------------|
| **Class names** | `identifier` | `type_identifier` |
| Function names | `identifier` | `identifier` |
| Variable names | `identifier` | `identifier` |
| Method names | `property_identifier` | `property_identifier` |

**This is the critical difference that prevents using TSX grammar for all files.**

The `type_identifier` vs `identifier` difference for class names means:
- JavaScript grammar: `(class_declaration name: (identifier) @name)`
- TypeScript grammar: `(class_declaration name: (type_identifier) @name)`

JSXMapping already handles this by overriding `get_class_query()` to use `type_identifier` (lines 110-119).

### Why Not Use TSX Grammar for Everything?

We investigated using TSX grammar for all JS-family files (Option 3). **This is not viable** because:

1. All class-related queries in JavaScriptMapping would break
2. `extract_class_name()` would fail to find class names
3. Would require significant query rewrites with no clear benefit

The current grammar separation is intentional and correct.

---

## Proposed Architecture: Inheritance Consolidation

```
BaseMapping (abstract)
│
└─ JavaScriptMapping + JSFamilyExtraction
   │  • All common JS patterns (uses `identifier` for class names)
   │  • IMPORT concept
   │  • TOP_LEVEL_VAR_CONFIG
   │  • var function/arrow patterns
   │  • Uses javascript grammar
   │
   ├─ JSXMapping + JSXExtras (mixin)
   │   │  • React/JSX patterns
   │   │  • Overrides class queries to use `type_identifier`
   │   │  • Uses tsx grammar
   │
   └─ TypeScriptMapping
       │  • Calls super() + adds TS constructs
       │  • interface, enum, type_alias, namespace
       │  • Overrides class queries to use `type_identifier`
       │  • Uses typescript grammar
       │
       ├─ TSXMapping + JSXExtras (mixin)
       │   │  • Inherits all TS + adds React patterns
       │   │  • Uses tsx grammar
       │
       └─ VueMapping
           │  • SFC section handling
           │  • Uses typescript grammar for scripts
```

### Benefits

1. **Single Source of Truth** - Common patterns defined once in JavaScriptMapping
2. **Automatic Fix Propagation** - Fix in JS automatically available to TS/TSX
3. **Reflects Language Relationship** - TS extends JS just like the actual languages
4. **Reduced Code** - Estimated ~200-300 lines less code
5. **Easier Maintenance** - One place to update common patterns
6. **Proper Grammar Handling** - Each language uses appropriate grammar with correct overrides

### New Shared Components

**`_shared/jsx_extras.py`** (new mixin):
```python
class JSXExtras:
    """Mixin providing React/JSX-specific queries and extraction."""

    def get_jsx_element_query(self) -> str: ...
    def get_jsx_expression_query(self) -> str: ...
    def get_hook_query(self) -> str: ...
    def extract_component_name(self, node, source) -> str: ...
    def extract_jsx_element_name(self, node, source) -> str: ...
    def extract_hook_name(self, node, source) -> str: ...
    def is_react_component(self, node, source) -> bool: ...
    def extract_jsx_props(self, node, source) -> list[str]: ...
```

---

## Implementation Plan

### Phase 1: Prepare Shared Components

1. **Create `_shared/jsx_extras.py`**
   - Extract React/JSX-specific methods from JSXMapping and TSXMapping
   - Make it a proper mixin class

2. **Enhance `_shared/js_query_patterns.py`**
   - Add IMPORT query pattern
   - Ensure all var patterns are included

3. **Add helper for node type differences**
   ```python
   # In js_query_patterns.py
   def class_declaration_query(name_type: str = "identifier") -> str:
       """Generate class declaration query with appropriate name node type."""
       return f"""
           (class_declaration
               name: ({name_type}) @name
           ) @definition
       """
   ```

### Phase 2: Refactor TypeScriptMapping

1. **Change inheritance**:
   ```python
   # Before
   class TypeScriptMapping(BaseMapping, JSFamilyExtraction):

   # After
   class TypeScriptMapping(JavaScriptMapping):
   ```

2. **Override `get_query_for_concept()`**:
   ```python
   def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
       if concept == UniversalConcept.DEFINITION:
           # Get base JS patterns
           base_query = super().get_query_for_concept(concept)

           if base_query is None:
               return None

           # CRITICAL: Override class_declaration to use type_identifier
           # TypeScript grammar uses type_identifier for class names,
           # while JavaScript grammar uses identifier
           base_query = base_query.replace(
               "(class_declaration\n                            name: (identifier)",
               "(class_declaration\n                            name: (type_identifier)"
           )

           # Add TypeScript-specific patterns
           ts_specific = """
               ; TypeScript-specific constructs
               (interface_declaration
                   name: (type_identifier) @name
               ) @definition

               (enum_declaration
                   name: (identifier) @name
               ) @definition

               (type_alias_declaration
                   name: (type_identifier) @name
               ) @definition

               ; Namespace declarations
               (internal_module
                   name: (identifier) @name
               ) @definition

               ; Module declarations (declare module 'name')
               (ambient_declaration
                   (module
                       name: (_) @name
                   )
               ) @definition
           """

           return base_query + ts_specific

       # Inherit IMPORT and COMMENT concepts from JavaScript
       return super().get_query_for_concept(concept)
   ```

3. **Override `extract_class_name()` for type_identifier**:
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

4. **Remove duplicated methods** - Inherit from JavaScript:
   - `extract_function_name()` - inherited
   - `extract_parameters()` - inherited
   - `should_include_node()` - inherited (or override if needed)

### Phase 3: Refactor JSXMapping

1. **Keep extending JavaScriptMapping, add JSXExtras mixin**:
   ```python
   # Before
   class JSXMapping(JavaScriptMapping):

   # After
   class JSXMapping(JavaScriptMapping, JSXExtras):
   ```

2. **Ensure class query override for type_identifier** (already exists):
   ```python
   def get_class_query(self) -> str:
       # JSX uses TSX grammar which needs type_identifier
       return """
           (class_declaration
               name: (type_identifier) @class_name
           ) @class_def
       """
   ```

3. **Remove duplicated React code** - Now in JSXExtras mixin

### Phase 4: Refactor TSXMapping

1. **Keep extending TypeScriptMapping, add JSXExtras mixin**:
   ```python
   # Before
   class TSXMapping(TypeScriptMapping):

   # After
   class TSXMapping(TypeScriptMapping, JSXExtras):
   ```

2. **Remove duplicated React code** - Now in JSXExtras mixin

3. **Inherits type_identifier handling** from TypeScriptMapping

### Phase 5: Testing

1. Run all existing parser tests
2. Add new tests for:
   - TypeScript parsing `var` patterns
   - JavaScript parsing imports
   - JSX parsing imports and `var` patterns
   - Class name extraction with correct node types
   - Inheritance chain verification

---

## Handling the identifier vs type_identifier Difference

This is the most critical implementation detail. Here's the strategy:

### Strategy: Override in TypeScript Layer

1. **JavaScriptMapping** uses `identifier` (matches javascript grammar)
2. **TypeScriptMapping** overrides to use `type_identifier` (matches typescript/tsx grammar)
3. **JSXMapping** overrides to use `type_identifier` (matches tsx grammar)

### Query Override Pattern

```python
# In TypeScriptMapping.get_query_for_concept()
base_query = super().get_query_for_concept(concept)

# Replace identifier with type_identifier for class names
base_query = base_query.replace(
    "name: (identifier)",
    "name: (type_identifier)"
)
```

### Extraction Override Pattern

```python
# In TypeScriptMapping.extract_class_name()
def extract_class_name(self, node, source) -> str:
    # Try type_identifier first (TypeScript grammar)
    name_node = self.find_child_by_type(node, "type_identifier")
    if name_node:
        return self.get_node_text(name_node, source)

    # Fallback to identifier (shouldn't happen with TS grammar)
    return super().extract_class_name(node, source)
```

---

## Risks and Mitigations

### Risk 1: MRO (Method Resolution Order) Complexity

**Issue:** Multiple inheritance can cause method resolution issues.

**Mitigation:**
- Use mixins only for truly independent functionality (JSXExtras)
- Test MRO explicitly in unit tests
- Document inheritance chain clearly
- Keep mixin methods distinct from base class methods

### Risk 2: Breaking Existing Behavior

**Issue:** Refactoring could break existing parsing.

**Mitigation:**
- Comprehensive test coverage before refactoring
- Run all tests after each phase
- Compare parsing results before/after for sample files
- Golden file tests for known good outputs

### Risk 3: Grammar-Specific Query Differences

**Issue:** The `identifier` vs `type_identifier` difference requires careful handling.

**Mitigation:**
- Document the difference prominently (in code comments)
- Use string replacement in TypeScriptMapping override
- Test class parsing explicitly for each language
- Add assertions to verify correct node types

### Risk 4: Performance Impact

**Issue:** Deeper inheritance chain and string replacement could affect performance.

**Mitigation:**
- Benchmark parsing performance before/after
- Cache query results if needed
- Profile method resolution time
- Use `__slots__` if memory becomes an issue

---

## Alternative Approaches Considered

### Alternative A: TSX Grammar for Everything (NOT VIABLE)

Use TSX grammar for all JS-family files including .js files.

**Why it doesn't work:**
- Class names use `type_identifier` in TSX but `identifier` in JS grammar
- All JavaScriptMapping class queries would break
- Would require rewriting all JS queries with no benefit

### Alternative B: Query Composition (String Templates)

```python
JS_BASE_QUERY = "... common patterns ..."
TS_QUERY = JS_BASE_QUERY + "... ts patterns ..."
```

**Pros:** Explicit, no MRO issues
**Cons:** String manipulation is error-prone, loses OOP benefits, doesn't handle node type differences well

### Alternative C: Keep Separate + Shared Query Module

Keep JS and TS as siblings but centralize all queries in a shared module.

**Pros:** Minimal code changes
**Cons:** Doesn't fix the propagation problem, still have duplicate code

### Alternative D: Single Mapping with Language Parameter

```python
class JSFamilyMapping(BaseMapping):
    def __init__(self, language: Language):
        self.language = language
        self.is_typescript = language in [Language.TYPESCRIPT, Language.TSX]
        self.has_jsx = language in [Language.JSX, Language.TSX]
```

**Pros:** Maximum code sharing
**Cons:** Lots of conditionals, harder to maintain, violates SRP

---

## Recommendation

**Implement Inheritance Consolidation** (TypeScript extends JavaScript) because:

1. It's the most semantically correct (TS extends JS)
2. It provides automatic fix propagation
3. It uses standard OOP patterns
4. It's easier to understand and maintain
5. It reduces code duplication significantly
6. It properly handles grammar differences through overrides

The inheritance approach should be combined with the JSXExtras mixin to avoid diamond inheritance issues while still sharing React-specific code between JSX and TSX.

---

## Success Metrics

After implementation, verify:

- [ ] All existing tests pass
- [ ] Code reduction of ~200-300 lines
- [ ] Single place to fix common patterns (in JavaScriptMapping)
- [ ] TypeScript parses all valid JS syntax (including `var`)
- [ ] Class names extracted correctly for all languages
- [ ] No performance regression (< 5% slowdown acceptable)
- [ ] Clear, documented inheritance chain

---

## Complexity Estimate

| Phase | Tasks | Complexity | Risk |
|-------|-------|------------|------|
| Phase 1 | Create JSXExtras mixin, update query patterns | Medium | Low |
| Phase 2 | Refactor TypeScriptMapping to extend JavaScript | High | Medium |
| Phase 3 | Refactor JSXMapping to use JSXExtras | Low | Low |
| Phase 4 | Refactor TSXMapping to use JSXExtras | Low | Low |
| Phase 5 | Testing and validation | Medium | Low |

**Note:** This is an architectural refactoring that touches core parsing logic. It should be done carefully with thorough testing.

---

## Quick Fix vs. Full Refactor

If you want a **quick fix** for the immediate issues (missing IMPORT, var patterns), you can:
1. Add the missing patterns to each file individually
2. Clean up duplicate imports
3. Add tests

Then do the **full refactor** as a separate, dedicated effort.

The quick fix addresses the symptoms; the refactor addresses the root cause.

---

## Files Affected

### Modified
- `chunkhound/parsers/mappings/typescript.py` - Major changes (extend JavaScript)
- `chunkhound/parsers/mappings/javascript.py` - Add missing patterns
- `chunkhound/parsers/mappings/jsx.py` - Use JSXExtras mixin
- `chunkhound/parsers/mappings/tsx.py` - Use JSXExtras mixin

### New
- `chunkhound/parsers/mappings/_shared/jsx_extras.py`

### Tests
- `tests/test_parsers.py` (additions)
- New inheritance chain tests
- Class name extraction tests for all languages

---

## Version History

- **Created:** 2025-11-19
- **Updated:** 2025-11-19 - Added grammar analysis findings, clarified node type differences, removed Option 3 (TSX for all) as not viable
- **Author:** Claude Code Review
- **Status:** Proposal - Recommended for Implementation
