# JS-Family Parser Test Specification

This document defines all constructs that must be tested across JS-family parsers to ensure complete and consistent parsing coverage.

## Test Matrix

| Category | JS | JSX | TS | TSX | Vue-JS | Vue-TS |
|----------|:--:|:---:|:--:|:---:|:------:|:------:|
| Imports | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Exports | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Functions | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Classes | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Variables | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Comments | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| CommonJS | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| TypeScript Types | :x: | :x: | :white_check_mark: | :white_check_mark: | :x: | :white_check_mark: |
| JSX/React | :x: | :white_check_mark: | :x: | :white_check_mark: | :x: | :x: |
| Vue Specific | :x: | :x: | :x: | :x: | :white_check_mark: | :white_check_mark: |

---

## 1. Imports

**Applies to: ALL (JS, JSX, TS, TSX, Vue-JS, Vue-TS)**

### ES6 Import Statements

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Default import | `import React from 'react'` | name: React, module: react, type: default |
| Named import | `import { useState } from 'react'` | names: [useState], module: react, type: named |
| Multiple named | `import { useState, useEffect } from 'react'` | names: [useState, useEffect] |
| Namespace import | `import * as utils from './utils'` | name: utils, type: namespace |
| Combined default + named | `import React, { useState } from 'react'` | default: React, named: [useState] |
| Side-effect import | `import './styles.css'` | module: ./styles.css, type: side-effect |
| Aliased import | `import { foo as bar } from 'module'` | name: bar, original: foo |
| Multi-line import | `import {\n  a,\n  b,\n  c\n} from 'module'` | names: [a, b, c] |

### TypeScript-Specific Imports (TS, TSX, Vue-TS only)

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Type import | `import type { Props } from './types'` | typeOnly: true |
| Inline type import | `import { type Props, useState } from 'react'` | typeOnly: [Props], value: [useState] |
| Type namespace | `import type * as Types from './types'` | typeOnly: true, type: namespace |

### Dynamic Imports

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Await import | `const module = await import('./module')` | dynamic: true, module: ./module |
| Promise import | `import('./module').then(m => m.default)` | dynamic: true |
| Conditional import | `if (condition) { await import('./module') }` | dynamic: true |

### CommonJS Imports

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Basic require | `const fs = require('fs')` | name: fs, module: fs |
| Destructured require | `const { readFile } = require('fs')` | names: [readFile] |
| Multiple requires | `const fs = require('fs'), path = require('path')` | multiple declarations |
| Aliased require | `const { readFile: read } = require('fs')` | name: read, original: readFile |

---

## 2. Exports

**Applies to: ALL (JS, JSX, TS, TSX, Vue-JS, Vue-TS)**

### Named Exports

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Export const | `export const name = 'value'` | name: name, type: const |
| Export let | `export let count = 0` | name: count, type: let |
| Export var | `export var legacy = true` | name: legacy, type: var |
| Export function | `export function foo() {}` | name: foo, type: function |
| Export class | `export class Bar {}` | name: Bar, type: class |
| Export list | `export { foo, bar }` | names: [foo, bar] |
| Export as default | `export { foo as default }` | name: foo, asDefault: true |
| Export aliased | `export { foo as bar }` | name: bar, original: foo |
| Export async function | `export async function fetch() {}` | async: true |

### Default Exports

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Default anonymous function | `export default function() {}` | type: function, anonymous: true |
| Default named function | `export default function named() {}` | name: named |
| Default anonymous class | `export default class {}` | type: class, anonymous: true |
| Default named class | `export default class Named {}` | name: Named |
| Default object | `export default { key: value }` | type: object |
| Default array | `export default [1, 2, 3]` | type: array |
| Default expression | `export default foo + bar` | type: expression |
| Default arrow function | `export default () => {}` | type: arrow |

### Re-exports

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Re-export named | `export { foo } from './module'` | source: ./module |
| Re-export all | `export * from './module'` | type: all, source: ./module |
| Re-export namespace | `export * as namespace from './module'` | name: namespace |
| Re-export default | `export { default } from './module'` | type: default |
| Re-export aliased | `export { foo as bar } from './module'` | name: bar, original: foo |

### TypeScript-Specific Exports (TS, TSX, Vue-TS only)

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Export type | `export type { Props }` | typeOnly: true |
| Export interface | `export interface Foo {}` | name: Foo, type: interface |
| Export enum | `export enum Status {}` | name: Status, type: enum |
| Export type alias | `export type Alias = string` | name: Alias, type: type_alias |

### CommonJS Exports

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| module.exports object | `module.exports = { foo, bar }` | type: object |
| module.exports function | `module.exports = function() {}` | type: function |
| module.exports property | `module.exports.foo = bar` | name: foo |
| exports shorthand | `exports.foo = bar` | name: foo |
| Nested exports | `module.exports.sub.prop = value` | path: sub.prop |

---

## 3. Functions

**Applies to: ALL (JS, JSX, TS, TSX, Vue-JS, Vue-TS)**

### Function Declarations

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Basic function | `function foo() {}` | name: foo |
| With parameters | `function foo(a, b) {}` | params: [a, b] |
| Async function | `async function foo() {}` | async: true |
| Generator function | `function* foo() {}` | generator: true |
| Async generator | `async function* foo() {}` | async: true, generator: true |

### Function Expressions

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Const function expression | `const foo = function() {}` | name: foo |
| Named function expression | `const foo = function bar() {}` | name: foo, innerName: bar |
| Let function expression | `let foo = function() {}` | name: foo, declarationType: let |
| Var function expression | `var foo = function() {}` | name: foo, declarationType: var |

### Arrow Functions

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Const arrow | `const foo = () => {}` | name: foo, arrow: true |
| Let arrow | `let foo = () => {}` | declarationType: let |
| Var arrow | `var foo = () => {}` | declarationType: var |
| With parameters | `const foo = (a, b) => {}` | params: [a, b] |
| Single param no parens | `const foo = a => a * 2` | params: [a] |
| Implicit return | `const foo = () => value` | implicitReturn: true |
| Async arrow | `const foo = async () => {}` | async: true |

### Parameters (All Function Types)

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Default values | `function foo(a = 1, b = 'default') {}` | defaults: [1, 'default'] |
| Rest parameters | `function foo(...args) {}` | rest: args |
| Destructured object | `function foo({ a, b }) {}` | destructured: true |
| Destructured array | `function foo([first, second]) {}` | destructured: true |
| Mixed parameters | `function foo(a, { b, c } = {}, ...rest) {}` | complex: true |
| Nested destructuring | `function foo({ a: { b } }) {}` | nested: true |
| Default with destructuring | `function foo({ a = 1 } = {}) {}` | defaults in destructuring |

### TypeScript Function Features (TS, TSX, Vue-TS only)

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Parameter types | `function foo(a: string, b: number) {}` | paramTypes: [string, number] |
| Return type | `function foo(): string {}` | returnType: string |
| Generic function | `function foo<T>(a: T): T {}` | typeParams: [T] |
| Multiple generics | `function foo<T, U>(a: T, b: U) {}` | typeParams: [T, U] |
| Generic constraints | `function foo<T extends Base>() {}` | constraints: [Base] |
| Optional parameters | `function foo(a?: string) {}` | optional: [a] |
| Function overloads | `function foo(a: string): string;` | overload: true |
| This parameter | `function foo(this: Context) {}` | thisType: Context |

---

## 4. Classes

**Applies to: ALL (JS, JSX, TS, TSX, Vue-JS, Vue-TS)**

### Class Declarations

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Basic class | `class Foo {}` | name: Foo |
| With extends | `class Foo extends Bar {}` | extends: Bar |
| Multiple inheritance simulation | `class Foo extends mixin(A, B) {}` | extends: expression |

### Class Expressions

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Anonymous class expression | `const Foo = class {}` | name: Foo, anonymous: true |
| Named class expression | `const Foo = class Bar {}` | name: Foo, innerName: Bar |

### Class Members - Methods

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Constructor | `constructor() {}` | name: constructor |
| Instance method | `method() {}` | name: method |
| Async method | `async method() {}` | async: true |
| Generator method | `*method() {}` | generator: true |
| Static method | `static method() {}` | static: true |
| Getter | `get prop() {}` | getter: true |
| Setter | `set prop(value) {}` | setter: true |
| Computed name | `[computed]() {}` | computed: true |
| Private method (ES2022) | `#privateMethod() {}` | private: true |

### Class Members - Properties

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Instance property | `prop = value` | name: prop |
| Static property | `static prop = value` | static: true |
| Private property | `#privateProp = value` | private: true |
| Without initializer | `prop;` | uninitialized: true |

### TypeScript Class Features (TS, TSX, Vue-TS only)

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Implements | `class Foo implements Bar {}` | implements: [Bar] |
| Multiple implements | `class Foo implements A, B {}` | implements: [A, B] |
| Access modifiers | `public method() {}` | access: public |
| Private modifier | `private method() {}` | access: private |
| Protected modifier | `protected method() {}` | access: protected |
| Readonly property | `readonly prop: string` | readonly: true |
| Abstract class | `abstract class Foo {}` | abstract: true |
| Abstract method | `abstract method(): void` | abstract: true |
| Parameter properties | `constructor(public name: string) {}` | paramProps: [name] |
| Generic class | `class Foo<T> {}` | typeParams: [T] |
| Class decorator | `@decorator class Foo {}` | decorators: [decorator] |
| Method decorator | `@decorator method() {}` | decorators: [decorator] |
| Property decorator | `@decorator prop: string` | decorators: [decorator] |

---

## 5. Variables

**Applies to: ALL (JS, JSX, TS, TSX, Vue-JS, Vue-TS)**

### Declaration Types

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Const declaration | `const name = value` | declarationType: const |
| Let declaration | `let name = value` | declarationType: let |
| Var declaration | `var name = value` | declarationType: var |
| Uninitialized let | `let name;` | uninitialized: true |
| Uninitialized var | `var name;` | uninitialized: true |

### Initializer Types

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| String literal | `const str = 'hello'` | initType: string |
| Number literal | `const num = 42` | initType: number |
| Boolean literal | `const bool = true` | initType: boolean |
| Object literal | `const obj = { key: value }` | initType: object |
| Array literal | `const arr = [1, 2, 3]` | initType: array |
| Function expression | `const fn = function() {}` | initType: function |
| Arrow function | `const fn = () => {}` | initType: arrow |
| Class expression | `const Cls = class {}` | initType: class |
| Null | `const n = null` | initType: null |
| Undefined | `const u = undefined` | initType: undefined |
| Template literal | `const tpl = \`hello ${name}\`` | initType: template |
| RegExp | `const re = /pattern/g` | initType: regexp |
| BigInt | `const big = 100n` | initType: bigint |

### Multiple Declarations

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Multiple const | `const a = 1, b = 2, c = 3` | names: [a, b, c] |
| Multiple let uninitialized | `let x, y, z` | names: [x, y, z] |
| Mixed initialization | `var i = 0, len = arr.length` | partial init |

### Destructuring

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Object destructuring | `const { a, b } = obj` | destructured: object |
| Array destructuring | `const [first, second] = arr` | destructured: array |
| Nested object | `const { a: { b } } = obj` | nested: true |
| With defaults | `const { a = 1 } = obj` | defaults: [a] |
| With rename | `const { a: renamed } = obj` | renames: {a: renamed} |
| Rest element | `const { a, ...rest } = obj` | rest: rest |
| Array rest | `const [first, ...rest] = arr` | rest: rest |
| Mixed | `const { a, b: [c, d] } = obj` | complex: true |

### TypeScript Variable Features (TS, TSX, Vue-TS only)

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Type annotation | `const name: string = 'value'` | type: string |
| Complex type | `const map: Map<string, number> = new Map()` | type: Map<string, number> |
| As const | `const obj = { key: 'value' } as const` | asConst: true |
| Satisfies | `const obj = {} satisfies Type` | satisfies: Type |
| Union type | `const value: string \| number = getValue()` | union: true |

---

## 6. TypeScript-Specific Constructs

**Applies to: TS, TSX, Vue-TS only**

### Interfaces

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Basic interface | `interface Foo { prop: string }` | name: Foo |
| Extends interface | `interface Foo extends Bar {}` | extends: Bar |
| Multiple extends | `interface Foo extends A, B {}` | extends: [A, B] |
| Generic interface | `interface Foo<T> { prop: T }` | typeParams: [T] |
| Optional properties | `interface Foo { prop?: string }` | optional: [prop] |
| Readonly properties | `interface Foo { readonly prop: string }` | readonly: [prop] |
| Index signature | `interface Foo { [key: string]: any }` | indexSignature: true |
| Call signature | `interface Foo { (a: string): void }` | callable: true |
| Construct signature | `interface Foo { new(): Bar }` | constructable: true |
| Method signature | `interface Foo { method(): void }` | methods: [method] |

### Type Aliases

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Simple alias | `type Foo = string` | name: Foo, aliasOf: string |
| Union type | `type Foo = A \| B` | union: [A, B] |
| Intersection type | `type Foo = A & B` | intersection: [A, B] |
| Generic alias | `type Foo<T> = T[]` | typeParams: [T] |
| Mapped type | `type Foo = { [K in keyof T]: T[K] }` | mapped: true |
| Conditional type | `type Foo<T> = T extends string ? A : B` | conditional: true |
| Template literal | `type Foo = \`prefix-${string}\`` | template: true |
| Tuple type | `type Foo = [string, number]` | tuple: true |
| Function type | `type Foo = (a: string) => void` | function: true |

### Enums

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Numeric enum | `enum Status { Active, Inactive }` | numeric: true |
| String enum | `enum Status { Active = 'ACTIVE' }` | string: true |
| Mixed enum | `enum Status { A = 0, B = 'B' }` | mixed: true |
| Const enum | `const enum Status { Active }` | const: true |
| Computed member | `enum Foo { A = 1 + 1 }` | computed: true |

### Namespaces

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Basic namespace | `namespace Foo { export const bar = 1 }` | name: Foo |
| Nested namespace | `namespace Foo.Bar {}` | name: Foo.Bar |
| Module declaration | `declare module 'foo' {}` | declare: true, module: foo |
| Ambient declaration | `declare const foo: string` | ambient: true |
| Global augmentation | `declare global {}` | global: true |

### Decorators

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Class decorator | `@decorator class Foo {}` | target: class |
| Method decorator | `@decorator method() {}` | target: method |
| Property decorator | `@decorator prop: string` | target: property |
| Parameter decorator | `method(@decorator param: string) {}` | target: parameter |
| Decorator factory | `@decorator() class Foo {}` | factory: true |
| Multiple decorators | `@a @b class Foo {}` | decorators: [a, b] |

---

## 7. JSX/React Patterns

**Applies to: JSX, TSX only**

### React Components

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Function component | `function App() { return <div /> }` | component: true, type: function |
| Arrow component | `const App = () => <div />` | component: true, type: arrow |
| Class component | `class App extends React.Component {}` | component: true, type: class |
| PureComponent | `class App extends React.PureComponent {}` | pure: true |
| Memo component | `const App = React.memo(() => <div />)` | memo: true |
| ForwardRef | `const App = React.forwardRef((props, ref) => <div />)` | forwardRef: true |
| Lazy component | `const App = React.lazy(() => import('./App'))` | lazy: true |

### JSX Elements

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Self-closing element | `<Component />` | selfClosing: true |
| Element with children | `<Component>children</Component>` | hasChildren: true |
| Fragment shorthand | `<>content</>` | fragment: true |
| Named fragment | `<React.Fragment>content</React.Fragment>` | fragment: true |
| Fragment with key | `<Fragment key={id}>content</Fragment>` | fragmentKey: true |
| Nested elements | `<div><span>text</span></div>` | nested: true |
| Component as child | `<Parent><Child /></Parent>` | componentChildren: true |

### JSX Attributes

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| String attribute | `<div className="foo" />` | attr: className |
| Expression attribute | `<div onClick={handler} />` | expression: true |
| Spread attributes | `<Component {...props} />` | spread: true |
| Boolean attribute | `<input disabled />` | boolean: true |
| Computed attribute | `<div {...{[key]: value}} />` | computed: true |
| Multiple attributes | `<div id="x" className="y" />` | multiple: true |

### React Hooks

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| useState | `const [state, setState] = useState(initial)` | hook: useState |
| useEffect | `useEffect(() => {}, [])` | hook: useEffect |
| useContext | `const value = useContext(Context)` | hook: useContext |
| useReducer | `const [state, dispatch] = useReducer(reducer, initial)` | hook: useReducer |
| useCallback | `const fn = useCallback(() => {}, [])` | hook: useCallback |
| useMemo | `const value = useMemo(() => compute(), [])` | hook: useMemo |
| useRef | `const ref = useRef(null)` | hook: useRef |
| useLayoutEffect | `useLayoutEffect(() => {}, [])` | hook: useLayoutEffect |
| useImperativeHandle | `useImperativeHandle(ref, () => ({}))` | hook: useImperativeHandle |
| Custom hook | `function useCustom() {}` | customHook: true |
| Custom hook call | `const data = useCustomHook()` | customHook: true |

### TypeScript + React (TSX only)

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| FC with props | `const App: React.FC<Props> = () => {}` | propsType: Props |
| Generic component | `function List<T>({ items }: Props<T>) {}` | generic: true |
| Typed useState | `const [state, setState] = useState<Type>()` | stateType: Type |
| Event handler type | `const onClick: MouseEventHandler = () => {}` | handlerType: MouseEventHandler |
| Ref type | `const ref = useRef<HTMLDivElement>(null)` | refType: HTMLDivElement |
| Props interface | `interface Props { name: string }` | propsInterface: true |
| Props type alias | `type Props = { name: string }` | propsType: true |

---

## 8. Vue-Specific Patterns

**Applies to: Vue-JS, Vue-TS only**

### Script Section Types

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Options API | `export default { data() {}, methods: {} }` | api: options |
| Composition API | `export default { setup() {} }` | api: composition |
| Script Setup | `<script setup>` | scriptSetup: true |
| Script Setup TS | `<script setup lang="ts">` | scriptSetup: true, ts: true |
| defineComponent | `export default defineComponent({})` | defineComponent: true |

### Vue Macros (Script Setup)

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| defineProps | `const props = defineProps<Props>()` | macro: defineProps |
| defineProps runtime | `const props = defineProps({ name: String })` | runtime: true |
| defineEmits | `const emit = defineEmits<Emits>()` | macro: defineEmits |
| defineEmits runtime | `const emit = defineEmits(['click'])` | runtime: true |
| defineExpose | `defineExpose({ method })` | macro: defineExpose |
| defineOptions | `defineOptions({ name: 'Component' })` | macro: defineOptions |
| defineSlots | `const slots = defineSlots<Slots>()` | macro: defineSlots |
| withDefaults | `withDefaults(defineProps<Props>(), {})` | withDefaults: true |
| defineModel | `const model = defineModel<string>()` | macro: defineModel |

### Composables

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| useRoute | `const route = useRoute()` | composable: useRoute |
| useRouter | `const router = useRouter()` | composable: useRouter |
| Custom composable def | `function useCounter() {}` | customComposable: true |
| Custom composable use | `const { count } = useCounter()` | composableCall: true |

### Vue Lifecycle

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| onMounted | `onMounted(() => {})` | lifecycle: mounted |
| onUpdated | `onUpdated(() => {})` | lifecycle: updated |
| onUnmounted | `onUnmounted(() => {})` | lifecycle: unmounted |
| onBeforeMount | `onBeforeMount(() => {})` | lifecycle: beforeMount |
| onBeforeUpdate | `onBeforeUpdate(() => {})` | lifecycle: beforeUpdate |
| onBeforeUnmount | `onBeforeUnmount(() => {})` | lifecycle: beforeUnmount |
| onErrorCaptured | `onErrorCaptured(() => {})` | lifecycle: errorCaptured |
| onActivated | `onActivated(() => {})` | lifecycle: activated |
| onDeactivated | `onDeactivated(() => {})` | lifecycle: deactivated |

### Options API Patterns

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| data function | `data() { return {} }` | option: data |
| computed property | `computed: { prop() {} }` | option: computed |
| computed getter/setter | `computed: { prop: { get() {}, set() {} } }` | getterSetter: true |
| method | `methods: { handler() {} }` | option: methods |
| watch | `watch: { prop() {} }` | option: watch |
| watch immediate | `watch: { prop: { handler() {}, immediate: true } }` | immediate: true |
| props | `props: { name: String }` | option: props |
| props with validation | `props: { name: { type: String, required: true } }` | validation: true |
| emits | `emits: ['click']` | option: emits |
| components | `components: { Child }` | option: components |

### Options API Lifecycle

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| created | `created() {}` | lifecycle: created |
| mounted | `mounted() {}` | lifecycle: mounted |
| updated | `updated() {}` | lifecycle: updated |
| destroyed/unmounted | `unmounted() {}` | lifecycle: unmounted |
| beforeCreate | `beforeCreate() {}` | lifecycle: beforeCreate |
| beforeMount | `beforeMount() {}` | lifecycle: beforeMount |
| beforeUpdate | `beforeUpdate() {}` | lifecycle: beforeUpdate |
| beforeDestroy/Unmount | `beforeUnmount() {}` | lifecycle: beforeUnmount |

### SFC Sections

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Script section | `<script>...</script>` | section: script |
| Script setup section | `<script setup>...</script>` | section: scriptSetup |
| Multiple scripts | `<script>` + `<script setup>` | multiple: true |
| Template section | `<template>...</template>` | section: template |
| Style section | `<style>...</style>` | section: style |
| Scoped style | `<style scoped>...</style>` | scoped: true |
| CSS modules | `<style module>...</style>` | module: true |
| Lang attribute | `<script lang="ts">` | lang: ts |

---

## 9. Comments

**Applies to: ALL (JS, JSX, TS, TSX, Vue-JS, Vue-TS)**

### Single-Line Comments

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Basic comment | `// comment` | type: single-line |
| End of line | `const x = 1 // comment` | inline: true |
| Multiple consecutive | `// line 1\n// line 2` | consecutive: true |
| Empty comment | `//` | empty: true |

### Multi-Line Comments

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| Basic multi-line | `/* comment */` | type: multi-line |
| Spanning lines | `/* line 1\nline 2 */` | multiline: true |
| Inline multi-line | `const x = /* comment */ 1` | inline: true |

### Documentation Comments

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| JSDoc basic | `/** Description */` | jsdoc: true |
| JSDoc with @param | `/** @param {string} name */` | tags: [param] |
| JSDoc with @returns | `/** @returns {number} */` | tags: [returns] |
| JSDoc with @throws | `/** @throws {Error} */` | tags: [throws] |
| JSDoc with @example | `/** @example code */` | tags: [example] |
| JSDoc with @deprecated | `/** @deprecated use other */` | tags: [deprecated] |
| JSDoc with @see | `/** @see reference */` | tags: [see] |
| JSDoc with @type | `/** @type {string} */` | tags: [type] |
| TSDoc format | `/** @param name - description */` | tsdoc: true |
| Multiple tags | `/** @param a\n@param b */` | multipleTags: true |

### JSX Comments (JSX, TSX only)

| Test Case | Code | Expected Extraction |
|-----------|------|---------------------|
| JSX comment | `{/* comment */}` | jsx: true |
| Multi-line JSX | `{/* line 1\nline 2 */}` | multiline: true |

---

## 10. Edge Cases and Special Patterns

**Applies to: ALL (JS, JSX, TS, TSX, Vue-JS, Vue-TS)**

### File Structure

| Test Case | Code | Expected Behavior |
|-----------|------|-------------------|
| Empty file | `` | No chunks, no errors |
| Only comments | `// comment` | Comment chunk only |
| Only imports | `import 'module'` | Import chunk only |
| Shebang | `#!/usr/bin/env node` | Ignored or metadata |
| Use strict | `'use strict'` | Directive recognized |

### Nesting Patterns

| Test Case | Code | Expected Behavior |
|-----------|------|-------------------|
| Function in function | `function a() { function b() {} }` | Both extracted, relationship tracked |
| Class in function | `function factory() { return class {} }` | Both extracted |
| Function in class | `class A { method() {} }` | Method extracted as part of class |
| Namespace nesting (TS) | `namespace A { namespace B {} }` | Both extracted |
| Arrow in arrow | `const a = () => () => {}` | Both extracted |

### Unicode

| Test Case | Code | Expected Behavior |
|-----------|------|-------------------|
| Unicode identifier | `const café = 'coffee'` | Name extracted correctly |
| Unicode in string | `const emoji = '🎉'` | Content preserved |
| Unicode in comment | `// 日本語コメント` | Comment extracted |
| Unicode in JSX | `<div>中文内容</div>` | Content preserved |

### Unusual Valid Patterns

| Test Case | Code | Expected Behavior |
|-----------|------|-------------------|
| Comma operator | `const x = (1, 2, 3)` | Value is 3 |
| IIFE | `(function() {}())` | Function extracted |
| IIFE arrow | `(() => {})()` | Function extracted |
| Labeled statement | `loop: for (;;) {}` | Label recognized |
| Sequence in export | `export default (a, b)` | Value is b |

### Module Patterns

| Test Case | Code | Expected Behavior |
|-----------|------|-------------------|
| UMD wrapper | `(function(root, factory) { ... })` | Factory extracted |
| AMD define | `define(['dep'], function(dep) {})` | Dependencies tracked |
| SystemJS | `System.register([], function() {})` | Module extracted |

### Metadata Completeness

For every extracted chunk, verify:

| Metadata | Description |
|----------|-------------|
| name | Identifier/symbol name |
| content | Full source text |
| startLine | 1-based start line number |
| endLine | 1-based end line number |
| startByte | Byte offset start |
| endByte | Byte offset end |
| chunkType | Classification (function, class, etc.) |
| language | Source language |
| filePath | Source file path |

---

## 11. Cross-Language Consistency Tests

These tests verify that the same construct produces consistent results across all applicable languages.

### Consistency Matrix

For each of these constructs, verify identical extraction across all applicable languages:

| Construct | Expected Consistency |
|-----------|---------------------|
| `function foo() {}` | Same name, type, location across JS/JSX/TS/TSX/Vue |
| `class Foo {}` | Same name, type, location across all |
| `const x = {}` | Same name, initType across all |
| `import { a } from 'b'` | Same import info across all |
| `export const x = 1` | Same export info across all |
| `// comment` | Same content across all |
| `/* multi */` | Same content across all |
| `async function f() {}` | Same async flag across all |
| `const f = () => {}` | Same arrow detection across all |

### Type-Enhanced Consistency (TS variants)

| Construct | JS/JSX | TS/TSX |
|-----------|--------|--------|
| `function foo(a) {}` | params: [a] | params: [a], types: [any] |
| `class Foo {}` | name: Foo | name: Foo, typeParams: [] |
| `const x = 1` | initType: number | type: number (inferred) |

---

## 12. Performance and Stress Tests

### Large Files

| Test Case | Expected Behavior |
|-----------|-------------------|
| 10,000 line file | Completes in < 5 seconds |
| 1,000 functions | All extracted correctly |
| Deeply nested (20 levels) | No stack overflow |
| 1MB file size | Memory usage < 500MB |

### Complex Structures

| Test Case | Expected Behavior |
|-----------|-------------------|
| 100 imports | All imports extracted |
| 50 exports | All exports extracted |
| Class with 100 methods | All methods extracted |
| 50 TypeScript generics | All type params extracted |

---

## Summary

This specification defines **~300 distinct test cases** across:

- **10 major categories** (Imports, Exports, Functions, Classes, Variables, TypeScript, JSX, Vue, Comments, Edge Cases)
- **6 language variants** (JS, JSX, TS, TSX, Vue-JS, Vue-TS)

Each test should verify:
1. The construct is **detected** (chunk is created)
2. The **metadata is correct** (name, type, location)
3. The **content is complete** (full source text)
4. **Cross-language consistency** (same construct = same result)

---

## Version

- **Created:** 2025-11-19
- **Author:** Claude Code Review
- **Purpose:** Comprehensive test specification for JS-family parser parity
