"""Vue 3 Compiler Macros and Composition API tests.

Tests for Vue 3 specific patterns including:
- Compiler macros (defineProps, defineEmits, defineExpose, defineSlots, defineOptions, withDefaults, defineModel)
- Composition API patterns
- Options API lifecycle hooks

Reference: docs/js-family-parser-test-specification.md
"""

import pytest
from pathlib import Path

from chunkhound.core.types.common import ChunkType, FileId, Language
from chunkhound.parsers.parser_factory import ParserFactory
from chunkhound.parsers.vue_parser import VueParser


@pytest.fixture
def parser_factory():
    """Create a parser factory instance."""
    return ParserFactory()


def _parse_vue(code: str, filename: str = "test.vue"):
    """Helper to parse Vue SFC code.

    Args:
        code: Vue SFC content
        filename: Filename for the parsed content

    Returns:
        List of parsed chunks
    """
    parser = VueParser()
    return parser.parse_content(code, Path(filename), file_id=FileId(1))


# =============================================================================
# VUE COMPILER MACROS
# =============================================================================


class TestVueCompilerMacros:
    """Test Vue 3 compiler macros in script setup.

    These tests verify the desired extraction behavior for Vue compiler macros.
    The parser should extract variables (props, emit, etc.) and functions from
    script setup sections with appropriate metadata.
    """

    def test_with_defaults_macro(self):
        """Test withDefaults macro for props."""
        code = """<script setup lang="ts">
interface Props {
    msg?: string;
    count?: number;
    disabled?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
    msg: 'hello',
    count: 0,
    disabled: false
})
</script>

<template>
    <div>{{ props.msg }} - {{ props.count }}</div>
</template>
"""
        chunks = _parse_vue(code, "WithDefaults.vue")

        # Verify Props interface extracted
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) == 1, "Should extract exactly one Props interface"
        assert interface_chunks[0].symbol == "Props", "Interface should be named Props"
        assert "msg?: string" in interface_chunks[0].code, "Interface should contain msg property"

        # Verify props variable extracted
        variable_chunks = [c for c in chunks if c.chunk_type == ChunkType.VARIABLE]
        props_chunks = [c for c in variable_chunks if c.symbol == "props"]
        assert len(props_chunks) == 1, "Should extract props variable"
        assert "withDefaults" in props_chunks[0].code, "props should contain withDefaults"
        assert "defineProps" in props_chunks[0].code, "props should contain defineProps"

        # Verify Vue metadata
        script_chunks = [c for c in chunks if c.metadata.get("vue_section") == "script"]
        assert len(script_chunks) > 0, "Should have script chunks"
        assert all(c.metadata.get("vue_script_setup") for c in script_chunks), \
            "All script chunks should be marked as setup"
        assert all(c.metadata.get("vue_script_lang") == "ts" for c in script_chunks), \
            "All script chunks should have lang=ts"

        # Verify macros detected in metadata
        macros_found = set()
        for c in script_chunks:
            macros_found.update(c.metadata.get("vue_macros", []))
        assert "defineProps" in macros_found, "defineProps should be in detected macros"
        assert "withDefaults" in macros_found, "withDefaults should be in detected macros"

    def test_define_model_macro(self):
        """Test defineModel macro for v-model support."""
        code = """<script setup>
const modelValue = defineModel()
const count = defineModel('count', { type: Number, default: 0 })
</script>

<template>
    <input v-model="modelValue" />
    <input v-model="count" type="number" />
</template>
"""
        chunks = _parse_vue(code, "DefineModel.vue")

        # Verify both model variables extracted
        variable_chunks = [c for c in chunks if c.chunk_type == ChunkType.VARIABLE]

        model_value_chunks = [c for c in variable_chunks if c.symbol == "modelValue"]
        assert len(model_value_chunks) == 1, "Should extract modelValue variable"
        assert "defineModel()" in model_value_chunks[0].code, \
            "modelValue should contain defineModel()"

        count_chunks = [c for c in variable_chunks if c.symbol == "count"]
        assert len(count_chunks) == 1, "Should extract count variable"
        assert "defineModel('count'" in count_chunks[0].code, \
            "count should contain defineModel with name"

        # Verify Vue metadata
        script_chunks = [c for c in chunks if c.metadata.get("vue_section") == "script"]
        assert len(script_chunks) > 0, "Should have script chunks"
        assert all(c.metadata.get("vue_script_setup") for c in script_chunks), \
            "All script chunks should be marked as setup"

        # Verify macros detected
        macros_found = set()
        for c in script_chunks:
            macros_found.update(c.metadata.get("vue_macros", []))
        assert "defineModel" in macros_found, "defineModel should be in detected macros"

    def test_define_slots_macro(self):
        """Test defineSlots macro with TypeScript."""
        code = """<script setup lang="ts">
const slots = defineSlots<{
    default: () => any;
    header: (props: { title: string }) => any;
    footer: (props: { year: number }) => any;
}>()
</script>

<template>
    <div>
        <header><slot name="header" :title="'Title'" /></header>
        <main><slot /></main>
        <footer><slot name="footer" :year="2024" /></footer>
    </div>
</template>
"""
        chunks = _parse_vue(code, "DefineSlots.vue")

        # Verify slots variable extracted
        variable_chunks = [c for c in chunks if c.chunk_type == ChunkType.VARIABLE]
        slots_chunks = [c for c in variable_chunks if c.symbol == "slots"]
        assert len(slots_chunks) == 1, "Should extract slots variable"
        assert "defineSlots" in slots_chunks[0].code, "slots should contain defineSlots"
        assert "header:" in slots_chunks[0].code, "slots should contain header slot type"
        assert "footer:" in slots_chunks[0].code, "slots should contain footer slot type"

        # Verify Vue metadata
        script_chunks = [c for c in chunks if c.metadata.get("vue_section") == "script"]
        assert all(c.metadata.get("vue_script_lang") == "ts" for c in script_chunks), \
            "All script chunks should have lang=ts"

        # Verify macros detected
        macros_found = set()
        for c in script_chunks:
            macros_found.update(c.metadata.get("vue_macros", []))
        assert "defineSlots" in macros_found, "defineSlots should be in detected macros"

    def test_define_options_macro(self):
        """Test defineOptions macro for component options."""
        code = """<script setup>
defineOptions({
    name: 'MyCustomComponent',
    inheritAttrs: false,
    customOptions: {
        customProperty: true
    }
})

const message = 'Hello'
</script>

<template>
    <div v-bind="$attrs">{{ message }}</div>
</template>
"""
        chunks = _parse_vue(code, "DefineOptions.vue")

        # Verify message variable extracted
        variable_chunks = [c for c in chunks if c.chunk_type == ChunkType.VARIABLE]
        message_chunks = [c for c in variable_chunks if c.symbol == "message"]
        assert len(message_chunks) == 1, "Should extract message variable"
        assert "'Hello'" in message_chunks[0].code, "message should contain 'Hello'"

        # Verify Vue metadata
        script_chunks = [c for c in chunks if c.metadata.get("vue_section") == "script"]
        assert len(script_chunks) > 0, "Should have script chunks"

        # Verify defineOptions detected in macros
        macros_found = set()
        for c in script_chunks:
            macros_found.update(c.metadata.get("vue_macros", []))
        assert "defineOptions" in macros_found, "defineOptions should be in detected macros"

    def test_define_props_runtime(self):
        """Test defineProps with runtime declaration."""
        code = """<script setup>
const props = defineProps({
    title: {
        type: String,
        required: true
    },
    count: {
        type: Number,
        default: 0
    },
    items: {
        type: Array,
        default: () => []
    }
})
</script>

<template>
    <div>
        <h1>{{ props.title }}</h1>
        <span>{{ props.count }}</span>
    </div>
</template>
"""
        chunks = _parse_vue(code, "DefinePropsRuntime.vue")

        # Verify props variable extracted
        variable_chunks = [c for c in chunks if c.chunk_type == ChunkType.VARIABLE]
        props_chunks = [c for c in variable_chunks if c.symbol == "props"]
        assert len(props_chunks) == 1, "Should extract props variable"
        assert "defineProps" in props_chunks[0].code, "props should contain defineProps"
        assert "title:" in props_chunks[0].code, "props should contain title property"
        assert "required: true" in props_chunks[0].code, "props should contain required validation"

        # Verify Vue metadata
        script_chunks = [c for c in chunks if c.metadata.get("vue_section") == "script"]
        assert all(c.metadata.get("vue_script_setup") for c in script_chunks), \
            "All script chunks should be marked as setup"

        # Verify macros detected
        macros_found = set()
        for c in script_chunks:
            macros_found.update(c.metadata.get("vue_macros", []))
        assert "defineProps" in macros_found, "defineProps should be in detected macros"

    def test_define_emits_with_validation(self):
        """Test defineEmits with validation functions."""
        code = """<script setup>
const emit = defineEmits({
    submit: (payload) => {
        return payload.email && payload.password
    },
    cancel: null,
    'update:modelValue': (value) => true
})

function handleSubmit() {
    emit('submit', { email: 'test@test.com', password: '123' })
}
</script>

<template>
    <form @submit.prevent="handleSubmit">
        <button type="submit">Submit</button>
        <button type="button" @click="emit('cancel')">Cancel</button>
    </form>
</template>
"""
        chunks = _parse_vue(code, "DefineEmitsValidation.vue")

        # Verify emit variable extracted
        variable_chunks = [c for c in chunks if c.chunk_type == ChunkType.VARIABLE]
        emit_chunks = [c for c in variable_chunks if c.symbol == "emit"]
        assert len(emit_chunks) == 1, "Should extract emit variable"
        assert "defineEmits" in emit_chunks[0].code, "emit should contain defineEmits"
        assert "submit:" in emit_chunks[0].code, "emit should contain submit event"

        # Verify handleSubmit function extracted
        function_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        handle_submit_chunks = [c for c in function_chunks if c.symbol == "handleSubmit"]
        assert len(handle_submit_chunks) == 1, "Should extract handleSubmit function"
        assert "emit('submit'" in handle_submit_chunks[0].code, \
            "handleSubmit should call emit with submit"

        # Verify Vue metadata
        script_chunks = [c for c in chunks if c.metadata.get("vue_section") == "script"]
        macros_found = set()
        for c in script_chunks:
            macros_found.update(c.metadata.get("vue_macros", []))
        assert "defineEmits" in macros_found, "defineEmits should be in detected macros"

    def test_define_expose_macro(self):
        """Test defineExpose macro."""
        code = """<script setup>
import { ref } from 'vue'

const count = ref(0)
const message = ref('Hello')

function increment() {
    count.value++
}

function reset() {
    count.value = 0
    message.value = 'Hello'
}

// Only expose specific properties/methods
defineExpose({
    count,
    increment,
    reset
})
</script>

<template>
    <div>{{ message }} - {{ count }}</div>
</template>
"""
        chunks = _parse_vue(code, "DefineExpose.vue")

        # Verify ref variables extracted
        variable_chunks = [c for c in chunks if c.chunk_type == ChunkType.VARIABLE]

        count_chunks = [c for c in variable_chunks if c.symbol == "count"]
        assert len(count_chunks) == 1, "Should extract count variable"
        assert "ref(0)" in count_chunks[0].code, "count should contain ref(0)"

        message_chunks = [c for c in variable_chunks if c.symbol == "message"]
        assert len(message_chunks) == 1, "Should extract message variable"
        assert "ref('Hello')" in message_chunks[0].code, "message should contain ref('Hello')"

        # Verify functions extracted
        function_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]

        increment_chunks = [c for c in function_chunks if c.symbol == "increment"]
        assert len(increment_chunks) == 1, "Should extract increment function"
        assert "count.value++" in increment_chunks[0].code, \
            "increment should modify count.value"

        reset_chunks = [c for c in function_chunks if c.symbol == "reset"]
        assert len(reset_chunks) == 1, "Should extract reset function"
        assert "count.value = 0" in reset_chunks[0].code, "reset should set count to 0"

        # Verify Vue metadata
        script_chunks = [c for c in chunks if c.metadata.get("vue_section") == "script"]
        macros_found = set()
        for c in script_chunks:
            macros_found.update(c.metadata.get("vue_macros", []))
        assert "defineExpose" in macros_found, "defineExpose should be in detected macros"


# =============================================================================
# VUE OPTIONS API LIFECYCLE
# =============================================================================


class TestVueOptionsAPILifecycle:
    """Test Vue Options API lifecycle hooks.

    These tests verify that the parser correctly extracts Options API
    components including lifecycle hooks, methods, computed properties,
    and watchers.
    """

    def test_options_api_lifecycle_hooks(self):
        """Test Options API with lifecycle hooks."""
        code = """<script>
export default {
    name: 'LifecycleComponent',
    data() {
        return {
            message: 'Hello',
            items: []
        }
    },
    created() {
        console.log('Component created')
        this.fetchData()
    },
    mounted() {
        console.log('Component mounted')
        this.setupEventListeners()
    },
    beforeUnmount() {
        console.log('Component will unmount')
        this.cleanup()
    },
    methods: {
        fetchData() {
            // Fetch data logic
        },
        setupEventListeners() {
            // Setup listeners
        },
        cleanup() {
            // Cleanup logic
        }
    }
}
</script>

<template>
    <div>{{ message }}</div>
</template>
"""
        chunks = _parse_vue(code, "LifecycleComponent.vue")

        # Verify script chunks exist
        script_chunks = [c for c in chunks if c.metadata.get("vue_section") == "script"]
        assert len(script_chunks) > 0, "Should have script chunks"

        # Verify the script contains lifecycle hooks
        script_code = " ".join([c.code for c in script_chunks])
        assert "export default" in script_code, "Should contain export default"
        assert "created()" in script_code or "created:" in script_code, \
            "Should contain created lifecycle hook"
        assert "mounted()" in script_code or "mounted:" in script_code, \
            "Should contain mounted lifecycle hook"
        assert "beforeUnmount()" in script_code or "beforeUnmount:" in script_code, \
            "Should contain beforeUnmount lifecycle hook"

        # Verify methods are present
        assert "fetchData" in script_code, "Should contain fetchData method"
        assert "setupEventListeners" in script_code, "Should contain setupEventListeners method"
        assert "cleanup" in script_code, "Should contain cleanup method"

        # Options API should NOT be marked as script setup
        assert not any(c.metadata.get("vue_script_setup") for c in script_chunks), \
            "Options API should not be marked as setup"

    def test_options_api_computed_watch(self):
        """Test Options API computed properties and watchers."""
        code = """<script>
export default {
    data() {
        return {
            firstName: '',
            lastName: '',
            items: []
        }
    },
    computed: {
        fullName() {
            return `${this.firstName} ${this.lastName}`
        },
        itemCount() {
            return this.items.length
        }
    },
    watch: {
        firstName(newVal, oldVal) {
            console.log(`firstName changed from ${oldVal} to ${newVal}`)
        },
        items: {
            handler(newItems) {
                console.log('Items updated:', newItems.length)
            },
            deep: true
        }
    }
}
</script>

<template>
    <div>
        <p>{{ fullName }}</p>
        <p>Items: {{ itemCount }}</p>
    </div>
</template>
"""
        chunks = _parse_vue(code, "ComputedWatch.vue")

        # Verify script chunks exist
        script_chunks = [c for c in chunks if c.metadata.get("vue_section") == "script"]
        assert len(script_chunks) > 0, "Should have script chunks"

        # Verify computed properties are present
        script_code = " ".join([c.code for c in script_chunks])
        assert "computed:" in script_code or "computed" in script_code, \
            "Should contain computed section"
        assert "fullName" in script_code, "Should contain fullName computed property"
        assert "itemCount" in script_code, "Should contain itemCount computed property"

        # Verify watchers are present
        assert "watch:" in script_code or "watch" in script_code, \
            "Should contain watch section"
        assert "firstName" in script_code, "Should contain firstName watcher"
        assert "deep: true" in script_code, "Should contain deep watcher option"

        # Verify data function
        assert "data()" in script_code or "data:" in script_code, \
            "Should contain data function"

    def test_options_api_mixins(self):
        """Test Options API with mixins."""
        code = """<script>
import { loggingMixin, validationMixin } from './mixins'

export default {
    mixins: [loggingMixin, validationMixin],
    data() {
        return {
            formData: {}
        }
    },
    methods: {
        submit() {
            if (this.validate(this.formData)) {
                this.log('Form submitted')
            }
        }
    }
}
</script>

<template>
    <form @submit.prevent="submit">
        <button type="submit">Submit</button>
    </form>
</template>
"""
        chunks = _parse_vue(code, "MixinsComponent.vue")

        # Verify script chunks exist
        script_chunks = [c for c in chunks if c.metadata.get("vue_section") == "script"]
        assert len(script_chunks) > 0, "Should have script chunks"

        # Verify imports and content are present
        script_code = " ".join([c.code for c in script_chunks])
        assert "loggingMixin" in script_code, "Should contain loggingMixin import"
        assert "validationMixin" in script_code, "Should contain validationMixin import"

        # Verify mixins and methods are present
        assert "mixins:" in script_code or "mixins" in script_code, \
            "Should contain mixins section"
        assert "submit()" in script_code or "submit:" in script_code, \
            "Should contain submit method"


# =============================================================================
# VUE COMPOSITION API ADVANCED
# =============================================================================


class TestVueCompositionAPIAdvanced:
    """Test advanced Composition API patterns.

    These tests verify that the parser correctly extracts interfaces,
    functions, and variables from advanced Composition API patterns
    including composables, provide/inject, and reactivity transforms.
    """

    def test_composable_with_typescript(self):
        """Test composable function with TypeScript."""
        code = """<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'

interface User {
    id: number;
    name: string;
    email: string;
}

function useUser(userId: number) {
    const user = ref<User | null>(null)
    const loading = ref(true)
    const error = ref<Error | null>(null)

    const fullName = computed(() => user.value?.name ?? 'Unknown')

    async function fetchUser() {
        try {
            loading.value = true
            const response = await fetch(`/api/users/${userId}`)
            user.value = await response.json()
        } catch (e) {
            error.value = e as Error
        } finally {
            loading.value = false
        }
    }

    onMounted(() => {
        fetchUser()
    })

    return { user, loading, error, fullName, fetchUser }
}

const { user, loading, error } = useUser(1)
</script>

<template>
    <div v-if="loading">Loading...</div>
    <div v-else-if="error">Error: {{ error.message }}</div>
    <div v-else>{{ user?.name }}</div>
</template>
"""
        chunks = _parse_vue(code, "ComposableTS.vue")

        # Verify User interface extracted
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) == 1, "Should extract exactly one User interface"
        assert interface_chunks[0].symbol == "User", "Interface should be named User"
        assert "id: number" in interface_chunks[0].code, "Interface should contain id property"
        assert "email: string" in interface_chunks[0].code, "Interface should contain email property"

        # Verify useUser composable function extracted
        function_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        use_user_chunks = [c for c in function_chunks if c.symbol == "useUser"]
        assert len(use_user_chunks) == 1, "Should extract useUser composable function"
        assert "userId: number" in use_user_chunks[0].code, \
            "useUser should have typed parameter"
        assert "return { user, loading, error" in use_user_chunks[0].code, \
            "useUser should return composable state"

        # Verify script chunks contain imports
        script_chunks = [c for c in chunks if c.metadata.get("vue_section") == "script"]
        script_code = " ".join([c.code for c in script_chunks])
        assert "ref" in script_code, "Should import ref"
        assert "computed" in script_code, "Should import computed"
        assert "onMounted" in script_code, "Should import onMounted"

        # Verify Vue metadata
        script_chunks = [c for c in chunks if c.metadata.get("vue_section") == "script"]
        assert all(c.metadata.get("vue_script_setup") for c in script_chunks), \
            "All script chunks should be marked as setup"
        assert all(c.metadata.get("vue_script_lang") == "ts" for c in script_chunks), \
            "All script chunks should have lang=ts"

        # Verify composables detected in metadata
        composables_found = set()
        for c in script_chunks:
            composables_found.update(c.metadata.get("vue_composables", []))
        assert "useUser" in composables_found, "useUser should be in detected composables"

    def test_provide_inject_typed(self):
        """Test provide/inject with TypeScript types."""
        code = """<script setup lang="ts">
import { provide, inject, ref, InjectionKey } from 'vue'

interface AppConfig {
    theme: string;
    apiUrl: string;
}

const configKey: InjectionKey<AppConfig> = Symbol('config')

// Provider component
const config = ref<AppConfig>({
    theme: 'dark',
    apiUrl: 'https://api.example.com'
})

provide(configKey, config.value)

// Consumer component would use:
// const config = inject(configKey)
</script>

<template>
    <div :class="config.theme">
        <slot />
    </div>
</template>
"""
        chunks = _parse_vue(code, "ProvideInject.vue")

        # Verify AppConfig interface extracted
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) == 1, "Should extract exactly one AppConfig interface"
        assert interface_chunks[0].symbol == "AppConfig", "Interface should be named AppConfig"
        assert "theme: string" in interface_chunks[0].code, \
            "Interface should contain theme property"
        assert "apiUrl: string" in interface_chunks[0].code, \
            "Interface should contain apiUrl property"

        # Verify script chunks contain imports
        script_chunks = [c for c in chunks if c.metadata.get("vue_section") == "script"]
        script_code = " ".join([c.code for c in script_chunks])
        assert "provide" in script_code, "Should import provide"
        assert "inject" in script_code, "Should import inject"
        assert "InjectionKey" in script_code, "Should import InjectionKey"

        # Verify Vue metadata
        assert all(c.metadata.get("vue_script_lang") == "ts" for c in script_chunks), \
            "All script chunks should have lang=ts"

    def test_reactive_transform(self):
        """Test Vue reactivity transform patterns."""
        code = """<script setup lang="ts">
import { reactive, toRefs } from 'vue'

interface State {
    count: number;
    message: string;
    items: string[];
}

const state = reactive<State>({
    count: 0,
    message: 'Hello',
    items: []
})

// Destructure with toRefs to maintain reactivity
const { count, message, items } = toRefs(state)

function increment() {
    count.value++
}

function addItem(item: string) {
    items.value.push(item)
}
</script>

<template>
    <div>
        <p>{{ message }}</p>
        <p>Count: {{ count }}</p>
        <button @click="increment">+</button>
    </div>
</template>
"""
        chunks = _parse_vue(code, "ReactiveTransform.vue")

        # Verify State interface extracted
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) == 1, "Should extract exactly one State interface"
        assert interface_chunks[0].symbol == "State", "Interface should be named State"
        assert "count: number" in interface_chunks[0].code, \
            "Interface should contain count property"

        # Verify state variable extracted
        variable_chunks = [c for c in chunks if c.chunk_type == ChunkType.VARIABLE]
        state_chunks = [c for c in variable_chunks if c.symbol == "state"]
        assert len(state_chunks) == 1, "Should extract state variable"
        assert "reactive<State>" in state_chunks[0].code, \
            "state should contain reactive<State>"

        # Verify functions extracted
        function_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]

        increment_chunks = [c for c in function_chunks if c.symbol == "increment"]
        assert len(increment_chunks) == 1, "Should extract increment function"
        assert "count.value++" in increment_chunks[0].code, \
            "increment should modify count.value"

        add_item_chunks = [c for c in function_chunks if c.symbol == "addItem"]
        assert len(add_item_chunks) == 1, "Should extract addItem function"
        assert "item: string" in add_item_chunks[0].code, \
            "addItem should have typed parameter"

        # Verify script chunks contain imports
        script_chunks = [c for c in chunks if c.metadata.get("vue_section") == "script"]
        script_code = " ".join([c.code for c in script_chunks])
        assert "reactive" in script_code, "Should import reactive"
        assert "toRefs" in script_code, "Should import toRefs"

        # Verify Vue metadata
        assert all(c.metadata.get("vue_script_lang") == "ts" for c in script_chunks), \
            "All script chunks should have lang=ts"


# =============================================================================
# VUE EDGE CASES
# =============================================================================


class TestVueEdgeCases:
    """Test edge cases in Vue SFC parsing.

    These tests verify that the parser correctly handles edge cases
    like multiple script blocks, empty scripts, and type-only imports.
    """

    def test_multiple_script_blocks(self):
        """Test SFC with both script and script setup."""
        code = """<script>
export default {
    name: 'HybridComponent',
    inheritAttrs: false
}
</script>

<script setup>
import { ref } from 'vue'

const count = ref(0)

function increment() {
    count.value++
}
</script>

<template>
    <div>
        <p>{{ count }}</p>
        <button @click="increment">+</button>
    </div>
</template>
"""
        chunks = _parse_vue(code, "HybridComponent.vue")

        # Verify we have script chunks
        script_chunks = [c for c in chunks if c.metadata.get("vue_section") == "script"]
        assert len(script_chunks) > 0, "Should have script chunks"

        # Verify content from regular script
        non_setup_chunks = [c for c in script_chunks if not c.metadata.get("vue_script_setup")]
        if non_setup_chunks:
            non_setup_code = " ".join([c.code for c in non_setup_chunks])
            assert "HybridComponent" in non_setup_code or "export default" in non_setup_code, \
                "Non-setup script should contain component definition"

        # Verify content from script setup
        setup_chunks = [c for c in script_chunks if c.metadata.get("vue_script_setup")]
        if setup_chunks:
            setup_code = " ".join([c.code for c in setup_chunks])
            assert "ref" in setup_code, "Setup script should import ref"

        # Verify function from script setup
        function_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        increment_chunks = [c for c in function_chunks if c.symbol == "increment"]
        assert len(increment_chunks) == 1, "Should extract increment function"
        assert "count.value++" in increment_chunks[0].code, \
            "increment should modify count.value"

        # Verify we have both setup and non-setup script chunks
        assert len(setup_chunks) > 0, "Should have setup script chunks"
        assert len(non_setup_chunks) > 0, "Should have non-setup script chunks"

    def test_empty_script_setup(self):
        """Test SFC with empty script setup."""
        code = """<script setup>
</script>

<template>
    <div>Static content</div>
</template>
"""
        chunks = _parse_vue(code, "EmptyScript.vue")

        # Should extract template chunk
        template_chunks = [c for c in chunks if c.metadata.get("vue_section") == "template"]
        assert len(template_chunks) == 1, "Should extract exactly one template chunk"
        assert "Static content" in template_chunks[0].code, \
            "Template should contain static content"

        # Should not have any script chunks with code (empty script)
        script_chunks = [c for c in chunks if c.metadata.get("vue_section") == "script"]
        # Empty script may or may not produce chunks, but if it does they should be minimal
        for chunk in script_chunks:
            # Empty script shouldn't have meaningful code
            assert len(chunk.code.strip()) < 50, "Script chunks should be minimal for empty script"

    def test_script_with_type_imports(self):
        """Test script with type-only imports."""
        code = """<script setup lang="ts">
import type { PropType } from 'vue'
import type { User, Config } from './types'

interface Props {
    user: User;
    config: Config;
}

const props = defineProps<Props>()
</script>

<template>
    <div>{{ props.user.name }}</div>
</template>
"""
        chunks = _parse_vue(code, "TypeImports.vue")

        # Verify Props interface extracted
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) == 1, "Should extract exactly one Props interface"
        assert interface_chunks[0].symbol == "Props", "Interface should be named Props"
        assert "user: User" in interface_chunks[0].code, \
            "Interface should contain user property"
        assert "config: Config" in interface_chunks[0].code, \
            "Interface should contain config property"

        # Verify Vue metadata
        script_chunks = [c for c in chunks if c.metadata.get("vue_section") == "script"]
        assert len(script_chunks) > 0, "Should have script chunks"

        # Verify type imports are in script
        script_code = " ".join([c.code for c in script_chunks])
        assert "import type" in script_code, "Should have type-only imports"
        assert "PropType" in script_code, "Should import PropType"
        assert "User" in script_code, "Should import User"
        assert "Config" in script_code, "Should import Config"

        # Verify script setup metadata
        assert all(c.metadata.get("vue_script_setup") for c in script_chunks), \
            "All script chunks should be marked as setup"
        assert all(c.metadata.get("vue_script_lang") == "ts" for c in script_chunks), \
            "All script chunks should have lang=ts"

        # Verify macros detected
        macros_found = set()
        for c in script_chunks:
            macros_found.update(c.metadata.get("vue_macros", []))
        assert "defineProps" in macros_found, "defineProps should be in detected macros"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
