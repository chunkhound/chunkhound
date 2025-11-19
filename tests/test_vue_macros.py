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
from chunkhound.parsers.parser_factory import ParserFactory, get_parser_factory


@pytest.fixture
def parser_factory():
    """Create a parser factory instance."""
    return ParserFactory()


# =============================================================================
# VUE COMPILER MACROS
# =============================================================================


class TestVueCompilerMacros:
    """Test Vue 3 compiler macros in script setup.

    NOTE: Vue compiler macros are typically const assignments which the
    current parser doesn't extract as separate chunks. These tests document
    the expected behavior for future enhancement.
    """

    @pytest.mark.xfail(reason="Vue parser does not extract const assignments with macros")
    def test_with_defaults_macro(self, parser_factory):
        """Test withDefaults macro for props."""
        parser = parser_factory.create_parser(Language.VUE)

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
        chunks = parser.parse_content(code, "WithDefaults.vue", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Should find the interface
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) > 0, "Should extract Props interface"

        # Verify withDefaults macro content is present
        all_code = " ".join([c.code for c in chunks])
        assert "withDefaults" in all_code or "defineProps" in all_code, \
            "Should capture withDefaults or defineProps macro"

    @pytest.mark.xfail(reason="Vue parser does not extract const assignments with macros")
    def test_define_model_macro(self, parser_factory):
        """Test defineModel macro for v-model support."""
        parser = parser_factory.create_parser(Language.VUE)

        code = """<script setup>
const modelValue = defineModel()
const count = defineModel('count', { type: Number, default: 0 })
</script>

<template>
    <input v-model="modelValue" />
    <input v-model="count" type="number" />
</template>
"""
        chunks = parser.parse_content(code, "DefineModel.vue", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Verify defineModel is present in content
        all_code = " ".join([c.code for c in chunks])
        assert "defineModel" in all_code, "Should capture defineModel macro"

    @pytest.mark.xfail(reason="Vue parser does not extract const assignments with macros")
    def test_define_slots_macro(self, parser_factory):
        """Test defineSlots macro with TypeScript."""
        parser = parser_factory.create_parser(Language.VUE)

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
        chunks = parser.parse_content(code, "DefineSlots.vue", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Verify defineSlots is present
        all_code = " ".join([c.code for c in chunks])
        assert "defineSlots" in all_code, "Should capture defineSlots macro"

    @pytest.mark.xfail(reason="Vue parser does not extract top-level macro calls")
    def test_define_options_macro(self, parser_factory):
        """Test defineOptions macro for component options."""
        parser = parser_factory.create_parser(Language.VUE)

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
        chunks = parser.parse_content(code, "DefineOptions.vue", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Verify defineOptions is present
        all_code = " ".join([c.code for c in chunks])
        assert "defineOptions" in all_code, "Should capture defineOptions macro"

    @pytest.mark.xfail(reason="Vue parser does not extract const assignments with macros")
    def test_define_props_runtime(self, parser_factory):
        """Test defineProps with runtime declaration."""
        parser = parser_factory.create_parser(Language.VUE)

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
        chunks = parser.parse_content(code, "DefinePropsRuntime.vue", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        all_code = " ".join([c.code for c in chunks])
        assert "defineProps" in all_code, "Should capture defineProps"

    @pytest.mark.xfail(reason="Vue parser does not extract const assignments with macros")
    def test_define_emits_with_validation(self, parser_factory):
        """Test defineEmits with validation functions."""
        parser = parser_factory.create_parser(Language.VUE)

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
        chunks = parser.parse_content(code, "DefineEmitsValidation.vue", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        all_code = " ".join([c.code for c in chunks])
        assert "defineEmits" in all_code, "Should capture defineEmits"

    @pytest.mark.xfail(reason="Vue parser does not extract top-level macro calls")
    def test_define_expose_macro(self, parser_factory):
        """Test defineExpose macro."""
        parser = parser_factory.create_parser(Language.VUE)

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
        chunks = parser.parse_content(code, "DefineExpose.vue", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        all_code = " ".join([c.code for c in chunks])
        assert "defineExpose" in all_code, "Should capture defineExpose"


# =============================================================================
# VUE OPTIONS API LIFECYCLE
# =============================================================================


class TestVueOptionsAPILifecycle:
    """Test Vue Options API lifecycle hooks."""

    def test_options_api_lifecycle_hooks(self, parser_factory):
        """Test Options API with lifecycle hooks."""
        parser = parser_factory.create_parser(Language.VUE)

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
        chunks = parser.parse_content(code, "LifecycleComponent.vue", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Should capture the export default object
        all_code = " ".join([c.code for c in chunks])
        assert "export default" in all_code or "created" in all_code, \
            "Should capture Options API component"

    def test_options_api_computed_watch(self, parser_factory):
        """Test Options API computed properties and watchers."""
        parser = parser_factory.create_parser(Language.VUE)

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
        chunks = parser.parse_content(code, "ComputedWatch.vue", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        all_code = " ".join([c.code for c in chunks])
        assert "computed" in all_code or "watch" in all_code, \
            "Should capture computed/watch properties"

    def test_options_api_mixins(self, parser_factory):
        """Test Options API with mixins."""
        parser = parser_factory.create_parser(Language.VUE)

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
        chunks = parser.parse_content(code, "MixinsComponent.vue", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"


# =============================================================================
# VUE COMPOSITION API ADVANCED
# =============================================================================


class TestVueCompositionAPIAdvanced:
    """Test advanced Composition API patterns."""

    def test_composable_with_typescript(self, parser_factory):
        """Test composable function with TypeScript."""
        parser = parser_factory.create_parser(Language.VUE)

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
        chunks = parser.parse_content(code, "ComposableTS.vue", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Should find interface and functions
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) > 0, "Should extract User interface"

    def test_provide_inject_typed(self, parser_factory):
        """Test provide/inject with TypeScript types."""
        parser = parser_factory.create_parser(Language.VUE)

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
        chunks = parser.parse_content(code, "ProvideInject.vue", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) > 0, "Should extract AppConfig interface"

    def test_reactive_transform(self, parser_factory):
        """Test Vue reactivity transform patterns."""
        parser = parser_factory.create_parser(Language.VUE)

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
        chunks = parser.parse_content(code, "ReactiveTransform.vue", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) > 0, "Should extract State interface"


# =============================================================================
# VUE EDGE CASES
# =============================================================================


class TestVueEdgeCases:
    """Test edge cases in Vue SFC parsing."""

    def test_multiple_script_blocks(self, parser_factory):
        """Test SFC with both script and script setup."""
        parser = parser_factory.create_parser(Language.VUE)

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
        chunks = parser.parse_content(code, "HybridComponent.vue", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"

    def test_empty_script_setup(self, parser_factory):
        """Test SFC with empty script setup."""
        parser = parser_factory.create_parser(Language.VUE)

        code = """<script setup>
</script>

<template>
    <div>Static content</div>
</template>
"""
        chunks = parser.parse_content(code, "EmptyScript.vue", FileId(1))

        # Should at least extract template
        assert len(chunks) >= 1, "Should extract at least template chunk"

    def test_script_with_type_imports(self, parser_factory):
        """Test script with type-only imports."""
        parser = parser_factory.create_parser(Language.VUE)

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
        chunks = parser.parse_content(code, "TypeImports.vue", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) > 0, "Should extract Props interface"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
