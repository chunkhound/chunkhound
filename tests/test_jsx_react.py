"""JSX/React pattern parsing tests.

Tests JSX (JavaScript with JSX) and TSX (TypeScript with JSX) parsing
for React components, hooks, and JSX elements.

This is a critical test file - JSX/React patterns had 0% coverage before
this file was created.
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
# REACT COMPONENTS - JSX
# =============================================================================


class TestJSXFunctionComponents:
    """Test function component extraction in JSX."""

    def test_function_component(self, parser_factory):
        """Test basic function component extraction."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function App() {
    return <div>Hello World</div>
}
"""
        chunks = parser.parse_content(code, "App.jsx", FileId(1))

        # Should extract the function component
        assert len(chunks) > 0, "Should extract at least one chunk"
        function_chunks = [c for c in chunks if "App" in c.symbol]
        assert len(function_chunks) > 0, "Should find App function component"

    def test_arrow_function_component(self, parser_factory):
        """Test arrow function component extraction."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
const Greeting = () => {
    return <h1>Hello</h1>
}
"""
        chunks = parser.parse_content(code, "Greeting.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        greeting_chunks = [c for c in chunks if "Greeting" in c.code]
        assert len(greeting_chunks) > 0, "Should find Greeting component"

    def test_arrow_function_implicit_return(self, parser_factory):
        """Test arrow function with implicit return."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
const Button = () => <button>Click me</button>
"""
        chunks = parser.parse_content(code, "Button.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        button_chunks = [c for c in chunks if "Button" in c.code]
        assert len(button_chunks) > 0, "Should find Button component"

    def test_component_with_props(self, parser_factory):
        """Test component with props parameter."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function UserCard(props) {
    return (
        <div className="user-card">
            <h2>{props.name}</h2>
            <p>{props.email}</p>
        </div>
    )
}
"""
        chunks = parser.parse_content(code, "UserCard.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        card_chunks = [c for c in chunks if "UserCard" in c.symbol]
        assert len(card_chunks) > 0, "Should find UserCard component"

    def test_component_with_destructured_props(self, parser_factory):
        """Test component with destructured props."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function Profile({ name, age, email }) {
    return (
        <div>
            <span>{name}</span>
            <span>{age}</span>
            <span>{email}</span>
        </div>
    )
}
"""
        chunks = parser.parse_content(code, "Profile.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        profile_chunks = [c for c in chunks if "Profile" in c.symbol]
        assert len(profile_chunks) > 0, "Should find Profile component"


class TestJSXClassComponents:
    """Test class component extraction in JSX."""

    @pytest.mark.xfail(reason="JSX parser does not extract classes - only functions are extracted")
    def test_class_component(self, parser_factory):
        """Test basic class component extraction.

        NOTE: Currently JSX parser only extracts functions, not classes.
        This test documents expected behavior for future enhancement.
        """
        parser = parser_factory.create_parser(Language.JSX)

        code = """
class Counter extends React.Component {
    constructor(props) {
        super(props);
        this.state = { count: 0 };
    }

    render() {
        return <div>{this.state.count}</div>
    }
}
"""
        chunks = parser.parse_content(code, "Counter.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, "Should find Counter class"
        assert any("Counter" in c.symbol for c in class_chunks), \
            "Class name should be captured"

    def test_class_component_with_component_import(self, parser_factory):
        """Test class component extending Component directly."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
import { Component } from 'react';

class App extends Component {
    render() {
        return <div>Hello</div>
    }
}
"""
        chunks = parser.parse_content(code, "App.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Should find import and class
        class_chunks = [c for c in chunks if "App" in c.code and "class" in c.code.lower()]
        assert len(class_chunks) > 0, "Should find App class component"

    def test_pure_component(self, parser_factory):
        """Test PureComponent class."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
class OptimizedList extends React.PureComponent {
    render() {
        return (
            <ul>
                {this.props.items.map(item => <li key={item.id}>{item.name}</li>)}
            </ul>
        )
    }
}
"""
        chunks = parser.parse_content(code, "OptimizedList.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        class_chunks = [c for c in chunks if "OptimizedList" in c.code]
        assert len(class_chunks) > 0, "Should find OptimizedList PureComponent"


class TestJSXHigherOrderComponents:
    """Test higher-order component patterns in JSX."""

    @pytest.mark.xfail(reason="JSX parser does not extract const with React.memo - inner function not detected")
    def test_memo_component(self, parser_factory):
        """Test React.memo component.

        NOTE: Parser doesn't extract the inner function inside React.memo.
        This documents expected behavior for future enhancement.
        """
        parser = parser_factory.create_parser(Language.JSX)

        code = """
const MemoizedCounter = React.memo(function Counter({ count }) {
    return <span>{count}</span>
})
"""
        chunks = parser.parse_content(code, "MemoizedCounter.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        memo_chunks = [c for c in chunks if "MemoizedCounter" in c.code or "Counter" in c.code]
        assert len(memo_chunks) > 0, "Should find memo component"

    @pytest.mark.xfail(reason="JSX parser does not extract const with React.forwardRef arrow function")
    def test_forward_ref_component(self, parser_factory):
        """Test React.forwardRef component.

        NOTE: Parser doesn't extract arrow functions inside React.forwardRef.
        This documents expected behavior for future enhancement.
        """
        parser = parser_factory.create_parser(Language.JSX)

        code = """
const FancyInput = React.forwardRef((props, ref) => (
    <input ref={ref} className="fancy-input" {...props} />
))
"""
        chunks = parser.parse_content(code, "FancyInput.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        ref_chunks = [c for c in chunks if "FancyInput" in c.code]
        assert len(ref_chunks) > 0, "Should find forwardRef component"

    @pytest.mark.xfail(reason="JSX parser does not extract const with React.lazy")
    def test_lazy_component(self, parser_factory):
        """Test React.lazy component.

        NOTE: Parser doesn't extract const assignments with React.lazy.
        This documents expected behavior for future enhancement.
        """
        parser = parser_factory.create_parser(Language.JSX)

        code = """
const LazyDashboard = React.lazy(() => import('./Dashboard'))
"""
        chunks = parser.parse_content(code, "LazyDashboard.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        lazy_chunks = [c for c in chunks if "LazyDashboard" in c.code]
        assert len(lazy_chunks) > 0, "Should find lazy component"


# =============================================================================
# JSX ELEMENTS
# =============================================================================


class TestJSXElements:
    """Test JSX element patterns."""

    def test_self_closing_element(self, parser_factory):
        """Test self-closing JSX elements."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function Icon() {
    return <img src="/icon.png" alt="icon" />
}
"""
        chunks = parser.parse_content(code, "Icon.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Verify the JSX content is preserved
        assert any("<img" in c.code for c in chunks), "Should preserve self-closing JSX"

    def test_element_with_children(self, parser_factory):
        """Test JSX elements with children."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function Container() {
    return (
        <div className="container">
            <Header />
            <Main />
            <Footer />
        </div>
    )
}
"""
        chunks = parser.parse_content(code, "Container.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        container_chunks = [c for c in chunks if "Container" in c.code]
        assert len(container_chunks) > 0, "Should find Container component"
        # Verify nested elements are preserved
        assert any("Header" in c.code and "Footer" in c.code for c in chunks), \
            "Should preserve nested JSX elements"

    def test_fragment_shorthand(self, parser_factory):
        """Test Fragment shorthand syntax."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function Wrapper() {
    return (
        <>
            <First />
            <Second />
        </>
    )
}
"""
        chunks = parser.parse_content(code, "Wrapper.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Fragment syntax should be preserved
        assert any("<>" in c.code for c in chunks), "Should preserve fragment shorthand"

    def test_named_fragment(self, parser_factory):
        """Test React.Fragment named syntax."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function List({ items }) {
    return (
        <React.Fragment>
            {items.map(item => (
                <li key={item.id}>{item.name}</li>
            ))}
        </React.Fragment>
    )
}
"""
        chunks = parser.parse_content(code, "List.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("React.Fragment" in c.code for c in chunks), \
            "Should preserve React.Fragment"


class TestJSXAttributes:
    """Test JSX attribute patterns."""

    def test_string_attribute(self, parser_factory):
        """Test string JSX attributes."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function Input() {
    return <input type="text" placeholder="Enter name" />
}
"""
        chunks = parser.parse_content(code, "Input.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any('type="text"' in c.code for c in chunks), \
            "Should preserve string attributes"

    def test_expression_attribute(self, parser_factory):
        """Test expression JSX attributes."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function Button({ onClick, disabled }) {
    return <button onClick={onClick} disabled={disabled}>Click</button>
}
"""
        chunks = parser.parse_content(code, "Button.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("onClick={onClick}" in c.code for c in chunks), \
            "Should preserve expression attributes"

    def test_spread_attributes(self, parser_factory):
        """Test spread JSX attributes."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function CustomInput(props) {
    return <input className="custom" {...props} />
}
"""
        chunks = parser.parse_content(code, "CustomInput.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("{...props}" in c.code for c in chunks), \
            "Should preserve spread attributes"

    def test_boolean_attribute(self, parser_factory):
        """Test boolean JSX attributes."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function Checkbox() {
    return <input type="checkbox" disabled checked />
}
"""
        chunks = parser.parse_content(code, "Checkbox.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Boolean attributes should be preserved
        assert any("disabled" in c.code and "checked" in c.code for c in chunks), \
            "Should preserve boolean attributes"


# =============================================================================
# REACT HOOKS
# =============================================================================


class TestReactHooks:
    """Test React hooks extraction."""

    def test_use_state_hook(self, parser_factory):
        """Test useState hook."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function Counter() {
    const [count, setCount] = useState(0)
    return (
        <div>
            <p>{count}</p>
            <button onClick={() => setCount(count + 1)}>Increment</button>
        </div>
    )
}
"""
        chunks = parser.parse_content(code, "Counter.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        counter_chunks = [c for c in chunks if "Counter" in c.symbol or "Counter" in c.code]
        assert len(counter_chunks) > 0, "Should find Counter component"
        assert any("useState" in c.code for c in chunks), "Should preserve useState hook"

    def test_use_effect_hook(self, parser_factory):
        """Test useEffect hook."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function DataFetcher({ url }) {
    const [data, setData] = useState(null)

    useEffect(() => {
        fetch(url)
            .then(res => res.json())
            .then(setData)
    }, [url])

    return <div>{data ? JSON.stringify(data) : 'Loading...'}</div>
}
"""
        chunks = parser.parse_content(code, "DataFetcher.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("useEffect" in c.code for c in chunks), "Should preserve useEffect hook"

    def test_use_context_hook(self, parser_factory):
        """Test useContext hook."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function ThemedButton() {
    const theme = useContext(ThemeContext)
    return <button style={{ background: theme.primary }}>Themed</button>
}
"""
        chunks = parser.parse_content(code, "ThemedButton.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("useContext" in c.code for c in chunks), "Should preserve useContext hook"

    def test_use_reducer_hook(self, parser_factory):
        """Test useReducer hook."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function TodoApp() {
    const [state, dispatch] = useReducer(todoReducer, { todos: [] })
    return (
        <div>
            <button onClick={() => dispatch({ type: 'ADD', text: 'New' })}>
                Add
            </button>
        </div>
    )
}
"""
        chunks = parser.parse_content(code, "TodoApp.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("useReducer" in c.code for c in chunks), "Should preserve useReducer hook"

    def test_use_callback_hook(self, parser_factory):
        """Test useCallback hook."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function SearchBox({ onSearch }) {
    const handleSearch = useCallback((term) => {
        onSearch(term)
    }, [onSearch])

    return <input onChange={(e) => handleSearch(e.target.value)} />
}
"""
        chunks = parser.parse_content(code, "SearchBox.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("useCallback" in c.code for c in chunks), "Should preserve useCallback hook"

    def test_use_memo_hook(self, parser_factory):
        """Test useMemo hook."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function ExpensiveList({ items, filter }) {
    const filteredItems = useMemo(() => {
        return items.filter(item => item.name.includes(filter))
    }, [items, filter])

    return <ul>{filteredItems.map(i => <li key={i.id}>{i.name}</li>)}</ul>
}
"""
        chunks = parser.parse_content(code, "ExpensiveList.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("useMemo" in c.code for c in chunks), "Should preserve useMemo hook"

    def test_use_ref_hook(self, parser_factory):
        """Test useRef hook."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function FocusInput() {
    const inputRef = useRef(null)

    const handleClick = () => {
        inputRef.current.focus()
    }

    return (
        <div>
            <input ref={inputRef} />
            <button onClick={handleClick}>Focus</button>
        </div>
    )
}
"""
        chunks = parser.parse_content(code, "FocusInput.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("useRef" in c.code for c in chunks), "Should preserve useRef hook"


class TestCustomHooks:
    """Test custom hook extraction."""

    def test_custom_hook_definition(self, parser_factory):
        """Test custom hook definition."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function useCounter(initialValue = 0) {
    const [count, setCount] = useState(initialValue)

    const increment = () => setCount(c => c + 1)
    const decrement = () => setCount(c => c - 1)
    const reset = () => setCount(initialValue)

    return { count, increment, decrement, reset }
}
"""
        chunks = parser.parse_content(code, "useCounter.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        hook_chunks = [c for c in chunks if "useCounter" in c.symbol or "useCounter" in c.code]
        assert len(hook_chunks) > 0, "Should find custom hook definition"

    def test_custom_hook_arrow_function(self, parser_factory):
        """Test custom hook as arrow function."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
const useLocalStorage = (key, initialValue) => {
    const [storedValue, setStoredValue] = useState(() => {
        const item = window.localStorage.getItem(key)
        return item ? JSON.parse(item) : initialValue
    })

    const setValue = (value) => {
        setStoredValue(value)
        window.localStorage.setItem(key, JSON.stringify(value))
    }

    return [storedValue, setValue]
}
"""
        chunks = parser.parse_content(code, "useLocalStorage.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        hook_chunks = [c for c in chunks if "useLocalStorage" in c.code]
        assert len(hook_chunks) > 0, "Should find custom hook arrow function"

    def test_custom_hook_usage(self, parser_factory):
        """Test custom hook usage in component."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function Dashboard() {
    const { data, loading, error } = useFetch('/api/dashboard')
    const [theme] = useLocalStorage('theme', 'light')
    const windowSize = useWindowSize()

    if (loading) return <div>Loading...</div>
    if (error) return <div>Error: {error.message}</div>

    return (
        <div className={theme}>
            <p>Window: {windowSize.width}x{windowSize.height}</p>
            <pre>{JSON.stringify(data)}</pre>
        </div>
    )
}
"""
        chunks = parser.parse_content(code, "Dashboard.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Multiple custom hooks should be preserved
        dashboard_chunks = [c for c in chunks if "useFetch" in c.code or "useLocalStorage" in c.code]
        assert len(dashboard_chunks) > 0, "Should preserve custom hook usage"


# =============================================================================
# TSX - TYPESCRIPT WITH JSX
# =============================================================================


class TestTSXTypedComponents:
    """Test TypeScript + React (TSX) patterns."""

    def test_fc_with_props_type(self, parser_factory):
        """Test React.FC with props type."""
        parser = parser_factory.create_parser(Language.TSX)

        code = """
interface Props {
    name: string;
    age: number;
}

const Greeting: React.FC<Props> = ({ name, age }) => {
    return <p>Hello {name}, you are {age} years old</p>
}
"""
        chunks = parser.parse_content(code, "Greeting.tsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Should find both interface and component
        greeting_chunks = [c for c in chunks if "Greeting" in c.code]
        assert len(greeting_chunks) > 0, "Should find typed FC component"
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) > 0, "Should find Props interface"

    def test_generic_component(self, parser_factory):
        """Test generic component."""
        parser = parser_factory.create_parser(Language.TSX)

        code = """
interface ListProps<T> {
    items: T[];
    renderItem: (item: T) => React.ReactNode;
}

function List<T>({ items, renderItem }: ListProps<T>) {
    return (
        <ul>
            {items.map((item, index) => (
                <li key={index}>{renderItem(item)}</li>
            ))}
        </ul>
    )
}
"""
        chunks = parser.parse_content(code, "List.tsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        list_chunks = [c for c in chunks if "List" in c.symbol or ("List" in c.code and "function" in c.code)]
        assert len(list_chunks) > 0, "Should find generic component"

    def test_typed_use_state(self, parser_factory):
        """Test typed useState hook."""
        parser = parser_factory.create_parser(Language.TSX)

        code = """
interface User {
    id: number;
    name: string;
    email: string;
}

function UserProfile() {
    const [user, setUser] = useState<User | null>(null)
    const [loading, setLoading] = useState<boolean>(true)

    return (
        <div>
            {loading ? <p>Loading...</p> : <p>{user?.name}</p>}
        </div>
    )
}
"""
        chunks = parser.parse_content(code, "UserProfile.tsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("useState<User" in c.code for c in chunks), "Should preserve typed useState"

    def test_typed_use_ref(self, parser_factory):
        """Test typed useRef hook."""
        parser = parser_factory.create_parser(Language.TSX)

        code = """
function VideoPlayer() {
    const videoRef = useRef<HTMLVideoElement>(null)
    const containerRef = useRef<HTMLDivElement>(null)

    const play = () => {
        videoRef.current?.play()
    }

    return (
        <div ref={containerRef}>
            <video ref={videoRef} />
            <button onClick={play}>Play</button>
        </div>
    )
}
"""
        chunks = parser.parse_content(code, "VideoPlayer.tsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("useRef<HTML" in c.code for c in chunks), "Should preserve typed useRef"

    def test_event_handler_types(self, parser_factory):
        """Test event handler types."""
        parser = parser_factory.create_parser(Language.TSX)

        code = """
interface FormProps {
    onSubmit: (data: FormData) => void;
}

const Form: React.FC<FormProps> = ({ onSubmit }) => {
    const handleClick: React.MouseEventHandler<HTMLButtonElement> = (event) => {
        event.preventDefault()
    }

    const handleChange: React.ChangeEventHandler<HTMLInputElement> = (event) => {
        console.log(event.target.value)
    }

    return (
        <form>
            <input onChange={handleChange} />
            <button onClick={handleClick}>Submit</button>
        </form>
    )
}
"""
        chunks = parser.parse_content(code, "Form.tsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("MouseEventHandler" in c.code for c in chunks), \
            "Should preserve event handler types"

    def test_props_type_alias(self, parser_factory):
        """Test props with type alias."""
        parser = parser_factory.create_parser(Language.TSX)

        code = """
type ButtonVariant = 'primary' | 'secondary' | 'danger';
type ButtonSize = 'sm' | 'md' | 'lg';

type ButtonProps = {
    variant: ButtonVariant;
    size?: ButtonSize;
    children: React.ReactNode;
    onClick?: () => void;
}

const Button: React.FC<ButtonProps> = ({ variant, size = 'md', children, onClick }) => {
    return (
        <button className={`btn btn-${variant} btn-${size}`} onClick={onClick}>
            {children}
        </button>
    )
}
"""
        chunks = parser.parse_content(code, "Button.tsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Should find type aliases
        type_chunks = [c for c in chunks if c.chunk_type == ChunkType.TYPE_ALIAS]
        assert len(type_chunks) > 0, "Should find type aliases"


class TestTSXAdvancedPatterns:
    """Test advanced TSX patterns."""

    def test_typed_custom_hook(self, parser_factory):
        """Test typed custom hook."""
        parser = parser_factory.create_parser(Language.TSX)

        code = """
interface UseApiReturn<T> {
    data: T | null;
    loading: boolean;
    error: Error | null;
    refetch: () => void;
}

function useApi<T>(url: string): UseApiReturn<T> {
    const [data, setData] = useState<T | null>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<Error | null>(null)

    const refetch = useCallback(() => {
        setLoading(true)
        fetch(url)
            .then(res => res.json())
            .then(setData)
            .catch(setError)
            .finally(() => setLoading(false))
    }, [url])

    useEffect(() => {
        refetch()
    }, [refetch])

    return { data, loading, error, refetch }
}
"""
        chunks = parser.parse_content(code, "useApi.tsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        hook_chunks = [c for c in chunks if "useApi" in c.symbol or "useApi" in c.code]
        assert len(hook_chunks) > 0, "Should find typed custom hook"

    def test_typed_forward_ref(self, parser_factory):
        """Test typed forwardRef component."""
        parser = parser_factory.create_parser(Language.TSX)

        code = """
interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
    label: string;
}

const LabeledInput = React.forwardRef<HTMLInputElement, InputProps>(
    ({ label, ...props }, ref) => (
        <label>
            {label}
            <input ref={ref} {...props} />
        </label>
    )
)
"""
        chunks = parser.parse_content(code, "LabeledInput.tsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Interface should be extracted, even if forwardRef isn't fully parsed
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) > 0, "Should extract InputProps interface"

    def test_typed_memo_component(self, parser_factory):
        """Test typed memo component."""
        parser = parser_factory.create_parser(Language.TSX)

        code = """
interface ItemProps {
    id: number;
    name: string;
    selected: boolean;
    onSelect: (id: number) => void;
}

const Item = React.memo<ItemProps>(({ id, name, selected, onSelect }) => (
    <li
        onClick={() => onSelect(id)}
        className={selected ? 'selected' : ''}
    >
        {name}
    </li>
))
"""
        chunks = parser.parse_content(code, "Item.tsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Interface should be extracted, even if memo isn't fully parsed
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) > 0, "Should extract ItemProps interface"

    def test_context_with_types(self, parser_factory):
        """Test typed React context."""
        parser = parser_factory.create_parser(Language.TSX)

        code = """
interface AuthContextValue {
    user: User | null;
    login: (credentials: Credentials) => Promise<void>;
    logout: () => void;
    isAuthenticated: boolean;
}

const AuthContext = React.createContext<AuthContextValue | undefined>(undefined)

function useAuth(): AuthContextValue {
    const context = useContext(AuthContext)
    if (context === undefined) {
        throw new Error('useAuth must be used within AuthProvider')
    }
    return context
}

const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [user, setUser] = useState<User | null>(null)

    const login = async (credentials: Credentials) => {
        const user = await api.login(credentials)
        setUser(user)
    }

    const logout = () => {
        setUser(null)
    }

    const value: AuthContextValue = {
        user,
        login,
        logout,
        isAuthenticated: user !== null,
    }

    return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
"""
        chunks = parser.parse_content(code, "AuthContext.tsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Should find multiple definitions
        assert any("AuthContextValue" in c.code for c in chunks), \
            "Should preserve context interface"
        assert any("useAuth" in c.code for c in chunks), \
            "Should preserve custom hook"
        assert any("AuthProvider" in c.code for c in chunks), \
            "Should preserve provider component"


# =============================================================================
# COMPLEX PATTERNS
# =============================================================================


class TestComplexJSXPatterns:
    """Test complex JSX patterns and combinations."""

    def test_multiple_components_in_file(self, parser_factory):
        """Test file with multiple components.

        NOTE: Parser currently extracts only the first function and includes
        all subsequent functions in its content. This documents current behavior.
        """
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function Header() {
    return <header><h1>Title</h1></header>
}

function Main({ children }) {
    return <main>{children}</main>
}

function Footer() {
    return <footer>&copy; 2024</footer>
}

function Layout({ children }) {
    return (
        <div className="layout">
            <Header />
            <Main>{children}</Main>
            <Footer />
        </div>
    )
}
"""
        chunks = parser.parse_content(code, "Layout.jsx", FileId(1))

        # Current behavior: all functions are in the first chunk
        assert len(chunks) >= 1, "Should extract at least one chunk"
        # All component names should be in the code
        component_names = ["Header", "Main", "Footer", "Layout"]
        for name in component_names:
            assert any(name in c.code for c in chunks), \
                f"Should find {name} in code"

    def test_component_with_all_hooks(self, parser_factory):
        """Test component using multiple hooks."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function ComplexComponent({ initialCount, onUpdate }) {
    const [count, setCount] = useState(initialCount)
    const [name, setName] = useState('')
    const prevCount = useRef(count)
    const theme = useContext(ThemeContext)

    const memoizedValue = useMemo(() => {
        return computeExpensiveValue(count)
    }, [count])

    const handleClick = useCallback(() => {
        setCount(c => c + 1)
        onUpdate(count + 1)
    }, [count, onUpdate])

    useEffect(() => {
        prevCount.current = count
    }, [count])

    useLayoutEffect(() => {
        document.title = `Count: ${count}`
    }, [count])

    return (
        <div style={{ background: theme.background }}>
            <p>Count: {count}, Previous: {prevCount.current}</p>
            <p>Expensive: {memoizedValue}</p>
            <input value={name} onChange={e => setName(e.target.value)} />
            <button onClick={handleClick}>Increment</button>
        </div>
    )
}
"""
        chunks = parser.parse_content(code, "ComplexComponent.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Verify component with all hooks is captured
        complex_chunk = [c for c in chunks if "ComplexComponent" in c.symbol]
        assert len(complex_chunk) > 0, "Should find ComplexComponent"
        # Hooks should be preserved in the code
        hooks = ["useState", "useRef", "useContext", "useMemo", "useCallback", "useEffect", "useLayoutEffect"]
        for hook in hooks:
            assert any(hook in c.code for c in chunks), f"Should preserve {hook}"

    def test_nested_jsx_expressions(self, parser_factory):
        """Test nested JSX expressions."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function ConditionalList({ items, loading, error }) {
    return (
        <div>
            {loading && <Spinner />}
            {error && <ErrorMessage message={error.message} />}
            {!loading && !error && (
                <ul>
                    {items.length > 0 ? (
                        items.map(item => (
                            <li key={item.id}>
                                {item.active ? (
                                    <ActiveBadge />
                                ) : (
                                    <InactiveBadge />
                                )}
                                <span>{item.name}</span>
                            </li>
                        ))
                    ) : (
                        <EmptyState />
                    )}
                </ul>
            )}
        </div>
    )
}
"""
        chunks = parser.parse_content(code, "ConditionalList.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        list_chunks = [c for c in chunks if "ConditionalList" in c.symbol]
        assert len(list_chunks) > 0, "Should find ConditionalList"
        # Complex JSX should be preserved
        assert any("items.map" in c.code for c in chunks), "Should preserve map expression"


class TestJSXImportsExports:
    """Test import/export patterns in JSX."""

    def test_react_imports(self, parser_factory):
        """Test React imports.

        NOTE: JSX parser extracts functions, and imports are included in the
        function chunk's code. This is the current behavior.
        """
        parser = parser_factory.create_parser(Language.JSX)

        code = """
import React, { useState, useEffect, useCallback } from 'react'
import { Button } from './Button'
import styles from './App.module.css'

function App() {
    const [count, setCount] = useState(0)
    return <Button onClick={() => setCount(c => c + 1)}>{count}</Button>
}

export default App
"""
        chunks = parser.parse_content(code, "App.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        # App function should be found
        app_chunks = [c for c in chunks if "App" in c.symbol or "App" in c.code]
        assert len(app_chunks) > 0, "Should find App function"

    def test_named_exports(self, parser_factory):
        """Test named exports of components.

        NOTE: Parser extracts the first export function, and other exports
        are included in its content. This documents current behavior.
        """
        parser = parser_factory.create_parser(Language.JSX)

        code = """
export function Card({ children }) {
    return <div className="card">{children}</div>
}

export const CardHeader = ({ title }) => <h2>{title}</h2>

export const CardBody = ({ children }) => <div className="body">{children}</div>
"""
        chunks = parser.parse_content(code, "Card.jsx", FileId(1))

        assert len(chunks) >= 1, "Should extract at least one chunk"
        # All exports should be present in the code
        for name in ["Card", "CardHeader", "CardBody"]:
            assert any(name in c.code for c in chunks), f"Should find {name} in code"

    def test_default_export_component(self, parser_factory):
        """Test default export component."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
export default function HomePage() {
    return (
        <div>
            <h1>Welcome</h1>
            <p>This is the home page</p>
        </div>
    )
}
"""
        chunks = parser.parse_content(code, "HomePage.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        home_chunks = [c for c in chunks if "HomePage" in c.symbol or "HomePage" in c.code]
        assert len(home_chunks) > 0, "Should find default export component"


# =============================================================================
# CROSS-LANGUAGE CONSISTENCY
# =============================================================================


class TestJSXTSXConsistency:
    """Test that same patterns produce consistent results in JSX and TSX."""

    @pytest.mark.parametrize("language,extension", [
        (Language.JSX, "jsx"),
        (Language.TSX, "tsx"),
    ])
    def test_function_component_consistency(self, parser_factory, language, extension):
        """Test function component is extracted consistently."""
        parser = parser_factory.create_parser(language)

        code = """
function Welcome({ name }) {
    return <h1>Hello, {name}</h1>
}
"""
        chunks = parser.parse_content(code, f"Welcome.{extension}", FileId(1))

        assert len(chunks) > 0, f"Should extract chunks for {language}"
        welcome_chunks = [c for c in chunks if "Welcome" in c.symbol or "Welcome" in c.code]
        assert len(welcome_chunks) > 0, f"Should find Welcome in {language}"

    def test_class_component_consistency_tsx(self, parser_factory):
        """Test class component is extracted in TSX."""
        parser = parser_factory.create_parser(Language.TSX)

        code = """
class App extends React.Component {
    render() {
        return <div>Hello</div>
    }
}
"""
        chunks = parser.parse_content(code, "App.tsx", FileId(1))

        assert len(chunks) > 0, "Should extract chunks for TSX"
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, "Should find class in TSX"

    @pytest.mark.xfail(reason="JSX parser does not extract class components")
    def test_class_component_consistency_jsx(self, parser_factory):
        """Test class component is extracted in JSX.

        NOTE: Currently JSX parser only extracts functions, not classes.
        """
        parser = parser_factory.create_parser(Language.JSX)

        code = """
class App extends React.Component {
    render() {
        return <div>Hello</div>
    }
}
"""
        chunks = parser.parse_content(code, "App.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract chunks for JSX"
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) > 0, "Should find class in JSX"

    @pytest.mark.parametrize("language,extension", [
        (Language.JSX, "jsx"),
        (Language.TSX, "tsx"),
    ])
    def test_hooks_consistency(self, parser_factory, language, extension):
        """Test hooks are preserved consistently."""
        parser = parser_factory.create_parser(language)

        code = """
function Counter() {
    const [count, setCount] = useState(0)
    useEffect(() => {
        console.log(count)
    }, [count])
    return <button onClick={() => setCount(count + 1)}>{count}</button>
}
"""
        chunks = parser.parse_content(code, f"Counter.{extension}", FileId(1))

        assert len(chunks) > 0, f"Should extract chunks for {language}"
        assert any("useState" in c.code for c in chunks), f"useState should be in {language}"
        assert any("useEffect" in c.code for c in chunks), f"useEffect should be in {language}"


# =============================================================================
# EDGE CASES
# =============================================================================


class TestJSXEdgeCases:
    """Test JSX edge cases and special patterns."""

    def test_empty_component(self, parser_factory):
        """Test empty component that returns null."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function EmptyComponent() {
    return null
}
"""
        chunks = parser.parse_content(code, "Empty.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("EmptyComponent" in c.symbol for c in chunks), \
            "Should find component returning null"

    def test_jsx_in_conditional(self, parser_factory):
        """Test JSX in conditional expression."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function Toggle({ isOn }) {
    return isOn ? <span>ON</span> : <span>OFF</span>
}
"""
        chunks = parser.parse_content(code, "Toggle.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"

    def test_jsx_comments(self, parser_factory):
        """Test JSX comments."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function CommentedComponent() {
    return (
        <div>
            {/* This is a JSX comment */}
            <span>Content</span>
            {/*
                Multi-line
                JSX comment
            */}
        </div>
    )
}
"""
        chunks = parser.parse_content(code, "Commented.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        # JSX comments should be preserved
        assert any("{/*" in c.code for c in chunks), "Should preserve JSX comments"

    def test_deeply_nested_jsx(self, parser_factory):
        """Test deeply nested JSX structure."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function DeepNest() {
    return (
        <div>
            <div>
                <div>
                    <div>
                        <div>
                            <span>Deep content</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
"""
        chunks = parser.parse_content(code, "DeepNest.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("Deep content" in c.code for c in chunks), \
            "Should preserve deeply nested content"

    def test_unicode_in_jsx(self, parser_factory):
        """Test unicode content in JSX."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function International() {
    return (
        <div>
            <p>English: Hello</p>
            <p>Spanish: Hola</p>
            <p>Japanese: こんにちは</p>
            <p>Emoji: 🎉🚀✨</p>
        </div>
    )
}
"""
        chunks = parser.parse_content(code, "International.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Unicode should be preserved
        assert any("こんにちは" in c.code for c in chunks), \
            "Should preserve Japanese text"


# =============================================================================
# ADDITIONAL TSX TYPED PATTERNS - Missing from specification
# =============================================================================


class TestTSXAdditionalTypedPatterns:
    """Additional TSX typed patterns from test specification."""

    def test_typed_use_layout_effect(self, parser_factory):
        """Test typed useLayoutEffect hook."""
        parser = parser_factory.create_parser(Language.TSX)

        code = """
function MeasureComponent() {
    const ref = useRef<HTMLDivElement>(null)

    useLayoutEffect(() => {
        if (ref.current) {
            const { width, height } = ref.current.getBoundingClientRect()
            console.log(width, height)
        }
    }, [])

    return <div ref={ref}>Measure me</div>
}
"""
        chunks = parser.parse_content(code, "Measure.tsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("useLayoutEffect" in c.code for c in chunks), "Should preserve useLayoutEffect hook"

    def test_typed_use_imperative_handle(self, parser_factory):
        """Test typed useImperativeHandle hook."""
        parser = parser_factory.create_parser(Language.TSX)

        code = """
interface InputHandle {
    focus: () => void;
    clear: () => void;
}

const CustomInput = React.forwardRef<InputHandle, {}>((props, ref) => {
    const inputRef = useRef<HTMLInputElement>(null)

    useImperativeHandle(ref, () => ({
        focus: () => inputRef.current?.focus(),
        clear: () => { if (inputRef.current) inputRef.current.value = '' }
    }))

    return <input ref={inputRef} {...props} />
})
"""
        chunks = parser.parse_content(code, "CustomInput.tsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Should find interface at minimum
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) > 0, "Should extract InputHandle interface"

    def test_fc_with_children_type(self, parser_factory):
        """Test React.FC with children type in props."""
        parser = parser_factory.create_parser(Language.TSX)

        code = """
interface CardProps {
    title: string;
    children: React.ReactNode;
}

const Card: React.FC<CardProps> = ({ title, children }) => (
    <div className="card">
        <h2>{title}</h2>
        <div className="content">{children}</div>
    </div>
)
"""
        chunks = parser.parse_content(code, "Card.tsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) > 0, "Should extract CardProps interface"
        assert any("ReactNode" in c.code for c in chunks), "Should preserve ReactNode type"

    def test_component_with_generic_props(self, parser_factory):
        """Test component with generic props pattern."""
        parser = parser_factory.create_parser(Language.TSX)

        code = """
interface SelectProps<T> {
    options: T[];
    value: T;
    onChange: (value: T) => void;
    getLabel: (item: T) => string;
}

function Select<T>({ options, value, onChange, getLabel }: SelectProps<T>) {
    return (
        <select value={String(value)} onChange={e => {
            const selected = options.find(o => String(o) === e.target.value)
            if (selected) onChange(selected)
        }}>
            {options.map(option => (
                <option key={String(option)} value={String(option)}>
                    {getLabel(option)}
                </option>
            ))}
        </select>
    )
}
"""
        chunks = parser.parse_content(code, "Select.tsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) > 0, "Should extract SelectProps generic interface"

    def test_typed_reducer_hook(self, parser_factory):
        """Test typed useReducer hook."""
        parser = parser_factory.create_parser(Language.TSX)

        code = """
interface State {
    count: number;
    step: number;
}

type Action =
    | { type: 'increment' }
    | { type: 'decrement' }
    | { type: 'setStep'; payload: number };

function reducer(state: State, action: Action): State {
    switch (action.type) {
        case 'increment':
            return { ...state, count: state.count + state.step }
        case 'decrement':
            return { ...state, count: state.count - state.step }
        case 'setStep':
            return { ...state, step: action.payload }
    }
}

function Counter() {
    const [state, dispatch] = useReducer(reducer, { count: 0, step: 1 })

    return (
        <div>
            <p>Count: {state.count}</p>
            <button onClick={() => dispatch({ type: 'increment' })}>+</button>
            <button onClick={() => dispatch({ type: 'decrement' })}>-</button>
        </div>
    )
}
"""
        chunks = parser.parse_content(code, "Counter.tsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Should find interface, type alias, and functions
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        type_chunks = [c for c in chunks if c.chunk_type == ChunkType.TYPE_ALIAS]
        assert len(interface_chunks) > 0 or len(type_chunks) > 0, "Should extract State/Action types"

    def test_typed_context_provider(self, parser_factory):
        """Test typed Context with Provider pattern."""
        parser = parser_factory.create_parser(Language.TSX)

        code = """
interface Theme {
    primaryColor: string;
    secondaryColor: string;
    fontSize: number;
}

const defaultTheme: Theme = {
    primaryColor: 'blue',
    secondaryColor: 'gray',
    fontSize: 14
}

const ThemeContext = React.createContext<Theme>(defaultTheme)

function useTheme(): Theme {
    return useContext(ThemeContext)
}

const ThemeProvider: React.FC<{ theme?: Theme; children: React.ReactNode }> = ({
    theme = defaultTheme,
    children
}) => (
    <ThemeContext.Provider value={theme}>
        {children}
    </ThemeContext.Provider>
)
"""
        chunks = parser.parse_content(code, "ThemeContext.tsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) > 0, "Should extract Theme interface"


# =============================================================================
# MISSING REACT HOOKS TESTS
# =============================================================================


class TestMissingReactHooks:
    """Test React hooks that were missing from original specification."""

    def test_use_memo_with_dependencies(self, parser_factory):
        """Test useMemo with dependencies array."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function FilteredList({ items, searchTerm }) {
    const filteredItems = useMemo(() => {
        return items.filter(item =>
            item.name.toLowerCase().includes(searchTerm.toLowerCase())
        )
    }, [items, searchTerm])

    return (
        <ul>
            {filteredItems.map(item => (
                <li key={item.id}>{item.name}</li>
            ))}
        </ul>
    )
}
"""
        chunks = parser.parse_content(code, "FilteredList.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("useMemo" in c.code for c in chunks), "Should preserve useMemo"
        assert any("[items, searchTerm]" in c.code for c in chunks), "Should preserve dependency array"

    def test_use_ref_for_mutable_value(self, parser_factory):
        """Test useRef for mutable value storage."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function Timer() {
    const intervalRef = useRef(null)
    const [count, setCount] = useState(0)

    const startTimer = () => {
        intervalRef.current = setInterval(() => {
            setCount(c => c + 1)
        }, 1000)
    }

    const stopTimer = () => {
        clearInterval(intervalRef.current)
    }

    return (
        <div>
            <p>Count: {count}</p>
            <button onClick={startTimer}>Start</button>
            <button onClick={stopTimer}>Stop</button>
        </div>
    )
}
"""
        chunks = parser.parse_content(code, "Timer.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("useRef(null)" in c.code for c in chunks), "Should preserve useRef"

    def test_use_layout_effect(self, parser_factory):
        """Test useLayoutEffect hook."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
function Tooltip({ text, targetRef }) {
    const tooltipRef = useRef(null)

    useLayoutEffect(() => {
        if (targetRef.current && tooltipRef.current) {
            const targetRect = targetRef.current.getBoundingClientRect()
            tooltipRef.current.style.left = `${targetRect.left}px`
            tooltipRef.current.style.top = `${targetRect.bottom}px`
        }
    }, [targetRef])

    return <div ref={tooltipRef} className="tooltip">{text}</div>
}
"""
        chunks = parser.parse_content(code, "Tooltip.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        assert any("useLayoutEffect" in c.code for c in chunks), "Should preserve useLayoutEffect"

    @pytest.mark.xfail(reason="JSX parser does not extract React.forwardRef arrow functions")
    def test_use_imperative_handle(self, parser_factory):
        """Test useImperativeHandle hook."""
        parser = parser_factory.create_parser(Language.JSX)

        code = """
const FancyInput = React.forwardRef((props, ref) => {
    const inputRef = useRef(null)

    useImperativeHandle(ref, () => ({
        focus: () => {
            inputRef.current.focus()
        },
        scrollIntoView: () => {
            inputRef.current.scrollIntoView()
        }
    }))

    return <input ref={inputRef} {...props} />
})
"""
        chunks = parser.parse_content(code, "FancyInput.jsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"
        # Should capture some content with useImperativeHandle
        all_code = " ".join([c.code for c in chunks])
        assert "useImperativeHandle" in all_code or "forwardRef" in all_code, \
            "Should capture forwardRef or useImperativeHandle"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
