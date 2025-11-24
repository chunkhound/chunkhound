"""TSX (TypeScript + React) pattern parsing tests.

Tests TypeScript React (TSX) specific patterns including:
- Typed functional components (FC<Props>)
- Generic components
- Props interfaces and type aliases
- Typed hooks (useState<T>, useRef<T>, etc.)
- Event handler types
- Custom typed hooks

This test file focuses on TypeScript-specific React patterns that add
type safety on top of standard JSX patterns.
"""

import pytest
from pathlib import Path

from chunkhound.core.types.common import ChunkType, FileId, Language
from chunkhound.parsers.parser_factory import ParserFactory


@pytest.fixture
def parser_factory():
    """Create a parser factory instance."""
    return ParserFactory()


# =============================================================================
# TYPED COMPONENTS
# =============================================================================


class TestTypedComponents:
    """Test typed React component patterns in TSX."""

    def test_fc_with_props_type(self, parser_factory):
        """Test React.FC with props type annotation.

        Pattern: const Comp: React.FC<Props> = (props) => <div>{props.name}</div>
        """
        parser = parser_factory.create_parser(Language.TSX)

        code = """
interface GreetingProps {
    name: string;
    age: number;
}

const Greeting: React.FC<GreetingProps> = (props) => {
    return <div>Hello {props.name}, you are {props.age} years old</div>
}
"""
        chunks = parser.parse_content(code, "Greeting.tsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"

        # Should find interface
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) > 0, "Should extract GreetingProps interface"
        assert any("GreetingProps" in c.symbol for c in interface_chunks), \
            "Interface should be named GreetingProps"

        # Should find component with typed props
        greeting_chunks = [c for c in chunks if "Greeting" in c.code and "React.FC" in c.code]
        assert len(greeting_chunks) > 0, "Should find typed FC component"
        assert any("GreetingProps" in c.code for c in chunks), \
            "Should preserve Props type annotation"

    def test_generic_component(self, parser_factory):
        """Test generic component with type parameters.

        Pattern: function List<T>(props: { items: T[] }) { return <ul>...</ul> }
        """
        parser = parser_factory.create_parser(Language.TSX)

        code = """
interface ListProps<T> {
    items: T[];
    renderItem: (item: T) => React.ReactNode;
    keyExtractor: (item: T) => string | number;
}

function List<T>(props: ListProps<T>) {
    return (
        <ul>
            {props.items.map((item) => (
                <li key={props.keyExtractor(item)}>
                    {props.renderItem(item)}
                </li>
            ))}
        </ul>
    )
}
"""
        chunks = parser.parse_content(code, "List.tsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"

        # Should find generic interface
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) > 0, "Should extract ListProps generic interface"
        assert any("<T>" in c.code for c in interface_chunks), \
            "Should preserve generic type parameter in interface"

        # Should find generic function component
        list_chunks = [c for c in chunks if "List" in c.symbol or "List" in c.code]
        assert len(list_chunks) > 0, "Should find generic List component"
        assert any("function List<T>" in c.code for c in chunks), \
            "Should preserve generic type parameter in function"

    def test_props_interface(self, parser_factory):
        """Test component with Props interface definition.

        Pattern: interface ButtonProps { onClick: () => void } + component using it
        """
        parser = parser_factory.create_parser(Language.TSX)

        code = """
interface ButtonProps {
    onClick: () => void;
    disabled?: boolean;
    variant?: 'primary' | 'secondary' | 'danger';
    children: React.ReactNode;
}

function Button({ onClick, disabled = false, variant = 'primary', children }: ButtonProps) {
    return (
        <button
            onClick={onClick}
            disabled={disabled}
            className={`btn btn-${variant}`}
        >
            {children}
        </button>
    )
}
"""
        chunks = parser.parse_content(code, "Button.tsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"

        # Should find interface with all properties
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) > 0, "Should extract ButtonProps interface"
        assert any("onClick" in c.code and "disabled" in c.code for c in interface_chunks), \
            "Should preserve interface properties"
        assert any("React.ReactNode" in c.code for c in interface_chunks), \
            "Should preserve React types in interface"

        # Should find function component
        button_chunks = [c for c in chunks if "Button" in c.symbol or "Button" in c.code]
        assert len(button_chunks) > 0, "Should find Button component"
        assert any("ButtonProps" in c.code for c in chunks), \
            "Should preserve props type annotation"

    def test_props_type_alias(self, parser_factory):
        """Test component with Props type alias definition.

        Pattern: type ButtonProps = { onClick: () => void } + component using it
        """
        parser = parser_factory.create_parser(Language.TSX)

        code = """
type Size = 'small' | 'medium' | 'large';
type Variant = 'primary' | 'secondary' | 'outline';

type CardProps = {
    title: string;
    description?: string;
    size?: Size;
    variant?: Variant;
    children: React.ReactNode;
    onClose?: () => void;
}

const Card: React.FC<CardProps> = ({
    title,
    description,
    size = 'medium',
    variant = 'primary',
    children,
    onClose
}) => {
    return (
        <div className={`card card-${size} card-${variant}`}>
            <header>
                <h3>{title}</h3>
                {onClose && <button onClick={onClose}>×</button>}
            </header>
            {description && <p>{description}</p>}
            <div className="card-content">{children}</div>
        </div>
    )
}
"""
        chunks = parser.parse_content(code, "Card.tsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"

        # Should find type aliases
        type_chunks = [c for c in chunks if c.chunk_type == ChunkType.TYPE_ALIAS]
        assert len(type_chunks) >= 1, "Should extract type aliases"

        # Should find union types
        type_code = " ".join([c.code for c in type_chunks])
        assert "'small'" in type_code or "'primary'" in type_code or "'secondary'" in type_code, \
            "Should preserve union type values"

        # Should find component with type alias props
        card_chunks = [c for c in chunks if "Card" in c.code and "React.FC" in c.code]
        assert len(card_chunks) > 0, "Should find typed Card component"
        assert any("CardProps" in c.code for c in chunks), \
            "Should preserve CardProps type annotation"


# =============================================================================
# TYPED HOOKS
# =============================================================================


class TestTypedHooks:
    """Test typed React hooks in TSX."""

    def test_typed_use_state(self, parser_factory):
        """Test typed useState hook.

        Pattern: const [state, setState] = useState<Type>(initial)
        """
        parser = parser_factory.create_parser(Language.TSX)

        code = """
interface User {
    id: number;
    name: string;
    email: string;
    role: 'admin' | 'user' | 'guest';
}

function UserProfile() {
    const [user, setUser] = useState<User | null>(null)
    const [loading, setLoading] = useState<boolean>(true)
    const [error, setError] = useState<string | null>(null)

    useEffect(() => {
        fetchUser()
            .then(setUser)
            .catch(err => setError(err.message))
            .finally(() => setLoading(false))
    }, [])

    if (loading) return <div>Loading...</div>
    if (error) return <div>Error: {error}</div>
    if (!user) return <div>No user found</div>

    return (
        <div>
            <h2>{user.name}</h2>
            <p>{user.email}</p>
            <span>Role: {user.role}</span>
        </div>
    )
}
"""
        chunks = parser.parse_content(code, "UserProfile.tsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"

        # Should find interface
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) > 0, "Should extract User interface"

        # Should find typed useState calls
        profile_code = " ".join([c.code for c in chunks])
        assert "useState<User | null>" in profile_code, \
            "Should preserve typed useState with union type"
        assert "useState<boolean>" in profile_code, \
            "Should preserve typed useState with boolean"
        assert "useState<string | null>" in profile_code, \
            "Should preserve typed useState with string | null"

    def test_typed_use_ref(self, parser_factory):
        """Test typed useRef hook.

        Pattern: const ref = useRef<HTMLDivElement>(null)
        """
        parser = parser_factory.create_parser(Language.TSX)

        code = """
function MediaPlayer() {
    const videoRef = useRef<HTMLVideoElement>(null)
    const audioRef = useRef<HTMLAudioElement>(null)
    const containerRef = useRef<HTMLDivElement>(null)
    const timeoutRef = useRef<NodeJS.Timeout | null>(null)

    const playVideo = () => {
        videoRef.current?.play()
    }

    const pauseVideo = () => {
        videoRef.current?.pause()
    }

    const setDelayedAction = () => {
        timeoutRef.current = setTimeout(() => {
            console.log('Action executed')
        }, 1000)
    }

    useEffect(() => {
        return () => {
            if (timeoutRef.current) {
                clearTimeout(timeoutRef.current)
            }
        }
    }, [])

    return (
        <div ref={containerRef}>
            <video ref={videoRef} src="/video.mp4" />
            <audio ref={audioRef} src="/audio.mp3" />
            <button onClick={playVideo}>Play</button>
            <button onClick={pauseVideo}>Pause</button>
        </div>
    )
}
"""
        chunks = parser.parse_content(code, "MediaPlayer.tsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"

        # Should find typed useRef calls with HTML element types
        player_code = " ".join([c.code for c in chunks])
        assert "useRef<HTMLVideoElement>" in player_code, \
            "Should preserve typed useRef with HTMLVideoElement"
        assert "useRef<HTMLAudioElement>" in player_code, \
            "Should preserve typed useRef with HTMLAudioElement"
        assert "useRef<HTMLDivElement>" in player_code, \
            "Should preserve typed useRef with HTMLDivElement"
        assert "useRef<NodeJS.Timeout | null>" in player_code or "useRef<" in player_code, \
            "Should preserve typed useRef with custom types"

    def test_event_handler_types(self, parser_factory):
        """Test event handler type annotations.

        Pattern: const onClick = (e: React.MouseEvent<HTMLButtonElement>) => {}
        """
        parser = parser_factory.create_parser(Language.TSX)

        code = """
interface FormData {
    username: string;
    password: string;
}

function LoginForm() {
    const [formData, setFormData] = useState<FormData>({ username: '', password: '' })

    const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault()
        console.log('Submitting:', formData)
    }

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = e.target
        setFormData(prev => ({ ...prev, [name]: value }))
    }

    const handleButtonClick = (e: React.MouseEvent<HTMLButtonElement>) => {
        console.log('Button clicked', e.currentTarget.value)
    }

    const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter') {
            handleSubmit(e as any)
        }
    }

    return (
        <form onSubmit={handleSubmit}>
            <input
                name="username"
                value={formData.username}
                onChange={handleInputChange}
                onKeyPress={handleKeyPress}
            />
            <input
                name="password"
                type="password"
                value={formData.password}
                onChange={handleInputChange}
            />
            <button onClick={handleButtonClick}>Login</button>
        </form>
    )
}
"""
        chunks = parser.parse_content(code, "LoginForm.tsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"

        # Should find interface
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) > 0, "Should extract FormData interface"

        # Should find event handler types
        form_code = " ".join([c.code for c in chunks])
        assert "React.FormEvent<HTMLFormElement>" in form_code, \
            "Should preserve FormEvent type"
        assert "React.ChangeEvent<HTMLInputElement>" in form_code, \
            "Should preserve ChangeEvent type"
        assert "React.MouseEvent<HTMLButtonElement>" in form_code, \
            "Should preserve MouseEvent type"
        assert "React.KeyboardEvent<HTMLInputElement>" in form_code, \
            "Should preserve KeyboardEvent type"

    def test_typed_custom_hook(self, parser_factory):
        """Test typed custom hook with return type.

        Pattern: function useCustom<T>(): [T, Dispatch<T>] {}
        """
        parser = parser_factory.create_parser(Language.TSX)

        code = """
interface FetchState<T> {
    data: T | null;
    loading: boolean;
    error: Error | null;
}

interface UseFetchReturn<T> {
    data: T | null;
    loading: boolean;
    error: Error | null;
    refetch: () => Promise<void>;
}

function useFetch<T>(url: string): UseFetchReturn<T> {
    const [state, setState] = useState<FetchState<T>>({
        data: null,
        loading: true,
        error: null
    })

    const fetchData = useCallback(async () => {
        setState(prev => ({ ...prev, loading: true }))
        try {
            const response = await fetch(url)
            const data = await response.json()
            setState({ data, loading: false, error: null })
        } catch (error) {
            setState({ data: null, loading: false, error: error as Error })
        }
    }, [url])

    useEffect(() => {
        fetchData()
    }, [fetchData])

    return {
        data: state.data,
        loading: state.loading,
        error: state.error,
        refetch: fetchData
    }
}
"""
        chunks = parser.parse_content(code, "useFetch.tsx", FileId(1))

        assert len(chunks) > 0, "Should extract at least one chunk"

        # Should find interfaces
        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) >= 1, "Should extract FetchState and/or UseFetchReturn interfaces"

        # Should find generic custom hook
        hook_chunks = [c for c in chunks if "useFetch" in c.symbol or "useFetch" in c.code]
        assert len(hook_chunks) > 0, "Should find custom hook definition"

        hook_code = " ".join([c.code for c in chunks])
        assert "function useFetch<T>" in hook_code, \
            "Should preserve generic type parameter in hook"
        assert "UseFetchReturn<T>" in hook_code, \
            "Should preserve return type annotation"
        assert "useState<FetchState<T>>" in hook_code, \
            "Should preserve nested generic types"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
