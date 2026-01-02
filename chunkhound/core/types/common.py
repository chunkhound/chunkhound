"""ChunkHound Core Types - Common type definitions and aliases.

This module contains type definitions, enums, and type aliases used throughout
the ChunkHound system. These types provide better code clarity, IDE support,
and runtime type checking capabilities.
"""

from enum import Enum
from pathlib import Path
from typing import NewType

# String-based type aliases for better semantic clarity
ProviderName = NewType("ProviderName", str)  # e.g., "openai"
ModelName = NewType("ModelName", str)  # e.g., "text-embedding-3-small"
FilePath = NewType("FilePath", str)  # File path as string

# Numeric type aliases
ChunkId = NewType("ChunkId", int)  # Database chunk ID
FileId = NewType("FileId", int)  # Database file ID
LineNumber = NewType("LineNumber", int)  # 1-based line numbers
ByteOffset = NewType("ByteOffset", int)  # Byte positions in files
Timestamp = NewType("Timestamp", float)  # Unix timestamp
Distance = NewType("Distance", float)  # Vector distance/similarity score
Dimensions = NewType("Dimensions", int)  # Embedding vector dimensions

# Complex types
EmbeddingVector = list[float]  # Vector embedding representation


class ChunkType(Enum):
    """Enumeration of semantic chunk types supported by ChunkHound."""

    # Code structure types
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    INTERFACE = "interface"
    STRUCT = "struct"
    ENUM = "enum"
    NAMESPACE = "namespace"
    CONSTRUCTOR = "constructor"
    PROPERTY = "property"
    FIELD = "field"
    TYPE_ALIAS = "type_alias"
    CLOSURE = "closure"
    TRAIT = "trait"
    SCRIPT = "script"
    OBJECT = "object"
    COMPANION_OBJECT = "companion_object"
    DATA_CLASS = "data_class"
    EXTENSION_FUNCTION = "extension_function"

    # C-specific types
    VARIABLE = "variable"
    TYPE = "type"
    MACRO = "macro"

    # Documentation types
    COMMENT = "comment"
    DOCSTRING = "docstring"
    HEADER_1 = "header_1"
    HEADER_2 = "header_2"
    HEADER_3 = "header_3"
    HEADER_4 = "header_4"
    HEADER_5 = "header_5"
    HEADER_6 = "header_6"
    PARAGRAPH = "paragraph"
    CODE_BLOCK = "code_block"

    # Configuration types
    TABLE = "table"
    KEY_VALUE = "key_value"
    ARRAY = "array"

    # Generic types
    BLOCK = "block"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str) -> "ChunkType":
        """Convert string to ChunkType enum, defaulting to UNKNOWN for
        invalid values."""
        try:
            return cls(value)
        except ValueError:
            return cls.UNKNOWN

    @property
    def is_code(self) -> bool:
        """Return True if this chunk type represents code structure."""
        return self in {
            ChunkType.FUNCTION,
            ChunkType.METHOD,
            ChunkType.CLASS,
            ChunkType.INTERFACE,
            ChunkType.STRUCT,
            ChunkType.ENUM,
            ChunkType.NAMESPACE,
            ChunkType.CONSTRUCTOR,
            ChunkType.PROPERTY,
            ChunkType.FIELD,
            ChunkType.TYPE_ALIAS,
            ChunkType.CLOSURE,
            ChunkType.TRAIT,
            ChunkType.SCRIPT,
            ChunkType.BLOCK,
            ChunkType.VARIABLE,
            ChunkType.TYPE,
            ChunkType.MACRO,
        }

    @property
    def is_documentation(self) -> bool:
        """Return True if this chunk type represents documentation."""
        return self in {
            ChunkType.COMMENT,
            ChunkType.DOCSTRING,
            ChunkType.HEADER_1,
            ChunkType.HEADER_2,
            ChunkType.HEADER_3,
            ChunkType.HEADER_4,
            ChunkType.HEADER_5,
            ChunkType.HEADER_6,
            ChunkType.PARAGRAPH,
            ChunkType.CODE_BLOCK,
        }


class Language(Enum):
    """Enumeration of programming languages and file types supported by ChunkHound."""

    # Programming languages
    PYTHON = "python"
    JAVA = "java"
    CSHARP = "csharp"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    TSX = "tsx"
    JSX = "jsx"
    GROOVY = "groovy"
    KOTLIN = "kotlin"
    GO = "go"
    HASKELL = "haskell"
    RUST = "rust"
    ZIG = "zig"
    BASH = "bash"
    MAKEFILE = "makefile"
    C = "c"
    CPP = "cpp"
    MATLAB = "matlab"
    HCL = "hcl"
    OBJC = "objc"
    PHP = "php"
    VUE = "vue"
    SVELTE = "svelte"
    SWIFT = "swift"
    DART = "dart"

    # Documentation languages
    MARKDOWN = "markdown"

    # Data/Configuration languages
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    TEXT = "text"
    PDF = "pdf"

    # Generic/unknown
    UNKNOWN = "unknown"

    @classmethod
    def from_file_extension(cls, file_path: str | Path) -> "Language":
        """Determine language from file extension and filename."""
        if isinstance(file_path, str):
            file_path = Path(file_path)

        # Check filename-based detection first (for Makefiles)
        basename = file_path.name.lower()
        filename_map = {
            "makefile": cls.MAKEFILE,
            "gnumakefile": cls.MAKEFILE,
        }

        if basename in filename_map:
            return filename_map[basename]

        # Check extension-based detection
        extension = file_path.suffix.lower()
        extension_map = {
            ".py": cls.PYTHON,
            ".java": cls.JAVA,
            ".cs": cls.CSHARP,
            ".ts": cls.TYPESCRIPT,
            ".js": cls.JAVASCRIPT,
            ".tsx": cls.TSX,
            ".jsx": cls.JSX,
            ".groovy": cls.GROOVY,
            ".gvy": cls.GROOVY,
            ".gy": cls.GROOVY,
            ".kt": cls.KOTLIN,
            ".kts": cls.KOTLIN,
            ".go": cls.GO,
            ".hs": cls.HASKELL,
            ".lhs": cls.HASKELL,
            ".hs-boot": cls.HASKELL,
            ".hsig": cls.HASKELL,
            ".hsc": cls.HASKELL,
            ".sh": cls.BASH,
            ".bash": cls.BASH,
            ".zsh": cls.BASH,
            ".mk": cls.MAKEFILE,
            ".make": cls.MAKEFILE,
            ".md": cls.MARKDOWN,
            ".markdown": cls.MARKDOWN,
            ".dart": cls.DART,
            ".hcl": cls.HCL,
            ".tf": cls.HCL,
            ".tfvars": cls.HCL,
            ".json": cls.JSON,
            ".yaml": cls.YAML,
            ".yml": cls.YAML,
            ".toml": cls.TOML,
            ".txt": cls.TEXT,
            ".pdf": cls.PDF,
            ".c": cls.C,
            ".h": cls.C,
            ".cpp": cls.CPP,
            ".cxx": cls.CPP,
            ".cc": cls.CPP,
            ".hpp": cls.CPP,
            ".hxx": cls.CPP,
            ".h++": cls.CPP,
            ".rs": cls.RUST,
            ".zig": cls.ZIG,
            ".m": cls.MATLAB,  # Note: .m is ambiguous, will use content detection
            ".mm": cls.OBJC,
            ".php": cls.PHP,
            ".phtml": cls.PHP,
            ".php3": cls.PHP,
            ".php4": cls.PHP,
            ".php5": cls.PHP,
            ".phps": cls.PHP,
            ".vue": cls.VUE,
            ".svelte": cls.SVELTE,
            ".swift": cls.SWIFT,
            ".swiftinterface": cls.SWIFT,
        }

        return extension_map.get(extension, cls.UNKNOWN)

    @classmethod
    def from_string(cls, value: str) -> "Language":
        """Convert string to Language enum, defaulting to UNKNOWN for invalid values."""
        try:
            return cls(value)
        except ValueError:
            return cls.UNKNOWN

    @property
    def is_programming_language(self) -> bool:
        """Return True if this is a programming language (not documentation)."""
        return self in {
            Language.PYTHON,
            Language.JAVA,
            Language.CSHARP,
            Language.TYPESCRIPT,
            Language.JAVASCRIPT,
            Language.TSX,
            Language.JSX,
            Language.GROOVY,
            Language.KOTLIN,
            Language.GO,
            Language.HASKELL,
            Language.RUST,
            Language.BASH,
            Language.MAKEFILE,
            Language.C,
            Language.CPP,
            Language.MATLAB,
            Language.OBJC,
            Language.PHP,
            Language.VUE,
            Language.SVELTE,
            Language.SWIFT,
            Language.DART,
        }

    @property
    def supports_classes(self) -> bool:
        """Return True if this language supports class definitions."""
        return self in {
            Language.PYTHON,
            Language.JAVA,
            Language.CSHARP,
            Language.TYPESCRIPT,
            Language.TSX,
            Language.GROOVY,
            Language.KOTLIN,
            Language.GO,
            Language.CPP,
            Language.MATLAB,
            Language.OBJC,
            Language.PHP,
            Language.VUE,
            Language.SVELTE,
            Language.SWIFT,
            Language.DART,
        }

    @property
    def supports_interfaces(self) -> bool:
        """Return True if this language supports interface definitions."""
        return self in {
            Language.JAVA,
            Language.CSHARP,
            Language.TYPESCRIPT,
            Language.TSX,
            Language.PHP,
            Language.VUE,
            Language.SVELTE,
            Language.SWIFT,
        }

    @property
    def is_structured_config_language(self) -> bool:
        """Return True if this is a structured config/data language.

        These languages are used for both:
        - Small config files (< 20KB typically): package.json, tsconfig.json
        - Large data files (> 100KB typically): package-lock.json, API dumps

        Size-based filtering helps distinguish between the two use cases.
        """
        return self in {Language.JSON, Language.YAML, Language.TOML, Language.HCL}

    @classmethod
    def get_all_extensions(cls) -> set[str]:
        """Get all supported file extensions.

        Returns all extensions from the parser system's EXTENSION_TO_LANGUAGE mapping.
        This ensures consistency between language detection and file pattern matching.
        Uses lazy import to avoid circular dependency (parser_factory imports Language).
        """
        # Lazy import to avoid circular dependency at module load time
        from chunkhound.parsers.parser_factory import EXTENSION_TO_LANGUAGE

        # Extract all extensions (keys starting with '.')
        # Filter out filename patterns like "Makefile" which don't start with '.'
        extensions = {
            ext for ext in EXTENSION_TO_LANGUAGE.keys() if ext.startswith(".")
        }

        return extensions

    @classmethod
    def get_all_filename_patterns(cls) -> set[str]:
        """Get all supported filename patterns (non-extension based).

        Returns filename patterns like "Makefile", "GNUmakefile", "Dockerfile" that are
        detected by exact filename match rather than file extension. All patterns are
        normalized to lowercase for case-insensitive matching.

        Uses lazy import to avoid circular dependency (parser_factory imports Language).
        """
        # Lazy import to avoid circular dependency at module load time
        from chunkhound.parsers.parser_factory import EXTENSION_TO_LANGUAGE

        # Extract filename patterns (keys not starting with '.')
        # Normalize to lowercase for case-insensitive filesystem compatibility
        patterns = {
            key.lower()
            for key in EXTENSION_TO_LANGUAGE.keys()
            if not key.startswith(".")
        }
        return patterns

    @classmethod
    def get_file_patterns(cls) -> list[str]:
        """Get glob patterns for all supported file types.

        Derives patterns from parser_factory.EXTENSION_TO_LANGUAGE to ensure
        consistency. Handles both extension-based patterns (.py) and filename-based
        patterns (Makefile) from the parser system.
        """
        # Lazy import to avoid circular dependency at module load time
        from chunkhound.parsers.parser_factory import EXTENSION_TO_LANGUAGE

        patterns = []

        # Add patterns for all keys in EXTENSION_TO_LANGUAGE
        for key in EXTENSION_TO_LANGUAGE.keys():
            if key.startswith("."):
                # Extension-based pattern (e.g., ".py" -> "**/*.py")
                patterns.append(f"**/*{key}")
            else:
                # Filename-based pattern (e.g., "Makefile" -> "**/Makefile")
                patterns.append(f"**/{key}")

        return patterns

    @classmethod
    def is_supported_file(cls, file_path: str | Path) -> bool:
        """Check if a file is supported based on its extension or name."""
        language = cls.from_file_extension(file_path)
        return language != cls.UNKNOWN
