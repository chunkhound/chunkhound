"""ChunkHound File Domain Model - Represents a source code file in the system.

This module contains the File domain model which represents a source code file
that has been indexed by ChunkHound. The File model encapsulates file metadata,
state, and provides methods for working with file data in a type-safe manner.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from chunkhound.core.detection import detect_language
from chunkhound.core.exceptions import ModelError, ValidationError
from chunkhound.core.types import FileId, FilePath, Language, Timestamp


@dataclass(frozen=True)
class File:
    """Domain model representing a source code file.

    This immutable model encapsulates all information about a file that has been
    indexed by ChunkHound, including its path, metadata, and language information.

    Attributes:
        id: Unique file identifier (None for new files)
        path: Relative path to the file (with forward slashes for
            cross-platform compatibility)
        mtime: Last modification time as Unix timestamp
        language: Programming language of the file
        size_bytes: File size in bytes
        content_hash: Fast checksum for change detection (None if not computed)
        created_at: When the file was first indexed
        updated_at: When the file record was last updated
    """

    path: FilePath
    mtime: Timestamp
    language: Language
    size_bytes: int
    id: FileId | None = None
    content_hash: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    # Class constant for content detection
    CONTENT_DETECTION_READ_BYTES = 1024

    def __post_init__(self) -> None:
        """Validate file model after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate file model attributes."""
        # Path validation
        if not self.path:
            raise ValidationError("path", self.path, "Path cannot be empty")

        # Size validation
        if self.size_bytes < 0:
            raise ValidationError(
                "size_bytes", self.size_bytes, "File size cannot be negative"
            )

        # mtime validation
        if self.mtime < 0:
            raise ValidationError(
                "mtime", self.mtime, "Modification time cannot be negative"
            )

    @staticmethod
    def _detect_language(file_path: Path) -> Language | None:
        """Detect language from file content for ambiguous extensions.

        DEPRECATED: Use chunkhound.core.detection.detect_language() directly.

        This method now delegates to the centralized language detector for
        consistency across the codebase.

        Args:
            file_path: Path to file for content analysis

        Returns:
            Detected Language, or None if content-based detection not needed
            (kept for backward compatibility with code expecting this signature)
        """
        from chunkhound.core.detection.language_detector import _is_ambiguous_extension

        # Only return language if content detection was needed
        if _is_ambiguous_extension(file_path):
            return detect_language(file_path)

        return None  # Let caller use extension detection

    @classmethod
    def from_path(cls, file_path: Path) -> "File":
        """Create a File model from a filesystem path.

        Args:
            file_path: Path to the file on disk

        Returns:
            File model with metadata extracted from the filesystem

        Raises:
            ModelError: If file doesn't exist or can't be read
            ValidationError: If extracted data is invalid
        """
        try:
            if not file_path.exists():
                raise ModelError("File", "create", f"File does not exist: {file_path}")

            if not file_path.is_file():
                raise ModelError("File", "create", f"Path is not a file: {file_path}")

            # Extract file stats
            stat = file_path.stat()

            # Detect language using centralized detector
            # (handles content-based detection for ambiguous extensions like .m files)
            detected_language = detect_language(file_path)

            return cls(
                path=FilePath(str(file_path)),
                mtime=Timestamp(stat.st_mtime),
                language=detected_language,
                size_bytes=stat.st_size,
            )

        except OSError as e:
            raise ModelError("File", "create", f"Failed to read file metadata: {e}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "File":
        """Create a File model from a dictionary.

        This method provides backward compatibility with existing code that
        uses dictionary representations of files.

        Args:
            data: Dictionary containing file data

        Returns:
            File model created from dictionary data

        Raises:
            ValidationError: If required fields are missing or invalid
        """
        try:
            # Extract required fields
            path = data.get("path")
            if not path:
                raise ValidationError("path", path, "Path is required")

            mtime = data.get("mtime")
            if mtime is None:
                raise ValidationError("mtime", mtime, "Modification time is required")

            size_bytes = data.get("size_bytes", 0)

            # Handle language - could be string or Language enum
            language_value = data.get("language")
            if isinstance(language_value, Language):
                language = language_value
            elif isinstance(language_value, str):
                language = Language.from_string(language_value)
            else:
                # Try to infer from path
                language = Language.from_file_extension(path)

            # Handle optional fields
            file_id = data.get("id")
            if file_id is not None:
                file_id = FileId(file_id)

            created_at = data.get("created_at")
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)

            updated_at = data.get("updated_at")
            if isinstance(updated_at, str):
                updated_at = datetime.fromisoformat(updated_at)

            content_hash = data.get("content_hash")

            return cls(
                id=file_id,
                path=FilePath(path),
                mtime=Timestamp(float(mtime)),
                language=language,
                size_bytes=int(size_bytes),
                content_hash=content_hash,
                created_at=created_at,
                updated_at=updated_at,
            )

        except (ValueError, TypeError) as e:
            raise ValidationError("data", data, f"Invalid data format: {e}")

    def to_dict(self) -> dict[str, Any]:
        """Convert File model to dictionary.

        This method provides backward compatibility with existing code that
        expects dictionary representations of files.

        Returns:
            Dictionary representation of the file
        """
        result = {
            "path": self.path,
            "mtime": self.mtime,
            "language": self.language.value,
            "size_bytes": self.size_bytes,
        }

        if self.id is not None:
            result["id"] = self.id

        if self.content_hash is not None:
            result["content_hash"] = self.content_hash

        if self.created_at is not None:
            result["created_at"] = self.created_at.isoformat()

        if self.updated_at is not None:
            result["updated_at"] = self.updated_at.isoformat()

        return result

    @property
    def name(self) -> str:
        """Get the file name (without directory path)."""
        return Path(self.path).name

    @property
    def extension(self) -> str:
        """Get the file extension."""
        return Path(self.path).suffix

    @property
    def stem(self) -> str:
        """Get the file name without extension."""
        return Path(self.path).stem

    @property
    def parent_dir(self) -> str:
        """Get the parent directory path."""
        return str(Path(self.path).parent)

    @property
    def relative_path(self) -> str:
        """Get relative path (path is already stored as relative)."""
        return self.path

    def is_supported_language(self) -> bool:
        """Check if the file's language is supported by ChunkHound.

        Returns:
            True if the language is supported for parsing
        """
        return self.language != Language.UNKNOWN

    def with_id(self, file_id: FileId) -> "File":
        """Create a new File instance with the specified ID.

        Args:
            file_id: File ID to set

        Returns:
            New File instance with the ID set
        """
        return File(
            id=file_id,
            path=self.path,
            mtime=self.mtime,
            language=self.language,
            size_bytes=self.size_bytes,
            content_hash=self.content_hash,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    def with_updated_mtime(self, new_mtime: Timestamp) -> "File":
        """Create a new File instance with updated modification time.

        Args:
            new_mtime: New modification time

        Returns:
            New File instance with updated mtime
        """
        return File(
            id=self.id,
            path=self.path,
            mtime=new_mtime,
            language=self.language,
            size_bytes=self.size_bytes,
            content_hash=self.content_hash,
            created_at=self.created_at,
            updated_at=datetime.utcnow(),
        )

    def __str__(self) -> str:
        """Return string representation of the file."""
        return (
            f"File(id={self.id}, path={self.relative_path}, "
            f"language={self.language.value})"
        )

    def __repr__(self) -> str:
        """Return detailed string representation of the file."""
        return (
            f"File(id={self.id}, path='{self.path}', "
            f"language={self.language.value}, size_bytes={self.size_bytes}, "
            f"mtime={self.mtime})"
        )
