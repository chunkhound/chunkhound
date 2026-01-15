"""Centralized language detection for ChunkHound.

ARCHITECTURE:
- Single source of truth for all language detection
- Multi-stage strategy: content → extension → filename → fallback
- Handles ambiguous extensions (.m for ObjC/MATLAB)

USAGE:
    from chunkhound.core.detection import detect_language
    language = detect_language(file_path)

RATIONALE:
- Prevents scattered detection logic across services
- Makes adding new ambiguous extensions trivial
- Single point of maintenance and testing
"""

from pathlib import Path

from loguru import logger

from chunkhound.core.types.common import Language

# Content detection byte limit (performance optimization)
# Only read first 1KB for language markers
CONTENT_DETECTION_READ_BYTES = 1024


def detect_language(file_path: Path) -> Language:
    """Detect programming language with content-based disambiguation.

    Strategy:
    1. Check for ambiguous extensions (.m) → content detection
    2. Check extension mapping
    3. Check filename mapping (Makefile)
    4. Return UNKNOWN for unsupported files

    Args:
        file_path: Path to file

    Returns:
        Detected Language enum
    """
    # Stage 1: Content-based detection for ambiguous extensions
    if _is_ambiguous_extension(file_path):
        content_lang = _detect_from_content(file_path)
        if content_lang:
            return content_lang

    # Stage 2: Extension-based detection (fast path)
    lang = Language.from_file_extension(file_path)
    if lang != Language.UNKNOWN:
        return lang

    # Stage 3: Return UNKNOWN for unsupported files
    return Language.UNKNOWN


def _is_ambiguous_extension(file_path: Path) -> bool:
    """Check if extension requires content-based detection.

    Add extensions here as needed:
    - .m: Objective-C vs MATLAB
    - .h: (future) C vs C++ vs Objective-C headers
    - .pl: (future) Perl vs Prolog
    """
    return file_path.suffix.lower() in {".m"}


def _detect_from_content(file_path: Path) -> Language | None:
    """Detect language by reading file content.

    Currently handles:
    - .m files: Objective-C vs MATLAB

    Future: Could handle other ambiguous extensions

    Args:
        file_path: Path to file

    Returns:
        Detected language, or None if detection failed
    """
    ext = file_path.suffix.lower()

    if ext == ".m":
        return _detect_objc_vs_matlab(file_path)

    return None


def _detect_objc_vs_matlab(file_path: Path) -> Language:
    """Disambiguate .m files between Objective-C and MATLAB.

    Objective-C markers:
    - @interface, @implementation, @protocol, @class (unique to ObjC)
    - #import (ObjC convention; MATLAB uses %)

    Default: MATLAB (more common in wild, predates ObjC .m usage)

    Args:
        file_path: Path to .m file

    Returns:
        Language.OBJC or Language.MATLAB
    """
    try:
        with open(file_path, "rb") as f:
            header = f.read(CONTENT_DETECTION_READ_BYTES).decode(
                "utf-8", errors="ignore"
            )

        # Objective-C markers (highly distinctive)
        # These directives are unique to Objective-C and never appear in MATLAB
        objc_markers = ["@interface", "@implementation", "@protocol", "@class"]
        if any(marker in header for marker in objc_markers):
            logger.debug(f"Detected Objective-C in {file_path.name} via @-directive")
            return Language.OBJC

        # #import is Objective-C convention (MATLAB uses % for imports)
        if "#import" in header:
            logger.debug(f"Detected Objective-C in {file_path.name} via #import")
            return Language.OBJC

        # Default to MATLAB for .m files without Objective-C markers
        # Rationale: MATLAB predates Objective-C's .m extension usage
        return Language.MATLAB

    except (OSError, UnicodeDecodeError) as e:
        # If file can't be read, default to MATLAB (backward compatibility)
        logger.debug(
            f"Failed to read {file_path.name} for language detection "
            f"({e}), defaulting to MATLAB"
        )
        return Language.MATLAB
