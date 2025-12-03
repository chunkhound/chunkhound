"""Single source of truth for ChunkHound version.

Version is automatically derived from git tags via hatch-vcs.
For development builds, the version includes the git commit hash.
"""

try:
    # Try to import from generated _version.py (created by hatch-vcs during build)
    from chunkhound._version import __version__
except ImportError:
    # Fallback for development environments before first build
    try:
        from importlib.metadata import version
        __version__ = version("chunkhound")
    except Exception:
        # Final fallback for editable installs without metadata
        __version__ = "0.0.0+unknown"
