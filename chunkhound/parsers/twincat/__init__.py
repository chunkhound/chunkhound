"""TwinCAT Structured Text parser for ChunkHound."""

# Conditional export - only when lark is available
try:
    from .twincat_mapping import TwinCATMapping
    from .twincat_parser import TwinCATParser

    __all__ = ["TwinCATParser", "TwinCATMapping"]
except ImportError:
    __all__ = []
