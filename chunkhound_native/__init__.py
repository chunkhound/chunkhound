from .chunkhound_native import scan_files

try:
    from .chunkhound_native import RustDbWriter
except ImportError:
    RustDbWriter = None
