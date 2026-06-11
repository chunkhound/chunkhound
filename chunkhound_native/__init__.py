try:
    from .chunkhound_native import scan_files
except ImportError as e:
    raise ImportError(
        "chunkhound_native extension not built. "
        "Run: maturin develop  (or install a wheel that includes the native extension)"
    ) from e
