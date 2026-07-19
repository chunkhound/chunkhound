from .chunkhound_native import scan_files

try:
    from .chunkhound_native import RustDbWriter
except ImportError:
    RustDbWriter = None  # native extension not built; callers should guard with `if RustDbWriter`

try:
    from .chunkhound_native import IndexingPipeline, PipelineReport  # type: ignore[import-untyped]
except ImportError:
    IndexingPipeline = None  # compiled without rust-pipeline feature
    PipelineReport = None
