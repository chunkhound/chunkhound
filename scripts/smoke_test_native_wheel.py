"""Smoke-test an installed chunkhound_native wheel's public API.

Run with the wheel installed into the interpreter (e.g. via
`uv run --with <wheel> python scripts/smoke_test_native_wheel.py`) from a
directory outside the repo, so the source-tree chunkhound_native/__init__.py
stub can't shadow the installed package. Shared by build-native-wheel's
post-build check and release.yml/release-rc.yml's post-download
validate-release job, which previously each carried their own byte-identical
copy of this script.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from chunkhound_native import (
    IndexingPipeline,
    ParseCallConfig,
    PipelineReport,
    scan_files,
)

_CONFIG_TEMPLATE = {
    "db_batch_size": 100,
    "compaction_threshold": 0.60,
    "compaction_min_size_mb": 10,
    "disk_usage_limit_mb": None,
    "parse_batch_size": 200,
    "parse_thread_pool_size": 2,
    "embed_batch_size": 200,
    "force_reindex": False,
    "mtime_epsilon_seconds": 0.01,
    "do_cleanup": True,
    "skip_embeddings": True,
    "per_file_timeout_secs": 3.0,
    "per_file_timeout_min_size_kb": 128,
    "detect_embedded_sql": True,
    "config_file_size_threshold_kb": 20,
    "embedding_provider": "",
    "embedding_model": "",
}


def main() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source_file = root / "sample.py"
        source_file.write_text("def sample():\n    return 1\n")
        discovered = scan_files(str(root), ["py"])
        assert str(source_file) in discovered

        db_dir = root / "db"
        db_dir.mkdir()
        config = {**_CONFIG_TEMPLATE, "db_path": str(db_dir)}
        seen_parse_configs: list[ParseCallConfig] = []

        def parse_batch_callback(
            file_paths: list[str], parse_config: ParseCallConfig
        ) -> list[tuple[str, list[Any], None]]:
            assert isinstance(parse_config, ParseCallConfig)
            seen_parse_configs.append(parse_config)
            return [("python", [], None) for _ in file_paths]

        pipeline = IndexingPipeline(config)
        report = pipeline.run(
            files=[(str(source_file), "sample.py")],
            parse_batch_callback=parse_batch_callback,
            embed_batch_callback=None,
            progress_callback=None,
            incremental=False,
        )

        assert isinstance(report, PipelineReport)
        assert not report.errors
        assert seen_parse_configs


if __name__ == "__main__":
    main()
