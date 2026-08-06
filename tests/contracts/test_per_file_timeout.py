"""Contract test: per-file parse timeout and config-file size threshold.

Gap this closes: `PipelineConfig` (src/pipeline/config.rs) has carried
`per_file_timeout_secs`, `per_file_timeout_min_size_kb`, and
`config_file_size_threshold_kb` since early in the Rust-pipeline work,
labeled "Pass-through (parse callback)" — but nothing ever read them back on
the Rust side, and Python's `parse_file_callback()` hardcoded a 20 KB config
threshold and never enforced a timeout at all, despite its own docstring
once claiming one was "applied by the caller via multiprocessing." A
pathological/hanging file could stall a whole parse batch (and every batch
queued behind it) indefinitely.

`ParseCallConfig` (a Rust #[pyclass], src/pipeline/parse_call_config.rs) now
carries all of these plus `parse_thread_pool_size` and `detect_embedded_sql`
as one object Rust constructs once per `IndexingPipeline.run()` and passes
to every `parse_batch_callback()` call, instead of a growing list of
positional arguments. `_parse_one_file()` in `chunkhound/pipeline_bridge.py`
now actually enforces the timeout (via `_parse_with_timeout()`, a dedicated
killable child process — mirroring
`chunkhound.services.batch_processor._parse_file_with_timeout`'s approach)
and honors the configured config-file size threshold instead of a
hardcoded 20 KB.
"""

from pathlib import Path

import pytest

import chunkhound.pipeline_bridge as pipeline_bridge
from chunkhound.pipeline_bridge import parse_file_callback
from tests.contracts.pipeline_harness import default_rust_config


class TestParseCallConfigWiring:
    """Rust must build ParseCallConfig from PipelineConfig and hand it to
    every parse_batch_callback() call — not just detect_embedded_sql."""

    def test_rust_forwards_configured_values_to_callback(self, tmp_path):
        try:
            from chunkhound_native import (  # type: ignore[import-untyped]
                IndexingPipeline,
            )
        except ImportError:
            pytest.fail(
                "Rust IndexingPipeline is not yet available in chunkhound_native."
            )

        good_file = tmp_path / "good.py"
        good_file.write_text("def a():\n    return 1\n")

        db_dir = tmp_path / "db"
        db_dir.mkdir(parents=True, exist_ok=True)

        seen_configs = []

        def capturing_batch_callback(file_paths, parse_config):
            seen_configs.append(parse_config)
            return [("python", [], None) for _ in file_paths]

        # Distinctive, unlikely-default values for every field ParseCallConfig
        # forwards to the callback — proves they're actually wired through,
        # not just falling back to defaults that would happen to match.
        config_dict = default_rust_config(
            tmp_path,
            db_dir,
            parse_thread_pool_size=7,
            per_file_timeout_secs=2.5,
            per_file_timeout_min_size_kb=99,
            detect_embedded_sql=False,
            config_file_size_threshold_kb=13,
        )

        pipeline = IndexingPipeline(config_dict)
        pipeline.run(
            files=[str(good_file)],
            parse_batch_callback=capturing_batch_callback,
            embed_batch_callback=None,
            progress_callback=None,
            incremental=False,
        )

        assert seen_configs, "parse_batch_callback should have been invoked"
        cfg = seen_configs[0]
        assert cfg.detect_embedded_sql is False
        assert cfg.per_file_timeout_secs == 2.5
        assert cfg.per_file_timeout_min_size_kb == 99
        assert cfg.config_file_size_threshold_kb == 13
        assert cfg.parse_thread_pool_size == 7


class TestConfigFileSizeThreshold:
    """config_file_size_threshold_kb must actually gate structured config files."""

    def _make_json_file(self, tmp_path: Path) -> Path:
        config_file = tmp_path / "data.json"
        config_file.write_text("{" + ",".join(f'"k{i}": {i}' for i in range(200)) + "}")
        assert config_file.stat().st_size / 1024 > 1, (
            "fixture file must be bigger than the small threshold used below"
        )
        return config_file

    def test_file_above_configured_threshold_is_skipped(self, tmp_path):
        config_file = self._make_json_file(tmp_path)

        lang, chunks = parse_file_callback(
            str(config_file), config_file_size_threshold_kb=1
        )

        assert lang == ""
        assert chunks == []

    def test_file_below_configured_threshold_is_processed(self, tmp_path):
        config_file = self._make_json_file(tmp_path)

        lang, chunks = parse_file_callback(
            str(config_file), config_file_size_threshold_kb=100
        )

        assert lang == "json"

    def test_threshold_disabled_when_non_positive(self, tmp_path):
        """<= 0 disables the gate — matches
        chunkhound.services.batch_processor's existing convention."""
        config_file = self._make_json_file(tmp_path)

        lang, chunks = parse_file_callback(
            str(config_file), config_file_size_threshold_kb=0
        )

        assert lang == "json"


class TestPerFileTimeout:
    """A file that can't be parsed within per_file_timeout_secs must be
    recorded as a per-file error, not hang the batch or the run."""

    def test_small_file_skips_timeout_path(self, monkeypatch, tmp_path):
        """Files below per_file_timeout_min_size_kb must use the direct
        fast path — no subprocess-spawn overhead for typical small files."""
        small_file = tmp_path / "small.py"
        small_file.write_text("def f():\n    return 1\n")

        def _fail_if_called(*args, **kwargs):
            raise AssertionError(
                "_parse_with_timeout should not be called for small files"
            )

        monkeypatch.setattr(pipeline_bridge, "_parse_with_timeout", _fail_if_called)

        cfg = pipeline_bridge._ParsePoolConfig(
            per_file_timeout_secs=3.0, per_file_timeout_min_size_kb=128
        )
        lang, chunks, error = pipeline_bridge._parse_one_file((str(small_file), cfg))

        assert error is None
        assert lang == "python"
        assert chunks

    def test_large_file_uses_timeout_path(self, monkeypatch, tmp_path):
        """Files at/above per_file_timeout_min_size_kb must route through
        _parse_with_timeout(), not the direct fast path."""
        big_file = tmp_path / "big.py"
        big_file.write_text("def f():\n    return 1\n")

        called = {}

        def _fake_parse_with_timeout(file_path, cfg):
            called["file_path"] = file_path
            return ("python", [], None)

        monkeypatch.setattr(
            pipeline_bridge, "_parse_with_timeout", _fake_parse_with_timeout
        )

        cfg = pipeline_bridge._ParsePoolConfig(
            per_file_timeout_secs=3.0, per_file_timeout_min_size_kb=0
        )
        pipeline_bridge._parse_one_file((str(big_file), cfg))

        assert called.get("file_path") == str(big_file)

    def test_timeout_disabled_when_non_positive(self, monkeypatch, tmp_path):
        """per_file_timeout_secs <= 0 must disable the timeout path entirely,
        regardless of file size."""
        big_file = tmp_path / "big.py"
        big_file.write_text("def f():\n    return 1\n")

        def _fail_if_called(*args, **kwargs):
            raise AssertionError(
                "_parse_with_timeout should not be called when disabled"
            )

        monkeypatch.setattr(pipeline_bridge, "_parse_with_timeout", _fail_if_called)

        cfg = pipeline_bridge._ParsePoolConfig(
            per_file_timeout_secs=0.0, per_file_timeout_min_size_kb=0
        )
        lang, chunks, error = pipeline_bridge._parse_one_file((str(big_file), cfg))

        assert error is None
        assert lang == "python"

    @pytest.mark.timeout(30)
    def test_file_that_cant_finish_in_time_returns_error_not_hang(self, tmp_path):
        """A real timeout (dedicated child process killed outright) must
        surface as a per-file error tuple, not raise or hang the caller.

        Uses an unrealistically tiny timeout rather than a slow/hanging
        file: multiprocessing's "spawn" start method re-imports modules
        fresh in the child, so a monkeypatched sleep in this (parent)
        process wouldn't be visible there anyway (the same trap documented
        in test_parse_error_per_file.py). A few-millisecond deadline is
        already tighter than spawning + importing + parsing can complete,
        which exercises the real timeout branch deterministically without
        needing a genuinely pathological input.
        """
        target = tmp_path / "target.py"
        target.write_text("def f():\n    return 1\n")

        cfg = pipeline_bridge._ParsePoolConfig(
            per_file_timeout_secs=0.001, per_file_timeout_min_size_kb=0
        )

        lang, chunks, error = pipeline_bridge._parse_one_file((str(target), cfg))

        assert chunks == []
        assert error is not None
        assert "timed out" in error
