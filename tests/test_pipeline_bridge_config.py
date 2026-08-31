"""Tests for run_rust_pipeline's config-dict mapping.

Covers the contract that user-facing settings (set via CLI, .chunkhound.json,
or env var) are actually forwarded to the Rust pipeline, not overridden by
internal defaults.
"""

from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _fake_report(**overrides: object) -> SimpleNamespace:
    defaults = dict(
        files_processed=0,
        chunks_written=0,
        embeddings_generated=0,
        elapsed_secs=0.0,
        files_skipped=0,
        errors=[],
        # Option<(f64, f64)> on the Rust side: None or (current_mb, limit_mb)
        disk_limit=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@dataclass
class _FakeNativePipeline:
    """Fake chunkhound_native.IndexingPipeline, installed via sys.modules.

    `captured_config` is populated with whatever config_dict
    IndexingPipeline.__init__ was called with; tests that only care about
    run_rust_pipeline's return value can ignore it. `instance.run.return_value`
    defaults to `_fake_report()` -- override it before calling
    run_rust_pipeline for disk-limit-style tests.
    """

    instance: MagicMock
    captured_config: dict


@pytest.fixture
def fake_native_pipeline(monkeypatch: pytest.MonkeyPatch) -> _FakeNativePipeline:
    captured_config: dict = {}
    instance = MagicMock()
    instance.run.return_value = _fake_report()

    def _init_side_effect(config_dict: dict) -> MagicMock:
        captured_config.update(config_dict)
        return instance

    fake_module = types.SimpleNamespace(
        IndexingPipeline=MagicMock(side_effect=_init_side_effect)
    )
    monkeypatch.setitem(sys.modules, "chunkhound_native", fake_module)
    return _FakeNativePipeline(instance=instance, captured_config=captured_config)


@pytest.mark.asyncio
async def test_fragmentation_threshold_pct_forwarded_as_compaction_ratio(
    tmp_path: Path, fake_native_pipeline: _FakeNativePipeline
) -> None:
    """A user-set --fragmentation-threshold-pct must reach Rust as a ratio.

    DatabaseConfig.fragmentation_threshold_pct is a percentage (e.g. 60.0)
    and is already honored by the Python indexing path (duckdb_provider.py).
    The Rust pipeline bridge must forward the same setting, converted to the
    ratio Rust's PipelineConfig.compaction_threshold expects (0.60) — not a
    hardcoded default.
    """
    from chunkhound import pipeline_bridge

    config = SimpleNamespace(
        database=SimpleNamespace(fragmentation_threshold_pct=60.0),
        indexing=SimpleNamespace(),
        embedding=None,
    )

    await pipeline_bridge.run_rust_pipeline(
        files_to_process=[],
        db_path=tmp_path,
        project_root=tmp_path,
        skip_embeddings=True,
        config=config,
    )

    assert fake_native_pipeline.captured_config["compaction_threshold"] == pytest.approx(0.60)


@pytest.mark.asyncio
async def test_embedding_configuration_reaches_native_pipeline(
    tmp_path: Path,
    fake_native_pipeline: _FakeNativePipeline,
) -> None:
    """Resolved provider settings cross the Python/Rust boundary intact."""
    from chunkhound import pipeline_bridge
    from chunkhound.core.config.embedding_config import EmbeddingConfig

    embedding = EmbeddingConfig(
        provider="openai",
        api_key="secret-test-key",
        base_url="http://127.0.0.1:9999/v1",
        output_dims=8,
        client_side_truncation=True,
        ssl_verify=False,
    )
    await pipeline_bridge.run_rust_pipeline(
        files_to_process=[],
        db_path=tmp_path,
        project_root=tmp_path,
        skip_embeddings=False,
        config=SimpleNamespace(
            database=SimpleNamespace(),
            indexing=SimpleNamespace(),
            embedding=embedding,
        ),
    )

    captured = fake_native_pipeline.captured_config
    assert captured["embedding_model"] == "text-embedding-3-small"
    assert captured["embedding_api_key"] == "secret-test-key"
    assert captured["embedding_base_url"] == "http://127.0.0.1:9999/v1"
    assert captured["embedding_output_dims"] == 8
    assert captured["embedding_client_side_truncation"] is True
    assert captured["embedding_ssl_verify"] is False
    assert captured["embed_max_tokens_per_batch"] == 8091


@pytest.mark.asyncio
async def test_missing_fragmentation_threshold_pct_defaults_to_30_pct(
    tmp_path: Path, fake_native_pipeline: _FakeNativePipeline
) -> None:
    """No database config at all falls back to the documented 30% default."""
    from chunkhound import pipeline_bridge

    await pipeline_bridge.run_rust_pipeline(
        files_to_process=[],
        db_path=tmp_path,
        project_root=tmp_path,
        skip_embeddings=True,
        config=None,
    )

    assert fake_native_pipeline.captured_config["compaction_threshold"] == pytest.approx(0.30)


@pytest.mark.asyncio
async def test_explicit_none_fragmentation_threshold_pct_disables_compaction(
    tmp_path: Path, fake_native_pipeline: _FakeNativePipeline
) -> None:
    """fragmentation_threshold_pct=None means "never auto-compact" (see
    DatabaseConfig's docstring), not "unset — use the 30% default". Rust's
    compaction_threshold must receive that None verbatim so it can disable
    the auto-compaction check entirely, matching what the Python indexing
    path already does via duckdb_provider.py's
    _fragmentation_exceeds_threshold(threshold=None) -> False.
    """
    from chunkhound import pipeline_bridge

    config = SimpleNamespace(
        database=SimpleNamespace(fragmentation_threshold_pct=None),
        indexing=SimpleNamespace(),
        embedding=None,
    )

    await pipeline_bridge.run_rust_pipeline(
        files_to_process=[],
        db_path=tmp_path,
        project_root=tmp_path,
        skip_embeddings=True,
        config=config,
    )

    assert fake_native_pipeline.captured_config["compaction_threshold"] is None


@pytest.mark.asyncio
async def test_explicit_zero_config_file_threshold_disables_gate(
    tmp_path: Path, fake_native_pipeline: _FakeNativePipeline
) -> None:
    """config_file_size_threshold_kb=0 (explicitly disabling the gate) must
    reach Rust as 0, not silently fall back to the default 20 -- matching
    the Python path's documented "<= 0 disables the gate" contract.
    """
    from chunkhound import pipeline_bridge

    config = SimpleNamespace(
        database=SimpleNamespace(),
        indexing=SimpleNamespace(config_file_size_threshold_kb=0),
        embedding=None,
    )

    await pipeline_bridge.run_rust_pipeline(
        files_to_process=[],
        db_path=tmp_path,
        project_root=tmp_path,
        skip_embeddings=True,
        config=config,
    )

    assert fake_native_pipeline.captured_config["config_file_size_threshold_kb"] == 0


@pytest.mark.asyncio
async def test_explicit_zero_mtime_epsilon_forwarded(
    tmp_path: Path, fake_native_pipeline: _FakeNativePipeline
) -> None:
    """mtime_epsilon_seconds=0.0 (exact mtime match) must reach Rust as 0.0,
    not silently fall back to the default 0.01.
    """
    from chunkhound import pipeline_bridge

    config = SimpleNamespace(
        database=SimpleNamespace(),
        indexing=SimpleNamespace(mtime_epsilon_seconds=0.0),
        embedding=None,
    )

    await pipeline_bridge.run_rust_pipeline(
        files_to_process=[],
        db_path=tmp_path,
        project_root=tmp_path,
        skip_embeddings=True,
        config=config,
    )

    assert fake_native_pipeline.captured_config["mtime_epsilon_seconds"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_disk_limit_exceeded_report_becomes_structured_error_dict(
    tmp_path: Path, fake_native_pipeline: _FakeNativePipeline
) -> None:
    """A tripped PipelineReport.disk_limit must surface as the same
    error-dict shape DiskUsageLimitExceededError.to_error_dict() builds --
    also used by IndexingCoordinator._store_parsed_results for the Python
    path (indexing_coordinator.py:1039-1042) -- {"file": None, "error": ...,
    "disk_limit_exceeded": True, "current_size_mb": ..., "limit_mb": ...} --
    so the coordinator's existing generic disk-limit scan
    (indexing_coordinator.py:2175-2183) picks it up with zero coordinator
    changes, regardless of which pipeline ran.
    """
    from chunkhound import pipeline_bridge

    fake_native_pipeline.instance.run.return_value = _fake_report(disk_limit=(12.0, 10.0))

    result = await pipeline_bridge.run_rust_pipeline(
        files_to_process=[],
        db_path=tmp_path,
        project_root=tmp_path,
        skip_embeddings=True,
        config=None,
    )

    assert result["errors"] == [
        {
            "file": None,
            "error": "Database disk usage limit exceeded: 12.0 MB >= 10.0 MB",
            "disk_limit_exceeded": True,
            "current_size_mb": 12.0,
            "limit_mb": 10.0,
        }
    ]


@pytest.mark.asyncio
async def test_disk_limit_not_exceeded_report_has_no_extra_error(
    tmp_path: Path, fake_native_pipeline: _FakeNativePipeline
) -> None:
    """A normal (non-tripped) report must not append any disk-limit error."""
    from chunkhound import pipeline_bridge

    result = await pipeline_bridge.run_rust_pipeline(
        files_to_process=[],
        db_path=tmp_path,
        project_root=tmp_path,
        skip_embeddings=True,
        config=None,
    )

    assert result["errors"] == []


def test_embed_thread_http_clients_are_closed_with_their_loops() -> None:
    """Per-thread embed clients must shut down before process exit.

    Leaving httpx AsyncClients bound to worker loops causes Windows
    Proactor transports to raise 'Event loop is closed' from __del__
    after a successful `chunkhound index` (Python 3.10).
    """
    import asyncio

    from chunkhound import pipeline_bridge

    loop = asyncio.new_event_loop()
    provider = MagicMock()
    provider.shutdown = AsyncMock()
    cache = pipeline_bridge._EmbedThreadCache()
    cache.providers[999001] = provider
    cache.loops[999001] = loop
    try:
        pipeline_bridge._shutdown_embed_thread_resources(cache)
        provider.shutdown.assert_called_once()
        assert loop.is_closed()
        assert cache.providers == {}
        assert cache.loops == {}
    finally:
        if not loop.is_closed():
            loop.close()


def test_hung_embed_shutdown_is_bounded_by_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider.shutdown() that never returns must not hang cleanup
    forever -- regression test for the missing-timeout half of the
    PR #380 review's embed-cache finding."""
    import asyncio
    import time

    from chunkhound import pipeline_bridge

    monkeypatch.setattr(
        pipeline_bridge, "_EMBED_PROVIDER_SHUTDOWN_TIMEOUT_SECS", 0.05
    )

    async def _hang() -> None:
        await asyncio.sleep(9999)

    loop = asyncio.new_event_loop()
    provider = MagicMock()
    provider.shutdown = _hang
    cache = pipeline_bridge._EmbedThreadCache()
    cache.providers[999004] = provider
    cache.loops[999004] = loop

    start = time.monotonic()
    pipeline_bridge._shutdown_embed_thread_resources(cache)
    elapsed = time.monotonic() - start

    assert elapsed < 5  # bounded by the (monkeypatched) 0.05s timeout, not 9999s
    assert cache.providers == {}
    assert cache.loops == {}
    if not loop.is_closed():
        loop.close()


@pytest.mark.asyncio
async def test_embed_callback_uses_caller_config_not_registry(
    tmp_path: Path,
    fake_native_pipeline: _FakeNativePipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rust embed callbacks must use run_rust_pipeline's config.embedding.

    Looking up get_registry()._config instead ignores the coordinator's
    embedding settings whenever those two configs diverge (MCP with more
    than one config, tests, explicit Config objects).
    """
    from chunkhound import pipeline_bridge
    from chunkhound.core.config.embedding_config import EmbeddingConfig
    from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory

    caller_cfg = EmbeddingConfig(
        provider="openai",
        model="caller-model",
        api_key="sk-caller",
        base_url="http://caller.example/v1",
        max_concurrent_batches=2,
    )
    registry_cfg = EmbeddingConfig(
        provider="openai",
        model="registry-model",
        api_key="sk-registry",
        base_url="http://registry.example/v1",
        max_concurrent_batches=2,
    )
    fake_registry = types.ModuleType("chunkhound.registry")
    fake_registry.get_registry = lambda: SimpleNamespace(  # type: ignore[attr-defined]
        _config=SimpleNamespace(embedding=registry_cfg)
    )
    monkeypatch.setitem(sys.modules, "chunkhound.registry", fake_registry)

    created_models: list[str | None] = []

    def _create_provider(config: EmbeddingConfig) -> MagicMock:
        created_models.append(config.model)
        provider = MagicMock()
        provider.embed = AsyncMock(return_value=[[0.1, 0.2]])
        provider.shutdown = AsyncMock()
        return provider

    monkeypatch.setattr(
        EmbeddingProviderFactory,
        "create_provider",
        staticmethod(_create_provider),
    )

    await pipeline_bridge.run_rust_pipeline(
        files_to_process=[],
        db_path=tmp_path,
        project_root=tmp_path,
        skip_embeddings=False,
        config=SimpleNamespace(
            database=SimpleNamespace(),
            indexing=SimpleNamespace(),
            embedding=caller_cfg,
        ),
    )

    cb = fake_native_pipeline.instance.run.call_args.kwargs["embed_batch_callback"]
    cache = cb.keywords["cache"]
    try:
        cb(["hello"])
    finally:
        cache.providers.clear()
        for loop in cache.loops.values():
            if not loop.is_closed():
                loop.close()
        cache.loops.clear()

    assert created_models == ["caller-model"]


@pytest.mark.asyncio
async def test_run_rust_pipeline_closes_its_own_embed_resources_after_run(
    tmp_path: Path,
    fake_native_pipeline: _FakeNativePipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider/loop actually created by this run's own embed callback
    must be shut down -- not just dropped for GC -- and cleared from that
    run's cache by the time run_rust_pipeline() returns. Regression test for
    PR #380 review finding #6: run_rust_pipeline's finally must actually
    invoke _shutdown_embed_thread_resources on the cache it created, not
    just leave entries for a future run to overwrite.
    """
    from chunkhound import pipeline_bridge
    from chunkhound.core.config.embedding_config import EmbeddingConfig
    from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory

    created: list[MagicMock] = []

    def _create_provider(config: EmbeddingConfig) -> MagicMock:
        provider = MagicMock()
        provider.embed = AsyncMock(return_value=[[0.1, 0.2]])
        provider.shutdown = AsyncMock()
        created.append(provider)
        return provider

    monkeypatch.setattr(
        EmbeddingProviderFactory, "create_provider", staticmethod(_create_provider)
    )

    captured_caches: list[pipeline_bridge._EmbedThreadCache] = []

    def _run_side_effect(
        *,
        files: object,
        parse_batch_callback: object,
        embed_batch_callback: object,
        progress_callback: object,
        incremental: object,
    ) -> SimpleNamespace:
        assert embed_batch_callback is not None
        captured_caches.append(embed_batch_callback.keywords["cache"])
        embed_batch_callback(["hello"])
        return _fake_report()

    fake_native_pipeline.instance.run.side_effect = _run_side_effect

    await pipeline_bridge.run_rust_pipeline(
        files_to_process=[],
        db_path=tmp_path,
        project_root=tmp_path,
        skip_embeddings=False,
        config=SimpleNamespace(
            database=SimpleNamespace(),
            indexing=SimpleNamespace(),
            embedding=EmbeddingConfig(
                provider="openai",
                model="m",
                api_key="sk-test",
                max_concurrent_batches=1,
            ),
        ),
    )

    assert len(created) == 1
    created[0].shutdown.assert_awaited_once()
    cache = captured_caches[0]
    assert cache.providers == {}
    assert cache.loops == {}


@pytest.mark.asyncio
async def test_run_rust_pipeline_does_not_touch_a_sibling_runs_embed_cache(
    tmp_path: Path, fake_native_pipeline: _FakeNativePipeline
) -> None:
    """run_rust_pipeline() must only drain/close its OWN embed cache, never
    a different, concurrently in-flight run's providers/loops.

    Regression test for the cross-run race in an earlier fix for PR #380
    review finding #6: that fix stopped a thread-id recycled from an OLDER,
    already-finished run from colliding with a newer one, but the cache it
    drained/closed was still a single module-level dict shared by every
    run_rust_pipeline() call -- so one run's cleanup could tear down a
    still-live provider/event loop belonging to a *different*, concurrently
    running call. Each run must own an isolated _EmbedThreadCache instance
    that no other run ever reaches into.
    """
    from chunkhound import pipeline_bridge

    sibling_cache = pipeline_bridge._EmbedThreadCache()
    sibling_provider = AsyncMock()
    sibling_loop = asyncio.new_event_loop()
    sibling_cache.providers[999999] = sibling_provider
    sibling_cache.loops[999999] = sibling_loop

    try:
        await pipeline_bridge.run_rust_pipeline(
            files_to_process=[],
            db_path=tmp_path,
            project_root=tmp_path,
            skip_embeddings=True,
            config=None,
        )

        sibling_provider.shutdown.assert_not_awaited()
        assert not sibling_loop.is_closed()
        assert 999999 in sibling_cache.providers
        assert 999999 in sibling_cache.loops
    finally:
        if not sibling_loop.is_closed():
            sibling_loop.close()
