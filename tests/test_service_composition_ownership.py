"""Caller-owned service graphs retain configuration and provider identity."""

import asyncio
import gc
import sys
import threading
import weakref

import duckdb
import pytest

from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services
from chunkhound.providers.database.lifecycle import DatabaseRecoveryRequiredError
from chunkhound.registry import ProviderRegistry, get_registry
from tests.hub.artifacts import build_database
from tests.hub.conftest import embedding_http as _embedding_http_fixture

embedding_http = _embedding_http_fixture


@pytest.mark.parametrize("deferred", [False, True])
def test_independent_graphs_keep_their_config_through_connect_and_retry(
    tmp_path, monkeypatch, deferred
):
    global_registry = get_registry()
    before = dict(global_registry._providers)
    bundles = []
    configs = []
    for name, batch in (("alpha", 111), ("beta", 222)):
        root = tmp_path / name
        root.mkdir()
        config = Config(
            target_dir=root,
            database={"path": root / "db"},
            indexing={"db_batch_size": batch},
            embeddings_disabled=True,
            analytics={"enabled": False},
            skip_layers={"env", "global", "local_config", "config_file", "cli"},
        )
        configs.append(config)
        bundles.append(
            create_services(
                config.database.get_db_path(), config, connect_database=not deferred
            )
        )
    real_connect = duckdb.connect
    target = str(configs[0].database.get_db_path())
    fail = deferred

    def connect(database, *args, **kwargs):
        nonlocal fail
        if fail and str(database) == target:
            fail = False
            raise OSError("one owner's transient open failure")
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(duckdb, "connect", connect)
    try:
        if deferred:
            with pytest.raises(OSError, match="transient open failure"):
                bundles[0].provider.connect()
        for services, config in zip(bundles, configs):
            assert services.search_service.search_regex("never_matches")[0] == []
            provider = services.provider
            # These identities protect the public convenience operation's
            # configuration, not how a helper is called.
            assert provider._indexing_coordinator is services.indexing_coordinator
            assert provider._search_service is services.search_service
            assert provider._embedding_service is services.embedding_service
            assert services.indexing_coordinator.config is config
            assert provider.get_base_directory() == config.target_dir
            assert (
                services.indexing_coordinator.config.indexing.db_batch_size
                == config.indexing.db_batch_size
            )
        assert global_registry._providers == before
    finally:
        for services in bundles:
            services.provider.close()


def test_registry_stays_eager_and_retains_its_graph(tmp_path):
    registry = ProviderRegistry()
    config = Config(
        target_dir=tmp_path,
        database={"path": tmp_path / "db"},
        embeddings_disabled=True,
        analytics={"enabled": False},
        skip_layers={"env", "global", "local_config", "config_file", "cli"},
    )
    registry.configure(config)
    provider = registry.get_provider("database")
    try:
        assert provider.is_connected
        assert registry.create_indexing_coordinator() is provider._indexing_coordinator
        provider.connect()
        assert registry.create_search_service() is provider._search_service
    finally:
        provider.close()


def test_supplied_embedding_manager_is_shared_by_lancedb_and_services(tmp_path):
    from chunkhound.embeddings import EmbeddingManager

    manager = EmbeddingManager()
    config = Config(
        target_dir=tmp_path,
        database={"path": tmp_path / "db", "provider": "lancedb"},
        embeddings_disabled=True,
        analytics={"enabled": False},
        skip_layers={"env", "global", "local_config", "config_file", "cli"},
    )
    services = create_services(
        config.database.get_db_path(),
        config,
        embedding_manager=manager,
        connect_database=False,
    )
    try:
        assert services.provider.embedding_manager is manager
        assert services.search_service.search_regex("never_matches")[0] == []
        assert services.provider.embedding_manager is manager
    finally:
        services.provider.close()


@pytest.fixture
def seeded_config(tmp_path):
    root = tmp_path / "database"
    build_database(
        root, provider="voyageai", model="voyage-code-3", dims=1024,
        symbol_prefix="completion_marker",
    )
    return Config(
        target_dir=root,
        database={"path": root},
        embedding={
            "provider": "voyageai", "model": "voyage-code-4",
            "output_dims": 2048, "api_key": "local-test-key",
        },
        analytics={"enabled": False},
        skip_layers={"env", "global", "local_config", "config_file", "cli"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["configure", "async-action"])
@pytest.mark.parametrize(
    "failure_type",
    [asyncio.CancelledError, KeyboardInterrupt, SystemExit, RuntimeError],
)
async def test_failed_decision_retries_on_next_action_without_reopening_or_replay(
    seeded_config, monkeypatch, entry, failure_type
):
    registry = ProviderRegistry()
    decisions = []
    searches = []
    payloads = []
    real_execute = duckdb.DuckDBPyConnection.execute

    class DecisionPayload:
        def __str__(self):
            return "decision interrupted https://private.invalid/decision-secret"

    def execute(conn, query, *args, **kwargs):
        if "regexp_matches" in query:
            searches.append(query)
        return real_execute(conn, query, *args, **kwargs)

    monkeypatch.setattr(duckdb.DuckDBPyConnection, "execute", execute)

    def decide(drift):
        database = registry.get_provider("database")
        decisions.append(drift)
        if len(decisions) == 1:
            database.execute_query("CREATE TEMP TABLE retained AS SELECT 42 AS value")
            payload = DecisionPayload()
            payloads.append(weakref.ref(payload))
            error = failure_type(payload)
            error.add_note("api_key=private-note-secret")
            raise error from ValueError("context https://private.invalid/cause-secret")
        # A reopened executor loses this session-local state.
        assert database.execute_query("SELECT value FROM retained") == [{"value": 42}]
        return False

    expected = (
        DatabaseRecoveryRequiredError if failure_type is RuntimeError else failure_type
    )
    try:
        if entry == "configure":
            with pytest.raises(expected, match="decision interrupted") as caught:
                registry.configure(seeded_config, decide)
        else:
            registry.configure(seeded_config, decide, connect_database=False)
            with pytest.raises(expected, match="decision interrupted") as caught:
                await registry.get_provider("database").search_regex_async(
                    "failed-action-must-not-replay"
                )
        if failure_type is not RuntimeError:
            assert type(caught.value) is failure_type
            assert str(caught.value) == (
                "decision interrupted [redacted URL] [api_key=[REDACTED]]"
            )
            assert caught.value.args == (str(caught.value),)
            if failure_type is SystemExit:
                # Exit intent survives; opaque/raw exit codes do not cross readiness.
                assert caught.value.code == str(caught.value)
            assert type(caught.value.__cause__) is ValueError
            assert str(caught.value.__cause__) == "context [redacted URL]"
            assert caught.value.__cause__.__traceback__ is None
        gc.collect()
        assert payloads[0]() is None, "readiness must not retain raw callback arguments"
        provider = registry.get_provider("database")
        coordinator = registry.create_indexing_coordinator()
        search = registry.create_search_service()
        embedding = registry.get_provider("embedding")
        assert len(decisions) == 1
        assert searches == [], "the failed caller's operation must not run"
        assert embedding.model == "voyage-code-4"
        assert provider.is_connected

        hits, _ = await provider.search_regex_async("completion_marker_0")
        assert hits
        assert len(decisions) == 2
        assert all("failed-action-must-not-replay" not in query for query in searches)
        assert (embedding.model, embedding.dims) == ("voyage-code-3", 1024)
        assert seeded_config.embedding.model == "voyage-code-3"
        assert seeded_config.embedding.output_dims == 1024
        assert registry.get_provider("database") is provider
        assert provider._indexing_coordinator is coordinator
        assert provider._search_service is search
        assert search.search_regex("completion_marker_0")[0]
        assert len(decisions) == 2
    finally:
        registry.get_provider("database").close()


@pytest.mark.asyncio
@pytest.mark.parametrize("refuse_recovery", [False, True])
async def test_semantic_action_admits_database_before_model_and_embedding_request(
    tmp_path, monkeypatch, embedding_http, refuse_recovery
):
    root = tmp_path / "database"
    target = build_database(root, symbol_prefix="ready_marker")
    config = Config(
        target_dir=root,
        database={"path": root},
        embedding={
            "provider": "openai", "model": "unresolved-model", "output_dims": 16,
            "api_key": "local-test-key", "base_url": embedding_http.url,
        },
        analytics={"enabled": False},
        skip_layers={"env", "global", "local_config", "config_file", "cli"},
    )
    services = create_services(target, config, connect_database=False)
    provider = services.provider
    real_connect = duckdb.connect
    fail = True

    def connect(database, *args, **kwargs):
        if fail and str(database) == str(target):
            raise OSError("database admission unavailable")
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(duckdb, "connect", connect)
    try:
        with pytest.raises(OSError, match="database admission unavailable"):
            provider.connect()
        assert config.embedding.model == "unresolved-model"
        assert embedding_http.requests == []
        fail = refuse_recovery
        if refuse_recovery:
            with pytest.raises(OSError, match="database admission unavailable"):
                await services.search_service.search_semantic(
                    "ready_marker", force_strategy="single_hop"
                )
            assert embedding_http.requests == [], "refused search must not spend tokens"
        else:
            hits, _ = await services.search_service.search_semantic(
                "ready_marker", force_strategy="single_hop"
            )
            assert hits, "query vectors and captured search model must match the index"
            assert all("ready_marker" in hit["content"] for hit in hits)
            assert embedding_http.requests
            assert all(r["model"] == "hub-test" for r in embedding_http.requests)
            assert all(r.get("dimensions", 8) == 8 for r in embedding_http.requests)
            assert config.embedding.model == "hub-test"
            assert config.embedding.output_dims == 8
            assert services.provider is provider
    finally:
        fail = False
        provider.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["initial", "recovery"])
@pytest.mark.parametrize("external_cancel", [False, True])
@pytest.mark.parametrize(
    "failure_type", [asyncio.CancelledError, KeyboardInterrupt, SystemExit]
)
async def test_rust_completion_preserves_sanitized_control_flow_and_retries(
    seeded_config, monkeypatch, entry, failure_type, external_cancel
):
    import chunkhound_native as native
    from tests.hub.test_next_action_recovery import run_empty_phase

    services = create_services(seeded_config.database.get_db_path(), seeded_config)
    provider = services.provider
    embedding = services.search_service._embedding_provider
    original_update = embedding.update_config
    original_update(model="voyage-code-4", output_dims=2048)
    payloads, decisions, searches = [], [], []
    mode = "ordinary" if entry == "recovery" else "control"
    entered, release = threading.Event(), threading.Event()
    cancelling = None
    caller = asyncio.current_task()

    async def cancel_at_completion():
        assert await asyncio.to_thread(entered.wait, 15)
        caller.cancel()
        await asyncio.sleep(0)
        assert not caller.done(), "cancellation must drain owner completion"
        release.set()

    real_execute = duckdb.DuckDBPyConnection.execute

    class Payload:
        def __str__(self):
            return "completion interrupted https://private.invalid/completion-secret"

    def update(**kwargs):
        decisions.append(kwargs)
        if mode == "ordinary":
            raise RuntimeError("initial owner completion unavailable")
        if mode == "control":
            if external_cancel:
                entered.set()
                assert release.wait(20), "completion boundary was not released"
            payload = Payload()
            payloads.append(weakref.ref(payload))
            error = failure_type(payload)
            error.add_note("api_key=private-note-secret")
            raise error from ValueError("context https://private.invalid/cause-secret")
        return original_update(**kwargs)

    def execute(conn, query, *args, **kwargs):
        if "regexp_matches" in query:
            searches.append(query)
        return real_execute(conn, query, *args, **kwargs)

    monkeypatch.setattr(embedding, "update_config", update)
    monkeypatch.setattr(duckdb.DuckDBPyConnection, "execute", execute)
    try:
        if entry == "recovery":
            with pytest.raises(DatabaseRecoveryRequiredError):
                await run_empty_phase(services, seeded_config, seeded_config.target_dir)
            mode = "control"
        if external_cancel:
            cancelling = asyncio.create_task(cancel_at_completion())
        expected = asyncio.CancelledError if external_cancel else failure_type
        with pytest.raises(expected) as caught:
            if entry == "initial":
                await run_empty_phase(services, seeded_config, seeded_config.target_dir)
            else:
                await services.search_service.search_regex_async("cancelled-action")
        assert type(caught.value) is expected
        if not external_cancel:
            assert str(caught.value) == (
                "completion interrupted [redacted URL] [api_key=[REDACTED]]"
            )
            assert caught.value.args == (str(caught.value),)
            if failure_type is SystemExit:
                assert caught.value.code == str(caught.value)
            assert type(caught.value.__cause__) is ValueError
            assert str(caught.value.__cause__) == "context [redacted URL]"
            assert caught.value.__cause__.__traceback__ is None
        assert caught.value.__context__ is None
        if cancelling is not None:
            await cancelling
        # Let completed executor/shield callbacks leave the event loop queue;
        # retained diagnostics and the delivered exception remain live below.
        await asyncio.sleep(0)
        gc.collect()
        assert payloads and all(ref() is None for ref in payloads)
        resources = provider._rust_pipeline_cleanup
        assert resources.token.cleanup_complete, "use actual native acknowledgment"
        assert resources.recovery_required and provider.is_rust_pipeline_in_progress()
        assert resources.cleanup_failure.kind is failure_type
        assert "completion interrupted [redacted URL]" in resources.cleanup_failure.message
        assert resources.failure_stage == "trusted database reconnect and configuration"
        assert resources.cleanup_failure.cause[0].message == "context [redacted URL]"
        assert resources.reconnect_future is None
        assert searches == [], "control-flow action must not run"
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(provider.db_path), "duckdb")
        mode = "success"
        hits, _ = await services.search_service.search_regex_async("completion_marker_0")
        assert hits
        assert len(decisions) == (3 if entry == "recovery" else 2)
        assert all("cancelled-action" not in query for query in searches)
        assert services.provider is provider
        assert provider._indexing_coordinator is services.indexing_coordinator
        assert provider._search_service is services.search_service
        assert provider._embedding_service is services.embedding_service
        assert services.search_service._embedding_provider is embedding
        assert (embedding.model, embedding.dims) == ("voyage-code-3", 1024)
        assert not provider.is_rust_pipeline_in_progress()
    finally:
        mode = "success"
        release.set()
        if cancelling is not None:
            await asyncio.gather(cancelling, return_exceptions=True)
        provider.close()
    native.acquire_install_lease(str(provider.db_path), "duckdb").close()


@pytest.mark.parametrize(
    ("initial_type", "recovery_type"),
    [(asyncio.CancelledError, RuntimeError), (RuntimeError, SystemExit)],
)
def test_rust_caller_keeps_own_failure_when_public_recovery_overtakes_delivery(
    seeded_config, monkeypatch, initial_type, recovery_type
):
    import chunkhound_native as native

    from chunkhound.pipeline_bridge import RustPipelineCleanupError
    from chunkhound.services.rust_pipeline_runner import run_rust_indexing_phase
    from tests.hub.test_next_action_recovery import run_empty_phase

    services = create_services(seeded_config.database.get_db_path(), seeded_config)
    provider = services.provider
    executor = provider._executor
    embedding = services.search_service._embedding_provider
    original_update = embedding.update_config
    original_update(model="voyage-code-4", output_dims=2048)
    published, resume = threading.Event(), threading.Event()
    runner_errors, decisions, searches, payloads = [], [], [], []
    mode = "initial"
    real_execute = duckdb.DuckDBPyConnection.execute

    class Payload:
        def __init__(self, owner):
            self.owner = owner

        def __str__(self):
            return f"{self.owner} completion https://private.invalid/payload-secret"

    def update(**kwargs):
        decisions.append(mode)
        if mode != "success":
            payload = Payload(mode)
            payloads.append(weakref.ref(payload))
            kind = initial_type if mode == "initial" else recovery_type
            error = kind(payload)
            error.add_note("api_key=private-note-secret")
            raise error from ValueError(
                f"{mode} cause https://private.invalid/cause-secret"
            )
        return original_update(**kwargs)

    def execute(conn, query, *args, **kwargs):
        if "regexp_matches" in query:
            searches.append(query)
        return real_execute(conn, query, *args, **kwargs)

    async def invoke():
        try:
            await run_empty_phase(services, seeded_config, seeded_config.target_dir)
        except BaseException as error:
            runner_errors.append(error)

    def runner():
        previous = sys.gettrace()
        target_code = run_rust_indexing_phase.__code__

        def trace(frame, event, arg):
            # Only this thread is instrumented. Delegate other frames unchanged,
            # and preserve the prior hook's per-frame trace for the target too.
            prior_local = previous(frame, event, arg) if previous else None
            if frame.f_code is not target_code:
                return prior_local

            def trace_runner(frame, event, arg):
                nonlocal prior_local
                if prior_local is not None:
                    prior_local = prior_local(frame, event, arg)
                resources = frame.f_locals.get("resources")
                if (
                    event == "line" and not published.is_set()
                    and resources is not None and resources.recovery_required
                ):
                    # Real native work and cleanup have completed. Park before
                    # the old caller delivers; do not replace lifecycle logic.
                    published.set()
                    assert resume.wait(20), "failed-resource publication not released"
                return trace_runner

            return trace_runner

        try:
            sys.settrace(trace)
            asyncio.run(invoke())
        finally:
            sys.settrace(previous)

    monkeypatch.setattr(embedding, "update_config", update)
    monkeypatch.setattr(duckdb.DuckDBPyConnection, "execute", execute)
    thread = threading.Thread(target=runner)
    thread.start()
    try:
        assert published.wait(15), "original caller did not publish failed resources"
        resources = provider._rust_pipeline_cleanup
        initial_failure = resources.cleanup_failure
        assert initial_failure.kind is initial_type
        assert resources.native_started and resources.token.cleanup_complete
        assert resources.cache_closed
        assert all(getattr(resources, name) is None for name in (
            "release_future", "native_future", "cleanup_future", "reconnect_future"
        ))
        mode = "recovery"
        expected_b = (
            DatabaseRecoveryRequiredError if recovery_type is RuntimeError
            else recovery_type
        )
        with pytest.raises(expected_b) as caught_b:
            services.search_service.search_regex("failed-recovery-must-not-run")
        assert type(caught_b.value) is expected_b
        latest_failure = resources.cleanup_failure
        assert latest_failure is not initial_failure
        assert latest_failure.kind is recovery_type
        assert provider._rust_pipeline_cleanup is resources
        assert resources.recovery_required and provider.is_rust_pipeline_in_progress()
        assert searches == [], "neither failed caller may execute a search"
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(provider.db_path), "duckdb")
        resume.set()
        thread.join(15)
        assert not thread.is_alive() and len(runner_errors) == 1
        assert resources.cleanup_failure is latest_failure
        assert resources.recovery_required and provider.is_rust_pipeline_in_progress()
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(provider.db_path), "duckdb")
        mode = "success"
        assert services.search_service.search_regex("completion_marker_0")[0]
        assert decisions == ["initial", "recovery", "success"]
        assert all("failed-recovery-must-not-run" not in query for query in searches)
        assert services.provider is provider and provider._executor is executor
        assert provider._indexing_coordinator is services.indexing_coordinator
        assert provider._search_service is services.search_service
        assert provider._embedding_service is services.embedding_service
        assert services.search_service._embedding_provider is embedding
        assert (embedding.model, embedding.dims) == ("voyage-code-3", 1024)
        assert provider._rust_pipeline_cleanup is None
        assert not provider.is_rust_pipeline_in_progress()
        gc.collect()
        assert all(ref() is None for ref in payloads), "raw payloads must be detached"

        expected_a = (
            RustPipelineCleanupError if initial_type is RuntimeError else initial_type
        )
        assert type(runner_errors[0]) is expected_a, "A must deliver A's category"
        for error, owner, kind, failure in (
            (runner_errors[0], "initial", initial_type, initial_failure),
            (caught_b.value, "recovery", recovery_type, latest_failure),
        ):
            assert error.__context__ is None
            message = f"{owner} completion [redacted URL] [api_key=[REDACTED]]"
            if kind is RuntimeError:
                assert error.failure.cleanup is failure
                detached = error.__cause__
                assert type(detached) is RuntimeError
                assert detached.__traceback__ is None
            else:
                detached = error
                assert detached.args == (message,)
                if kind is SystemExit:
                    assert detached.code == message
            assert str(detached) == message
            assert type(detached.__cause__) is ValueError
            assert str(detached.__cause__) == f"{owner} cause [redacted URL]"
            assert detached.__cause__.__traceback__ is None
            assert detached.__cause__.__context__ is None
    finally:
        mode = "success"
        resume.set()
        thread.join(20)
        provider.close()
    native.acquire_install_lease(str(provider.db_path), "duckdb").close()
    with pytest.raises(RuntimeError, match="clos"):
        services.search_service.search_regex("completion_marker_0")


def test_old_rust_completion_cannot_skip_newer_failed_model_resolution(
    tmp_path, monkeypatch, embedding_http
):
    from chunkhound.providers.database.serial_database_provider import (
        SerialDatabaseProvider,
    )
    from tests.hub.test_next_action_recovery import run_empty_phase

    root = tmp_path / "database"
    target = build_database(root, symbol_prefix="ready_marker")
    config = Config(
        target_dir=root, database={"path": root},
        embedding={
            "provider": "openai", "model": "hub-test", "output_dims": 8,
            "api_key": "local-test-key", "base_url": embedding_http.url,
        },
        analytics={"enabled": False},
        skip_layers={"env", "global", "local_config", "config_file", "cli"},
    )
    services = create_services(target, config)
    provider = services.provider
    embedding = services.search_service._embedding_provider
    original_update = embedding.update_config
    handoff, resume = threading.Event(), threading.Event()
    runner_errors = []
    decisions = []
    fail = True

    def update(**kwargs):
        decisions.append(kwargs)
        if fail:
            raise RuntimeError("newer model completion failed")
        return original_update(**kwargs)

    def runner():
        previous = sys.gettrace()
        target_code = SerialDatabaseProvider.set_rust_pipeline_in_progress.__code__

        def trace(frame, event, arg):
            if frame.f_code is target_code and event == "return":
                if frame.f_locals["self"] is provider and not frame.f_locals["active"]:
                    # The real guard release completed, but the old runner has
                    # not resumed. No readiness/backend behavior is replaced.
                    handoff.set()
                    assert resume.wait(20), "guard handoff was not released"
            return trace

        try:
            sys.settrace(trace)
            asyncio.run(run_empty_phase(services, config, root))
        except BaseException as error:
            runner_errors.append(error)
        finally:
            sys.settrace(previous)

    thread = threading.Thread(target=runner)
    thread.start()
    try:
        assert handoff.wait(15)
        assert not provider.is_rust_pipeline_in_progress()
        original_update(model="unresolved-model", output_dims=16)
        monkeypatch.setattr(embedding, "update_config", update)
        with pytest.raises(DatabaseRecoveryRequiredError, match="newer model"):
            provider.connect()
        assert len(decisions) == 1
        assert embedding_http.requests == []
        fail = False
        resume.set()
        thread.join(15)
        assert not thread.is_alive() and runner_errors == []
        hits, _ = asyncio.run(services.search_service.search_semantic(
            "ready_marker", force_strategy="single_hop"
        ))
        assert len(decisions) == 2, "later action must retry the newer completion"
        assert hits
        assert all(r["model"] == "hub-test" for r in embedding_http.requests)
        assert all(r.get("dimensions", 8) == 8 for r in embedding_http.requests)
        assert services.provider is provider
        assert provider._search_service is services.search_service
        assert provider._indexing_coordinator is services.indexing_coordinator
        assert provider._embedding_service is services.embedding_service
    finally:
        fail = False
        resume.set()
        thread.join(20)
        provider.close()
    with pytest.raises(RuntimeError, match="clos"):
        services.search_service.search_regex("ready_marker")


def test_pending_completion_is_not_skipped_when_manager_handle_is_missing(
    seeded_config,
):
    registry = ProviderRegistry()
    decisions = []

    def decide(drift):
        decisions.append(drift)
        if len(decisions) == 1:
            raise RuntimeError("decision unavailable")
        return False

    try:
        with pytest.raises(DatabaseRecoveryRequiredError, match="decision unavailable"):
            registry.configure(seeded_config, decide)
        provider = registry.get_provider("database")
        provider._connection_manager.disconnect(skip_checkpoint=True)
        with pytest.raises(DatabaseRecoveryRequiredError, match="Connect the database"):
            provider.search_regex("completion_marker_0")
        assert len(decisions) == 1
        assert registry.get_provider("embedding").model == "voyage-code-4"
        provider.connect()
        assert provider.search_regex("completion_marker_0")[0]
        assert len(decisions) == 2
        assert registry.get_provider("embedding").model == "voyage-code-3"
    finally:
        registry.get_provider("database").close()


def test_refused_reconnect_does_not_repeat_completed_drift_on_retained_session(
    seeded_config,
):
    from chunkhound.core.index_format import IndexFormatError

    registry = ProviderRegistry()
    registry.configure(seeded_config)
    provider = registry.get_provider("database")
    embedding = registry.get_provider("embedding")
    try:
        embedding.update_config(model="voyage-code-4", output_dims=2048)
        provider.begin_transaction()
        provider.execute_query("CREATE TEMP TABLE retained AS SELECT 42 AS value")
        provider._connection_manager.disconnect(skip_checkpoint=True)
        with pytest.raises(IndexFormatError, match="transaction"):
            provider.connect()
        assert provider.execute_query("SELECT value FROM retained") == [{"value": 42}]
        assert embedding.model == "voyage-code-4", "no new completion was owed"
        provider.rollback_transaction()
        provider.connect()
        assert (embedding.model, embedding.dims) == ("voyage-code-3", 1024)
        assert provider.search_regex("completion_marker_0")[0]
    finally:
        provider.close()
