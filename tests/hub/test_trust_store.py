from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import stat
from pathlib import Path
from typing import Any

import pytest

import chunkhound.hub.trust as trust_module
from chunkhound.hub.trust import (
    HubTrustError,
    forget_checkout_trust,
    grant_trust,
    is_trusted,
    trust_store_path,
)


def _grant_worker(
    home: str,
    checkout: str,
    manifest_url: str,
    barrier: Any,
    results: Any,
) -> None:
    os.environ["HOME"] = home
    try:
        barrier.wait(timeout=10)
        grant_trust(Path(checkout), manifest_url)
    except BaseException as exc:
        results.put((type(exc).__name__, str(exc)))
    else:
        results.put(None)


def test_missing_store_is_untrusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    assert not is_trusted(tmp_path / "checkout", "https://hub.invalid/manifest")
    assert not trust_store_path().exists()


def test_grant_matches_only_canonical_checkout_and_exact_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    alias = tmp_path / "checkout-alias"
    try:
        alias.symlink_to(checkout, target_is_directory=True)
    except (NotImplementedError, OSError):
        alias = checkout / "."
    url = "https://hub.invalid/manifest?token=secret"

    grant_trust(alias, url)

    assert is_trusted(checkout.resolve(), url)
    assert not is_trusted(checkout, url + "&revision=2")
    assert not is_trusted(tmp_path / "other-checkout", url)
    text = trust_store_path().read_text()
    assert "token=secret" not in text
    assert hashlib.sha256(url.encode("utf-8")).hexdigest() in text
    assert str(checkout) not in text


def test_moving_checkout_invalidates_its_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    url = "https://hub.invalid/manifest"
    grant_trust(checkout, url)

    moved = tmp_path / "moved-checkout"
    checkout.rename(moved)

    assert not is_trusted(moved, url)


def test_forget_is_idempotent_and_does_not_create_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    checkout = tmp_path / "checkout"

    assert not forget_checkout_trust(checkout)
    assert not trust_store_path().exists()


def test_forget_removes_only_selected_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    checkout_a = tmp_path / "checkout-a"
    checkout_b = tmp_path / "checkout-b"
    url_a = "https://hub.invalid/a"
    url_b = "https://hub.invalid/b"
    grant_trust(checkout_a, url_a)
    grant_trust(checkout_b, url_b)

    assert forget_checkout_trust(checkout_a)
    assert not forget_checkout_trust(checkout_a)
    assert not is_trusted(checkout_a, url_a)
    assert is_trusted(checkout_b, url_b)


@pytest.mark.parametrize(
    "payload",
    [
        "{broken",
        "[]",
        '{"version": 2, "grants": {}}',
        '{"version": true, "grants": {}}',
        '{"version": 1.0, "grants": {}}',
        '{"version": 1, "grants": []}',
        '{"version": 1, "grants": {7: []}}',
        json.dumps({"version": 1, "grants": {"checkout": ["a" * 64]}}),
        json.dumps({"version": 1, "grants": {"A" * 64: ["a" * 64]}}),
        json.dumps({"version": 1, "grants": {"a" * 63: ["a" * 64]}}),
        '{"version": 1, "grants": {"checkout": "digest"}}',
        '{"version": 1, "grants": {"checkout": [7]}}',
        json.dumps({"version": 1, "grants": {"checkout": ["A" * 64]}}),
        '{"version": 1, "grants": {"checkout": ["abc"]}}',
    ],
)
def test_invalid_store_fails_closed_and_is_not_overwritten(
    payload: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    store = trust_store_path()
    store.parent.mkdir(parents=True)
    store.write_text(payload)

    with pytest.raises(HubTrustError, match="^Hub trust store cannot be read safely$"):
        is_trusted(tmp_path / "checkout", "https://hub.invalid/m")
    with pytest.raises(HubTrustError, match="^Hub trust store cannot be read safely$"):
        grant_trust(tmp_path / "checkout", "https://hub.invalid/m")
    assert store.read_text() == payload


@pytest.mark.parametrize(
    ("operation_name", "expected_message"),
    [
        ("is_trusted", "Hub trust store cannot be read safely"),
        ("grant_trust", "Hub trust store cannot be updated safely"),
        ("forget_checkout_trust", "Hub trust store cannot be updated safely"),
    ],
)
def test_trust_store_path_failures_are_sanitized(
    operation_name: str,
    expected_message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    grant_trust(tmp_path / "original", "https://hub.invalid/original")
    store = trust_store_path()
    original_text = store.read_text()
    secret_path = tmp_path / "secret-home"

    def fail_trust_store_path() -> Path:
        raise OSError(f"must not escape: {secret_path}")

    monkeypatch.setattr(trust_module, "trust_store_path", fail_trust_store_path)
    operation = getattr(trust_module, operation_name)

    with pytest.raises(HubTrustError) as caught:
        if operation_name == "forget_checkout_trust":
            operation(tmp_path / "checkout")
        else:
            operation(tmp_path / "checkout", "https://hub.invalid/manifest")

    assert str(caught.value) == expected_message
    assert str(secret_path) not in str(caught.value)
    assert store.read_text() == original_text


@pytest.mark.parametrize(
    ("operation_name", "expected_message"),
    [
        ("is_trusted", "Hub trust store cannot be read safely"),
        ("grant_trust", "Hub trust store cannot be updated safely"),
        ("forget_checkout_trust", "Hub trust store cannot be updated safely"),
    ],
)
def test_checkout_resolution_failures_are_sanitized_without_overwriting(
    operation_name: str,
    expected_message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    grant_trust(tmp_path / "original", "https://hub.invalid/original")
    store = trust_store_path()
    original_text = store.read_text()
    secret_path = tmp_path / "secret-checkout"

    def fail_resolve(self: Path, strict: bool = False) -> Path:
        raise OSError(f"must not escape: {secret_path}")

    monkeypatch.setattr(Path, "resolve", fail_resolve)
    operation = getattr(trust_module, operation_name)

    with pytest.raises(HubTrustError) as caught:
        if operation_name == "forget_checkout_trust":
            operation(secret_path)
        else:
            operation(secret_path, "https://hub.invalid/manifest")

    assert str(caught.value) == expected_message
    assert str(secret_path) not in str(caught.value)
    assert store.read_text() == original_text


def test_unreadable_store_path_has_sanitized_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    store = trust_store_path()
    store.mkdir(parents=True)
    secret_url = "https://hub.invalid/manifest?token=secret"

    with pytest.raises(HubTrustError) as caught:
        is_trusted(tmp_path / "checkout", secret_url)

    assert str(caught.value) == "Hub trust store cannot be read safely"
    assert str(store) not in str(caught.value)
    assert secret_url not in str(caught.value)


@pytest.mark.skipif(os.name == "nt", reason="POSIX file mode contract")
def test_store_and_lock_have_private_posix_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    grant_trust(tmp_path / "checkout", "https://hub.invalid/manifest")

    store = trust_store_path()
    lock = store.with_name("hub-trust.lock")
    assert stat.S_IMODE(store.stat().st_mode) == 0o600
    assert stat.S_IMODE(lock.stat().st_mode) == 0o600


def test_atomic_write_failure_is_sanitized_and_preserves_existing_grants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    original_checkout = tmp_path / "original"
    original_url = "https://hub.invalid/original"
    grant_trust(original_checkout, original_url)
    original_text = trust_store_path().read_text()
    secret_url = "https://hub.invalid/new?token=secret"

    def fail_write(*args: object, **kwargs: object) -> None:
        raise OSError(f"must not escape: {secret_url}")

    monkeypatch.setattr(trust_module, "write_json_atomically", fail_write)

    with pytest.raises(HubTrustError) as caught:
        grant_trust(tmp_path / "new", secret_url)

    assert str(caught.value) == "Hub trust store cannot be updated safely"
    assert secret_url not in str(caught.value)
    assert trust_store_path().read_text() == original_text


def test_two_concurrent_writers_preserve_both_grants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    context = multiprocessing.get_context("spawn")
    try:
        barrier = context.Barrier(2)
        results = context.Queue()
    except (OSError, RuntimeError, ValueError) as exc:
        pytest.skip(f"spawn synchronization unavailable: {exc}")
    checkout_a = tmp_path / "checkout-a"
    checkout_b = tmp_path / "checkout-b"
    url_a = "https://hub.invalid/a"
    url_b = "https://hub.invalid/b"
    processes = [
        context.Process(
            target=_grant_worker,
            args=(str(home), str(checkout_a), url_a, barrier, results),
        ),
        context.Process(
            target=_grant_worker,
            args=(str(home), str(checkout_b), url_b, barrier, results),
        ),
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        if process.is_alive():
            process.terminate()
            process.join()
            pytest.fail("concurrent trust writer did not terminate")
    outcomes = [results.get(timeout=5), results.get(timeout=5)]

    assert outcomes == [None, None]
    assert is_trusted(checkout_a, url_a)
    assert is_trusted(checkout_b, url_b)
    data = json.loads(trust_store_path().read_text())
    assert data["version"] == 1
    assert len(data["grants"]) == 2
