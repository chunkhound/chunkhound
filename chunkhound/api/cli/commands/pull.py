import argparse
import asyncio
from pathlib import Path

from loguru import logger

from chunkhound.core.config.config import Config
from chunkhound.hub.client import binding_unchanged, fetch_validated_manifest, pull
from chunkhound.hub.errors import (
    FATAL_HUB_ERRORS,
    HUB_PULL_CANCELLATION_DIAGNOSTIC,
    HUB_PULL_FAILURE_GUIDANCE,
    HubAuthorizationError,
    HubLocalStateError,
    HubUnavailable,
    hub_origin,
    safe_label,
)
from chunkhound.hub.manifest import configured_key, select_manifest
from chunkhound.hub.trust import (
    HubTrustError,
    checkout_digest,
    forget_checkout_trust,
    grant_trust,
    is_trusted,
    manifest_url_digest,
)


def forget_trust_command(args: argparse.Namespace) -> None:
    """Remove the stored hub trust grant for a checkout, without contacting the hub."""
    if args.version != "latest":
        raise SystemExit("--version cannot be used with --forget-trust")
    try:
        checkout = Path(args.path).expanduser().resolve()
        removed = forget_checkout_trust(checkout)
    except (HubTrustError, OSError) as exc:
        logger.error(
            "{}",
            str(exc) if isinstance(exc, HubTrustError) else "Checkout is unavailable",
        )
        raise SystemExit(1) from None
    if removed:
        logger.bind(hub_client=True).info("Removed hub trust grant for {}", checkout)
    else:
        logger.warning(
            "No hub trust grant found for {}; check that the path is the "
            "checkout you indexed",
            checkout,
        )


async def pull_command(args: argparse.Namespace, config: Config) -> None:
    """Run an explicit hub pull; any failure once it begins stops nonzero."""
    pre_client_fatal = True
    pull_attempted = False
    try:
        authorized_url = config.hub.url
        if not authorized_url:
            raise HubUnavailable("No hub manifest URL configured")
        try:
            checkout = (config.target_dir or Path(args.path)).expanduser().resolve()
            checkout_binding = checkout_digest(checkout)
        except OSError:
            raise HubLocalStateError("Hub checkout identity is unavailable") from None
        url_binding = manifest_url_digest(authorized_url)
        if not args.trust:
            try:
                trusted = is_trusted(checkout, authorized_url)
            except HubTrustError:
                raise HubAuthorizationError(
                    "Hub source authorization could not be verified"
                ) from None
            if not trusted:
                raise HubAuthorizationError(
                    "This hub source is not trusted. Review the configured source "
                    "and run chunkhound pull --trust to authorize and pull it."
                )
        # Once a plain pull is authorized, failures resolving its configured
        # embedding key should retain the established retry guidance. Keep this
        # after the trust gate so untrusted pulls remain entirely offline.
        pull_attempted = not args.trust
        # Resolve the key before any hub request so unknown dimensions never
        # probe the hub or embedding service.
        key = configured_key(config)
        logger.info("Hub source origin={}", hub_origin(authorized_url))
        try:
            source = await fetch_validated_manifest(config, authorized_url)
        except asyncio.CancelledError:
            if pull_attempted:
                logger.error("{}", HUB_PULL_CANCELLATION_DIAGNOSTIC)
                logger.error("{}", HUB_PULL_FAILURE_GUIDANCE)
                raise SystemExit(1) from None
            raise
        still_bound = binding_unchanged(
            config, checkout_binding, authorized_url, Path(args.path)
        )
        if (
            source.checkout_digest != checkout_binding
            or source.url_digest != url_binding
            or not still_bound
        ):
            raise HubUnavailable(
                "Validated hub source no longer matches authorized configuration"
            )
        if args.trust:
            try:
                grant_trust(checkout, authorized_url)
            except HubTrustError:
                logger.warning(
                    "Hub manifest was authorized for this pull, but future "
                    "automatic hub use was not enabled"
                )
        pre_client_fatal = False
        selected = select_manifest(source.manifest, key, args.version)
        logger.bind(hub_client=True).info(
            "Hub source origin={} embedding provider={} model={} dimensions={}",
            hub_origin(authorized_url),
            safe_label(selected.variant.provider),
            safe_label(selected.variant.model),
            selected.variant.dims,
        )
        pull_attempted = True
        try:
            installed = await pull(
                config, rel=args.version, source=source, explicit_trust=args.trust
            )
        except asyncio.CancelledError:
            logger.error("{}", HUB_PULL_CANCELLATION_DIAGNOSTIC)
            logger.error("{}", HUB_PULL_FAILURE_GUIDANCE)
            raise SystemExit(1) from None
    except HubUnavailable as exc:
        logger.error("Hub pull unavailable: {}", str(exc))
        if pull_attempted:
            logger.error("{}", HUB_PULL_FAILURE_GUIDANCE)
        raise SystemExit(1) from None
    except FATAL_HUB_ERRORS as exc:
        if pre_client_fatal:
            logger.error("{}", str(exc))
        if pull_attempted:
            logger.error("{}", HUB_PULL_FAILURE_GUIDANCE)
        raise SystemExit(1) from None
    if not installed:
        logger.error("{}", HUB_PULL_FAILURE_GUIDANCE)
        raise SystemExit(1)
