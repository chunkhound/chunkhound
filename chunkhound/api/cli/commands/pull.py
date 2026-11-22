import argparse
import asyncio
from pathlib import Path
from urllib.parse import urlsplit

from loguru import logger

from chunkhound.core.config.config import Config
from chunkhound.hub.client import fetch_validated_manifest, pull
from chunkhound.hub.errors import (
    FATAL_HUB_ERRORS,
    HUB_PULL_CANCELLATION_DIAGNOSTIC,
    HUB_PULL_FAILURE_GUIDANCE,
    HubAuthorizationError,
    HubLocalStateError,
    HubUnavailable,
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
    if args.version != "latest":
        raise SystemExit("--version cannot be used with --forget-trust")
    try:
        forget_checkout_trust(Path(args.path))
    except HubTrustError as exc:
        logger.error("{}", str(exc))
        raise SystemExit(1) from None


def hub_origin(manifest_url: str) -> str:
    """Return only scheme, normalized host, and a non-default port."""
    parts = urlsplit(manifest_url)
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    port = parts.port
    default_port = 80 if parts.scheme == "http" else 443
    suffix = f":{port}" if port not in (None, default_port) else ""
    return f"{parts.scheme}://{host}{suffix}"


async def pull_command(args: argparse.Namespace, config: Config) -> None:
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
        key = configured_key(config)
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
        pre_client_fatal = bool(args.trust)
        source = await fetch_validated_manifest(config)
        try:
            current_checkout_digest = checkout_digest(
                config.target_dir or Path(args.path)
            )
        except OSError:
            raise HubLocalStateError("Hub checkout identity is unavailable") from None
        if (
            source.checkout_digest != checkout_binding
            or source.url_digest != url_binding
            or current_checkout_digest != checkout_binding
            or config.hub.url != authorized_url
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
            installed = await pull(config, rel=args.version, source=source)
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
