import argparse
from pathlib import Path

from loguru import logger

from chunkhound.core.config.config import Config
from chunkhound.hub.client import fetch_validated_manifest, pull
from chunkhound.hub.errors import (
    FATAL_HUB_ERRORS,
    HubAuthorizationError,
    HubLocalStateError,
    HubUnavailable,
)
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


async def pull_command(args: argparse.Namespace, config: Config) -> None:
    pre_client_fatal = False
    try:
        source = None
        authorized_url = config.hub.url
        if not authorized_url:
            raise HubUnavailable("No hub manifest URL configured")
        if not args.trust:
            pre_client_fatal = True
            try:
                checkout = (
                    config.target_dir or Path(args.path)
                ).expanduser().resolve()
                trusted = is_trusted(checkout, authorized_url)
            except OSError:
                raise HubLocalStateError(
                    "Hub checkout identity is unavailable"
                ) from None
            except HubTrustError:
                raise HubAuthorizationError(
                    "Hub source authorization could not be verified"
                ) from None
            if not trusted:
                raise HubAuthorizationError(
                    "This hub source is not trusted. Review the configured source "
                    "and run chunkhound pull --trust to authorize and pull it."
                )
        if args.trust:
            pre_client_fatal = True
            try:
                authorized_checkout = (
                    config.target_dir or Path(args.path)
                ).expanduser().resolve()
                authorized_checkout_digest = checkout_digest(authorized_checkout)
            except OSError:
                pre_client_fatal = True
                raise HubLocalStateError(
                    "Hub checkout identity is unavailable"
                ) from None
            authorized_url_digest = manifest_url_digest(authorized_url)
            source = await fetch_validated_manifest(config)
            try:
                current_checkout_digest = checkout_digest(
                    config.target_dir or Path(args.path)
                )
            except OSError:
                raise HubLocalStateError(
                    "Hub checkout identity is unavailable"
                ) from None
            if (
                source.checkout_digest != authorized_checkout_digest
                or source.url_digest != authorized_url_digest
                or current_checkout_digest != authorized_checkout_digest
                or config.hub.url != authorized_url
            ):
                raise HubUnavailable(
                    "Validated hub source no longer matches authorized configuration"
                )
            try:
                grant_trust(authorized_checkout, authorized_url)
            except HubTrustError:
                logger.warning(
                    "Hub manifest was authorized for this pull, but future "
                    "automatic hub use was not enabled"
                )
        pre_client_fatal = False
        installed = await pull(config, rel=args.version, source=source)
    except HubUnavailable as exc:
        logger.error("Hub pull unavailable: {}", str(exc))
        raise SystemExit(1) from None
    except FATAL_HUB_ERRORS as exc:
        # Client fatal outcomes are already logged; pre-client identity failures
        # are formatted here exactly once.
        if pre_client_fatal:
            logger.error("{}", str(exc))
        raise SystemExit(1) from None
    if not installed:
        raise SystemExit(1)
