import argparse
from pathlib import Path

from loguru import logger

from chunkhound.core.config.config import Config
from chunkhound.hub.client import fetch_validated_manifest, pull
from chunkhound.hub.errors import FATAL_HUB_ERRORS, HubUnavailable
from chunkhound.hub.trust import HubTrustError, forget_checkout_trust, grant_trust


def forget_trust_command(args: argparse.Namespace) -> None:
    if args.version != "latest":
        raise SystemExit("--version cannot be used with --forget-trust")
    try:
        forget_checkout_trust(Path(args.path))
    except HubTrustError as exc:
        logger.error("{}", str(exc))
        raise SystemExit(1) from None


async def pull_command(args: argparse.Namespace, config: Config) -> None:
    try:
        source = None
        if args.trust:
            source = await fetch_validated_manifest(config)
            hub_url = config.hub.url
            if hub_url:
                try:
                    grant_trust(config.target_dir or Path(args.path), hub_url)
                except HubTrustError:
                    logger.warning(
                        "Hub manifest was authorized for this pull, but future "
                        "automatic hub use was not enabled"
                    )
        installed = await pull(config, rel=args.version, source=source)
    except HubUnavailable as exc:
        logger.error("Hub pull unavailable: {}", str(exc))
        raise SystemExit(1) from None
    except FATAL_HUB_ERRORS:
        # The hub already logged a sanitized error; no CLI traceback/causes.
        raise SystemExit(1) from None
    if not installed:
        raise SystemExit(1)
