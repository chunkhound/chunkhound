import argparse

from chunkhound.core.config.config import Config
from chunkhound.hub.client import pull
from chunkhound.hub.errors import FATAL_HUB_ERRORS


async def pull_command(args: argparse.Namespace, config: Config) -> None:
    try:
        installed = await pull(config, rel=args.version)
    except FATAL_HUB_ERRORS:
        # The hub already logged a sanitized error; no CLI traceback/causes.
        raise SystemExit(1) from None
    if not installed:
        raise SystemExit(1)
