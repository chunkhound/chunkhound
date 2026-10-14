import argparse
from pathlib import Path
from typing import Any, cast

from .common_arguments import add_common_arguments, add_config_arguments


def add_pull_subparser(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("pull", help="Install a matching hub base index")
    parser.add_argument("path", nargs="?", default=Path("."), type=Path)
    parser.add_argument("--version", choices=("latest", "-1", "-2"), default="latest")
    add_common_arguments(parser)
    add_config_arguments(parser, ["database", "embedding"])
    return cast(argparse.ArgumentParser, parser)
