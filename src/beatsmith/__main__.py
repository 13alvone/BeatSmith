"""
BeatSmith CLI dispatcher.

Design intent:
- `beatsmith harvest ...` routes to the v4 harvester CLI.
- `beatsmith inspect ...` routes to the pack inspector CLI.
- If no explicit subcommand is provided, we default to harvest-mode and pass
  argv through unchanged so harvest can parse its own rich argument set,
  including positional out_dir and all harvest options.
"""

import argparse
import sys
from typing import Sequence

from .cli import main as harvest_main
from .inspect import main as inspect_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="beatsmith")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser(
        "harvest",
        help="Harvest sample packs (v4). If omitted, harvest is the default mode.",
    )
    sub.add_parser(
        "inspect",
        help="Inspect a pack.json and summarize pack contents.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    argv_list = list(sys.argv[1:] if argv is None else argv)

    # We intentionally use parse_known_args so that:
    # - `beatsmith --seed ...` (no subcommand) passes through to harvest.
    # - `beatsmith <out_dir>` (no subcommand) is treated as harvest positional.
    parser = build_parser()
    args, remainder = parser.parse_known_args(argv_list)

    if args.command == "inspect":
        inspect_main(remainder)
        return

    if args.command == "harvest":
        harvest_main(remainder)
        return

    # No explicit subcommand: default to harvest, passing argv through unchanged.
    harvest_main(argv_list)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        # Preserve argparse/CLI exit codes from downstream entrypoints.
        raise

