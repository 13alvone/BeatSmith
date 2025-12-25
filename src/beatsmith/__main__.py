import argparse
import sys

from .cli import main as harvest_main
from .inspect import main as inspect_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="beatsmith")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("harvest", help="Harvest sample packs for MPC workflows.")
    sub.add_parser("inspect", help="Summarize latest SampleSmith pack.")
    return parser


def main(argv: list[str] | None = None):
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args, remainder = parser.parse_known_args(argv)
    if args.command == "inspect":
        inspect_main(remainder)
    elif args.command == "harvest":
        harvest_main(remainder)
    else:
        harvest_main(argv)


if __name__ == "__main__":
    main()
