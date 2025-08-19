import argparse
import sys

from .cli import main as run_main
from .inspect import main as inspect_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="beatsmith")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("inspect", help="Summarize latest BeatSmith run.")
    return parser


def main(argv: list[str] | None = None):
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args, remainder = parser.parse_known_args(argv)
    if args.command == "inspect":
        inspect_main(remainder)
    else:
        sys.argv = [sys.argv[0]] + remainder
        run_main()


if __name__ == "__main__":
    main()
