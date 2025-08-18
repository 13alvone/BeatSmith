import sys

from .cli import main as run_main
from .inspect import main as inspect_main


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "inspect":
        sys.argv.pop(1)
        inspect_main()
    else:
        run_main()


if __name__ == "__main__":
    main()
