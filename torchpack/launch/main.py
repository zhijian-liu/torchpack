import argparse
import sys

from torchpack.launch.launchers import drunner

__all__ = ['main']


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('launcher', choices=['dist-run'])
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    sys.argv = [f'torchpack {args.launcher}'] + args.command
    if args.launcher == 'dist-run':
        drunner.main()
