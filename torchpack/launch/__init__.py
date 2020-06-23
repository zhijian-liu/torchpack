import argparse
import sys

from . import distributed_run

__all__ = ['main']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['dist-run'])
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    sys.argv = [f'torchpack {args.mode}'] + args.command
    if args.mode == 'dist-run':
        distributed_run.main()
