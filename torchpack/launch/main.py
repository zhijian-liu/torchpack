import argparse
import sys

from .launchers import drunner

__all__ = ['main']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['dist-run'])
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    sys.argv = [f'torchpack {args.mode}'] + args.command
    if args.mode == 'dist-run':
        drunner.launch()
