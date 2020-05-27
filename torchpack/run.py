import argparse
import os
import subprocess
import sys

from .utils.device import set_cuda_visible_devices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices',
                        '-d',
                        type=str,
                        default='*',
                        help='list of device(s) to use.')
    args, command = parser.parse_known_args()

    command = [sys.executable, '-u'] + command

    environ = os.environ.copy()
    set_cuda_visible_devices(args.devices, environ=environ)

    process = subprocess.Popen(command, env=environ)
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(returncode=process.returncode,
                                            cmd=command)


if __name__ == '__main__':
    main()
