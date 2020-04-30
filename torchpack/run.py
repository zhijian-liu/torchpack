import argparse
import os
import subprocess
import sys

from .utils.device import set_cuda_visible_devices

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices',
                        '-d',
                        type=str,
                        default='*',
                        help='list of device(s) to use.')
    args, command = parser.parse_known_args()

    cmd = [sys.executable, '-u'] + command

    env = os.environ.copy()
    set_cuda_visible_devices(args.devices, environ=env)

    process = subprocess.Popen(cmd, env=env)
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(returncode=process.returncode,
                                            cmd=cmd)
