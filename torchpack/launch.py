import argparse
import os
import subprocess
import sys


def run(opts):
    pass


def dist_run(opts):
    parser = argparse.ArgumentParser()
    parser.add_argument('-np',
                        '--nproc',
                        dest='np',
                        type=int,
                        help='Total number of training processes.')
    parser.add_argument('-p',
                        '--port',
                        default=12345,
                        type=int,
                        help='SSH port on all the hosts.')
    parser.add_argument(
        '-H',
        '--hosts',
        action='store',
        dest='hosts',
        help='List of host names and the number of available slots '
        'for running processes on each, of the form: <hostname>:<slots> '
        '(e.g.: host1:2,host2:4,host3:1 indicating 2 processes can run on host1, '
        '4 on host2, and 1 on host3). If not specified, defaults to using '
        'localhost:<np>')
    parser.add_argument('command',
                        nargs=argparse.REMAINDER,
                        help='Command to be executed.')

    args = parser.parse_args(opts)

    if args.hosts is None:
        args.hosts = 'localhost:' + str(args.np)

    print(args)
    # return

    environ = os.environ.copy()
    environ['MASTER_ADDR'] = '127.0.0.1'
    environ['MASTER_PORT'] = str(args.port)

    processes = []
    for rank in range(0, args.np):
        environ['WORLD_SIZE'] = str(args.np)
        environ['WORLD_RANK'] = str(rank)
        environ['LOCAL_SIZE'] = str(args.np)
        environ['LOCAL_RANK'] = str(rank)
        # env['CUDA_VISIBLE_DEVICES'] = str(local_rank)

        if rank == 0:
            process = subprocess.Popen(args.command, env=environ)
        else:
            process = subprocess.Popen(args.command,
                                       env=environ,
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode,
                                                cmd=args.command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['run', 'dist-run'])
    args, opts = parser.parse_known_args()

    if args.mode == 'run':
        run(opts)
    elif args.mode == 'dist-run':
        dist_run(opts)
