import argparse
import copy
import os
import subprocess
import sys


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def parse_hosts():
    pass


def dist_run(opts):
    parser = argparse.ArgumentParser()
    parser.add_argument('-np',
                        '--nproc',
                        dest='nproc',
                        type=int,
                        help='Total number of training processes.')
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
        args.hosts = 'localhost:' + str(args.nproc)

    master_addr = args.hosts.split(':')[0]
    master_port = _find_free_port()

    command = [
        'mpirun', '-np',
        str(args.nproc), '-H',
        str(args.hosts), '-bind-to', 'none', '-map-by', 'slot', '-x',
        'LD_LIBRARY_PATH', '-x', 'PATH', '-x',
        'MASTER_ADDR=' + str(master_addr), '-x',
        'MASTER_PORT=' + str(master_port), '-mca', 'pml', 'ob1', '-mca', 'btl',
        '^openib', '-mca', 'btl_tcp_if_exclude', 'docker0,lo'
    ] + args.command
    command = ' '.join(command)
    print(command)

    env = os.environ.copy()
    for var in ['PATH', 'PYTHONPATH']:
        if var not in env and var in os.environ:
            # copy env so we do not leak env modifications
            env = copy.copy(env)
            # copy var over from os.environ
            env[var] = os.environ[var]

    os.execve('/bin/sh', ['/bin/sh', '-c', command], env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['dist-run'])
    args, opts = parser.parse_known_args()

    if args.mode == 'dist-run':
        dist_run(opts)
