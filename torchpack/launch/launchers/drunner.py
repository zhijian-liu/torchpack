import argparse
import os
import re
import socket
from shlex import quote

__all__ = ['main']


def is_exportable(v):
    IGNORE_REGEXES = ['BASH_FUNC_.*', 'OLDPWD']
    return not any(re.match(r, v) for r in IGNORE_REGEXES)


def get_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp:
        tcp.bind(('0.0.0.0', 0))
        port = tcp.getsockname()[1]
    return port


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-np',
        '--nproc',
        type=int,
        required=True,
        help='total number of processes.',
    )
    parser.add_argument(
        '-H',
        '--hosts',
        help='list of host names and the number of available slots '
        'in the form of <hostname>:<slots>, defaults to localhost:<np>.',
    )
    parser.add_argument(
        '-hostfile',
        '--hostfile',
        help='path to a host file containing the list of host names and '
        'the number of available slots, where each line must be '
        'in the form of <hostname> slots=<slots>.',
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='extra messages will be printed if this flag is set.',
    )
    parser.add_argument(
        'command',
        nargs=argparse.REMAINDER,
        help='command to be executed.',
    )
    args = parser.parse_args()

    if not args.hosts:
        if args.hostfile:
            hosts = []
            with open(args.hostfile, 'r') as fp:
                for line in fp.read().splitlines():
                    hostname = line.split()[0]
                    slots = line.split('=')[1]
                    hosts.append(f'{hostname}:{slots}')
            args.hosts = ','.join(hosts)
        else:
            args.hosts = f'localhost:{args.nproc}'

    hosts = []
    for host in args.hosts.split(','):
        if not re.match(r'^[\w.-]+:[0-9]+$', host.strip()):
            raise ValueError(
                'Host input is not in the form of <hostname>:<slots>.')
        hostname, slots = host.strip().split(':')
        hosts.append(hostname)

    master_addr = hosts[0]
    master_port = get_free_tcp_port()

    environ = os.environ.copy()
    environ['MASTER_HOST'] = f'{master_addr}:{master_port}'

    command = ' '.join(map(quote, args.command))
    if not args.verbose:
        command = 'python -m torchpack.launch.assets.silentrun ' + command

    command = ('mpirun --allow-run-as-root '
               '-np {nproc} -H {hosts} '
               '-bind-to none -map-by slot '
               '{environ} '
               '-mca pml ob1 -mca btl ^openib '
               '-mca btl_tcp_if_exclude docker0,lo '
               '{command}'.format(nproc=args.nproc,
                                  hosts=args.hosts,
                                  environ=' '.join(
                                      f'-x {key}'
                                      for key in sorted(environ.keys())
                                      if is_exportable(key)),
                                  command=command))

    if args.verbose:
        print(command)
    os.execve('/bin/sh', ['/bin/sh', '-c', command], environ)
