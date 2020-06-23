import argparse
import copy
import os
import re
import socket
import sys
from shlex import quote

__all__ = ['drun']


def drun(opts):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-np',
        '--nproc',
        type=int,
        help='Total number of training processes.',
    )
    parser.add_argument(
        '-H',
        '--hosts',
        help='List of host names and the number of available slots '
        'for running processes on each, of the form: <hostname>:<slots> '
        '(e.g.: host1:2,host2:4,host3:1 indicating 2 processes can run on host1, '
        '4 on host2, and 1 on host3). If not specified, defaults to using '
        'localhost:<np>',
    )
    parser.add_argument(
        '-hostfile',
        '--hostfile',
        help=
        'Path to a host file containing the list of host names and the number of '
        'available slots. Each line of the file must be of the form: '
        '<hostname> slots=<slots>',
    )
    # parser.add_argument(
    #     '-d',
    #     '--devices',
    #     help='list of device(s) to use.',
    # )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='If this flag is set, extra messages will be printed.',
    )
    parser.add_argument(
        'command',
        nargs=argparse.REMAINDER,
        help='Command to be executed.',
    )
    args = parser.parse_args(opts)

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
            raise ValueError('Invalid host input, please make sure it has '
                             'format as : worker-0:2,worker-1:2.')
        hostname, slots = host.strip().split(':')
        hosts.append(hostname)

    environ = os.environ.copy()

    environ['MASTER_ADDR'] = hosts[0]
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('', 0))
        environ['MASTER_PORT'] = str(sock.getsockname()[1])

    command = ' '.join(map(quote, args.command))
    if not args.verbose:
        command = 'python -m torchpack.launch.utils.redirect ' + command

    command = ('mpirun --allow-run-as-root '
               '-np {nproc} -H {hosts} '
               '-bind-to none -map-by slot '
               '{environ} '
               '-mca pml ob1 -mca btl ^openib '
               '-mca btl_tcp_if_exclude docker0,lo '
               '{command}'.format(nproc=args.nproc,
                                  hosts=args.hosts,
                                  environ=' '.join(
                                      '-x %s' % key
                                      for key in sorted(environ.keys())),
                                  command=command))

    if args.verbose:
        print(command)
    os.execve('/bin/sh', ['/bin/sh', '-c', command], environ)
