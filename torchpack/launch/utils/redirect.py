import os
import subprocess
import sys


def main():
    rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))

    env = os.environ.copy()

    print(rank)
    print(sys.argv)

    command = ' '.join(sys.argv[1:])

    if rank == 0:
        command += ' 2>&1'
    else:
        command += ' 1>/dev/null 2>/dev/null'

    print(command)
    os.execve('/bin/sh', ['/bin/sh', '-c', command], env)


if __name__ == '__main__':
    main()
