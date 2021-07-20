import os
import sys


def main() -> None:
    rank = 0
    for name in ['OMPI_COMM_WORLD_RANK', 'PMI_RANK']:
        if name in os.environ:
            rank = int(os.environ[name])
            break

    command = ' '.join(sys.argv[1:])
    if rank == 0:
        command += ' 2>&1'
    else:
        command += ' 1>/dev/null'

    os.execv('/bin/sh', ['/bin/sh', '-c', command])


if __name__ == '__main__':
    main()
