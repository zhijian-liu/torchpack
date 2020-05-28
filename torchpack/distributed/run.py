import os
import subprocess
import sys
from argparse import REMAINDER, ArgumentParser


def main():
    parser = ArgumentParser()

    parser.add_argument('--nnodes',
                        type=int,
                        default=1,
                        help='The number of nodes to use for distributed '
                        'training')
    parser.add_argument('--node_rank',
                        type=int,
                        default=0,
                        help='The rank of the node for multi-node distributed '
                        'training')
    parser.add_argument('--nproc_per_node',
                        type=int,
                        default=1,
                        help='The number of processes to launch on each node, '
                        'for GPU training, this is recommended to be set '
                        'to the number of GPUs in your system so that '
                        'each process can be bound to a single GPU.')
    parser.add_argument('--master_addr',
                        default='127.0.0.1',
                        type=str,
                        help='Master node (rank 0) address, should be either '
                        'the IP address or the hostname of node 0, for '
                        'single node multi-proc training, the '
                        '--master_addr can simply be 127.0.0.1')
    parser.add_argument('--master_port',
                        default=29500,
                        type=int,
                        help='Master node (rank 0) free port that needs to '
                        'be used for communication during distributed '
                        'training')

    # positional
    parser.add_argument('training_script',
                        type=str,
                        help='The full path to the single GPU training '
                        'program/script to be launched in parallel, '
                        'followed by all the arguments for the '
                        'training script')

    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    args = parser.parse_args()

    dist_world_size = args.nproc_per_node * args.nnodes

    env = os.environ.copy()
    env['MASTER_ADDR'] = args.master_addr
    env['MASTER_PORT'] = str(args.master_port)
    env['WORLD_SIZE'] = str(dist_world_size)

    processes = []
    for local_rank in range(0, args.nproc_per_node):
        dist_rank = args.nproc_per_node * args.node_rank + local_rank

        env['RANK'] = str(dist_rank)
        # env['LOCAL_RANK'] = str(local_rank)
        env['CUDA_VISIBLE_DEVICES'] = str(local_rank)

        cmd = [sys.executable, '-u']
        cmd.append(args.training_script)
        cmd.extend(args.training_script_args)

        if dist_rank == 0:
            process = subprocess.Popen(cmd, env=env)
        else:
            process = subprocess.Popen(cmd,
                                       env=env,
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode,
                                                cmd=cmd)


if __name__ == '__main__':
    main()
