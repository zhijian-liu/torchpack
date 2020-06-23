import os

import torch.distributed

__all__ = ['init', 'size', 'rank', 'local_size', 'local_rank', 'is_master']

_world_comm = None
_local_comm = None


def init():
    from mpi4py import MPI

    global _world_comm, _local_comm
    _world_comm = MPI.COMM_WORLD
    _local_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)

    master_port = 'tcp://' + os.environ['MASTER_HOST']
    torch.distributed.init_process_group(backend='nccl',
                                         init_method=master_port,
                                         world_size=size(),
                                         rank=rank())


def size():
    return _world_comm.Get_size() if _world_comm else 1


def rank():
    return _world_comm.Get_rank() if _world_comm else 0


def local_size():
    return _local_comm.Get_size() if _local_comm else 1


def local_rank():
    return _local_comm.Get_rank() if _local_comm else 0


def is_master():
    return rank() == 0
