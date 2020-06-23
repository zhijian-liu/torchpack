import os
import pickle
import time

import torch
import torch.distributed as dist

__all__ = ['init', 'size', 'rank', 'local_size', 'local_rank', 'is_master']

_world_comm = None
_local_comm = None


def init():
    from mpi4py import MPI

    global _world_comm, _local_comm

    _world_comm = MPI.COMM_WORLD
    _local_comm = _world_comm.Split_type(MPI.COMM_TYPE_SHARED)

    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=size(),
                            rank=rank())


def size():
    if _world_comm is None:
        return 1
    return _world_comm.Get_size()


def rank():
    if _world_comm is None:
        return 0
    return _world_comm.Get_rank()


def local_size():
    if _local_comm is None:
        return 1
    return _local_comm.Get_size()


def local_rank():
    if _local_comm is None:
        return 0
    return _local_comm.Get_rank()


def is_master():
    return rank() == 0
