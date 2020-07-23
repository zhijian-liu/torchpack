import os

import torch.distributed

__all__ = ['init', 'size', 'rank', 'local_size', 'local_rank', 'is_master']

_world_size, _world_rank = 1, 0
_local_size, _local_rank = 1, 0


def init() -> None:
    from mpi4py import MPI  # type: ignore
    world_comm = MPI.COMM_WORLD
    local_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)

    global _world_size, _world_rank, _local_size, _local_rank
    _world_size, _world_rank = world_comm.Get_size(), world_comm.Get_rank()
    _local_size, _local_rank = local_comm.Get_size(), local_comm.Get_rank()

    master_host = 'tcp://' + os.environ['MASTER_HOST']
    torch.distributed.init_process_group(backend='nccl',
                                         init_method=master_host,
                                         world_size=_world_size,
                                         rank=_world_rank)


def size() -> int:
    return _world_size


def rank() -> int:
    return _world_rank


def local_size() -> int:
    return _local_size


def local_rank() -> int:
    return _local_rank


def is_master() -> bool:
    return _world_rank == 0
