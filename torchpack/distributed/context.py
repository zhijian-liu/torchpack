import os
from datetime import timedelta

import torch.distributed
from torch.distributed.constants import default_pg_timeout

from torchpack.utils.logging import logger
from torchpack.utils.network import get_free_tcp_port

__all__ = ['init', 'size', 'rank', 'local_size', 'local_rank', 'is_master']

_world_size, _world_rank = 1, 0
_local_size, _local_rank = 1, 0


def init(
    backend: str = 'nccl',
    timeout: timedelta = default_pg_timeout,
) -> None:
    from mpi4py import MPI
    world_comm = MPI.COMM_WORLD
    local_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)

    global _world_size, _world_rank, _local_size, _local_rank
    _world_size, _world_rank = world_comm.Get_size(), world_comm.Get_rank()
    _local_size, _local_rank = local_comm.Get_size(), local_comm.Get_rank()

    if 'MASTER_HOST' in os.environ:
        master_host = 'tcp://' + os.environ['MASTER_HOST']
    else:
        master_host = f'tcp://localhost:{get_free_tcp_port()}'
        logger.warning(
            'Distributed environment not detected, fall back to default')

    torch.distributed.init_process_group(
        backend=backend,
        init_method=master_host,
        timeout=timeout,
        world_size=_world_size,
        rank=_world_rank,
    )


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
