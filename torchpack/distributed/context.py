__all__ = ['size', 'rank', 'local_size', 'local_rank', 'is_master']

_world_size, _world_rank = 1, 0
_local_size, _local_rank = 1, 0


def size():
    return _world_size


def rank():
    return _world_rank


def local_size():
    return _local_size


def local_rank():
    return _local_rank


def is_master():
    return _world_rank == 0
