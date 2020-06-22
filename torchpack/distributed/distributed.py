import os
import pickle
import time

import torch
import torch.distributed as dist

__all__ = ['init', 'size', 'rank', 'local_size', 'local_rank', 'is_master']


def init():
    dist.init_process_group(backend='nccl',
                            init_method='tcp://' +
                            os.environ.get('MASTER_ADDR') + ':' +
                            os.environ.get('MASTER_PORT'),
                            world_size=size(),
                            rank=rank())


def size():
    return os.environ.get('WORLD_SIZE', 1)


def rank():
    return os.environ.get('WORLD_RANK', 0)


def local_size():
    return os.environ.get('LOCAL_SIZE', 1)


def local_rank():
    return os.environ.get('LOCAL_RANK', 0)


def is_master():
    return rank() == 0
