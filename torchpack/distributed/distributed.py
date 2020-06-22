import os
import pickle
import time

import torch
import torch.distributed as dist

__all__ = ['init', 'size', 'rank', 'local_size', 'local_rank', 'is_master']


def init():
    dist.init_process_group(backend='nccl')


def size():
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def local_size():
    return os.environ.get('LOCAL_SIZE', 1)


def local_rank():
    return os.environ.get('LOCAL_RANK', 0)


def is_master():
    return rank() == 0
