import os
import pickle
import time

import torch
import torch.distributed as dist

__all__ = ['is_master']

is_master = True
world_size = 1


def init():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return

    global is_master, world_size
    is_master = dist.get_rank()
    world_size = dist.get_world_size()


def barrier():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    if world_size == 1:
        return
    dist.barrier()
