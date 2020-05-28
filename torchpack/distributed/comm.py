import os
import pickle
import time

import torch
import torch.distributed as dist

__all__ = ['allreduce', 'allgather', 'barrier']


def reduce(data):
    pass


def allreduce(data):
    data = allgather(data)
    return sum(data)


def gather(data):
    pass


def allgather(data):
    world_size = dist.get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).cuda()

    # obtain tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).cuda()
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving tensors from all ranks
    tensors = []
    for _ in size_list:
        tensors.append(torch.ByteTensor(size=(max_size, )).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size, )).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensors, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensors):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))
    return data_list


def barrier():
    if not dist.is_available() or not dist.is_initialized():
        return
    dist.barrier()
