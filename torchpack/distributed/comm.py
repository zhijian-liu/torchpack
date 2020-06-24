import os
import pickle

import torch
import torch.distributed

from .context import _world_size

__all__ = ['allreduce', 'allgather', 'barrier']


def allreduce(data, reduction='sum'):
    data = allgather(data)
    return sum(data)


def allgather(data):
    if _world_size == 1:
        return [data]

    # serialized to a tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).cuda()

    # obtain tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).cuda()
    sizes = [torch.LongTensor([0]).cuda() for _ in range(_world_size)]
    torch.distributed.all_gather(sizes, local_size)
    sizes = [int(size.item()) for size in sizes]
    max_size = max(sizes)

    # receiving tensors from all ranks
    tensors = []
    for _ in sizes:
        tensors.append(torch.ByteTensor(size=(max_size, )).cuda())
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size, )).cuda()
        tensor = torch.cat((tensor, padding), dim=0)
    torch.distributed.all_gather(tensors, tensor)

    data_list = []
    for size, tensor in zip(sizes, tensors):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))
    return data_list


def barrier():
    if _world_size == 1:
        return
    torch.distributed.barrier()
