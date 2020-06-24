import os
import pickle

import torch
import torch.distributed

from .context import _world_size

__all__ = ['allreduce', 'allgather', 'barrier']


def allreduce(data):
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
    size_list = [torch.LongTensor([0]).cuda() for _ in range(_world_size)]
    torch.distributed.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving tensors from all ranks
    tensors = []
    for _ in size_list:
        tensors.append(torch.ByteTensor(size=(max_size, )).cuda())
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size, )).cuda()
        tensor = torch.cat((tensor, padding), dim=0)
    torch.distributed.all_gather(tensors, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensors):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))
    return data_list


def barrier():
    if _world_size == 1:
        return
    torch.distributed.barrier()
