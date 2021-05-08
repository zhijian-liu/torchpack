import pickle
from typing import Any, List

import torch
import torch.distributed

from . import context

__all__ = ['allreduce', 'allgather', 'barrier']


def allreduce(data: Any, reduction: str = 'sum') -> Any:
    data = allgather(data)
    if reduction == 'sum':
        return sum(data)


def allgather(data: Any) -> List:
    world_size = context.size()
    if world_size == 1:
        return [data]

    # serialized to a tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).cuda()

    # obtain tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).cuda()
    sizes = [torch.LongTensor([0]).cuda() for _ in range(world_size)]
    torch.distributed.all_gather(sizes, local_size)
    sizes = [int(size.item()) for size in sizes]
    max_size = max(sizes)

    # receiving tensors from all ranks
    tensors = [torch.ByteTensor(size=(max_size, )).cuda() for _ in sizes]
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size, )).cuda()
        tensor = torch.cat((tensor, padding), dim=0)
    torch.distributed.all_gather(tensors, tensor)

    data = []
    for size, tensor in zip(sizes, tensors):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data.append(pickle.loads(buffer))
    return data


def barrier() -> None:
    world_size = context.size()
    if world_size == 1:
        return
    torch.distributed.barrier()
