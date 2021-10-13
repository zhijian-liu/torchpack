import pickle
from typing import Any, List, Optional

import torch
import torch.distributed

from . import context

__all__ = ['broadcast', 'allgather', 'allreduce', 'barrier']


def _serialize(obj: Any) -> torch.Tensor:
    buffer = pickle.dumps(obj)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage)
    return tensor


def _deserialize(tensor: torch.Tensor, size: Optional[int] = None) -> Any:
    buffer = tensor.numpy().tobytes()
    if size is not None:
        buffer = buffer[:size]
    obj = pickle.loads(buffer)
    return obj


def broadcast(obj: Any, src: int = 0) -> Any:
    world_size = context.size()
    if world_size == 1:
        return obj

    # serialize
    if context.rank() == src:
        tensor = _serialize(obj).cuda()

    # broadcast the tensor size
    if context.rank() == src:
        size = torch.LongTensor([tensor.numel()]).cuda()
    else:
        size = torch.LongTensor([0]).cuda()
    torch.distributed.broadcast(size, src=src)

    # broadcast the tensor
    if context.rank() != src:
        tensor = torch.ByteTensor(size=(size.item(),)).cuda()
    torch.distributed.broadcast(tensor, src=src)

    # deserialize
    if context.rank() != src:
        obj = _deserialize(tensor.cpu())
    return obj


def allgather(obj: Any) -> List[Any]:
    world_size = context.size()
    if world_size == 1:
        return [obj]

    # serialize
    tensor = _serialize(obj).cuda()

    # gather the tensor size
    local_size = torch.LongTensor([tensor.numel()]).cuda()
    sizes = [torch.LongTensor([0]).cuda() for _ in range(world_size)]
    torch.distributed.all_gather(sizes, local_size)
    sizes = [int(size.item()) for size in sizes]
    max_size = max(sizes)

    # gather the tensor
    tensors = [torch.ByteTensor(size=(max_size,)).cuda() for _ in sizes]
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).cuda()
        tensor = torch.cat((tensor, padding), dim=0)
    torch.distributed.all_gather(tensors, tensor)

    # deserialize
    objs = []
    for size, tensor in zip(sizes, tensors):
        obj = _deserialize(tensor.cpu(), size=size)
        objs.append(obj)
    return objs


def allreduce(obj: Any, reduction: str = 'sum') -> Any:
    objs = allgather(obj)
    if reduction == 'sum':
        return sum(objs)
    else:
        raise NotImplementedError(reduction)


def barrier() -> None:
    world_size = context.size()
    if world_size == 1:
        return
    torch.distributed.barrier()
