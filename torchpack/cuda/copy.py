import collections

import torch

__all__ = ['async_copy_to']


def async_copy_to(obj, device, stream=None):
    """
    Copy an object to a specific device asynchronizedly.
    Args:
        obj: a structure (e.g., a list or a dict) containing pytorch tensors.
        device: the target device.
        stream: the main stream to be synchronized.
    Returns:
        a deep copy of the data structure, with each tensor copied to the device.
    """
    if torch.is_tensor(obj):
        v = obj.to(device, non_blocking=True)
        if stream is not None:
            v.record_stream(stream)
        return v
    elif isinstance(obj, collections.Mapping):
        return {k: async_copy_to(o, device, stream) for k, o in obj.items()}
    elif isinstance(obj, (tuple, list, collections.UserList)):
        return [async_copy_to(o, device, stream) for o in obj]
    else:
        return obj
