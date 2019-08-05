import collections

import torch

__all__ = ['async_copy_to']


# from https://github.com/vacancy/Jacinle/blob/master/jactorch/cuda/copy.py
def async_copy_to(obj, dev, stream=None):
    if torch.is_tensor(obj):
        v = obj.to(dev, non_blocking=True)
        if stream is not None:
            v.record_stream(stream)
        return v
    elif isinstance(obj, collections.Mapping):
        return {k: async_copy_to(v, dev, stream) for k, v in obj.items()}
    elif isinstance(obj, (tuple, list, collections.UserList)):
        return [async_copy_to(v, dev, stream) for v in obj]
    else:
        return obj
