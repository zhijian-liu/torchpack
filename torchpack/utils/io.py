import json

import numpy as np
import six
import torch

__all__ = [
    'load', 'save',
    'load_json', 'save_json',
    'load_npy', 'save_npy',
    'load_npz', 'save_npz',
    'load_pth', 'save_pth'
]


def file_descriptor(f, mode='r'):
    if isinstance(f, six.string_types):
        f = open(f, mode)
    return f


def load_json(f, **kwargs):
    with file_descriptor(f, 'r') as fp:
        return json.load(fp, **kwargs)


def save_json(obj, f, **kwargs):
    with file_descriptor(f, 'w') as fp:
        return json.dump(obj, fp, **kwargs)


def load_npy(f, **kwargs):
    return np.load(f, **kwargs)


def save_npy(obj, f, **kwargs):
    return np.save(f, obj, **kwargs)


def load_npz(f, **kwargs):
    return np.load(f, **kwargs)


def save_npz(obj, f, **kwargs):
    return np.savez(f, obj, **kwargs)


def load_pth(f, **kwargs):
    return torch.load(f, **kwargs)


def save_pth(obj, f, **kwargs):
    return torch.save(obj, f, **kwargs)


funcs = {
    '.json': dict(load_func=load_json, save_func=save_json),
    '.npy': dict(load_func=load_npy, save_func=save_npy),
    '.npz': dict(load_func=load_npz, save_func=save_npz),
    '.pth': dict(load_func=load_pth, save_func=save_pth),
    '.pth.tar': dict(load_func=load_pth, save_func=save_pth)
}


def load(filename, **kwargs):
    for suffix in sorted(funcs.keys(), key=len, reverse=True):
        if filename.endswith(suffix):
            load_func = funcs[suffix]['load_func']
            return load_func(filename, **kwargs)
    raise NotImplementedError


def save(obj, filename, **kwargs):
    for suffix in sorted(funcs.keys(), key=len, reverse=True):
        if filename.endswith(suffix):
            save_func = funcs[suffix]['save_func']
            return save_func(obj, filename, **kwargs)
    raise NotImplementedError
