import json
import os.path as osp
import pickle
from contextlib import contextmanager

import numpy as np
import scipy.io
import torch
import yaml

from . import fs

__all__ = [
    'load', 'save', 'load_json', 'save_json', 'load_jsonl', 'save_jsonl',
    'load_mat', 'save_mat', 'load_npy', 'save_npy', 'load_npz', 'save_npz',
    'load_pt', 'save_pt', 'load_yaml', 'save_yaml'
]


@contextmanager
def file_descriptor(f, mode='r'):
    opened = False
    try:
        if isinstance(f, str):
            f = open(f, mode)
            opened = True
        yield f
    finally:
        if opened:
            f.close()


def load_json(f, **kwargs):
    with file_descriptor(f, 'r') as fd:
        return json.load(fd, **kwargs)


def save_json(f, obj, **kwargs):
    with file_descriptor(f, 'w') as fd:
        return json.dump(obj, fd, **kwargs)


def load_jsonl(f, **kwargs):
    with file_descriptor(f, 'r') as fd:
        return [json.loads(datum, **kwargs) for datum in fd.readlines()]


def save_jsonl(f, obj, **kwargs):
    with file_descriptor(f, 'w') as fd:
        fd.write('\n'.join(json.dumps(datum, **kwargs) for datum in obj))


def load_mat(f, **kwargs):
    return scipy.io.loadmat(f, **kwargs)


def save_mat(f, obj, **kwargs):
    return scipy.io.savemat(f, obj, **kwargs)


def load_npy(f, **kwargs):
    return np.load(f, **kwargs)


def save_npy(f, obj, **kwargs):
    return np.save(f, obj, **kwargs)


def load_npz(f, **kwargs):
    return np.load(f, **kwargs)


def save_npz(f, obj, **kwargs):
    return np.savez(f, obj, **kwargs)


def load_pkl(f, **kwargs):
    with file_descriptor(f, 'rb') as fd:
        try:
            return pickle.load(fd, **kwargs)
        except UnicodeDecodeError:
            if 'encoding' in kwargs:
                raise
            return pickle.load(fd, encoding='latin1', **kwargs)


def save_pkl(f, obj, **kwargs):
    with file_descriptor(f, 'wb') as fd:
        return pickle.dump(obj, fd, **kwargs)


def load_pt(f, **kwargs):
    return torch.load(f, **kwargs)


def save_pt(f, obj, **kwargs):
    return torch.save(obj, f, **kwargs)


def load_yaml(f, **kwargs):
    with file_descriptor(f, 'r') as fd:
        return yaml.safe_load(fd, **kwargs)


def save_yaml(f, obj, **kwargs):
    with file_descriptor(f, 'w') as fd:
        return yaml.safe_dump(obj, fd, **kwargs)


__io_registry = {
    '.json': (load_json, save_json),
    '.jsonl': (load_jsonl, save_jsonl),
    '.mat': (load_mat, save_mat),
    '.npy': (load_npy, save_npy),
    '.npz': (load_npz, save_npz),
    '.pkl': (load_pkl, save_pkl),
    '.pt': (load_pt, save_pt),
    '.pth': (load_pt, save_pt),
    '.pth.tar': (load_pt, save_pt),
    '.yml': (load_yaml, save_yaml),
    '.yaml': (load_yaml, save_yaml)
}


def load(fpath, **kwargs):
    assert isinstance(fpath, str), type(fpath)

    for extension in sorted(__io_registry.keys(), key=len, reverse=True):
        if fpath.endswith(extension):
            return __io_registry[extension][0](fpath, **kwargs)

    extension = osp.splitext(fpath)[1]
    raise NotImplementedError(f'Unsupported file format: \'{extension}\'')


def save(fpath, obj, **kwargs):
    assert isinstance(fpath, str), type(fpath)

    dirpath = osp.dirname(fpath)
    if not osp.exists(dirpath):
        fs.makedir(dirpath)

    for extension in sorted(__io_registry.keys(), key=len, reverse=True):
        if fpath.endswith(extension):
            return __io_registry[extension][1](fpath, obj, **kwargs)

    extension = osp.splitext(fpath)[1]
    raise NotImplementedError(f'Unsupported file format: \'{extension}\'')
