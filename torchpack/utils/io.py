import json
import os.path as osp
import pickle
from contextlib import contextmanager

import numpy as np
import torch
import yaml

from . import fs

__all__ = [
    'load', 'save', 'load_txt', 'save_txt', 'load_json', 'save_json',
    'load_npy', 'save_npy', 'load_npz', 'save_npz', 'load_pt', 'save_pt'
]


@contextmanager
def file_descriptor(f, mode='r'):
    new_fp = False
    try:
        if isinstance(f, str):
            f = open(f, mode)
            new_fp = True
        yield f
    finally:
        if new_fp:
            f.close()


def load_txt(f, **kwargs):
    with file_descriptor(f, 'r') as fd:
        return fd.readlines(**kwargs)


def save_txt(f, obj, **kwargs):
    with file_descriptor(f, 'w') as fd:
        raise NotImplementedError()


def load_json(f, **kwargs):
    with file_descriptor(f, 'r') as fd:
        return json.load(fd, **kwargs)


def save_json(f, obj, **kwargs):
    with file_descriptor(f, 'w') as fd:
        return json.dump(obj, fd, **kwargs)


def load_yaml(f, **kwargs):
    with file_descriptor(f, 'r') as fd:
        return yaml.safe_load(fd, **kwargs)


def save_yaml(f, obj, **kwargs):
    with file_descriptor(f, 'w') as fd:
        return yaml.safe_dump(obj, fd, default_flow_style=False, **kwargs)


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


def load_npy(f, **kwargs):
    return np.load(f, **kwargs)


def save_npy(f, obj, **kwargs):
    return np.save(f, obj, **kwargs)


def load_npz(f, **kwargs):
    return np.load(f, **kwargs)


def save_npz(f, obj, **kwargs):
    return np.savez(f, obj, **kwargs)


def load_pt(f, **kwargs):
    return torch.load(f, **kwargs)


def save_pt(f, obj, **kwargs):
    return torch.save(obj, f, **kwargs)


load_funcs = {
    '.txt': load_txt,
    '.json': load_json,
    '.yml': load_yaml,
    '.yaml': load_yaml,
    '.pkl': load_pkl,
    '.npy': load_npy,
    '.npz': load_npz,
    '.pt': load_pt,
    '.pth': load_pt,
    '.pth.tar': load_pt
}
save_funcs = {
    '.txt': save_txt,
    '.json': save_json,
    '.yml': save_yaml,
    '.yaml': save_yaml,
    '.pkl': save_pkl,
    '.npy': save_npy,
    '.npz': save_npz,
    '.pt': save_pt,
    '.pth': save_pt,
    '.pth.tar': save_pt
}


def load(fpath, **kwargs):
    assert isinstance(fpath, str), type(fpath)
    for extension in sorted(load_funcs.keys(), key=len, reverse=True):
        if fpath.endswith(extension):
            return load_funcs[extension](fpath, **kwargs)
    raise NotImplementedError()


def save(fpath, obj, **kwargs):
    dirpath = osp.dirname(fpath)
    if not osp.exists(dirpath):
        fs.makedir(dirpath)

    assert isinstance(fpath, str), type(fpath)
    for ext in sorted(save_funcs.keys(), key=len, reverse=True):
        if fpath.endswith(ext):
            return save_funcs[ext](fpath, obj, **kwargs)
    raise NotImplementedError()
