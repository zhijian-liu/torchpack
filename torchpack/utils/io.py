import json
import pickle
from contextlib import contextmanager
from os import path as osp
from typing import IO, Any, Callable, Dict, Iterator, Tuple, Union

import numpy as np
import scipy.io
import torch
import yaml

from torchpack.utils import fs

__all__ = [
    'load', 'save', 'load_json', 'save_json', 'load_jsonl', 'save_jsonl',
    'load_mat', 'save_mat', 'load_npy', 'save_npy', 'load_npz', 'save_npz',
    'load_pt', 'save_pt', 'load_yaml', 'save_yaml'
]


@contextmanager
def file_descriptor(f: Union[str, IO], mode: str = 'r') -> Iterator[IO]:
    opened = False
    try:
        if isinstance(f, str):
            f = open(f, mode)
            opened = True
        yield f
    finally:
        if opened:
            f.close()


def load_json(f: Union[str, IO[str]], **kwargs) -> Any:
    with file_descriptor(f, 'r') as fd:
        return json.load(fd, **kwargs)


def save_json(f: Union[str, IO[str]], obj: Any, **kwargs) -> None:
    with file_descriptor(f, 'w') as fd:
        json.dump(obj, fd, **kwargs)


def load_jsonl(f: Union[str, IO[str]], **kwargs) -> Any:
    with file_descriptor(f, 'r') as fd:
        return [json.loads(datum, **kwargs) for datum in fd.readlines()]


def save_jsonl(f: Union[str, IO[str]], obj: Any, **kwargs) -> None:
    with file_descriptor(f, 'w') as fd:
        fd.write('\n'.join(json.dumps(datum, **kwargs) for datum in obj))


def load_mat(f: Union[str, IO[bytes]], **kwargs) -> Any:
    return scipy.io.loadmat(f, **kwargs)


def save_mat(f: Union[str, IO[bytes]], obj: Any, **kwargs) -> None:
    scipy.io.savemat(f, obj, **kwargs)


def load_npy(f: Union[str, IO[bytes]], **kwargs) -> Any:
    return np.load(f, **kwargs)


def save_npy(f: Union[str, IO[bytes]], obj: Any, **kwargs) -> None:
    np.save(f, obj, **kwargs)


def load_npz(f: Union[str, IO[bytes]], **kwargs) -> Any:
    return np.load(f, **kwargs)


def save_npz(f: Union[str, IO[bytes]], obj: Any, **kwargs) -> None:
    np.savez(f, obj, **kwargs)


def load_pkl(f: Union[str, IO[bytes]], **kwargs) -> Any:
    with file_descriptor(f, 'rb') as fd:
        try:
            return pickle.load(fd, **kwargs)
        except UnicodeDecodeError:
            if 'encoding' in kwargs:
                raise
            return pickle.load(fd, encoding='latin1', **kwargs)


def save_pkl(f: Union[str, IO[bytes]], obj: Any, **kwargs) -> None:
    with file_descriptor(f, 'wb') as fd:
        pickle.dump(obj, fd, **kwargs)


def load_pt(f: Union[str, IO[bytes]], **kwargs) -> Any:
    return torch.load(f, **kwargs)


def save_pt(f: Union[str, IO[bytes]], obj: Any, **kwargs) -> None:
    torch.save(obj, f, **kwargs)


def load_yaml(f: Union[str, IO[str]], **kwargs) -> Any:
    with file_descriptor(f, 'r') as fd:
        return yaml.safe_load(fd, **kwargs)


def save_yaml(f: Union[str, IO[str]], obj: Any, **kwargs) -> None:
    with file_descriptor(f, 'w') as fd:
        yaml.safe_dump(obj, fd, **kwargs)


__io_registry: Dict[str, Tuple[Callable, Callable]] = {
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


def load(fpath: str, **kwargs) -> Any:
    assert isinstance(fpath, str), type(fpath)

    for extension in sorted(__io_registry.keys(), key=len, reverse=True):
        if fpath.endswith(extension):
            return __io_registry[extension][0](fpath, **kwargs)

    extension = osp.splitext(fpath)[1]
    raise NotImplementedError(f'"{extension}" is not supported.')


def save(fpath: str, obj: Any, **kwargs) -> None:
    assert isinstance(fpath, str), type(fpath)
    fs.makedir(osp.dirname(fpath))

    for extension in sorted(__io_registry.keys(), key=len, reverse=True):
        if fpath.endswith(extension):
            __io_registry[extension][1](fpath, obj, **kwargs)
            return

    extension = osp.splitext(fpath)[1]
    raise NotImplementedError(f'"{extension}" is not supported.')
