import os
import os.path as osp
import shutil

__all__ = ['makedir', 'remove']


def makedir(path):
    path = osp.normpath(path)
    os.makedirs(path, exist_ok=True)
    return path


def remove(path):
    path = osp.normpath(path)
    if osp.exists(path):
        if osp.isdir(path):
            return shutil.rmtree(path, ignore_errors=True)
        if osp.isfile(path):
            return os.remove(path)
    return None
