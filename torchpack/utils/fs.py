import os
import os.path as osp
import shutil

__all__ = ['makedir', 'remove']


def makedir(dirname):
    dirname = osp.normpath(dirname)
    os.makedirs(dirname, exist_ok=True)
    return dirname


def remove(path):
    path = osp.normpath(path)
    if not osp.exists(path):
        return None
    if osp.isdir(path):
        return shutil.rmtree(path, ignore_errors=True)
    elif osp.isfile(path):
        return os.remove(path)
