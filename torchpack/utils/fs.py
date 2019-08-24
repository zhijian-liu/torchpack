import os
import os.path as osp
import shutil

__all__ = ['mkdir', 'remove']


def mkdir(dirname):
    dirname = osp.normpath(dirname)
    os.makedirs(dirname, exist_ok=True)
    return dirname


def remove(file):
    if osp.exists(file):
        if osp.isdir(file):
            return shutil.rmtree(file, ignore_errors=True)
        if osp.isfile(file):
            return os.remove(file)
    return None
