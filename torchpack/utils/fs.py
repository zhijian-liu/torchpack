import os
import os.path as osp
import shutil

__all__ = ['makedir', 'remove']


def makedir(dirpath):
    dirpath = osp.normpath(dirpath)
    os.makedirs(dirpath, exist_ok=True)
    # TODO: update the error information
    if not osp.isdir(dirpath):
        raise IOError()


def remove(path):
    path = osp.normpath(path)
    if osp.exists(path):
        if osp.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif osp.isfile(path):
            os.remove(path)
    # TODO: update the error information
    if osp.exists(path):
        raise IOError()
