import os
import os.path as osp
import shutil
from urllib.parse import urlparse, urlunparse

__all__ = ['normpath', 'makedir', 'remove']


def normpath(path):
    if '://' in path:
        scheme, netloc, path, params, query, fragment = urlparse(path)
        return urlunparse((scheme, netloc, osp.normpath(path), params, query, fragment))
    else:
        return osp.normpath(path)


def makedir(dirpath):
    dirpath = normpath(dirpath)
    os.makedirs(dirpath, exist_ok=True)
    if not osp.isdir(dirpath):
        raise OSError(f'"{dirpath}" cannot be created.')


def remove(path):
    path = normpath(path)
    if osp.exists(path):
        if osp.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif osp.isfile(path):
            os.remove(path)
    if osp.exists(path):
        raise OSError(f'"{path}" cannot be removed.')
