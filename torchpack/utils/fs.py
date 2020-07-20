import os
import shutil
from urllib.parse import urlparse, urlunparse

__all__ = ['normpath', 'makedir', 'remove']


def normpath(path: str) -> str:
    if '://' in path:
        scheme, netloc, path, params, query, fragment = urlparse(path)
        return urlunparse(
            (scheme, netloc, os.path.normpath(path), params, query, fragment))
    else:
        return os.path.normpath(path)


def makedir(dirpath: str) -> None:
    dirpath = normpath(dirpath)
    os.makedirs(dirpath, exist_ok=True)
    if not os.path.isdir(dirpath):
        raise OSError(f'"{dirpath}" cannot be created.')


def remove(path: str) -> None:
    path = normpath(path)
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.isfile(path):
            os.remove(path)
    if os.path.exists(path):
        raise OSError(f'"{path}" cannot be removed.')
