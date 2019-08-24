import os
import shutil

__all__ = ['mkdir', 'remove']


def mkdir(dirname):
    os.makedirs(dirname, exist_ok=True)


def remove(file):
    if os.path.exists(file):
        if os.path.isdir(file):
            shutil.rmtree(file, ignore_errors=True)
        if os.path.isfile(file):
            os.remove(file)
