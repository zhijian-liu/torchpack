import os.path as osp
import time

from ..logging import logger
from ..utils import fs

__all__ = ['get_run_dir', 'set_run_dir']

_run_dir = None


def get_run_dir():
    return _run_dir


def set_run_dir(dirpath):
    global _run_dir
    _run_dir = osp.normpath(dirpath)
    fs.makedir(_run_dir)
    logger.add(
        osp.join(_run_dir, 'logging',
                 time.strftime('%Y%m%d-%H%M%S') + '.log'))
