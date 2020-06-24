import os.path as osp

from .. import distributed as dist
from ..utils import fs
from ..utils.logging import logger

__all__ = ['get_run_dir', 'set_run_dir']

_run_dir = None


def get_run_dir():
    return _run_dir


def set_run_dir(dirpath):
    global _run_dir
    _run_dir = fs.normpath(dirpath)
    fs.makedir(_run_dir)

    prefix = '{time}'
    if dist.world_size() > 1:
        prefix += '_{:04d}'.format(dist.world_rank())
    logger.add(osp.join(_run_dir, 'logging', prefix + '.log'),
               format=('{time:YYYY-MM-DD HH:mm:ss.SSS} | '
                       '{name}:{function}:{line} | '
                       '{level} | {message}'))
