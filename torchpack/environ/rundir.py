import os.path as osp

from torchpack import distributed as dist
from torchpack.utils import fs
from torchpack.utils.logging import logger

__all__ = ['get_run_dir', 'set_run_dir']


def get_run_dir() -> str:
    global _run_dir
    return _run_dir


def set_run_dir(dirpath: str) -> None:
    global _run_dir
    _run_dir = fs.normpath(dirpath)
    fs.makedir(_run_dir)

    prefix = '{time}'
    if dist.size() > 1:
        prefix += '_{:04d}'.format(dist.rank())
    logger.add(osp.join(_run_dir, 'logging', prefix + '.log'),
               format=('{time:YYYY-MM-DD HH:mm:ss.SSS} | '
                       '{name}:{function}:{line} | '
                       '{level} | {message}'))
