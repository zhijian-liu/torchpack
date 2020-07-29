import os

from torchpack import distributed as dist
from torchpack.utils import fs, git
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

__all__ = ['get_run_dir', 'set_run_dir', 'auto_set_run_dir']


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
    logger.add(os.path.join(_run_dir, 'logging', prefix + '.log'),
               format=('{time:YYYY-MM-DD HH:mm:ss.SSS} | '
                       '{name}:{function}:{line} | '
                       '{level} | {message}'))


def auto_set_run_dir() -> str:
    names = ['run']
    if git.is_inside_work_tree():
        names.append(git.get_commit_hash()[:8])
    if configs:
        names.append(configs.hash()[:8])
    run_dir = os.path.join('runs', '-'.join(names))
    set_run_dir(run_dir)
    return run_dir
