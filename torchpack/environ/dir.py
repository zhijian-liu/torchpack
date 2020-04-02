import os.path as osp
import time

import torchpack.utils.fs as fs
from torchpack.logging import set_handler

__all__ = ['get_run_dir', 'set_run_dir', 'get_logger_dir', 'set_logger_dir']

_run_dir = None
_logger_dir = None


def get_run_dir():
    return _run_dir


def set_run_dir(dirpath):
    global _run_dir
    _run_dir = osp.normpath(dirpath)
    fs.makedir(_run_dir)
    set_logger_dir(osp.join(_run_dir, 'logging'))


def get_logger_dir():
    return _logger_dir


def set_logger_dir(dirpath):
    global _logger_dir
    _logger_dir = osp.normpath(dirpath)
    fs.makedir(_logger_dir)
    set_handler(osp.join(_logger_dir, time.strftime('%Y%m%d-%H%M%S') + '.log'))
