import os.path as osp
import time

import torchpack.utils.fs as fs
from torchpack.logging import set_handler

__all__ = ['get_run_dir', 'set_run_dir', 'get_logger_dir', 'set_logger_dir']

_run_dir = None
_logger_dir = None


def get_run_dir():
    return _run_dir


def set_run_dir(dirname):
    global _run_dir
    _run_dir = fs.makedir(dirname)
    set_logger_dir(osp.join(_run_dir, 'logging'))


def get_logger_dir():
    return _logger_dir


def set_logger_dir(dirname):
    global _logger_dir
    _logger_dir = fs.makedir(dirname)
    set_handler(osp.join(_logger_dir, time.strftime('%Y%m%d-%H%M%S') + '.log'))


# TODO: implement auto_set_dir
# def auto_set_dir(name=None):
#     mod = sys.modules['__main__']
#     basename = os.path.basename(mod.__file__)
#     auto_dirname = os.path.join('runs', basename[:basename.rfind('.')])
#     if name:
#         auto_dirname += '_%s' % name if os.name == 'nt' else ':%s' % name
#     set_logger_dir(auto_dirname)
