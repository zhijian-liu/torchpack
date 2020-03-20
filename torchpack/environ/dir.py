import os.path as osp
import time

import torchpack.utils.fs as fs
from torchpack.logging import set_handler

__all__ = [
    'get_default_dir', 'set_default_dir', 'get_logger_dir', 'set_logger_dir'
]

_default_dir = None
_logger_dir = None


def get_default_dir():
    return _default_dir


def set_default_dir(dirname):
    global _default_dir
    _default_dir = fs.makedir(dirname)
    set_logger_dir(osp.join(_default_dir, 'logging'))


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
