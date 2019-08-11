import logging
import os
import sys
import time

from termcolor import colored

__all__ = ['logger', 'get_logger', 'set_logger_dir', 'auto_set_dir', 'get_logger_dir']

_all_loggers = []
_default_level = logging.INFO

_logger_dir = None
_file_handler = None


class Formatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%', colored=True):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.colored = colored

    def format(self, record):
        date = '[%(asctime)s @%(filename)s:%(lineno)d]'
        if self.colored:
            date = colored(date, 'green')
        msg = '%(message)s'
        if record.levelno == logging.WARNING:
            fmt = date + ' ' + colored('WRN', 'red', attrs=['blink']) + ' ' + msg
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            fmt = date + ' ' + colored('ERR', 'red', attrs=['blink', 'underline']) + ' ' + msg
        elif record.levelno == logging.DEBUG:
            fmt = date + ' ' + colored('DBG', 'yellow', attrs=['blink']) + ' ' + msg
        else:
            fmt = date + ' ' + msg
        self._style._fmt = fmt
        return super().format(record)


def get_logger(name=None, formatter=Formatter):
    logger = logging.getLogger(name)
    _all_loggers.append(logger)

    logger.propagate = False
    logger.setLevel(_default_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter(datefmt='%m/%d %H:%M:%S'))
    logger.addHandler(handler)
    return logger


def _set_logger_file(filename):
    handler = logging.FileHandler(filename=filename, encoding='utf-8', mode='w')
    handler.setFormatter(Formatter(datefmt='%m/%d %H:%M:%S'))

    global _file_handler
    if _file_handler:
        for logger in _all_loggers:
            logger.removeHandler(_file_handler)
        del _file_handler

    _file_handler = handler
    for logger in _all_loggers:
        logger.addHandler(handler)


def get_logger_dir():
    return _logger_dir


def set_logger_dir(dirname):
    global _logger_dir
    _logger_dir = dirname

    os.makedirs(dirname, exist_ok=True)
    _set_logger_file(os.path.join(dirname, time.strftime('%Y-%m-%d-%H-%M-%S') + '.log'))


# def auto_set_dir(action=None, name=None):
#     mod = sys.modules['__main__']
#     basename = os.path.basename(mod.__file__)
#     auto_dirname = os.path.join('runs', basename[:basename.rfind('.')])
#     if name:
#         auto_dirname += '_%s' % name if os.name == 'nt' else ':%s' % name
#     set_logger_dir(auto_dirname, action=action)


def get_default_level():
    return _default_level


def set_default_level(level):
    global _default_level
    _default_level = level
    for logger in _all_loggers:
        logger.setLevel(level)


logger = get_logger('torchpack')
