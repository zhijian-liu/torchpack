import logging
import os
import sys
import time

from termcolor import colored

__all__ = ['logger', 'get_logger', 'set_logger_dir', 'auto_set_dir', 'get_logger_dir']

_ALL_LOGGERS = []
_DEFAULT_LEVEL = logging.INFO

_LOGGER_DIR = None
_FILE_HANDLER = None


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
    _ALL_LOGGERS.append(logger)

    logger.propagate = False
    logger.setLevel(_DEFAULT_LEVEL)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter(datefmt='%m/%d %H:%M:%S'))
    logger.addHandler(handler)
    return logger


def _set_logger_file(filename):
    handler = logging.FileHandler(filename=filename, encoding='utf-8', mode='w')
    handler.setFormatter(Formatter(datefmt='%m/%d %H:%M:%S'))

    global _FILE_HANDLER
    if _FILE_HANDLER:
        for _logger in _ALL_LOGGERS:
            _logger.removeHandler(_FILE_HANDLER)
        del _FILE_HANDLER

    _FILE_HANDLER = handler
    for _logger in _ALL_LOGGERS:
        _logger.addHandler(handler)


def get_logger_dir():
    return _LOGGER_DIR


def set_logger_dir(dirname, action=None):
    global _LOGGER_DIR
    _LOGGER_DIR = dirname

    os.makedirs(dirname, exist_ok=True)
    _set_logger_file(os.path.join(dirname, time.strftime('%Y-%m-%d-%H-%M-%S') + '.log'))


def auto_set_dir(action=None, name=None):
    mod = sys.modules['__main__']
    basename = os.path.basename(mod.__file__)
    auto_dirname = os.path.join('runs', basename[:basename.rfind('.')])
    if name:
        auto_dirname += '_%s' % name if os.name == 'nt' else ':%s' % name
    set_logger_dir(auto_dirname, action=action)


def get_default_level():
    return _DEFAULT_LEVEL


def set_level(level):
    global _DEFAULT_LEVEL
    _DEFAULT_LEVEL = level
    for logger in _ALL_LOGGERS:
        logger.setLevel(level)


logger = get_logger('torchpack')
