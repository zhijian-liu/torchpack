import logging
import os
import sys
import time

from termcolor import colored

from .formatter import Formatter

__all__ = ['logger', 'get_logger', 'set_handler', 'get_level', 'set_level']

_loggers = []
_handler = None
_level = logging.INFO


def get_logger(name=None, formatter=Formatter):
    logger = logging.getLogger(name)
    _loggers.append(logger)

    logger.propagate = False
    logger.setLevel(_level)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter(datefmt='%m/%d %H:%M:%S'))
    logger.addHandler(handler)
    return logger


def set_handler(filename):
    handler = logging.FileHandler(filename=filename,
                                  encoding='utf-8',
                                  mode='w')
    handler.setFormatter(Formatter(datefmt='%m/%d %H:%M:%S'))

    global _handler
    if _handler:
        for logger in _loggers:
            logger.removeHandler(_handler)
        del _handler

    _handler = handler
    for logger in _loggers:
        logger.addHandler(handler)


def get_level():
    return _level


def set_level(level):
    global _level
    _level = level
    for logger in _loggers:
        logger.setLevel(level)


logger = get_logger('torchpack')
