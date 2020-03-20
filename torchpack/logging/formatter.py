import logging

from termcolor import colored

__all__ = ['Formatter']


class Formatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)

    def format(self, record):
        fmt = colored('[%(asctime)s @%(filename)s:%(lineno)d]', 'green') + ' '
        if record.levelno == logging.WARNING:
            fmt += colored('WRN', 'red', attrs=['blink']) + ' '
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            fmt += colored('ERR', 'red', attrs=['blink', 'underline']) + ' '
        elif record.levelno == logging.DEBUG:
            fmt += colored('DBG', 'yellow', attrs=['blink']) + ' '
        fmt += '%(message)s'
        self._style._fmt = fmt
        return super().format(record)
