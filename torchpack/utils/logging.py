import sys

from torchpack.utils.typing import Logger

__all__ = ['logger']


def __get_logger() -> Logger:
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr,
               level='DEBUG',
               format=('<green>[{time:YYYY-MM-DD HH:mm:ss.SSS}]</green> '
                       '<level>{message}</level>'))
    return logger


logger: Logger = __get_logger()
