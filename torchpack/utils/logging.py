from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loguru import Logger

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
