import sys

__all__ = ['logger']


def __get_logger():
    from loguru import logger
    logger.remove()
    logger.add(
        sys.stdout,
        level='DEBUG',
        format=
        '<green>[{time:YYYY-MM-DD HH:mm:ss.SSS}]</green> <level>{message}</level>'
    )
    return logger


logger = __get_logger()
