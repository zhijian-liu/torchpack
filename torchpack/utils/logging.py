import sys

__all__ = ['logger']


def _get_logger():
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr,
               format='<green>[{time}]</green> <level>{message}</level>')
    return logger


logger = _get_logger()
