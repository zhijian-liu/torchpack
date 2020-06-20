import sys

from loguru import logger

__all__ = ['logger']

logger.remove()
logger.add(
    sys.stderr,
    format=
    '<green>[{time:YYYY-MM-DD HH:mm:ss.SSS}]</green> <level>{message}</level>')
