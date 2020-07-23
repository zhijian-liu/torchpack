import typing

__all__ = ['Logger', 'Trainer']

Logger = None
Trainer = None

# https://stackoverflow.com/questions/39740632/python-type-hinting-without-cyclic-imports
if typing.TYPE_CHECKING:
    from loguru import Logger
    from torchpack.train import Trainer
