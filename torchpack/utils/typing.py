import typing

from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from torch.optim.optimizer import Optimizer

__all__ = ['Optimizer', 'Scheduler', 'Trainer', 'Logger']

Trainer = None
Logger = None

# https://stackoverflow.com/questions/39740632/python-type-hinting-without-cyclic-imports
if typing.TYPE_CHECKING:
    from torchpack.train import Trainer
    from loguru import Logger
