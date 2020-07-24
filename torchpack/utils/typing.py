import typing

__all__ = ['Logger', 'Dataset', 'Optimizer', 'Scheduler', 'Trainer']

Logger = None
Dataset = None
Optimizer = None
Scheduler = None
Trainer = None

# https://stackoverflow.com/questions/39740632/python-type-hinting-without-cyclic-imports
if typing.TYPE_CHECKING:
    from loguru import Logger
    from torchpack.datasets.dataset import Dataset
    from torch.optim.optimizer import Optimizer
    from torch.optim.lr_scheduler import _LRScheduler as Scheduler
    from torchpack.train import Trainer
