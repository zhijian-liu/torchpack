import typing
from typing import Union

import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from torch.optim.optimizer import Optimizer

__all__ = ['Scalar', 'Tensor', 'Optimizer', 'Scheduler', 'Logger', 'Trainer']

Scalar = Union[int, float, np.integer, np.floating]
Tensor = Union[torch.Tensor, np.ndarray]

Logger = None
Trainer = None

# https://stackoverflow.com/questions/39740632/python-type-hinting-without-cyclic-imports
if typing.TYPE_CHECKING:
    from loguru import Logger
    from torchpack.train import Trainer
